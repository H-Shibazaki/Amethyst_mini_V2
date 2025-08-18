"""疵検知: リアルタイム推論モジュール（mini版）"""
import cv2
import numpy as np
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from check_brightness import top_brightness, bottom_brightness
from image_process import save_photo
# Hailo ランナーと前処理関数（crop 引数付き）をインポート
from hailo_runner import HailoRunner, preprocess_rgb224
import math

# -------------------- GStreamer パイプラインを動的生成 --------------------
def build_pipeline(
    width: int = 224,
    height: int = 448,
    cam_names = ("SENTECH-142124706912-24G6912", "SENTECH-142125602842-25F2842"),#1台目 / 2台目
) -> str | None:
    """
    SENTECHカメラ2台のうち、接続されているものを優先順で自動選択し、
    GStreamerパイプライン文字列を返す。
    どちらもなければ None を返す。
    """
    try:
        import gi
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst
        Gst.init(None)
        has_aravis = Gst.ElementFactory.find("aravissrc") is not None
    except Exception:
        has_aravis = False

    if has_aravis:
        for cam_name in cam_names:
            pipeline = (
                f"aravissrc camera-name={cam_name} ! "
                "videoconvert ! videoscale ! "
                f"video/x-raw,format=BGR,width={width},height={height} ! "
                "appsink drop=true max-buffers=1"
            )
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                cap.release()
                print(f"Aravisカメラ '{cam_name}' で接続しました")
                return pipeline
            cap.release()
        print("SENTECHカメラ2台とも接続されていません。")
        return None
    else:
        print("Aravisドライバが利用できません。")
        return None

# -----------------------------------------------------------------------

def draw_text_with_japanese(image, text, position, font_path, font_size, color):
    """OpenCV画像上に日本語テキストを描画"""
    pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, fill=color, font=font)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def create_gauge_bar(percent, length=20):
    """ビビリ率に応じたゲージバー文字列を生成"""
    filled = int(length * percent / 100)
    bar = "|" * filled + "-" * (length - filled)
    return f"[{bar}]\n0%{' ' * (length//2 - 2)}50%{' ' * (length//2 - 3)}100%"

def get_gauge_color(percent):
    """ビビリ率に応じたバーの色を返す"""
    if percent <= 50:
        return (0, 255, 0)
    elif percent <= 80:
        return (255, 150, 0)
    else:
        return (255, 0, 0)

def display_realtime_suiron_with_separate_windows(
    model_rasen_path: str,
    model_kurokawa_path: str,
    *,
    auto_save: bool,
    auto_save_threshold: tuple[int, int],
    cool_time_seconds: float,
    ng_rate_diff_threshold: float,
    brightness_threshold: int,
    area_ratio: float,
    use_ng_diff: bool
):
    import traceback

    runner_rasen = HailoRunner(model_rasen_path)
    runner_kurokawa = HailoRunner(model_kurokawa_path)

    # ここでパイプライン生成時にデバッグ表示
    pipeline = build_pipeline(width=224, height=448)
    print(f"【デバッグ】GStreamer pipeline: {pipeline}")
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("【エラー】cap.isOpened() == False → カメラが開けません！")
        return

    # ==== 変数の初期化 ====
    last_save_time_rasen = datetime.now()
    last_save_time_kurokawa = datetime.now()
    last_ng_rate_rasen = None
    last_ng_rate_kurokawa = None

    cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Preview", 448, 448)
    cv2.namedWindow("Information", cv2.WINDOW_NORMAL)
    font_path = "fonts/PixelMplus10-Regular.ttf"
    font_size = 24

    while True:
        try:
            ret, frame = cap.read()
            
            ####print(f"【デバッグ】cap.read() ret: {ret}")
            
            if not ret:
                print("【エラー】フレームが取得できません（cap.read()失敗）")
                break

            # フレームshape, dtype確認
            ####print(f"【デバッグ】frame.shape: {getattr(frame, 'shape', None)}, dtype: {getattr(frame, 'dtype', None)}")

            # サイズ確認（想定外なら例外）
            if not (isinstance(frame, np.ndarray) and frame.shape == (448, 224, 3)):
                print(f"【エラー】フレームshapeが不正: {getattr(frame, 'shape', None)}（想定: (448, 224, 3)）")
                break

            # テンソル生成（エラー箇所追跡）
            tensor_top = preprocess_rgb224(frame, crop='top')
            ####print(f"【デバッグ】tensor_top.shape: {tensor_top.shape}, nbytes: {tensor_top.nbytes}")

            tensor_bot = preprocess_rgb224(frame, crop='bottom')
            ####print(f"【デバッグ】tensor_bot.shape: {tensor_bot.shape}, nbytes: {tensor_bot.nbytes}")

            # 推論（try-exceptで例外追跡）
            try:
                probs_r = runner_rasen.infer(tensor_top)
                ####print("【デバッグ】らせん推論raw出力:", probs_r)
            except Exception as e:
                print("【エラー】らせん推論で例外発生:")
                traceback.print_exc()
                break

            try:
                probs_k = runner_kurokawa.infer(tensor_bot)
                ####print("【デバッグ】黒皮推論raw出力:", probs_k)
            except Exception as e:
                print("【エラー】黒皮推論で例外発生:")
                traceback.print_exc()
                break

            # らせん疵モデル（上段）
            if probs_r.sum() == 0:
                print("【警告】らせん: probs_r.sum()==0 → ÷0対策ブランチに入ります（ng_r=0.0, ok_r=0.0）")
                ng_r, ok_r = 0.0, 0.0
            else:
                ng_r, ok_r = probs_r / probs_r.sum()
                
            pred_r = "NG" if ng_r > ok_r else "OK"
            val_r = int(ng_r * 100) if not math.isnan(ng_r) else 0

            # 黒皮残りモデル（下段）
            if probs_k.sum() == 0:
                print("【警告】黒皮: probs_k.sum()==0 → ÷0対策ブランチに入ります（ng_k=0.0, ok_k=0.0）")
                ng_k, ok_k = 0.0, 0.0  # または適切な初期値
            else:
                ng_k, ok_k = probs_k / probs_k.sum()

            pred_k = "NG" if ng_k > ok_k else "OK"
            val_k = int(ng_k * 100) if not math.isnan(ng_k) else 0

            # -------- 描 画 --------
            result_text = f"【らせん】{pred_r} {val_r}%  【黒皮】{pred_k} {val_k}%"
            result_img = np.zeros((180, 600, 3), dtype=np.uint8)

            # メインテキスト
            result_img = draw_text_with_japanese(
                result_img, result_text, (10, 10),
                font_path, font_size, (255, 255, 255)
            )
            # ゲージ：らせん
            result_img = draw_text_with_japanese(
                result_img, create_gauge_bar(val_r, length=20),
                (10, 50), font_path, font_size, get_gauge_color(val_r)
            )
            # ゲージ：黒皮
            result_img = draw_text_with_japanese(
                result_img, create_gauge_bar(val_k, length=20),
                (320, 50), font_path, font_size, get_gauge_color(val_k)
            )
            # 検出マテリアル
            if not top_brightness(frame, brightness_threshold, area_ratio) or \
               not bottom_brightness(frame, brightness_threshold, area_ratio):
                detection_text = "材料: ナシ"
                color_det = (0, 128, 0)
            else:
                detection_text = "材料: アリ"
                color_det = (0, 255, 0)
            result_img = draw_text_with_japanese(
                result_img, detection_text, (10, 130),
                font_path, font_size, color_det
            )
            cv2.imshow("Preview", frame)
            cv2.imshow("Information", result_img)

            key = cv2.waitKey(1) & 0xFF

            # --- 手動保存 ---
            if key == ord('s'):
                save_photo(frame[:224, :], f"rasen_{pred_r}", f"{val_r}", "images/rasen/Photo/manual")
                print(f"[manual] らせん疵: {pred_r} ({val_r}%) を images/rasen/Photo/manual に保存しました。")
            elif key == ord('d'):
                save_photo(frame[224:, :], f"kurokawa_{pred_k}", f"{val_k}", "images/kurokawa/Photo/manual")
                print(f"[manual] 黒皮残り: {pred_k} ({val_k}%) を images/kurokawa/Photo/manual に保存しました。")

            # --- 自動保存 ---
            now = datetime.now()
            # らせん疵（上半分）自動保存
            if (auto_save and detection_text == "材料: アリ"
                and auto_save_threshold[0] <= val_r <= auto_save_threshold[1]):
                elapsed_rasen = (now - last_save_time_rasen).total_seconds()
                ng_diff_rasen = abs(val_r - last_ng_rate_rasen) if last_ng_rate_rasen is not None else float('inf')
                if elapsed_rasen >= cool_time_seconds and (not use_ng_diff or ng_diff_rasen > ng_rate_diff_threshold):
                    save_photo(frame[:224, :], f"rasen_{pred_r}", f"{val_r}", "images/rasen/Photo/auto")
                    print(f"[auto] らせん疵: {pred_r} ({val_r}%) を images/rasen/Photo/auto に保存しました。")
                    last_save_time_rasen = now
                    last_ng_rate_rasen = val_r

            # 黒皮残り（下半分）自動保存
            if (auto_save and detection_text == "材料: アリ"
                and auto_save_threshold[0] <= val_k <= auto_save_threshold[1]):
                elapsed_kurokawa = (now - last_save_time_kurokawa).total_seconds()
                ng_diff_kurokawa = abs(val_k - last_ng_rate_kurokawa) if last_ng_rate_kurokawa is not None else float('inf')
                if elapsed_kurokawa >= cool_time_seconds and (not use_ng_diff or ng_diff_kurokawa > ng_rate_diff_threshold):
                    save_photo(frame[224:, :], f"kurokawa_{pred_k}", f"{val_k}", "images/kurokawa/Photo/auto")
                    print(f"[auto] 黒皮残り: {pred_k} ({val_k}%) を images/kurokawa/Photo/auto に保存しました。")
                    last_save_time_kurokawa = now
                    last_ng_rate_kurokawa = val_k

            if key == ord('q'):
                break

        except Exception as e:
            print("【致命的エラー】ループ内で例外発生:")
            import traceback
            traceback.print_exc()
            break

    cap.release()
    cv2.destroyAllWindows()
