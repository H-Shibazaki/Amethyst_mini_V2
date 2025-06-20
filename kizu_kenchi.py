"""疵検知の全て"""

import cv2
import numpy as np
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

from check_brightness import top_brightness, bottom_brightness
from image_process import save_photo
# from common_utils import get_data_transforms    # ← PyTorch 依存の前処理は不要になったので削除
from hailo_runner import HailoRunner, preprocess_rgb224

# 既定 HEF をあらかじめ常駐させたい場合は下記を利用
runner_rasen    = HailoRunner("/home/pi/models/rasen.hef")
runner_kurokawa = HailoRunner("/home/pi/models/kurokawa.hef")


def draw_text_with_japanese(image, text, position, font_path, font_size, color_result):
    """Pillowを使用して日本語テキストを描画"""
    # OpenCVの画像をPillow形式に変換
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    # 日本語フォントを設定
    font = ImageFont.truetype(font_path, font_size)

    # 日本語テキストを描画
    draw.text(position, text, fill=color_result, font=font)

    # Pillow形式をOpenCV形式に戻す
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def create_gauge_bar(percent, length=20):
    """パーセンテージに応じたゲージバー（文字列）を生成"""
    filled_length = int(length * percent / 100)
    bar = "|" * filled_length + "-" * (length - filled_length)
    return f"[{bar}]\n0%        50%       100%"


def get_gauge_color(percent):
    """ビビリ率に応じたゲージの色を返す"""
    if percent <= 50:
        return (0, 255, 0)  # 緑
    elif percent <= 80:
        return (255, 150, 0)  # オレンジ
    else:
        return (255, 0, 0)  # 赤


"""リアルタイムで推論処理を行い、結果をウィンドウに表示する"""
def display_realtime_suiron_with_separate_windows(
        model_save_path, auto_save=True, auto_save_threshold=(50, 100),
        cool_time_seconds=1, ng_rate_diff_threshold=5,
        brightness_threshold=15, area_ratio=0.1
        ):
    auto_save_threshold = auto_save_threshold  # 自動保存のNG率範囲（デフォルトは50%～100%）

    # HEF をロードして推論ランナーを生成
    runner = HailoRunner(model_save_path)      # ★ 旧 HefModel → HailoRunner へ置換 :contentReference[oaicite:0]{index=0}

    cap = cv2.VideoCapture(0)
    last_save_time = datetime.now()  # 最後に保存した時間を初期化
    last_ng_rate = None             # 前回のNG率を初期化
    if not cap.isOpened():
        print("カメラを開けませんでした")
        return

    result_img = np.zeros((200, 300, 3), dtype=np.uint8)

    # ウィンドウをリサイズ可能に設定
    cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Preview", 448, 448)  # ウィンドウの初期サイズを設定

    while True:
        ret, frame = cap.read()

        if not ret:
            print("フレームを取得できませんでした")
            break

        cv2.imshow("Preview", frame)  # プレビューウィンドウを表示

        # 被写体の検知
        if not top_brightness(frame, brightness_threshold, area_ratio):  # 平均輝度が規定値以下
            detection_text = "材料: ナシ"
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            color_detection = (0, 128, 0)  # 暗い緑

        elif not bottom_brightness(frame, brightness_threshold, area_ratio):
            detection_text = "材料: ナシ"
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            color_detection = (0, 128, 0)  # 暗い緑

        else:
            detection_text = "材料: アリ"
            color_detection = (0, 255, 0)  # 緑

        # ★ HEFモデルによる推論部分（NumPy + HailoRT）
        tensor = preprocess_rgb224(frame)
        probs = runner.infer(tensor)
        ng_prob, ok_prob = probs / probs.sum()
        predicted_class = 'NG' if ng_prob > ok_prob else 'OK'
        bibiri_value = int(ng_prob * 100)

        result_text = f"判定: {predicted_class}, NG率: {bibiri_value}%"
        result_img = np.zeros((150, 300, 3), dtype=np.uint8)
        color_result = (0, 255, 0) if predicted_class == "OK" else (255, 0, 0)

        # フォント設定
        font_path = "fonts/PixelMplus10-Regular.ttf"  # 日本語対応フォントを指定
        font_size = 24

        # Informationウィンドウの描画
        result_img = draw_text_with_japanese(result_img, result_text, (10, 10), font_path, font_size, color_result)
        gauge_text = create_gauge_bar(bibiri_value)
        gauge_color = get_gauge_color(bibiri_value)
        result_img = draw_text_with_japanese(result_img, gauge_text, (10, 45), font_path, 24, gauge_color)
        result_img = draw_text_with_japanese(result_img, detection_text, (10, 110), font_path, font_size, color_detection)

        cv2.imshow("Information", result_img)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            frame_copy = frame.copy()
            save_photo(frame_copy, predicted_class, bibiri_value, "images/Photo/manual")

        # 自動保存機能
        if auto_save and detection_text == "材料: アリ" and auto_save_threshold[0] <= bibiri_value <= auto_save_threshold[1]:
            current_time = datetime.now()
            elapsed_time = (current_time - last_save_time).total_seconds()
            ng_rate_diff = abs(bibiri_value - last_ng_rate) if last_ng_rate is not None else float('inf')

            # クールタイムとNG率差分条件を確認
            if elapsed_time < cool_time_seconds:
                print(f"クールタイム未達: elapsed_time={elapsed_time:.2f}s < {cool_time_seconds:.2f}s")
            elif ng_rate_diff <= ng_rate_diff_threshold:
                print(f"NG率差分未達: ng_rate_diff={ng_rate_diff:.2f} <= {ng_rate_diff_threshold:.2f}")
            else:
                # 保存処理
                save_photo(frame.copy(), predicted_class, bibiri_value, "images/photo/auto")
                last_save_time = current_time
                last_ng_rate = bibiri_value

        if key == ord('q'):
            break
