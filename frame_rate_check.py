import time
import cv2
import numpy as np
from hailo_runner import HailoRunner, preprocess_rgb224


def build_pipeline(width: int = 224, height: int = 448, cam_id: str = None) -> str:
    try:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        Gst.init(None)
        has_aravis = Gst.ElementFactory.find("aravissrc") is not None
    except Exception:
        has_aravis = False

    if has_aravis and cam_id is not None:
        return (
            f"aravissrc camera-name={cam_id} ! "
            "videoconvert ! videoscale ! "
            f"video/x-raw,format=BGR,width={width},height={height} ! "
            "appsink drop=true max-buffers=1"
        )
    elif has_aravis:
        return (
            "aravissrc ! "
            "videoconvert ! videoscale ! "
            f"video/x-raw,format=BGR,width={width},height={height} ! "
            "appsink drop=true max-buffers=1"
        )
    else:
        return (
            "v4l2src device=/dev/video0 ! "
            "videoconvert ! videoscale ! "
            f"video/x-raw,width={width},height={height} ! "
            "appsink drop=true max-buffers=1"
        )


model_rasen_path = "models/net_250611_ResNet34_rasen.hef"
model_kurokawa_path = "models/net_250619_ResNet34_kurokawa.hef"

runner_rasen = HailoRunner(model_rasen_path)
runner_kurokawa = HailoRunner(model_kurokawa_path)

pipeline = build_pipeline(width=224, height=448)
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
if not cap.isOpened():
    print("カメラを開けませんでした")
    exit()

while True:
    loop_start = time.time()
    print("\n---- 新フレーム ----")
    print(f"[{time.strftime('%H:%M:%S', time.localtime(loop_start))}] ループ開始")

    # フレーム取得
    t_get0 = time.time()
    ret, frame = cap.read()
    t_get1 = time.time()
    print(
        f"[{time.strftime('%H:%M:%S', time.localtime(t_get1))}] フレーム取得完了  所要: {t_get1 - t_get0:.4f} 秒"
    )

    if not ret:
        print("フレームを取得できませんでした")
        break

    # 必要ならリサイズ
    t_resize0 = time.time()
    frame = cv2.resize(frame, (224, 448))
    t_resize1 = time.time()
    print(
        f"[{time.strftime('%H:%M:%S', time.localtime(t_resize1))}] リサイズ完了     所要: {t_resize1 - t_resize0:.4f} 秒"
    )

    # 前処理（上下分割＆テンソル化）
    t_pre0 = time.time()
    tensor_rasen = preprocess_rgb224(frame, crop="top")
    tensor_kurokawa = preprocess_rgb224(frame, crop="bottom")
    t_pre1 = time.time()
    print(
        f"[{time.strftime('%H:%M:%S', time.localtime(t_pre1))}] 前処理完了       所要: {t_pre1 - t_pre0:.4f} 秒"
    )

    # 推論：上（らせん疵）
    t_inf_r0 = time.time()
    probs_rasen = runner_rasen.infer(tensor_rasen)
    t_inf_r1 = time.time()
    print(
        f"[{time.strftime('%H:%M:%S', time.localtime(t_inf_r1))}] 推論(上)完了    所要: {t_inf_r1 - t_inf_r0:.4f} 秒"
    )

    # ÷0チェック
    if probs_rasen.sum() == 0:
        ng_r, ok_r = 0.0, 0.0
    else:
        ng_r, ok_r = probs_rasen / probs_rasen.sum()
        
    predicted_class_rasen = "NG" if ng_r > ok_r else "OK"
    rasen_value = int(ng_r * 100)

    # 推論：下（黒皮残り）
    t_inf_k0 = time.time()
    probs_kurokawa = runner_kurokawa.infer(tensor_kurokawa)
    t_inf_k1 = time.time()
    print(
        f"[{time.strftime('%H:%M:%S', time.localtime(t_inf_k1))}] 推論(下)完了    所要: {t_inf_k1 - t_inf_k0:.4f} 秒"
    )

    # ÷0チェック
    if probs_kurokawa.sum() == 0:
        ng_k, ok_k = 0.0, 0.0
    else:
        ng_k, ok_k = probs_kurokawa / probs_kurokawa.sum()

    predicted_class_kurokawa = "NG" if ng_k > ok_k else "OK"
    kurokawa_value = int(ng_k * 100)

    loop_end = time.time()
    fps = 1 / (loop_end - loop_start) if (loop_end - loop_start) > 0 else 0

    print(f"[{time.strftime('%H:%M:%S', time.localtime(loop_end))}] ループ終了")
    print(f"合計ループ時間: {loop_end - loop_start:.4f} 秒 (FPS: {fps:.2f})")
    print(f"らせん疵: {predicted_class_rasen} ({rasen_value}%)")
    print(f"黒皮残り: {predicted_class_kurokawa} ({kurokawa_value}%)")

    if fps < 30:
        print("⚠ 推論処理が30fpsに追いついていません")
    else:
        print(":white_check_mark: 推論処理は30fpsに追いついています")

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
