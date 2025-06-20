"""画像ファイルに関する処理"""

import os
import cv2
from datetime import datetime
from hailo_runner import HailoRunner, preprocess_rgb224  # 新モジュール

# """【前処理】画像の左右25%ずつを黒く塗りつぶす"""
# def mask_left_right(frame, mask_ratio=0.25):
#     height, width, _ = frame.shape
#     mask_width = int(width * mask_ratio)
#     frame[:, :mask_width] = 0
#     frame[:, -mask_width:] = 0
#     return frame

"""【前処理＋推論】NumPy & HailoRT 版"""
def maeshori_and_predict(frame_bgr):
    runner = HailoRunner(model_hef_path)
    tensor = preprocess_rgb224(frame_bgr)
    probs  = runner.infer(tensor)
    ng_prob, ok_prob = probs / probs.sum()

    predicted = 'NG' if ng_prob > ok_prob else 'OK'
    result_txt = f"判定: {predicted}, ビビリ率: {int(ng_prob*100)}%"
    return result_txt, predicted, int(ng_prob*100), frame_bgr

"""画像を保存する際のファイル名を指定"""
def save_photo(frame, predicted_class, bibiri_value, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    date_str = datetime.now().strftime("%y%m%d_%H%M")  # 「日付_時間_分」形式
    file_count = len([f for f in os.listdir(save_dir) if f.endswith(".bmp")]) + 1
    file_name = f"{date_str}_{file_count}_{predicted_class}_{bibiri_value}.bmp"
    save_path = os.path.join(save_dir, file_name)
    cv2.imwrite(save_path, frame)
    print(f"写真を保存しました: {save_path}")
