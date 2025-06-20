import hailort as hp
import numpy as np
import cv2                       # 画像前処理で必要

def preprocess_rgb224(bgr_frame: np.ndarray) -> np.ndarray:
    """OpenCV だけで 224×224 RGB 正規化 → NCHW flat float32."""
    rgb   = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    rgb   = cv2.resize(rgb, (224, 224))          # Hailo は 224×224 固定
    norm  = rgb.astype(np.float32) / 255.0
    norm  = (norm - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    nchw  = np.transpose(norm, (2, 0, 1))        # HWC → CHW
    return nchw.flatten().astype(np.float32)

class HailoRunner:
    """HEF をロードして 1×3×224×224 Float32 を推論する極小ラッパー"""
    def __init__(self, hef_path: str):
        self.rt  = hp.InferenceRunner(hef_path)
        self.inp = self.rt.get_input_tensors()[0]
        self.out = self.rt.get_output_tensors()[0]

    def infer(self, nchw_flat: np.ndarray) -> np.ndarray:
        self.inp[:] = nchw_flat
        self.rt.infer()
        return self.out.copy()  # shape=(2,)
