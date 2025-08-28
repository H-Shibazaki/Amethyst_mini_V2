# preprocess.py
# 疵強調の共通入口を提供する。PIL(Image)↔OpenCV(BGR)間の変換もここで完結させる。

from PIL import Image
import numpy as np
import cv2

# デフォルト方式（"horizontal": 横線強調, "blackhat": ブラックハット）
DEFAULT_METHOD = "horizontal"

class DefectEnhanceTransform:
    """pickle可能なPIL→PIL変換。DataLoaderのworkerでも安全"""
    def __init__(self, method="horizontal", **kwargs):
        self.method = method
        self.kwargs = kwargs
    def __call__(self, img):
        return enhance_defect(img, method=self.method, **self.kwargs)

def _to_bgr(image: Image.Image) -> np.ndarray:
    """PILのRGB画像をOpenCVのBGR配列に変換"""
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def _to_pil(bgr: np.ndarray) -> Image.Image:
    """OpenCVのBGR配列をPILのRGB画像に戻す"""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def emphasize_horizontal_lines_bgr(bgr: np.ndarray, rotate_deg=None) -> np.ndarray:
    """横線を強調したBGR画像を返す"""
    if rotate_deg is not None and abs(rotate_deg) > 0.1:
        h, w = bgr.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), rotate_deg, 1.0)
        bgr = cv2.warpAffine(bgr, M, (w, h), flags=cv2.INTER_LINEAR)

    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 1))
    tophat = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, se)

    grad_y = cv2.Sobel(g, cv2.CV_16S, dx=0, dy=1, ksize=3)
    grad_y = cv2.convertScaleAbs(grad_y)

    th = cv2.normalize(tophat, None, 0, 255, cv2.NORM_MINMAX)
    gy = cv2.normalize(grad_y, None, 0, 255, cv2.NORM_MINMAX)
    enh = cv2.addWeighted(th, 0.6, gy, 0.4, 0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enh = clahe.apply(enh)

    out = cv2.merge([enh, enh, enh])
    return out

# ===== 黒皮：指定の実装に置換（縦長SE、横方向なし） =====
def blackhat_bgr(bgr: np.ndarray, ksize: int = 15, to_rgb3: bool = True) -> np.ndarray:
    """
    ブラックハット（縦長SE）で「明るい背景に対して局所的に暗い縦塊」を抽出・強調する。
    - クロージング（膨張→収縮）で暗い凹みを“埋める”
    - (クロージング結果 − 元画像) を取り、暗い縦塊を白で浮かせる
    Parameters
    ----------
    bgr : np.ndarray
        入力画像（BGR, uint8）
    ksize : int, default=31
        縦方向のカーネル高さ（例: 15/21/31/61/101）。横幅は3固定
    to_rgb3 : bool, default=True
        出力を3chにして可視化・下流互換性を確保
    Returns
    -------
    np.ndarray
        ブラックハット強調画像（uint8, 1ch or 3ch）
    """
    # 1) グレイ化
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # 2) 縦長SE: 3×ksize
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (3, ksize))
    # 3) ブラックハット
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, se)
    # 4) 0-255正規化
    blackhat = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)
    # 5) 出力
    if to_rgb3:
        return cv2.merge([blackhat, blackhat, blackhat])
    else:
        return blackhat

def enhance_defect(image: Image.Image, method: str = None, **kwargs) -> Image.Image:
    """
    疵強調の共通入口（PIL→PIL）
    - horizontal: 横線強調（3ch BGR返し）
    - blackhat  : 黒皮強調（常に3chで返すよう to_rgb3=True を強制し下流整合）
    """
    method = method or DEFAULT_METHOD
    bgr = _to_bgr(image)
    if method == "blackhat":
        # 下流（ToTensor/Normalize）が3ch前提のため to_rgb3=True を強制
        kwargs = dict(kwargs)
        kwargs["to_rgb3"] = True
        out = blackhat_bgr(bgr, **kwargs)
    else:
        out = emphasize_horizontal_lines_bgr(bgr, **kwargs)
    return _to_pil(out)

