from pathlib import Path
import numpy as np
import cv2
import hailo_platform as hp           # python3-hailort

# ---------- 画像前処理 -------------------------------------------------
def preprocess_rgb224(bgr448x224: np.ndarray, crop: str = "full") -> np.ndarray:
    """448×224 BGR → 224×224 RGB flat(uint8)"""
    if crop == "top":
        roi = bgr448x224[0:224, :224]
    elif crop == "bottom":
        roi = bgr448x224[224:448, :224]
    else:                               # central 224 px
        y0 = (bgr448x224.shape[0] - 224) // 2
        roi = bgr448x224[y0:y0 + 224, :224]
    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    # デバッグ出力
    #print(f"preprocess_rgb224: crop={crop} roi.shape={roi.shape} dtype={roi.dtype}")
    tensor = rgb.astype(np.uint8).ravel()
    #print(f"tensor.shape={tensor.shape} nbytes={tensor.nbytes}")
    return tensor          # 150 528 bytes

# ----------------------------------------------------------------------

_GLOBAL_VD = hp.VDevice()   # プロセス内共有デバイス

class HailoRunner:
    """Input-1 / Output-1 HEF 用軽量ラッパ"""

    def __init__(self, hef_path: str, vdev: hp.VDevice | None = None):
        self.hef = hp.HEF(str(Path(hef_path).expanduser()))
        self.dev = vdev or _GLOBAL_VD

        cfg = hp.ConfigureParams.create_from_hef(
            self.hef, interface=hp.HailoStreamInterface.PCIe)
        self.ng = self.dev.configure(self.hef, cfg)[0]

        # ストリーム情報 -------------------------------------------------
        ivs_info = self.hef.get_input_vstream_infos()[0]
        ovs_info = self.hef.get_output_vstream_infos()[0]

        self.in_name,  self.in_shape  = ivs_info.name, ivs_info.shape
        self.out_name, self.out_shape = ovs_info.name, ovs_info.shape

        # ---------- フレームサイズ（バイト数 & 要素数）を手動計算 ---------------------
        # dtype は量子化入力が UINT8、出力が FLOAT32
        self.in_byte_size  = int(np.prod(self.in_shape)) * np.dtype(np.uint8).itemsize
        self.out_size      = int(np.prod(self.out_shape))
        self.out_byte_size = self.out_size * np.dtype(np.float32).itemsize
        # ---------------------------------------------------------------

        # --- すべての VStreamParams をそのまま取得 ---------------------
        in_params  = hp.InputVStreamParams.make_from_network_group(
            self.ng, quantized=True,  format_type=hp.FormatType.UINT8)
        out_params = hp.OutputVStreamParams.make_from_network_group(
            self.ng, quantized=False, format_type=hp.FormatType.UINT8)

        # --- VStream ホルダーを生成し **enter** しておく ---------------
        _in_cm  = hp.InputVStreams (self.ng, in_params)
        _out_cm = hp.OutputVStreams(self.ng, out_params)
        self._in_holder  = _in_cm.__enter__()
        self._out_holder = _out_cm.__enter__()

        # 先頭ストリーム取得（失敗時は例外）
        try:
            self.ivs = next(iter(self._in_holder))
            self.ovs = next(iter(self._out_holder))
        except StopIteration:
            raise RuntimeError("VStream が 0 本です。HEF または SDK バージョンを確認してください。")

        # -------- ActivationParams を必ず生成 --------
        self.act_params = self.ng.create_params()

        # コンテキストを抜ける（__exit__）処理を忘れないようにデストラクタで管理
        self._in_cm  = _in_cm
        self._out_cm = _out_cm

        print("HEF in_shape:", self.in_shape, "out_shape:", self.out_shape)
        print("in_byte_size:", self.in_byte_size, "out_byte_size:", self.out_byte_size)
        print("Input dtype: uint8? Out dtype: float32?")
        print("Input stream name:", self.in_name, "Input shape:", self.in_shape)
        print("Output stream name:", self.out_name, "Output shape:", self.out_shape)

    def __del__(self):
        # プロセス終了時にストリームを解放
        try:
            self._in_cm.__exit__(None, None, None)
            self._out_cm.__exit__(None, None, None)
        except Exception:
            pass   # 終了時の解放失敗は無視

    # ------------------------------------------------------------------
    def infer(self, flat_tensor: np.ndarray) -> np.ndarray:
        # 1バッチ分の次元を追加
        data = flat_tensor.reshape((1,)+self.in_shape)
        #print("SEND FINAL: type=", type(data),
        #     "dtype=", data.dtype,
        #     "shape=", data.shape,
        #     "nbytes=", data.nbytes,
        #     "C_CONTIGUOUS?", data.flags['C_CONTIGUOUS'])
        #print("SEND FINAL first 10 values:", data.flat[:10])
        with self.ng.activate(self.act_params):
            self.ivs.send(data)
            out_buf = self.ovs.recv()
        return out_buf.copy()
