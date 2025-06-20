# hailo_runner.py
import numpy as np
import hailo_platform as hp   # システムの python3-hailort で提供されるモジュール
def preprocess_rgb224(bgr_frame: np.ndarray) -> np.ndarray:
    """OpenCV だけで 224×224 RGB 正規化 → NCHW flat float32."""
    import cv2
    rgb   = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    rgb   = cv2.resize(rgb, (224, 224))
    norm  = rgb.astype(np.float32) / 255.0
    norm  = (norm - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    nchw  = np.transpose(norm, (2, 0, 1))
    return nchw.flatten().astype(np.float32)
class HailoRunner:
    """Hailo-8L 用 HEF をロードし、推論パイプラインをラップするクラス"""
    def __init__(self, hef_path: str):
        # HEFファイルを読み込む
        self.hef = hp.HEF(hef_path)
        # デバイスを開き、モデルを configure する
        self.device = hp.VDevice()
        params = hp.ConfigureParams.create_from_hef(
            self.hef,
            interface=hp.HailoStreamInterface.PCIe
        )
        # network_group = target.configure(...) の戻り配列から最初の要素を使う
        self.network_group = self.device.configure(self.hef, params)[0]
        self.ng_params    = self.network_group.create_params()
        # 入出力 vstream 情報とバッファ設定
        ivs_info = self.hef.get_input_vstream_infos()[0]
        ovs_info = self.hef.get_output_vstream_infos()[0]
        self.input_name  = ivs_info.name
        self.output_name = ovs_info.name
        self.ivs_params = hp.InputVStreamParams.make_from_network_group(
            self.network_group,
            quantized=False,
            format_type=hp.FormatType.FLOAT32
        )
        self.ovs_params = hp.OutputVStreamParams.make_from_network_group(
            self.network_group,
            quantized=False,
            format_type=hp.FormatType.FLOAT32
        )
    def infer(self, nchw_flat: np.ndarray) -> np.ndarray:
        """
        1×C×H×W の flat array (float32) を与えて
        NumPy array を返す（例: shape=(2,) など、モデルによる）
        """
        # バッファを dict 形式で渡す
        input_data = { self.input_name: nchw_flat }
        # 推論を実行
        with self.network_group.activate(self.ng_params):
            with hp.InferVStreams(
                self.network_group,
                self.ivs_params,
                self.ovs_params
            ) as pipeline:
                results = pipeline.infer(input_data)
        return results[self.output_name].copy()
