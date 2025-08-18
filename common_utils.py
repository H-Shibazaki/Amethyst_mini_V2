"""汎用的に使う関数のまとめ"""

# import torch
# from torchvision import transforms   # ← PyTorch / TorchVision を使用しない構成に変更

"""GPU を使って演算するように指定（Pi では CPU 固定）"""
def get_device() -> str:
    return "cpu"

"""画像前処理（旧 PyTorch transforms 互換ダミー）

   * 現在の Raspberry Pi + HailoRT ビルドでは未使用。
   * もし誤って呼ばれた場合は例外で知らせる。
"""
def get_data_transforms():
    raise NotImplementedError("get_data_transforms() は HailoRT 版で使用しません。")
