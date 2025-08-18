"""モデルファイル（.pth）を作成"""
#import torch
import torch.nn as nn
#import torch.optim as optim
#from torchvision import models, datasets
#from torch.utils.data import DataLoader
from torchvision.models import ResNet34_Weights  # 最新の重み設定をインポート

from common_utils import get_data_transforms, get_device


"""初期状態のモデルを作成"""
def create_model():
    weights = ResNet34_Weights.IMAGENET1K_V1  # 重みの選択
    model = models.resnet34(weights=weights)  # 最新仕様で重みを指定
    num_features = model.fc.in_features  # 最終層の入力特徴量数を取得
    model.fc = nn.Linear(num_features, 2)  # 出力層を2クラス（NG, OK）に変更
    return model 
