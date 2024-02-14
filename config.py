# config.py
from src.model_builder import HandFlapping_GRUModel, HM_GRUModel, PS_GRUModel
import torch.optim as optim
import torch.nn as nn

# 모델 정의
models = {
    'hand_flapping': HandFlapping_GRUModel,
    'hm': HM_GRUModel,
    'ps': PS_GRUModel
}

# 옵티마이저 정의
optimizers = {
    'hand_flapping': optim.Adam(models['hand_flapping'].parameters(), lr=0.001),
    'hm': optim.Adam(models['hm'].parameters(), lr=0.001),
    'ps': optim.Adam(models['ps'].parameters(), lr=0.001)
}

# 손실 함수 정의
criterions = {
    'hand_flapping': nn.BCEWithLogitsLoss(),
    'hm': nn.CrossEntropyLoss(),
    'ps': nn.CrossEntropyLoss()
}

# 데이터셋 타입 정의
dataset_types = {
    'hm': 'hand_manipulation',
    'ps': 'pose_stability'
}

# 루트 디렉토리 정의
root_dirs = {
    'hm': [f"/project/DB/AssessmentClip/HM/HM{i}" for i in range(1, 9)],
    'ps': [f"/project/DB/AssessmentClip/PS/PS{i}" for i in range(1, 4)]
}


hand_manipulation_landmark_names= [
            "left_hand_wrist", "left_thumb_cmc", "left_thumb_mcp", "left_thumb_ip", "left_thumb_tip",
            "left_indexfinger_mcp", "left_indexfinger_pip", "left_indexfinger_dip", "left_indexfinger_tip",
            "left_middlefinger_mcp", "left_middlefinger_pip", "left_middlefinger_dip", "left_middlefinger_tip",
            "left_ringfinger_mcp", "left_ringfinger_pip", "left_ringfinger_dip", "left_ringfinger_tip",
            "left_pinkyfinger_mcp", "left_pinkyfinger_pip", "left_pinkyfinger_dip", "left_pinkyfinger_tip",
            "right_hand_wrist", "right_thumb_cmc", "right_thumb_mcp", "right_thumb_ip", "right_thumb_tip",
            "right_indexfinger_mcp", "right_indexfinger_pip", "right_indexfinger_dip", "right_indexfinger_tip",
            "right_middlefinger_mcp", "right_middlefinger_pip", "right_middlefinger_dip", "right_middlefinger_tip",
            "right_ringfinger_mcp", "right_ringfinger_pip", "right_ringfinger_dip", "right_ringfinger_tip",
            "right_pinkyfinger_mcp", "right_pinkyfinger_pip", "right_pinkyfinger_dip", "right_pinkyfinger_tip"
        ]
pose_stability_landmark_names = []