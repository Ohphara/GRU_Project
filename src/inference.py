# inference.py
import torch
from src.model_builder import HandFlapping_GRUModel, HM_GRUModel, PS_GRUModel

class Inference:
    def __init__(self, model_path, model_type):
        self.model_path = model_path
        self.model = self.load_model(model_type)
        self.model.eval()  # 모델을 평가 모드로 설정

    def load_model(self, model_type):
        if model_type == 'hm':
            model = HM_GRUModel
        elif model_type == 'ps':
            model = PS_GRUModel
        else:
            raise ValueError("Model type should be 'hm' or 'ps'")
        
        model.load_state_dict(torch.load(self.model_path))
        return model

    def predict(self, input_data):
        with torch.no_grad():  # 그래디언트 계산을 비활성화
            outputs = self.model(input_data)
            _, predicted = torch.max(outputs, 1)
            return predicted + 1  # 1부터 시작하는 레이블로 변환