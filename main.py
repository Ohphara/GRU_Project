# main.py
import os
from src.data_loader import create_dataset_and_loader, InferenceDataset, DataLoader
from src.train import ModelTrain
from src.inference import Inference
from config import optimizers, criterions, models, dataset_types, root_dirs
import torch
import json

# 모델을 저장할 경로를 지정합니다.
MODEL_SAVE_DIR = '/YourProjectPATH/models'
TDVideos_path = '/YourProjectPATH/DB/TDvideos'
ASDVideos_path = '/YourProjectPATH/DB/ASDvideos'

def main(mode):
    # 사용자가 선택한 모드에 따라 훈련 모드 또는 추론 모드를 실행합니다.
    if mode == "train":
        execute_train_mode()
    elif mode == "inference":
        execute_inference_mode()
    else:
        print("올바른 실행 모드를 선택하지 않았습니다. (train/inference 중 하나를 선택해야 합니다.)")

def execute_train_mode():
    # 사용자로부터 훈련할 데이터셋의 타입을 입력받습니다.
    dataset_type = input("훈련할 데이터셋 타입을 선택하세요 (hm/ps): ").strip().lower()

    # 데이터셋 타입에 따라 모델, 옵티마이저, 손실 함수를 선택합니다.
    model = models[dataset_type]
    optimizer = optimizers[dataset_type]
    criterion = criterions[dataset_type]

    # 각 디렉토리의 데이터셋을 생성합니다.
    datasets = [create_dataset_and_loader(root_dir, dataset_types[dataset_type])[0] 
                for root_dir in root_dirs[dataset_type]]

    # 생성한 데이터셋들을 하나의 데이터셋으로 합칩니다.
    combined_dataset = torch.utils.data.ConcatDataset(datasets)

    # 합친 데이터셋으로 데이터 로더를 생성합니다.
    dataloader = DataLoader(combined_dataset, batch_size=1, shuffle=True)

    # 모델 훈련 객체를 생성하고 훈련을 시작합니다.
    model_train = ModelTrain(model=model, dataloader=dataloader, criterion=criterion, optimizer=optimizer, num_epochs=10)
    model_train.train()

    # 훈련된 모델을 저장합니다.
    model_train.save_model(MODEL_SAVE_DIR, f'{dataset_type}_model.pth')

def execute_inference_mode():
    # 사용자로부터 분석할 비디오 타입, 클립 번호, 추론할 데이터셋 타입을 입력받습니다.
    video_type = input("분석할 비디오 타입을 선택하세요 (TD/ASD): ").strip().lower()
    clip_number = input("분석할 클립 넘버를 선택하세요: ").strip()
    dataset_type = input("추론할 데이터셋 타입을 선택하세요 (HM/PS): ").strip().lower()
    video_path = TDVideos_path if video_type == "td" else ASDVideos_path

    # 저장된 모델을 로드합니다.
    model_path = os.path.join(MODEL_SAVE_DIR, f'{dataset_type}_model.pth')
    inference_model = Inference(model_path=model_path, model_type=dataset_type)

    # 추론할 데이터셋을 생성합니다.
    inference_dataset = InferenceDataset(root_dir=video_path, dataset_type=dataset_types[dataset_type], clip_number=clip_number)

    # 결과를 생성하고 JSON 파일로 저장합니다.
    results = generate_results(inference_dataset, inference_model)
    filename = f'/YourProjectPATH/predict_data/InferenceResult_{video_type.upper()}Video_{clip_number}_{dataset_type}.json'
    with open(filename, 'w') as f:
        json.dump(results, f)

def generate_results(inference_dataset, inference_model):
    # 결과를 저장할 리스트를 초기화합니다.
    results = []
    for i, input_data in enumerate(inference_dataset):
        # 입력 데이터를 윈도우로 나눕니다.
        windows = create_windows(input_data, window_size=60, overlap_ratio=0.5)
        for j, window in enumerate(windows):
            # 윈도우에 차원을 추가하여 모델에 입력할 수 있는 형태로 만듭니다.
            window = window.unsqueeze(0)
            # 모델을 사용하여 예측을 수행하고 결과를 저장합니다.
            prediction = inference_model.predict(window)
            results.append({"Window": j, "Prediction": prediction.item()})
    return results

def create_windows(input_data, window_size, overlap_ratio):
    # 입력 데이터를 윈도우로 나누는 함수입니다.
    windows = []
    step_size = int(window_size * (1 - overlap_ratio))
    for i in range(0, input_data.shape[0] - window_size + 1, step_size):
        windows.append(input_data[i:i+window_size, :])
    return windows

if __name__ == '__main__':
    # 사용자로부터 실행 모드를 입력받습니다.
    MODE = input("실행 모드를 선택하세요 (train/inference): ").strip().lower()
    main(MODE)
