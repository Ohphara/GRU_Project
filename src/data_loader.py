# data_loader.py
#파일 이름 형식은 '동영상 클립 타입_클립넘버_user id_관절이름' ex)HandManipulation1_1_101010_left_hand_wrist or _right_pinkyfinger_tip/
import os
import json
import torch
from config import hand_manipulation_landmark_names, pose_stability_landmark_names
from torch.utils.data import Dataset, DataLoader

# 데이터셋과 데이터로더를 생성하는 함수
def create_dataset_and_loader(root_dir, dataset_type, batch_size=1, shuffle=False):
    # 주어진 경로에서 데이터셋을 생성하고 이를 통해 데이터 로더를 생성합니다.
    dataset = TrainDataset(root_dir=root_dir, dataset_type=dataset_type)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataset, dataloader

# 모든 데이터셋 클래스의 상위 클래스
class BaseDataset(Dataset):
    def __init__(self, root_dir, dataset_type):
        # 데이터셋의 루트 디렉토리와 타입을 저장합니다.
        self.root_dir = root_dir
        self.dataset_type = dataset_type
        # 파일 목록과 데이터를 로드합니다.
        self.file_list = self.get_file_list()
        self.data = self.load_data()

    def get_file_list(self):
        raise NotImplementedError
    
    def load_data(self):
        # 각 파일에서 데이터를 로드하고 텐서로 변환하여 딕셔너리에 저장합니다.
        data = {}
        for file_name in self.file_list:
            identifier = file_name.split('_')[1]
            if identifier not in data:
                data[identifier] = []
            with open(os.path.join(self.root_dir, file_name), 'r') as file:
                json_data = json.load(file)
                data[identifier].append(torch.tensor(json_data))
        return data

    def calculate_diff_and_standardize(self, frames):
        # 프레임 간 차분 계산
        diff_frames = torch.diff(frames, dim=1)
        # 프레임별로 평균과 표준편차 계산
        mean = diff_frames.mean(dim=(0, 1), keepdim=True)
        std = diff_frames.std(dim=(0, 1), unbiased=False, keepdim=True)
        # 표준화
        standardized_frames = (diff_frames - mean) / std
        # 차원 변경
        if self.dataset_type == 'hand_flapping':
            standardized_frames = standardized_frames.view(-1, 36)
        elif self.dataset_type == 'hand_manipulation':
            standardized_frames = standardized_frames.view(-1, 126)
        elif self.dataset_type == 'pose_stability':
            standardized_frames = standardized_frames.view(-1,9)
        return standardized_frames

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 주어진 인덱스에 해당하는 데이터를 반환합니다.
        identifier = list(self.data.keys())[idx]
        frames = torch.stack(self.data[identifier])
        standardized_frames = self.calculate_diff_and_standardize(frames)
        return standardized_frames

class TrainDataset(BaseDataset):
    def __init__(self, root_dir, dataset_type):
        super().__init__(root_dir, dataset_type)
        self.labels = self.generate_labels()

    def get_file_list(self):
        file_list = [f for f in os.listdir(self.root_dir) if f.endswith('.json')]
        print("Total files found:", len(file_list))
        return file_list

    def generate_labels(self):
        labels = {}
        for file_name in self.file_list:
            parts = file_name.split('_')
            clip_type = parts[0]
            clip_number = int(clip_type[-1])
            labels[file_name] = clip_number
        return labels

    def __getitem__(self, idx):
        file_name = self.file_list[idx] 
        label = self.labels[file_name] -1
        identifier = list(self.data.keys())[idx]
        frames = torch.stack(self.data[identifier])
        standardized_frames = self.calculate_diff_and_standardize(frames)
        return standardized_frames, label

class InferenceDataset(BaseDataset):
    def __init__(self, root_dir, dataset_type, clip_number):
        self.clip_number = clip_number
        super().__init__(root_dir, dataset_type)

    def get_file_list(self):
        if self.dataset_type == 'hand_manipulation':
            file_list = [f for f in os.listdir(self.root_dir) if f.endswith('.json') and 
                         '_'.join(os.path.splitext(f)[0].split('_')[3:]) in hand_manipulation_landmark_names and
                         f.split('_')[1] == str(self.clip_number)]
        elif self.dataset_type == 'pose_stability':
            file_list = [f for f in os.listdir(self.root_dir) if f.endswith('.json') and 
                         '_'.join(os.path.splitext(f)[0].split('_')[3:]) in pose_stability_landmark_names and
                         f.split('_')[1] == str(self.clip_number)]
        print("Total files found:", len(file_list))
        return file_list