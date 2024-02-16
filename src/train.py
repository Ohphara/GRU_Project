# train.py
import os
import torch

class ModelTrain:
    def __init__(self, model, dataloader, criterion, optimizer, num_epochs=10):
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs

    def set_dataloader(self, dataloader):  # 새로운 데이터 로더를 설정하는 메소드
        self.dataloader = dataloader

    def train(self):
        for epoch in range(self.num_epochs):
            total_loss = 0.0  # 각 epoch의 전체 손실을 누적할 변수

            for i, data in enumerate(self.dataloader, 1):
                inputs, labels = data
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

                if i % 10 == 0:  # 매 10번째 미니배치마다 손실 출력
                    avg_loss = total_loss / 10
                    print(f'Epoch [{epoch + 1}/{self.num_epochs}], Batch [{i}/{len(self.dataloader)}], Loss: {avg_loss:.4f}')
                    total_loss = 0.0
    def save_model(self, save_path, model_name):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(self.model.state_dict(), os.path.join(save_path, model_name))

        print('Training finished.')