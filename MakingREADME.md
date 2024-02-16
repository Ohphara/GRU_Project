# Development of GRU-Based Anomaly Detection and Assessment Techniques in a Multisensory Program for ASD Children

## 1. Introduction
#### 이 프로젝트는 크리모의 ASD 아동 대상 다감각 치료 프로그램에서 학습 아동의 성취도 평가(Hand Manipulation, Pose Stability, Bilateral Hand use) 와 프로그램 중 Abnormal Behavior(Hand Flapping, Body Rocking, Sit up & Sit down,,,) Detection을 위해 개발하였다.





## 2. Installation

### 1) 가상환경 생성

#### 아나콘다 프롬프트를 사용하여 가상환경을 구축할 수 있다. 이 프로젝트에서 개발에 사용한 파이썬 버전은 3.11.5이다.

```Anaconda Prompt
conda create --name 가상환경이름 python=3.11.5
```

#### 이후 가상환경을 활성화 한다

```Anaconda Prompt
conda activate 가상환경이름
```

#### 아래 명령을 따라 Pytorch module을 설치한다. 

##### CPU 버전 : 
```Anaconda Prompt
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```
##### GPU 버전 (CUDA 지원이 있는 경우):
```Anaconda Prompt
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch
```

### 2) Model 동작 확인을 위한 dummy file 생성법

#### 현재는 DB폴더에 .json 파일들이 비어있으므로, dummy data를 생성하여 모델의 동작을 확인하여야 한다.

#### 일단 HM에 대해서 dummy data를 만드는 코드는 다음과 같다

```
import os
import json
import torch
import random

# 함수: JSON 파일 생성
def create_json_file(root_dir, videocliptype, clipnumber, userid, hand, landmarkname):
    # 파일 이름 생성
    file_name = f"{videocliptype}_{clipnumber}_{userid}_{hand}_{landmarkname}.json"
    file_path = os.path.join(root_dir, file_name)

    # (60, 3) 모양의 랜드마크 데이터 생성 (평균=0, 표준편차=1의 정규분포)
    landmark_data = torch.randn(60, 3)

    # JSON 파일에 랜드마크 데이터 저장
    with open(file_path, 'w') as file:
        json.dump(landmark_data.tolist(), file)

# 랜드마크 이름 목록
landmark_names = [
    "hand_wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "indexfinger_mcp", "indexfinger_pip", "indexfinger_dip", "indexfinger_tip",
    "middlefinger_mcp", "middlefinger_pip", "middlefinger_dip", "middlefinger_tip",
    "ringfinger_mcp", "ringfinger_pip", "ringfinger_dip", "ringfinger_tip",
    "pinkyfinger_mcp", "pinkyfinger_pip", "pinkyfinger_dip", "pinkyfinger_tip"
]

# 루트 디렉토리
root_dir_template = "YourProjectPATH/DB/AssessmentClip/HM/HM{}"

# 클립 타입, HM 넘버, 클립 번호에 따라 파일 생성
for hm_number in range(1, 9):
    for i in range(1, 21):
        videocliptype = f"HandManipulation{hm_number}"
        root_dir = root_dir_template.format(hm_number)
        
        # 같은 HM 넘버 내에서 같은 클립 넘버에 대해 같은 사용자 ID를 유지
        userid = str(random.randint(100000, 999999))  # 6자리 숫자 랜덤 생성
        for hand in ["left", "right"]:
            for _ in range(21):  # 21개의 랜드마크 (left + right)
                for landmarkname in landmark_names:
                    create_json_file(root_dir, videocliptype, i, userid, hand, landmarkname)
```



## 3. Execution

### 1) 초기 경로 설정






## 4. Usage Example






## 5. Lisence
