# GRU 기반 이상 탐지 및 평가 기법 개발: ASD 아동의 다감각 프로그램 적용

## 요약

   이 연구는 Creamo의 ASD 아동 대상 다감각 치료 프로그램에서 아동의 성취도 평가(Hand Manipulation, Pose Stability, Bilateral Hand use)와 프로그램 도중 이상행동(Hand Flapping, Body Rocking, Sit up & Sit down 등) 탐지를 위해 GRU 기반의 이상 탐지 및 평가 기법을 개발을 목적으로 진행하였습니다.

## 서론
   Autism Spectrum Disorder은 전 세계적으로 빠르게 증가하는 발달 장애로 조기 치료와 개입을 통해 그 증세를 크게 완화시킬 수 있습니다[1]. 이에 따라, ASD 아동 대상 다감각 치료 프로그램에 대한 평가와 이상행동 탐지에 대한 필요성이 높아지고 있습니다. 그러나 ASD 아동에 대한 조기 진단과 치료에는 많은 시간이 소요되고, 환자 수의 증가와 진단 및 치료를 담당할 전문인력의 부족이 맞물려 많은 ASD 아동들이 제때 적절한 조치를 받지 못하는 상황으로 이어졌습니다. 이러한 배경 속에서 ASD의 진단 및 치료 인력의 부족 문제를 해결하고자 머신 러닝을 ASD 아동의 진단에 사용하려는 선행연구들이 있었고 이들을 바탕으로 연구를 진행하게 되었습니다. 먼저"Development and Validation of a Joint Attention-Based Deep Learning System for Detection and Symptom Severity Assessment of Autism Spectrum Disorder"[1]에서는 CNN-LSTM 구조를 기반으로 TD와 ASD 아동들의 video data를 분석하여 아동들의 ASD severity를 높은 정확도로 예측하는데 성공하였습니다. 또한 "The Classification of Abnormal Hand Movement to Aid in Autism Detection: Machine Learning Study"[3]에서는 Google mediapipe를 기반으로, ASD 아동들에게서 흔히 보여지는 이상행동인 Hand Clapping을 detect하는데 성공하였습니다. 이러한 선행 연구들을 바탕으로 머신러닝을 기반으로한 ASD 아동들의 성취도 평가와 이상행동 탐지의 가능성을 확인하였고 이를 Creamo에서도 사용할 수 있게 만들고자 개발을 시작하였습니다.

## 연구 방법
본 연구에서는 Google Mediapipe의 Hand Gesture & Pose Detection에서 몸의 landmark들의 좌표 데이터를 input으로, output은 동영상에서의 성취도 평가 동작과 이상행동을 추출하는 것으로 설정하였다. 또한, 연구를 위한 프로젝트 구조는 아래와 같습니다.

## 프로그램 구조
    /YourProjectPATH
    │
    ├── main.py
    ├── config.py
    ├── requirements.txt
    ├── src
    │ ├── data_loader.py
    │ ├── model_builder.py
    │ ├── train.py
    │ └── inference.py
    └── DB
    │ ├── AssessmentClip
    │ │ ├── HM
    │ │ └── PS
    │ ├── DetectionClip
    │ ├── TDvideos
    │ └── ASDvideos

    

## 설치 및 실행 방법
본 연구를 진행하기 위한 환경 구축 및 실행 방법은 아래와 같다. 

### 1) 가상환경 생성

#### 아나콘다 프롬프트를 사용하여 가상환경을 구축할 수 있다. 이 프로젝트에서 개발에 사용한 파이썬 버전은 3.11.5이다.

```Anaconda Prompt
conda create --name 가상환경이름 python=3.11.5
```

이후 가상환경을 활성화 한다

```Anaconda Prompt
conda activate 가상환경이름
```

아래 명령을 따라 Pytorch module을 설치한다. 

CPU 버전 : 
```Anaconda Prompt
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```
GPU 버전 (CUDA 지원이 있는 경우):
```Anaconda Prompt
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch
```

### 2) Model 동작 확인을 위한 dummy file 생성법

현재는 DB폴더에 .json 파일들이 비어있으므로, dummy data를 생성하여 모델의 동작을 확인하여야 한다.

일단 HM에 대해서 dummy data를 만드는 코드는 다음과 같다

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

Inference 실행을 위한 dummy data를 만드는 코드는 다음과 같다

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

    # (50000, 3) 모양의 랜드마크 데이터 생성 (평균=0, 표준편차=1의 정규분포)
    landmark_data = torch.randn(50000, 3)

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
root_dir_template = r"C:/Users/PJO/23kist/GRU_project_kist_PiljunOh/DB/TDvideos"

# 클립 타입, HM 넘버, 클립 번호에 따라 파일 생성
for clipnumber in range(1, 6):  # Modified to iterate over clip numbers from 1 to 5
    videocliptype = f"TDVideos"
    root_dir = root_dir_template
    
    # 같은 HM 넘버 내에서 같은 클립 넘버에 대해 같은 사용자 ID를 유지
    userid = str(random.randint(100000, 999999))  # 6자리 숫자 랜덤 생성
    for hand in ["left", "right"]:
        for landmarkname in landmark_names:
                create_json_file(root_dir, videocliptype, clipnumber, userid, hand, landmarkname)
```
---

## 3. Execution






---
## 4. Usage Example





---
## 5. Lisence

## 6. Reference
1. Estes A, Munson J, Rogers SJ, Greenson J, Winter J, Dawson G. Long-term outcomes of early intervention in 6-year-old children with autism spectrum disorder. J Am Acad Child Adolesc Psychiatry 2015 Jul;54(7):580-587 [FREE Full text] doi: 10.1016/j.jaac.2015.04.005 [Medline: 26088663]
2. Ko C, Lim JH, Hong J, Hong SB, Park YR. Development and Validation of a Joint Attention-Based Deep Learning System for Detection and Symptom Severity Assessment of Autism Spectrum Disorder. JAMA Netw Open. 2023 May 1;6(5):e2315174. doi: 10.1001/jamanetworkopen.2023.15174. Erratum in: JAMA Netw Open. 2023 Jul 3;6(7):e2324944. PMID: 37227727; PMCID: PMC10214037.
3. Lakkapragada A, Kline A, Mutlu O, Paskov K, Chrisman B, Stockham N, Washington P, Wall D The Classification of Abnormal Hand Movement to Aid in Autism Detection: Machine Learning Study. JMIR Biomed Eng 2022;7(1):e33771
doi: 10.2196/33771
