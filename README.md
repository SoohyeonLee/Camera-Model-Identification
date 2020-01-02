# Camera-Model-Identification
 
## 서론

**계기**
- 소속 연구실 연구과제

**주제**
- 딥러닝을 활용한 카메라 모델 판별

**설명**
- 12개의 카메라 모델로 찍은 영상을 학습하여 어느 카메라에서 찍은 영상인지 판별

## 성과

**논문**
- 학회 / 학회지 : 한국정보처리학회 (KIPS) / KIPS TRANSACTIONS ON SOFTWARE AND DATA ENGINEERING (KTSDE)
- 국문 제목 : 딥러닝 기반 카메라 모델 판별
- 영문 제목 : Camera Model Identification based on Deep Learning
- Published : 2019.10.31
- DOI : https://doi.org/10.3745/KTSDE.2019.8.10.411

## 실험 환경

**하드웨어 스팩**
- 그래픽 카드 : NVIDIA GTX 1080
- RAM : 16GB

**소프트웨어 버전**
- OS : Windows 10 64bit
- CUDA / cuDNN : 9.0 / 7.2.1
- Python / Tensorflow-gpu : 3.6 / 1.10.0

## 실험 내용

**모델 구조**

<img src="https://github.com/SoohyeonLee/Camera-Model-Identification/blob/master/resource/model structure.png" width="90%"></img>

**전처리**
- HPF : 학습 모델 초기에 입력 데이터에 대해 HPF 적용

**사용 데이터**
- Dresden Open Dataset

<img src="https://github.com/SoohyeonLee/Camera-Model-Identification/blob/master/resource/Dresden.PNG" width="50%"></img>

- Camera List using Experiments

<img src="https://github.com/SoohyeonLee/Camera-Model-Identification/blob/master/resource/camera-12.PNG" width="50%"></img>

- Non Overlap Slice

<img src="https://github.com/SoohyeonLee/Camera-Model-Identification/blob/master/resource/slice.png" width="50%"></img>

- Data Sample

<img src="https://github.com/SoohyeonLee/Camera-Model-Identification/blob/master/resource/data sample.PNG" width="50%"></img>


## 실험 결과
- 레이어 갯수에 따른 성능 비교

<img src="https://github.com/SoohyeonLee/Camera-Model-Identification/blob/master/resource/accuracy by layers.png" width="80%"></img>

- 제안한 모델의 카메라별 정확도

<img src="https://github.com/SoohyeonLee/Camera-Model-Identification/blob/master/resource/accuracy per camera.PNG" width="80%"></img>

___

**작성일 : 2019.08.19**
