# **Hand Bone Image Segmentation**

## Project Overview
### 프로젝트 목표
 - X-ray 이미지에서 사람의 뼈를 Segmentation 하는 인공지능 만들기

### 개요
- 뼈는 우리 몸의 구조와 기능에 중요한 영향을 미치기 때문에, 정확한 뼈 분할은 의료 진단 및 치료 계획을 개발하는 데 필수적이다. Bone Segmentation은 인공지능 분야에서 중요한 응용 분야 중 하나로, 특히, 딥러닝 기술을 이용한 뼈 Segmentation은 많은 연구가 이루어지고 있으며 다양한 목적으로 도움을 줄 수 있다.

### Dataset
- 크게 손가락, 손등, 팔로 구성된 xray 이미지 및 annotation *(총 29개의 class)*
- 해상도 : 2048, 2048
- 총 이미지 수 : 1,100장(Train 이미지 수 : 800장)

### 협업 tools
- Slack, Notion, Github, Wandb

### GPU
- V100(vram 32GB) 5개

### 평가기준
- Dice


<br>
  
## Team Introduction
### Members
| 고금강 | 김동우 | 박준일 | 임재규 | 최지욱 |
|:--:|:--:|:--:|:--:|:--:|
|<img  src='https://avatars.githubusercontent.com/u/101968683?v=4'  height=80  width=80px></img>|<img  src='https://avatars.githubusercontent.com/u/113488324?v=4'  height=80  width=80px></img>|<img  src='https://avatars.githubusercontent.com/u/106866130?v=4'  height=80  width=80px></img>|<img  src='https://avatars.githubusercontent.com/u/77265704?v=4'  height=80  width=80px></img>|<img  src='https://avatars.githubusercontent.com/u/78603611?v=4'  height=80  width=80px></img>|
|[Github](https://github.com/TwinKay)|[Github](https://github.com/dwkim8155)|[Github](https://github.com/Parkjoonil)|[Github](https://github.com/Peachypie98)|[Github](https://github.com/guk98)|
|twinkay@yonsei.ac.kr|dwkim8155@gmail.com|joonil2613@gmail.com|jaekyu.1998.bliz@gmail.com|guk9898@gmail.com|

### Members' Role

| 팀원 | 역할 |
| -- | -- |
| 고금강_T5011 | - Multi-Label Dataset 사용을 위한 MMSegmentation 수정 및 환경 구축 <br> - Segformer, Upernet_swin, Mask2former_swin 모델 실험 <br> - Image Crop시, 정보 손실 방지를 위한 최적의 Crop Size를 구하는 코드 구현 |
| 김동우_T5026 | - HRNet, UNet3+ 모델 실험 및 개조 <br> - Auxiliary Meta Data 학습 모델 구현 <br> - Pseudo labeling, CutMix 구현 |
| 박준일_T5094 | - 최적 Augmentation 탐색 및 비교 분석 <br> - Unet++을 활용한 Encoder 비교 분석 <br> - 하이퍼파라미터 최적화 |
| 임재규_T5174 | - CBAM, Self-Attention, Squeeze-Excitation 모듈 구현 <br> - Copy-Paste 데이터 증강 구현 <br> - PSPNet, ResUnet++, ResUnet-a 직접 구현 <br> - Unet, DeepLabV3+ 모델 실험 및 개조 |
| 최지욱_T5219 | - Unet 모델 실험 <br> - Non-local Blocks, Squeeze-Excitation Blocks 모듈 적용 실험 <br> - Auxiliary task를 추가하여 Meta Data를 활용하는 실험 <br> - Soft Voting 및 Hard Voting 구현 |

<br>

## Procedure & Techniques

  

| 분류 | 내용 |
| :--: | -- |
|Model|**HRNet** <br> - 전체 Stage에서 높은 해상도를 유지하면서도 Multi- Resolution Subnetworks 연결을 통해 다양한 해상도의 정보 수신이 가능한 HRNet 모델을 적용 <br> &nbsp;&nbsp;&nbsp;&nbsp;=> 좋지 못한 실험 결과 <br> &nbsp;&nbsp;&nbsp;&nbsp;=> HRNet 모델 특성 상 Output이 1/4로 줄어들고, 정답 제출을 위한 Output을 다시 원본 크기로 키워야 하는 과정에서 사용된 Interpolation(bilinear) 및 Transpose Convolution로 인해 유의미한 정보 손실이 발생한 것으로 추측 <br> <br> **DeepLabV3+** <br> - 이미지를 다양한 크기의 조각으로 바라볼 수 있는 ASPP와 물체의 경계선을 모호해지지 않도록 Unet과 유사한 구조를 가진 Encoder-Decoder를 적용한 DeepLabV3+ 모델을 사용 <br> - CBAM과 NLP 분야에서 자주 사용되는 Transformer의 Self-Attention 기법으로 모델을 개조<br> &nbsp;&nbsp;&nbsp;&nbsp;=> 좋지 못한 실험 결과<br> &nbsp;&nbsp;&nbsp;&nbsp;=> CBAM과 Self-Attention 모듈을 모델 네트워크에서 잘못된 위치에 적용했기 때문이라 판단 <br> <br> **U-Net 계열** <br> - 해당 Task는 의료 데이터를 다루기 때문에 의료 영상 분야에서 높은 성능을 보여주는 U-Net 계열을 적극 활용함  <br> &nbsp;&nbsp;&nbsp;&nbsp;=> U-Net 계열 중 Full-scale의 Skip Connection과 Deep Supervision을 활용한 U-Net3+에서 최고 성능을 보일 것으로 예상했으나, 오히려 기본적인 U-Net 모델에서 가장 높은 결과를 얻음<br> 
|Loss|**Focal Loss(0.5) + Dice Loss(0.5)** <br> - Focal Loss와 Dice Loss에 동일 가중치(0.5)를 주고 더한 Combined Loss 적용하였을 때, 단순히 각각의 Loss를 사용할 때보다 더 좋은 성능을 보임
|Optimizer|**AdamW** <br> - Adam, AdamW, AdamP 중에서 Adam의 Weight Decay 기능을 개선한 AdamW에서 가장 안정된 학습과 높은 성능을 확인함
|Augmentation|**Copy-Paste 데이터 증강** <br> - 29개의 클래스 중 4개의 클래스, 즉 Trapezium, Trapezoid, Pisiform, Triquetrum 등이 Dice Score 측면에서 다소 낮은 Dice 점수를 보임 <br> &nbsp;&nbsp;&nbsp;&nbsp;=> 한 이미지의 객체를 다른 이미지에 복사하여 붙여넣는 방식의 Copy-Paste 데이터 증강 기법 사용<br> <br>  **Elastic Transformation**<br>  **Sharpen**<br>  **CLAHE**<br>  **Rotate**<br>  **Flip** <br>
|Ensemble|**Soft Voting** <br> - 기존 학습된 모델로 추론하여 각 지점들마다 확률값을 평균내어 Threshold 기준에 따라 분류하는 방식 사용 <br> - 각 모델의 성능에 따라 가중치를 다르게 설정하여 좋은 성능을 보이는 모델에 더 높은 가중치를 주어 Voting하도록 하여 성능 개선 <br> - Hard Voting에 비해 압도적으로 좋은 성능을 보임
<br>

## Results

### 단일모델

| Model | Loss | Optimizer | Batch-Size | Dice-Score |
| :--: | :--: | :--: | :--: | :--: |
|HRNet| BCE + Dice| AdamW| 32 |0.9536|
|DeepLabV3+| Focal + Dice| AdamW |4 |0.9662|
|U-Net3+ |Focal + Dice| AdamW| 2| 0.9687|
|U-Net++| Focal + Dice|AdamW|1| 0.9692|
|**U-Net**| **Focal + Dice**| **AdamW** |**2**| **0.9732**|

<br>

### 앙상블 (최종 결과)

<img  src='https://i.esdrop.com/d/f/JnUd4EePHi/vEgXyHpg1P.png'  height=380  width=900px></img>

### 최종 순위
- 🥈 **Public LB : 3rd / 19**
<img  src='https://i.esdrop.com/d/f/JnUd4EePHi/I31Luue32f.png'  height=150  width=900px></img>
- 🥈 **Private LB : 3rd / 19**
<img  src='https://i.esdrop.com/d/f/JnUd4EePHi/PW9u9XEBpH.png'  height=150  width=900px></img>
