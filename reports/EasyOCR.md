# OCR (Optical Character Recognition)

- 광학 문자 인식(OCR, Optical Character Recognition)은 사람이 쓰거나 기계로 인쇄한 문자의 영상을 이미지 스캐너로 획득하여 기계가 읽을 수 있는 문자로 변환하는 것

# EasyOCR

[EasyOCR](https://github.com/JaidedAI/EasyOCR) 

EasyOCR은 문자 영역 인식(Detection) + 문자 인식(Recognition)기능을 모두 하는 프레임워크

현재 [80개이상의 언어를 지원](https://www.jaided.ai/easyocr/)하고 있으며, 꾸준히 Releases 되고 있다.

## 구조

![easyocr_framework](image/framework.jpeg)

### Text Detection

- Clova AI의 [CRAFT](https://github.com/clovaai/CRAFT-pytorch)

### Recognition

**CRNN 모델**

- Feature Extraction: **ResNet**
- Sequence Modeling: **LSTM**
- Prediction: **CTC**

**Training pipeline**

- EasyOCR에서 사용하고 있는 신경망 모델은 학습의 각 단계별로 다음과 같은 또 다른 오픈소스 기반 프로젝트를 이용
- 학습데이터 생성: [TextRecognitionDataGenerator](https://github.com/Belval/TextRecognitionDataGenerator)
- 학습데이터 변환: [TRDG2DTRB](https://github.com/DaveLogs/TRDG2DTRB)
- 모델 학습 및 배포: [Deep-Text-Recognition-Benchmark](https://github.com/clovaai/deep-text-recognition-benchmark)
- 사용자 학습 모델 사용: [EasyOCR](https://github.com/JaidedAI/EasyOCR)

<aside>
💡 회색 테두리로 표시된 영역은 사용자들이 커스텀해서 사용할 수 있는 부분
Recognition 부분을 Fine-tuning !

</aside>