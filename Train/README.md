# YAMNet + LSTM 모듈형 훈련 시스템

고급 데이터 생성과 증강을 통한 환경음 분류 모델 훈련 시스템입니다.

## 🎯 주요 특징

- **📊 클래스별 균등 프레임 분석**: 각 클래스별 가용 데이터 자동 스캔 및 최적 샘플 수 계산
- **🎵 고급 데이터 증강**: 클래스별 맞춤 증강 전략 (볼륨, 리버브, SNR 제어, 노이즈 혼합)
- **🔄 전환 시나리오 생성**: 다양한 상황 전환 데이터 자동 생성
- **🧠 지능형 모델 훈련**: 클래스 불균형 자동 해결 및 최적화된 LSTM 아키텍처
- **📈 종합적 평가 시스템**: 자동 테스트 데이터 생성 및 실제 오디오 파일 검증

## 📁 파일 구조

```
Train/
├── config.py              # 중앙 설정 관리
├── data_generator.py       # 고급 데이터 생성기
├── model_trainer.py        # 모델 훈련기
├── evaluation.py           # 모델 평가기
├── main.py                # 메인 실행 인터페이스
├── install_requirements.py # 패키지 설치 스크립트
├── requirements.txt        # 필수 패키지 목록
└── README.md              # 이 파일
```

## 🛠️ 설치 및 설정

### 1. 필수 패키지 설치

```bash
# 방법 1: 자동 설치 스크립트 사용
python install_requirements.py

# 방법 2: pip를 통한 수동 설치
pip install -r requirements.txt
```

### 2. 데이터 폴더 구조 확인

```
Realtime_Sound_Detection/
├── envsound/
│   ├── fire/       # 화재 소리 파일들
│   ├── gas/        # 가스누출 소리 파일들
│   └── scream/     # 비명 소리 파일들
└── mixture/        # 공장소리 파일들
```

## 🚀 사용 방법

### 기본 실행

```bash
cd Train
python main.py
```

### 단계별 실행

```bash
# 1. 데이터 생성만
python data_generator.py

# 2. 모델 훈련만
python model_trainer.py

# 3. 모델 평가만
python evaluation.py
```

## ⚙️ 주요 설정

### config.py에서 수정 가능한 설정들:

```python
# 오디오 설정
MODEL_CONFIG = {
    'audio_duration': 8.0,    # 오디오 길이 (5.0~10.0초)
    'sample_rate': 16000,     # 샘플링 주파수
}

# 훈련 설정
TRAINING_CONFIG = {
    'epochs': 50,             # 훈련 에포크
    'batch_size': 16,         # 배치 크기
    'learning_rate': 0.001,   # 학습률
}

# 데이터 생성 목표
DATA_GENERATION_CONFIG = {
    'target_frames_per_class': 2000,  # 클래스당 목표 프레임 수
    'transition_data_ratio': 0.2,     # 전환 데이터 비율
}
```

## 📊 데이터 생성 전략

### 1. 클래스별 균등 프레임 분석
- 각 클래스별 가용 오디오 파일 자동 스캔
- YAMNet 프레임 수 계산 (0.48초당 1프레임)
- 목표 프레임 수 달성을 위한 필요 샘플 수 계산

### 2. 고급 데이터 증강
- **무음 클래스**: 다양한 노이즈 타입 (white, pink, brown)
- **위험소리 클래스**: 볼륨 변화, 리버브, 공장소리 믹싱 (SNR 제어)
- **공장소리 클래스**: 속도 변화, 룸 효과, 노이즈 추가

### 3. 전환 시나리오 생성
- `silence_to_silence`: 다양한 무음 패턴
- `silence_to_factory`: 무음에서 공장소리
- `silence_to_danger`: 무음에서 위험소리
- `factory_to_factory`: 공장소리 간 전환
- `factory_to_danger`: 공장에서 위험 상황 (가장 중요)

## 🧠 모델 아키텍처

```
YAMNet (Google Pre-trained)
    ↓ (1024차원 임베딩)
Dense(256) + ReLU + Dropout(0.3)
    ↓
Dense(128) + ReLU + Dropout(0.3)
    ↓
Reshape → (1, 128)
    ↓
LSTM(128) + Dropout(0.3)
    ↓
LSTM(64) + Dropout(0.3)
    ↓
Dense(5) + Softmax
```

## 📈 훈련 및 평가 기능

### 훈련 기능
- 클래스 가중치 자동 계산으로 불균형 해결
- 체크포인트 저장 및 조기 종료
- 학습률 스케줄링
- 실시간 성능 모니터링

### 평가 기능
- 자동 테스트 데이터셋 생성 (클래스별 균등 분할)
- 상세한 성능 분석 (정확도, 정밀도, 재현율, F1 점수)
- 혼동 행렬 시각화
- 예측 신뢰도 분석
- 실제 오디오 파일 테스트

## 📁 결과 파일

### 훈련 결과 (`Train/results/training/`)
- `yamnet_lstm_model_YYYYMMDD_HHMMSS.h5`: 훈련된 모델
- `model_info.json`: 훈련 정보 및 설정
- `class_mapping.npy`: 클래스 매핑 정보
- `training_history.png`: 훈련 과정 그래프
- `confusion_matrix.png`: 혼동 행렬

### 평가 결과 (`Train/results/evaluation/`)
- `evaluation_report.md`: 종합 평가 리포트
- `evaluation_results.json`: 상세 평가 결과
- `detailed_confusion_matrix.png`: 상세 혼동 행렬
- `class_performance.png`: 클래스별 성능 차트
- `prediction_confidence.png`: 예측 신뢰도 분석

### 데이터셋 정보 (`Train/results/datasets/`)
- `dataset_info_YYYYMMDD_HHMMSS.json`: 생성된 데이터셋 정보
- `dataset_YYYYMMDD_HHMMSS.npz`: 생성된 데이터 (임베딩 + 라벨)

## 🔧 문제 해결

### 메모리 부족
```python
# config.py에서 조정
TRAINING_CONFIG['batch_size'] = 8  # 배치 크기 감소
DATA_GENERATION_CONFIG['target_frames_per_class'] = 1000  # 목표 프레임 감소
```

### 훈련 시간 과다
```python
# config.py에서 조정
TRAINING_CONFIG['epochs'] = 20  # 에포크 감소
MODEL_CONFIG['audio_duration'] = 5.0  # 오디오 길이 단축
```

### 낮은 정확도
- 더 많은 훈련 데이터 추가
- 데이터 증강 강화
- 하이퍼파라미터 튜닝

## 💡 사용 팁

1. **첫 실행**: `python main.py`로 전체 파이프라인 실행
2. **설정 조정**: 메인 메뉴에서 "5. 설정 확인 및 수정" 선택
3. **결과 확인**: 메인 메뉴에서 "6. 결과 폴더 열기" 선택
4. **재훈련**: 기존 데이터로 다른 설정으로 재훈련 가능

## 📋 시스템 요구사항

- Python 3.8 이상
- TensorFlow 2.8 이상
- 최소 8GB RAM (권장 16GB)
- GPU 권장 (NVIDIA CUDA 지원)

## 🤝 지원되는 오디오 형식

- WAV, MP3, FLAC
- 권장: 16kHz, 모노
- 최소 길이: 5초

## 📞 문제 신고

코드 실행 중 문제가 발생하면:
1. 에러 메시지와 함께 상황 설명
2. 사용 중인 Python 버전 및 패키지 버전
3. 시스템 환경 (OS, RAM, GPU 등)

---

🎵 **YAMNet + LSTM 모듈형 훈련 시스템**으로 고품질 환경음 분류 모델을 만들어보세요!
