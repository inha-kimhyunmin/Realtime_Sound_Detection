# 🚀 실시간 위험소리 감지 시스템 (병렬 처리 버전)

## 📖 개요

이 시스템은 **오디오 입력 수집**과 **모델 추론**을 병렬로 처리하여 끊김없는 실시간 위험소리 감지를 구현합니다. 기존의 순차 처리 방식과 달리, 지속적인 오디오 스트림 수집과 동시에 AI 모델 분석이 이루어져 더 빠르고 정확한 감지가 가능합니다.

## 🔧 병렬 처리 아키텍처

### 시스템 구조
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   오디오 입력   │    │   순환 버퍼     │    │   모델 추론     │
│   (스레드 1)    │───▶│   (공유 메모리)  │───▶│   (스레드 2)    │
│                 │    │                 │    │                 │
│ • 0.5초 청크    │    │ • 10초 세그먼트 │    │ • YAMNet+LSTM   │
│ • 마이크 감도   │    │ • 50% 겹침      │    │ • 위험도 분석   │
│ • 전처리        │    │ • 자동 관리     │    │ • 결과 출력     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 주요 구성요소

#### 1. **오디오 입력 스레드 (Audio Callback Thread)**
- **역할**: 마이크로부터 지속적인 오디오 데이터 수집
- **동작 주기**: 0.5초마다 오디오 청크 처리
- **주요 기능**:
  - 실시간 마이크 입력 수집
  - 마이크 감도 자동 적용
  - 클리핑 방지 처리
  - 순환 버퍼에 데이터 저장

#### 2. **순환 버퍼 (Circular Buffer)**
- **역할**: 오디오 데이터의 임시 저장 및 관리
- **크기**: 20초 분량 (10초 세그먼트 × 2배 여유분)
- **특징**:
  - 메모리 효율적 관리 (자동 오버플로우 처리)
  - 스레드 안전성 보장 (Lock 기반)
  - 세그먼트 준비 상태 자동 감지

#### 3. **모델 추론 스레드 (Inference Worker Thread)**
- **역할**: AI 모델을 사용한 위험소리 분석
- **처리 방식**: 큐 기반 비동기 처리
- **주요 기능**:
  - YAMNet 임베딩 추출
  - LSTM 모델 예측
  - 결과 분석 및 출력

#### 4. **추론 큐 (Inference Queue)**
- **역할**: 분석 대기 중인 오디오 세그먼트 관리
- **크기**: 최대 5개 세그먼트
- **오버플로우 처리**: 오래된 데이터 자동 제거

## ⚡ 병렬 처리의 장점

### 1. **끊김없는 실시간 처리**
- **기존 방식**: 10초 녹음 → 분석 → 10초 대기 → 반복
- **병렬 방식**: 지속적인 수집 + 겹침 분석으로 연속 감지

### 2. **응답 속도 향상**
- **기존 지연**: 최대 10초 + 추론시간
- **병렬 지연**: 최대 0.5초 + 추론시간

### 3. **메모리 효율성**
- **순환 버퍼**: 고정 크기로 메모리 사용량 일정
- **자동 관리**: 오래된 데이터 자동 정리

### 4. **연속성 보장**
- **50% 겹침**: 세그먼트 간 연결성 유지
- **스텝 처리**: 새로운 데이터 축적시 즉시 분석

## 🔄 데이터 플로우

### 단계별 처리 과정

#### Step 1: 오디오 수집
```python
# 0.5초마다 실행되는 콜백 함수
def audio_callback(indata, frames, time_info, status):
    audio_chunk = indata[:, 0]  # 모노 변환
    audio_chunk *= mic_gain     # 감도 적용
    
    with buffer_lock:
        audio_buffer.extend(audio_chunk)
        
        if len(audio_buffer) >= segment_length:
            # 10초 세그먼트 추출
            segment = audio_buffer[-segment_length:]
            inference_queue.put(segment)
            
            # 5초 분량 제거 (50% 겹침)
            for _ in range(step_length):
                audio_buffer.popleft()
```

#### Step 2: 세그먼트 처리
```python
def inference_worker():
    while is_running:
        segment = inference_queue.get(timeout=1.0)
        
        # 전처리
        processed_audio = preprocess_audio(segment)
        
        # AI 분석
        embeddings = extract_yamnet_embeddings(processed_audio)
        prediction = lstm_model.predict(embeddings)
        
        # 결과 출력
        handle_prediction_result(prediction)
```

#### Step 3: 겹침 관리
```
시간축:  0    5    10   15   20   25   30
세그먼트1: [-----10초----]
세그먼트2:      [-----10초----]
세그먼트3:           [-----10초----]
세그먼트4:                [-----10초----]

겹침률: 50% (5초 겹침)
분석주기: 5초마다 새로운 세그먼트
```

## 🎛️ 고급 설정

### 병렬 처리 파라미터
```python
SAMPLE_RATE = 16000           # 샘플링 주파수
SEGMENT_DURATION = 10.0       # 분석 세그먼트 길이
CHUNK_DURATION = 0.5          # 오디오 청크 간격
OVERLAP_RATIO = 0.5           # 겹침 비율 (50%)
```

### 성능 최적화 설정
```python
# 버퍼 크기 (메모리 사용량)
buffer_maxlen = SAMPLE_RATE * SEGMENT_DURATION * 2  # 20초 버퍼

# 큐 크기 (처리 지연 제어)
inference_queue_maxsize = 5  # 최대 5개 세그먼트 대기

# 스레드 설정
audio_thread = daemon=True   # 메인 종료시 자동 종료
inference_thread = daemon=True
```

### 마이크 캘리브레이션
```python
AUTO_CALIBRATION_MODE = True        # 자동 캘리브레이션
CALIBRATION_SILENCE_DURATION = 5.0  # 무음 측정 시간
CALIBRATION_FACTORY_DURATION = 3.0  # 공장소리 측정 시간
CALIBRATION_MAX_ATTEMPTS = 10       # 최대 시도 횟수
```

## 📊 성능 모니터링

### 실시간 정보 출력
```
🕒 14:32:15 | 📊 확률: [0.02 0.05 0.88 0.03 0.02] | 🔊 RMS=0.0234, Max=0.0567 | ⚡ 처리시간: 0.234초
🚨 위험 감지: 🔥 화재 (88.3% 확률)
================================================================================
```

### 시스템 상태 정보
```
📋 시스템 정보:
  - 샘플링 주파수: 16,000 Hz
  - 세그먼트 길이: 10초 (160,000샘플)
  - 청크 길이: 0.5초 (8,000샘플)
  - 겹침 비율: 50%
  - 스텝 길이: 80,000샘플
  - 버퍼 최대 크기: 320,000샘플
  - 추론 큐 크기: 5
```

## 🛠️ 문제 해결

### 일반적인 문제

#### 1. **처리 지연 발생**
**증상**: `⚡ 처리시간`이 0.5초보다 길어짐
**해결책**:
- 추론 큐 크기 증가: `inference_queue_maxsize = 10`
- 세그먼트 길이 단축: `SEGMENT_DURATION = 5.0`
- 오버랩 비율 감소: `OVERLAP_RATIO = 0.3`

#### 2. **메모리 사용량 증가**
**증상**: 시간이 지날수록 메모리 사용량 증가
**해결책**:
- 버퍼 크기 조정: `buffer_maxlen` 감소
- 큐 크기 제한: `inference_queue_maxsize` 감소
- 가비지 컬렉션 강제 실행

#### 3. **오디오 드롭아웃**
**증상**: `⚠️ 오디오 상태` 경고 메시지
**해결책**:
- 청크 크기 증가: `CHUNK_DURATION = 1.0`
- 시스템 우선순위 조정
- 다른 프로그램 종료

### 성능 튜닝

#### CPU 사용량 최적화
```python
# 모델 예측 배치 크기 조정
predictions = lstm_model.predict(embeddings, batch_size=1, verbose=0)

# TensorFlow GPU 메모리 제한
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
```

#### 메모리 사용량 최적화
```python
# 주기적 가비지 컬렉션
import gc
if frame_count % 100 == 0:
    gc.collect()

# NumPy 배열 메모리 해제
del processed_audio, embeddings
```

## 📋 API 참조

### 주요 클래스

#### `RealTimeAudioDetector`
병렬 처리 기반 실시간 오디오 감지기

**주요 메서드**:
- `__init__()`: 시스템 초기화 및 모델 로드
- `start_detection()`: 실시간 감지 시작
- `stop_detection()`: 시스템 종료
- `get_system_info()`: 시스템 정보 출력

**콜백 함수**:
- `audio_callback()`: 오디오 입력 처리
- `inference_worker()`: 모델 추론 워커
- `process_audio_segment()`: 세그먼트 분석
- `handle_prediction_result()`: 결과 처리

### 설정 변수

#### 핵심 파라미터
| 변수명 | 기본값 | 설명 |
|--------|--------|------|
| `SEGMENT_DURATION` | 10.0 | 분석 세그먼트 길이 (초) |
| `CHUNK_DURATION` | 0.5 | 오디오 청크 간격 (초) |
| `OVERLAP_RATIO` | 0.5 | 세그먼트 겹침 비율 |
| `DANGER_THRESHOLD` | 0.7 | 위험 감지 확률 임계값 |

#### 캘리브레이션 파라미터
| 변수명 | 기본값 | 설명 |
|--------|--------|------|
| `AUTO_CALIBRATION_MODE` | True | 자동 캘리브레이션 사용 |
| `MIC_GAIN` | 3.0 | 마이크 감도 배율 |
| `SILENCE_RMS_THRESHOLD` | 0.005 | 무음 판단 RMS 임계값 |

## 🚀 사용 예제

### 기본 사용법
```python
# 시스템 초기화 및 실행
detector = RealTimeAudioDetector()
detector.get_system_info()
detector.start_detection()
```

### 설정 커스터마이징
```python
# 설정 수정
SEGMENT_DURATION = 5.0    # 더 빠른 반응
OVERLAP_RATIO = 0.3       # 적은 겹침
DANGER_THRESHOLD = 0.8    # 더 엄격한 감지

# 시스템 실행
detector = RealTimeAudioDetector()
detector.start_detection()
```

---

## 💡 팁과 권장사항

1. **초기 설정**: 첫 실행시 자동 캘리브레이션 권장
2. **성능 최적화**: CPU 사용률이 높으면 세그먼트 길이 증가
3. **정확도 향상**: 겹침 비율을 높이면 더 정확한 감지 가능
4. **실시간성 향상**: 청크 간격을 줄이면 더 빠른 반응
5. **안정성 확보**: 큐 크기를 적절히 설정하여 메모리 관리

이 병렬 처리 시스템은 기존 순차 처리 방식 대비 **훨씬 뛰어난 실시간성과 연속성**을 제공하며, 산업 현장에서의 실용적인 위험소리 감지에 최적화되어 있습니다.
