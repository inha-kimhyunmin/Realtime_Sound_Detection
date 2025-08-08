"""
5클래스 LSTM 모델 학습 스크립트
================================

이 스크립트는 YAMNet + LSTM을 사용하여 5개 클래스(무음, 정상, 화재, 가스누출, 비명)를 
분류하는 모델을 학습합니다.

데이터 가중치 수정 방법:
- SILENCE_SAMPLES: 무음 데이터 샘플 수
- NORMAL_SAMPLES: 정상(공장소음) 데이터 샘플 수  
- DANGER_WEIGHTS: 위험 소음별 가중치 딕셔너리
  - 'fire': 화재 소음 가중치 (확률적 샘플링 사용)
  - 'gas': 가스누출 소음 가중치 (파일당 샘플 수)
  - 'scream': 비명 소음 가중치 (파일당 샘플 수)

출력 파일:
- yamnet_lstm_model_5class_with_silence.h5: 학습된 모델
- model_info_5class.pkl: 모델 정보 (클래스 매핑, 이름 등)
- model_performance_5class.txt: 상세한 학습 결과 보고서
"""

import numpy as np
import librosa
import soundfile as sf
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split
import os
import glob
import pickle

# ================================
# 데이터 생성 가중치 설정 (수정 가능)
# ================================
SILENCE_SAMPLES = 150      # 무음 데이터 샘플 수
NORMAL_SAMPLES = 120       # 정상(공장소음) 데이터 샘플 수

# 위험 소음별 가중치 (파일당 생성할 샘플 수)
DANGER_WEIGHTS = {
    'fire': 0.3,    # 화재: 확률적 샘플링 (30% 확률)
    'gas': 20,      # 가스누출: 파일당 20개 샘플
    'scream': 12    # 비명: 파일당 12개 샘플
}

# ---------------------------
# 1) 무음 제거 함수
# ---------------------------
def remove_silence(y, sr, top_db=20):
    intervals = librosa.effects.split(y, top_db=top_db)
    if len(intervals) == 0:
        return y
    non_silent_audio = np.concatenate([y[start:end] for start, end in intervals])
    return non_silent_audio

# ---------------------------
# 2) 무음 생성 함수
# ---------------------------
def generate_silence(duration_sec, sr):
    """무음 데이터 생성"""
    length = int(sr * duration_sec)
    # 완전한 무음이 아닌 매우 작은 노이즈 추가 (현실적인 무음)
    silence = np.random.normal(0, 0.001, length).astype(np.float32)
    return silence

# ---------------------------
# 3) 저음량 배경 소음 생성 함수
# ---------------------------
def generate_background_noise(duration_sec, sr):
    """저음량 배경 소음 생성 (에어컨, 미세한 소음 등)"""
    length = int(sr * duration_sec)
    # 저주파 노이즈 생성
    noise = np.random.normal(0, 0.01, length).astype(np.float32)
    # 저역 통과 필터 효과 (간단한 이동평균)
    window_size = 50
    noise_filtered = np.convolve(noise, np.ones(window_size)/window_size, mode='same')
    return noise_filtered

# ---------------------------
# 4) 공장 소리와 위험 소리 합성 함수
# ---------------------------
def mix_factory_and_event(factory_audio, event_audio, sr, desired_length=10.0):
    event_len = len(event_audio)
    factory_len = len(factory_audio)
    target_len = int(sr * desired_length)
    
    if factory_len < target_len:
        factory_audio = np.pad(factory_audio, (0, target_len - factory_len))
    else:
        factory_audio = factory_audio[:target_len]
    
    if event_len > target_len:
        event_audio = event_audio[:target_len] 
        event_len = target_len
    
    if event_len == 0:
        return factory_audio, 0, 0
        
    max_start = max(0, target_len - event_len)
    insert_pos = random.randint(0, max_start) if max_start > 0 else 0
    
    rms_factory = np.sqrt(np.mean(factory_audio**2))
    rms_event = np.sqrt(np.mean(event_audio**2))
    if rms_event > 0:
        # 위험 소리의 볼륨을 랜덤하게 조정 (더 현실적)
        volume_factor = random.uniform(0.3, 0.8)
        event_audio = event_audio * (rms_factory / rms_event) * volume_factor
    
    mixed_audio = factory_audio.copy()
    mixed_audio[insert_pos:insert_pos+event_len] += event_audio
    
    max_val = np.max(np.abs(mixed_audio))
    if max_val > 1.0:
        mixed_audio = mixed_audio / max_val
    
    return mixed_audio, insert_pos / sr, (insert_pos + event_len) / sr

# ---------------------------
# 5) YAMNet 임베딩 추출 함수
# ---------------------------
def extract_yamnet_embeddings(audio, sr, yamnet_model):
    waveform = tf.convert_to_tensor(audio, dtype=tf.float32)
    waveform = tf.squeeze(waveform)
    yamnet_fn = yamnet_model.signatures['serving_default']
    yamnet_output = yamnet_fn(waveform=waveform)
    embeddings = yamnet_output['output_1'].numpy()  # (frames, 1024)
    return embeddings

# ---------------------------
# 6) 라벨 생성 함수 (5-클래스)
# ---------------------------
def generate_labels(start_sec, end_sec, class_id, total_duration=10.0, frame_length=0.48):
    """
    class_id: 0=무음, 1=정상(공장소리), 2=화재, 3=가스누출, 4=비명
    """
    num_frames = int(total_duration / frame_length)
    labels = np.zeros(num_frames, dtype=int)  # 기본값: 무음(0)
    
    if class_id > 0:  # 무음이 아닌 경우
        if start_sec is not None and end_sec is not None and class_id > 1:
            # 위험 소리가 있는 경우 (화재, 가스, 비명)
            start_frame = int(start_sec / frame_length)
            end_frame = int(end_sec / frame_length) + 1
            labels[start_frame:end_frame] = class_id
            # 나머지 구간은 정상(공장소리)로 설정
            labels[:start_frame] = 1
            labels[end_frame:] = 1
        else:
            # 전체가 해당 클래스 (무음 또는 정상)
            labels[:] = class_id
    
    return labels

# ---------------------------
# 7) LSTM 모델 정의 함수 (5-클래스)
# ---------------------------
def create_lstm_model(input_shape, num_classes=5):
    model = Sequential([
        LSTM(128, input_shape=input_shape, return_sequences=True),
        Dropout(0.4),
        LSTM(64, return_sequences=True),
        Dropout(0.4),
        LSTM(32, return_sequences=True),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    optimizer = Adam(learning_rate=0.001)
    
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ---------------------------
# 8) 메인 실행 코드
# ---------------------------
def main():
    sr = 16000
    frame_length = 0.48
    total_duration = 10.0
    num_classes = 5  # 무음(0), 정상(1), 화재(2), 가스누출(3), 비명(4)
    
    # 클래스 매핑
    class_mapping = {
        'fire': 2,
        'gas': 3, 
        'scream': 4
    }
    
    # 1) 오디오 파일 경로 설정
    mixture_folder = 'mixture'
    envsound_folder = 'envsound'
    
    # 2) 공장 소리 파일들 불러오기
    factory_paths = glob.glob(os.path.join(mixture_folder, '*.wav'))
    print(f"공장 소리 파일 수: {len(factory_paths)}")
    
    # 3) 위험 소리 파일들 불러오기 (클래스별로 분리)
    event_folders = ['fire', 'gas', 'scream']
    event_data = {}
    
    for folder in event_folders:
        folder_path = os.path.join(envsound_folder, folder)
        wav_files = glob.glob(os.path.join(folder_path, '*.wav'))
        mp3_files = glob.glob(os.path.join(folder_path, '*.mp3'))
        event_data[folder] = wav_files + mp3_files
        print(f"{folder}: {len(event_data[folder])}개 파일")
    
    # YAMNet 모델을 한 번만 로드
    print("YAMNet 모델 로딩 중...")
    yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
    yamnet_model = hub.load(yamnet_model_handle)
    print("YAMNet 모델 로딩 완료!")
    
    X_data = []
    y_data = []
    
    # 4) 무음 데이터 생성 (증가)
    print(f"\n무음 데이터 생성 중... (총 {SILENCE_SAMPLES}개 샘플)")
    
    for i in range(SILENCE_SAMPLES):
        try:
            if (i + 1) % 30 == 0:  # 진행률 표시 간격 조정
                print(f"  무음 데이터 진행률: {i+1}/{SILENCE_SAMPLES}")
            
            # 완전 무음과 저음량 배경 소음을 섞어서 생성
            if i < SILENCE_SAMPLES // 2:
                audio = generate_silence(total_duration, sr)
            else:
                audio = generate_background_noise(total_duration, sr)
            
            embeddings = extract_yamnet_embeddings(audio, sr, yamnet_model)
            labels = generate_labels(None, None, 0, total_duration=total_duration, frame_length=frame_length)  # 무음 클래스
            
            X_data.append(embeddings)
            y_data.append(labels)
            
        except Exception as e:
            print(f"  무음 데이터 생성 중 오류: {e}")
            continue
    
    # 5) 정상 데이터 생성 (공장 소리만) - 증가
    print(f"\n정상 데이터 생성 중... (총 {NORMAL_SAMPLES}개 샘플)")
    
    for i in range(NORMAL_SAMPLES):
        try:
            if (i + 1) % 30 == 0:  # 진행률 표시 간격 조정
                print(f"  정상 데이터 진행률: {i+1}/{NORMAL_SAMPLES}")
                
            factory_path = random.choice(factory_paths)
            factory_audio, _ = librosa.load(factory_path, sr=sr)
            
            target_len = int(sr * total_duration)
            if len(factory_audio) > target_len:
                start_pos = random.randint(0, len(factory_audio) - target_len)
                factory_audio = factory_audio[start_pos:start_pos + target_len]
            else:
                factory_audio = np.pad(factory_audio, (0, max(0, target_len - len(factory_audio))))
            
            embeddings = extract_yamnet_embeddings(factory_audio, sr, yamnet_model)
            labels = generate_labels(None, None, 1, total_duration=total_duration, frame_length=frame_length)  # 정상 클래스
            
            X_data.append(embeddings)
            y_data.append(labels)
            
        except Exception as e:
            print(f"  정상 데이터 처리 중 오류: {e}")
            continue
    
    # 6) 위험 소리 데이터 생성 (균형 맞춤) - 화재 가중치 대폭 감소
    samples_per_class = DANGER_WEIGHTS
    
    for class_name, event_paths in event_data.items():
        class_id = class_mapping[class_name]
        samples_per_event = samples_per_class[class_name]
        print(f"\n{class_name} 클래스 데이터 생성 중... (클래스 ID: {class_id}, 파일당 {samples_per_event}개 샘플)")
        
        for idx, event_path in enumerate(event_paths):
            try:
                print(f"  처리 중: [{idx+1}/{len(event_paths)}] {os.path.basename(event_path)}")
                event_audio, _ = librosa.load(event_path, sr=sr)
                event_audio_ns = remove_silence(event_audio, sr, top_db=20)
                
                # 소수점 샘플링 처리
                if samples_per_event < 1:
                    # 확률적 샘플링 (예: 0.3이면 30% 확률로 1개 생성)
                    if random.random() < samples_per_event:
                        factory_path = random.choice(factory_paths)
                        factory_audio, _ = librosa.load(factory_path, sr=sr)
                        
                        mixed_audio, start_sec, end_sec = mix_factory_and_event(factory_audio, event_audio_ns, sr, desired_length=total_duration)
                        embeddings = extract_yamnet_embeddings(mixed_audio, sr, yamnet_model)
                        labels = generate_labels(start_sec, end_sec, class_id, total_duration=total_duration, frame_length=frame_length)
                        
                        X_data.append(embeddings)
                        y_data.append(labels)
                else:
                    # 기존 정수 샘플링
                    for i in range(int(samples_per_event)):
                        factory_path = random.choice(factory_paths)
                        factory_audio, _ = librosa.load(factory_path, sr=sr)
                        
                        mixed_audio, start_sec, end_sec = mix_factory_and_event(factory_audio, event_audio_ns, sr, desired_length=total_duration)
                        embeddings = extract_yamnet_embeddings(mixed_audio, sr, yamnet_model)
                        labels = generate_labels(start_sec, end_sec, class_id, total_duration=total_duration, frame_length=frame_length)
                        
                        X_data.append(embeddings)
                        y_data.append(labels)
                    
            except Exception as e:
                print(f"    파일 처리 중 오류 발생: {event_path}, 오류: {e}")
                continue
    
    # 7) 데이터 배열화 및 전처리
    print(f"\n총 데이터 샘플 수: {len(X_data)}")
    
    if len(X_data) == 0:
        print("데이터가 없습니다. 프로그램을 종료합니다.")
        return
    
    try:
        # 동일한 길이로 맞추기 (패딩)
        max_len = max([x.shape[0] for x in X_data])
        print(f"최대 프레임 길이: {max_len}")
        
        X_data_padded = []
        y_data_padded = []
        
        for i in range(len(X_data)):
            x = X_data[i]
            y = y_data[i]
            
            # 데이터 형태 검증
            if len(x.shape) != 2 or x.shape[1] != 1024:
                print(f"경고: 샘플 {i}의 X 데이터 형태가 이상합니다: {x.shape}")
                continue
                
            if len(y) != x.shape[0]:
                print(f"경고: 샘플 {i}의 X와 y 길이가 맞지 않습니다: X={x.shape[0]}, y={len(y)}")
                if len(y) < x.shape[0]:
                    y = np.pad(y, (0, x.shape[0] - len(y)), mode='constant')
                else:
                    y = y[:x.shape[0]]
            
            if x.shape[0] < max_len:
                x_padded = np.pad(x, ((0, max_len - x.shape[0]), (0, 0)), mode='constant')
                y_padded = np.pad(y, (0, max_len - len(y)), mode='constant')
            else:
                x_padded = x[:max_len]
                y_padded = y[:max_len]
            
            X_data_padded.append(x_padded)
            y_data_padded.append(y_padded)
        
        print(f"패딩 후 유효한 샘플 수: {len(X_data_padded)}")
        
        X_data = np.array(X_data_padded, dtype=np.float32)
        y_data = np.array(y_data_padded, dtype=np.int32)
        
        print(f"데이터 형태 - X: {X_data.shape}, y: {y_data.shape}")
        
        # One-hot 인코딩 (5-클래스)
        y_data_oh = to_categorical(y_data, num_classes=num_classes)
        print(f"One-hot 인코딩 후 y 형태: {y_data_oh.shape}")
        
        # 클래스별 데이터 분포 확인
        unique, counts = np.unique(y_data, return_counts=True)
        print("클래스별 프레임 수:")
        class_names = ['무음', '정상(공장)', '화재', '가스누출', '비명']
        for cls, count in zip(unique, counts):
            if cls < len(class_names):
                print(f"  {class_names[cls]}: {count}개 프레임")
        
        # 클래스별 비율 확인
        total_frames = np.sum(counts)
        print("\n클래스별 비율:")
        for cls, count in zip(unique, counts):
            if cls < len(class_names):
                ratio = count / total_frames * 100
                print(f"  {class_names[cls]}: {ratio:.1f}%")
        
        # 8) 학습/검증 분리
        X_train, X_val, y_train, y_val = train_test_split(X_data, y_data_oh, test_size=0.2, random_state=42)
        
        print(f"\n훈련 데이터: {X_train.shape}, 레이블: {y_train.shape}")
        print(f"검증 데이터: {X_val.shape}, 레이블: {y_val.shape}")
        
        # 9) LSTM 모델 생성
        input_shape = X_train.shape[1:]
        model = create_lstm_model(input_shape, num_classes)
        model.summary()
        
        # 콜백 설정
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=0.00001
            )
        ]
        
        # 10) 학습
        print(f"\n모델 학습 시작...")
        history = model.fit(
            X_train, y_train, 
            epochs=20,
            batch_size=8, 
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # 11) 모델 평가
        print("\n=== 모델 평가 ===")
        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        print(f"검증 손실: {val_loss:.4f}")
        print(f"검증 정확도: {val_acc:.4f}")
        
        # 12) 모델 저장
        model.save('yamnet_lstm_model_5class_with_silence.h5')
        print("\n5클래스 모델 저장 완료: yamnet_lstm_model_5class_with_silence.h5")
        
        # 클래스 매핑 정보 저장
        model_info = {
            'class_mapping': {**class_mapping, 'silence': 0, 'normal': 1},
            'class_names': class_names,
            'num_classes': num_classes
        }
        
        # pickle을 사용하여 딕셔너리 저장
        with open('model_info_5class.pkl', 'wb') as f:
            pickle.dump(model_info, f)
        print("모델 정보 저장 완료: model_info_5class.pkl")
        
        # 13) 모델 성능 및 정보 텍스트 파일로 저장
        from datetime import datetime
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        performance_log = f"""
========================================
5클래스 LSTM 모델 학습 보고서
========================================
학습 완료 시간: {current_time}

=== 모델 구성 ===
- 모델 타입: YAMNet + LSTM
- 클래스 수: {num_classes}개
- 입력 형태: {input_shape}
- 음성 샘플링 주파수: {sr} Hz
- 프레임 길이: {frame_length}초
- 총 오디오 길이: {total_duration}초

=== 클래스 정보 ===
클래스 매핑:
  - 0: 무음 (silence)
  - 1: 정상 (공장소음)
  - 2: 화재 (fire)
  - 3: 가스누출 (gas)
  - 4: 비명 (scream)

=== 데이터 생성 설정 ===
무음 데이터: {SILENCE_SAMPLES}개 샘플
정상 데이터: {NORMAL_SAMPLES}개 샘플
위험 소음 가중치:
  - 화재: {DANGER_WEIGHTS['fire']} (파일당 샘플 수)
  - 가스누출: {DANGER_WEIGHTS['gas']} (파일당 샘플 수)
  - 비명: {DANGER_WEIGHTS['scream']} (파일당 샘플 수)

=== 데이터 분포 ===
총 샘플 수: {len(X_data)}개
훈련 데이터: {X_train.shape[0]}개
검증 데이터: {X_val.shape[0]}개

클래스별 프레임 수:
"""
        # 클래스별 분포 정보 추가
        unique, counts = np.unique(y_data, return_counts=True)
        total_frames = np.sum(counts)
        for cls, count in zip(unique, counts):
            if cls < len(class_names):
                ratio = count / total_frames * 100
                performance_log += f"  - {class_names[cls]}: {count:,}개 프레임 ({ratio:.1f}%)\n"
        
        performance_log += f"""
=== 모델 성능 ===
최종 검증 손실: {val_loss:.6f}
최종 검증 정확도: {val_acc:.6f} ({val_acc*100:.2f}%)

=== 학습 설정 ===
에포크 수: 20
배치 크기: 8
최적화 알고리즘: Adam (learning_rate=0.001)
콜백:
  - EarlyStopping (patience=5)
  - ReduceLROnPlateau (patience=3, factor=0.5)

=== 저장된 파일 ===
- 모델 파일: yamnet_lstm_model_5class_with_silence.h5
- 모델 정보: model_info_5class.pkl
- 성능 보고서: model_performance_5class.txt

========================================
"""
        
        # 텍스트 파일로 저장
        with open('model_performance_5class.txt', 'w', encoding='utf-8') as f:
            f.write(performance_log)
        
        print("모델 성능 보고서 저장 완료: model_performance_5class.txt")
        
    except Exception as e:
        print(f"데이터 처리 중 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
