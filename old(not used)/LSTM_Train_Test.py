import numpy as np
import librosa
import soundfile as sf
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow_hub as hub
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import os
import glob

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
# 2) 공장 소리와 위험 소리 합성 함수
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
        event_audio = event_audio * (rms_factory / rms_event) * 0.5
    
    mixed_audio = factory_audio.copy()
    mixed_audio[insert_pos:insert_pos+event_len] += event_audio
    
    max_val = np.max(np.abs(mixed_audio))
    if max_val > 1.0:
        mixed_audio = mixed_audio / max_val
    
    return mixed_audio, insert_pos / sr, (insert_pos + event_len) / sr

# ---------------------------
# 3) YAMNet 임베딩 추출 함수
# ---------------------------
def extract_yamnet_embeddings(audio, sr, yamnet_model):
    waveform = tf.convert_to_tensor(audio, dtype=tf.float32)
    waveform = tf.squeeze(waveform)
    yamnet_fn = yamnet_model.signatures['serving_default']
    yamnet_output = yamnet_fn(waveform=waveform)
    embeddings = yamnet_output['output_1'].numpy()  # (frames, 1024)
    return embeddings

# ---------------------------
# 4) 라벨 생성 함수
# ---------------------------
def generate_labels(start_sec, end_sec, total_duration=10.0, frame_length=0.48):
    num_frames = int(total_duration / frame_length)
    labels = np.zeros(num_frames, dtype=int)
    start_frame = int(start_sec / frame_length)
    end_frame = int(end_sec / frame_length) + 1
    labels[start_frame:end_frame] = 1
    return labels

# ---------------------------
# 5) LSTM 모델 정의 함수
# ---------------------------
def create_lstm_model(input_shape, num_classes=2):
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=True),  # 작은 모델로 시작
        Dropout(0.3),
        LSTM(32, return_sequences=True),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ---------------------------
# 6) 메인 실행 코드 (테스트 버전)
# ---------------------------
def main():
    sr = 16000
    frame_length = 0.48
    total_duration = 10.0
    num_classes = 2  # 정상(0), 위험(1)
    
    # 1) 오디오 파일 경로 설정
    mixture_folder = 'mixture'  # mixture 폴더
    envsound_folder = 'envsound'  # envsound 폴더
    
    # 2) 공장 소리 파일들 불러오기 (처음 10개만 테스트)
    factory_paths = glob.glob(os.path.join(mixture_folder, '*.wav'))[:10]
    print(f"테스트용 공장 소리 파일 수: {len(factory_paths)}")
    
    # 3) 위험 소리 파일들 불러오기 (각 폴더에서 2개씩만)
    event_folders = ['fire', 'gas', 'scream']
    event_paths = []
    for folder in event_folders:
        folder_path = os.path.join(envsound_folder, folder)
        wav_files = glob.glob(os.path.join(folder_path, '*.wav'))[:2]  # 각 폴더에서 2개씩만
        mp3_files = glob.glob(os.path.join(folder_path, '*.mp3'))[:1]  # mp3는 1개씩만
        event_paths.extend(wav_files + mp3_files)
    
    print(f"테스트용 위험 소리 파일 수: {len(event_paths)}")
    
    # YAMNet 모델을 한 번만 로드
    print("YAMNet 모델 로딩 중...")
    yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
    yamnet_model = hub.load(yamnet_model_handle)
    print("YAMNet 모델 로딩 완료!")
    
    X_data = []
    y_data = []
    
    # 4) 위험 소리와 공장 소리 합성 데이터 생성 (샘플 수 대폭 축소)
    samples_per_event = 2  # 각 위험 소리당 2개 샘플만
    
    print(f"위험 데이터 생성 시작... (총 {len(event_paths) * samples_per_event}개 샘플)")
    
    for idx, event_path in enumerate(event_paths):
        try:
            print(f"처리 중: [{idx+1}/{len(event_paths)}] {os.path.basename(event_path)}")
            event_audio, _ = librosa.load(event_path, sr=sr)
            event_audio_ns = remove_silence(event_audio, sr, top_db=20)
            
            for i in range(samples_per_event):
                factory_path = random.choice(factory_paths)
                factory_audio, _ = librosa.load(factory_path, sr=sr)
                
                mixed_audio, start_sec, end_sec = mix_factory_and_event(factory_audio, event_audio_ns, sr, desired_length=total_duration)
                embeddings = extract_yamnet_embeddings(mixed_audio, sr, yamnet_model)
                labels = generate_labels(start_sec, end_sec, total_duration=total_duration, frame_length=frame_length)
                
                X_data.append(embeddings)
                y_data.append(labels)
                
        except Exception as e:
            print(f"파일 처리 중 오류 발생: {event_path}, 오류: {e}")
            continue
    
    # 5) 정상 데이터 추가 (소량)
    normal_samples = 10  # 정상 데이터 10개만
    print(f"정상 데이터 생성 시작... (총 {normal_samples}개 샘플)")
    
    for i in range(normal_samples):
        try:
            factory_path = random.choice(factory_paths)
            factory_audio, _ = librosa.load(factory_path, sr=sr)
            
            target_len = int(sr * total_duration)
            if len(factory_audio) > target_len:
                factory_audio = factory_audio[:target_len]
            else:
                factory_audio = np.pad(factory_audio, (0, max(0, target_len - len(factory_audio))))
            
            embeddings = extract_yamnet_embeddings(factory_audio, sr, yamnet_model)
            labels = np.zeros(embeddings.shape[0], dtype=int)
            X_data.append(embeddings)
            y_data.append(labels)
            
        except Exception as e:
            print(f"정상 데이터 처리 중 오류 발생: {factory_path}, 오류: {e}")
            continue
    
    # 6) 데이터 배열화 및 전처리
    print(f"총 데이터 샘플 수: {len(X_data)}")
    
    if len(X_data) == 0:
        print("데이터가 없습니다. 프로그램을 종료합니다.")
        return
    
    try:
        # 동일한 길이로 맞추기 (패딩)
        max_len = max([x.shape[0] for x in X_data])
        X_data_padded = []
        y_data_padded = []
        
        for i in range(len(X_data)):
            x = X_data[i]
            y = y_data[i]
            
            if x.shape[0] < max_len:
                x_padded = np.pad(x, ((0, max_len - x.shape[0]), (0, 0)), mode='constant')
                y_padded = np.pad(y, (0, max_len - len(y)), mode='constant')
            else:
                x_padded = x[:max_len]
                y_padded = y[:max_len]
            
            X_data_padded.append(x_padded)
            y_data_padded.append(y_padded)
        
        X_data = np.array(X_data_padded)
        y_data = np.array(y_data_padded)
        
        print(f"데이터 형태 - X: {X_data.shape}, y: {y_data.shape}")
        
        # One-hot 인코딩
        y_data_oh = tf.keras.utils.to_categorical(y_data, num_classes=num_classes)
        
        # 7) 학습/검증 분리
        X_train, X_val, y_train, y_val = train_test_split(X_data, y_data_oh, test_size=0.2, random_state=42)
        
        print(f"훈련 데이터: {X_train.shape}, 레이블: {y_train.shape}")
        print(f"검증 데이터: {X_val.shape}, 레이블: {y_val.shape}")
        
        # 8) LSTM 모델 생성
        input_shape = X_train.shape[1:]  # (time_steps, 1024)
        model = create_lstm_model(input_shape, num_classes)
        model.summary()
        
        # 9) 학습 (적은 에폭으로)
        model.fit(X_train, y_train, epochs=5, batch_size=4, validation_data=(X_val, y_val))
        
        # 10) 모델 저장
        model.save('yamnet_lstm_model_test.h5')
        print("테스트 모델 저장 완료: yamnet_lstm_model_test.h5")
        
    except Exception as e:
        print(f"데이터 처리 중 오류: {e}")
        print("메모리 부족이거나 데이터 형태 문제일 수 있습니다.")

if __name__ == '__main__':
    main()
