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
    
    max_start = target_len - event_len
    insert_pos = random.randint(0, max_start)
    
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
def extract_yamnet_embeddings(audio, sr, yamnet_model=None):
    if yamnet_model is None:
        yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
        yamnet_model = hub.load(yamnet_model_handle)
    
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
        LSTM(128, input_shape=input_shape, return_sequences=True),
        Dropout(0.3),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ---------------------------
# 6) 메인 실행 코드
# ---------------------------
def main():
    sr = 16000
    frame_length = 0.48
    total_duration = 10.0
    num_classes = 2  # 정상(0), 위험(1)
    
    # 1) 오디오 파일 경로 설정
    mixture_folder = 'mixture'  # mixture 폴더
    envsound_folder = 'envsound'  # envsound 폴더
    
    # 2) 공장 소리 파일들 불러오기 (mixture 폴더에서 .wav 파일들)
    factory_paths = glob.glob(os.path.join(mixture_folder, '*.wav'))
    print(f"공장 소리 파일 수: {len(factory_paths)}")
    
    # 3) 위험 소리 파일들 불러오기
    event_folders = ['fire', 'gas', 'scream']
    event_paths = []
    for folder in event_folders:
        folder_path = os.path.join(envsound_folder, folder)
        # .wav와 .mp3 파일 모두 포함
        wav_files = glob.glob(os.path.join(folder_path, '*.wav'))
        mp3_files = glob.glob(os.path.join(folder_path, '*.mp3'))
        event_paths.extend(wav_files + mp3_files)
    
    print(f"위험 소리 파일 수: {len(event_paths)}")
    print(f"위험 소리 폴더별 파일 수:")
    for folder in event_folders:
        folder_path = os.path.join(envsound_folder, folder)
        files = glob.glob(os.path.join(folder_path, '*.wav')) + glob.glob(os.path.join(folder_path, '*.mp3'))
        print(f"  {folder}: {len(files)}개")
    
    # YAMNet 모델을 한 번만 로드
    print("YAMNet 모델 로딩 중...")
    yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
    yamnet_model = hub.load(yamnet_model_handle)
    print("YAMNet 모델 로딩 완료!")
    
    X_data = []
    y_data = []
    
    # 4) 위험 소리와 공장 소리 합성 데이터 생성
    samples_per_event = 5  # 각 위험 소리당 5개 샘플 (메모리 효율성을 위해 20→5로 줄임)
    total_event_samples = len(event_paths) * samples_per_event
    
    print(f"위험 데이터 생성 시작... (총 {total_event_samples}개 샘플)")
    
    for idx, event_path in enumerate(event_paths):
        try:
            print(f"처리 중: [{idx+1}/{len(event_paths)}] {os.path.basename(event_path)}")
            event_audio, _ = librosa.load(event_path, sr=sr)
            event_audio_ns = remove_silence(event_audio, sr, top_db=20)
            
            for i in range(samples_per_event):
                # 랜덤하게 공장 소리 선택
                factory_path = random.choice(factory_paths)
                factory_audio, _ = librosa.load(factory_path, sr=sr)
                
                mixed_audio, start_sec, end_sec = mix_factory_and_event(factory_audio, event_audio_ns, sr, desired_length=total_duration)
                embeddings = extract_yamnet_embeddings(mixed_audio, sr, yamnet_model)  # shape: (frames, 1024)
                labels = generate_labels(start_sec, end_sec, total_duration=total_duration, frame_length=frame_length)
                
                X_data.append(embeddings)
                y_data.append(labels)
                
            if (idx + 1) % 10 == 0:
                print(f"진행률: {idx+1}/{len(event_paths)} 파일 완료")
                
        except Exception as e:
            print(f"파일 처리 중 오류 발생: {event_path}, 오류: {e}")
            continue
    
    # 5) 정상 데이터 추가 (공장 소리만)
    normal_samples = len(event_paths) * samples_per_event // 2  # 위험 데이터의 절반 정도
    print(f"정상 데이터 생성 시작... (총 {normal_samples}개 샘플)")
    
    for i in range(normal_samples):
        try:
            if (i + 1) % 100 == 0:
                print(f"정상 데이터 진행률: {i+1}/{normal_samples}")
                
            factory_path = random.choice(factory_paths)
            factory_audio, _ = librosa.load(factory_path, sr=sr)
            
            # 원본 길이에 맞게 자르거나 패딩
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
    print("데이터를 numpy 배열로 변환 중...")
    
    # 메모리 효율성을 위해 배치별로 처리
    try:
        X_data = np.array(X_data)
        y_data = np.array(y_data)
        print(f"데이터 형태 - X: {X_data.shape}, y: {y_data.shape}")
    except MemoryError:
        print("메모리 부족으로 인해 데이터 샘플 수를 줄여주세요.")
        return
    
    # One-hot 인코딩 (time_steps, 2)
    y_data_oh = tf.keras.utils.to_categorical(y_data, num_classes=num_classes)
    
    # 7) 학습/검증 분리
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data_oh, test_size=0.2, random_state=42)
    
    print(f"훈련 데이터: {X_train.shape}, 레이블: {y_train.shape}")
    print(f"검증 데이터: {X_val.shape}, 레이블: {y_val.shape}")
    
    # 8) LSTM 모델 생성
    input_shape = X_train.shape[1:]  # (time_steps, 1024)
    model = create_lstm_model(input_shape, num_classes)
    model.summary()
    
    # 9) 학습
    model.fit(X_train, y_train, epochs=15, batch_size=8, validation_data=(X_val, y_val))
    
    # 10) 모델 저장
    model.save('yamnet_lstm_model.h5')
    print("모델 저장 완료: yamnet_lstm_model.h5")

if __name__ == '__main__':
    main()
