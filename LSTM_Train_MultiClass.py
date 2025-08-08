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
from sklearn.utils.class_weight import compute_class_weight
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
# 4) 라벨 생성 함수 (4-클래스)
# ---------------------------
def generate_labels(start_sec, end_sec, class_id, total_duration=10.0, frame_length=0.48):
    """
    class_id: 0=정상, 1=화재, 2=가스누출, 3=비명
    """
    num_frames = int(total_duration / frame_length)
    labels = np.zeros(num_frames, dtype=int)  # 기본값: 정상(0)
    
    if class_id > 0:  # 위험 소리가 있는 경우
        start_frame = int(start_sec / frame_length)
        end_frame = int(end_sec / frame_length) + 1
        labels[start_frame:end_frame] = class_id
    
    return labels

# ---------------------------
# 5) LSTM 모델 정의 함수 (4-클래스, 불균형 데이터 고려)
# ---------------------------
def create_lstm_model(input_shape, num_classes=4):
    model = Sequential([
        LSTM(128, input_shape=input_shape, return_sequences=True),
        Dropout(0.4),
        LSTM(64, return_sequences=True),
        Dropout(0.4),
        LSTM(32, return_sequences=True),  # 추가 LSTM 레이어
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    # 학습률 조정된 옵티마이저 사용
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # 학습률 증가
    
    # 가중치가 적용된 손실 함수 사용
    model.compile(optimizer=optimizer,
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
    num_classes = 4  # 정상(0), 화재(1), 가스누출(2), 비명(3)
    
    # 클래스 매핑
    class_mapping = {
        'fire': 1,
        'gas': 2, 
        'scream': 3
    }
    
    # 1) 오디오 파일 경로 설정
    mixture_folder = 'mixture'  # mixture 폴더
    envsound_folder = 'envsound'  # envsound 폴더
    
    # 2) 공장 소리 파일들 불러오기
    factory_paths = glob.glob(os.path.join(mixture_folder, '*.wav'))
    print(f"공장 소리 파일 수: {len(factory_paths)}")
    
    # 3) 위험 소리 파일들 불러오기 (클래스별로 분리)
    event_folders = ['fire', 'gas', 'scream']
    event_data = {}  # {class_name: [file_paths]}
    
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
    
    # 4) 각 클래스별로 위험 소리와 공장 소리 합성 데이터 생성
    # 클래스별 균형을 맞추기 위해 샘플 수 조정 (더 균등하게)
    samples_per_class = {
        'fire': 1,     # 화재: 130개 파일 × 1 = 130개 샘플
        'gas': 25,     # 가스: 3개 파일 × 25 = 75개 샘플  
        'scream': 15   # 비명: 7개 파일 × 15 = 105개 샘플
    }
    
    for class_name, event_paths in event_data.items():
        class_id = class_mapping[class_name]
        samples_per_event = samples_per_class[class_name]  # 클래스별 샘플 수
        print(f"\n{class_name} 클래스 데이터 생성 중... (클래스 ID: {class_id}, 파일당 {samples_per_event}개 샘플)")
        
        for idx, event_path in enumerate(event_paths):
            try:
                print(f"  처리 중: [{idx+1}/{len(event_paths)}] {os.path.basename(event_path)}")
                event_audio, _ = librosa.load(event_path, sr=sr)
                event_audio_ns = remove_silence(event_audio, sr, top_db=20)
                
                for i in range(samples_per_event):
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
    
    # 5) 정상 데이터 추가 (공장 소리만)
    # 위험 데이터와 균형을 맞추기 위해 적절한 정상 데이터 개수 계산
    total_danger_samples = 0
    for class_name, paths in event_data.items():
        total_danger_samples += len(paths) * samples_per_class[class_name]
    
    normal_samples = total_danger_samples // 3  # 위험 데이터의 1/3 정도로 줄임
    print(f"\n정상 데이터 생성 시작... (총 {normal_samples}개 샘플)")
    
    for i in range(normal_samples):
        try:
            if (i + 1) % 50 == 0:
                print(f"  정상 데이터 진행률: {i+1}/{normal_samples}")
                
            factory_path = random.choice(factory_paths)
            factory_audio, _ = librosa.load(factory_path, sr=sr)
            
            target_len = int(sr * total_duration)
            if len(factory_audio) > target_len:
                factory_audio = factory_audio[:target_len]
            else:
                factory_audio = np.pad(factory_audio, (0, max(0, target_len - len(factory_audio))))
            
            embeddings = extract_yamnet_embeddings(factory_audio, sr, yamnet_model)
            labels = np.zeros(embeddings.shape[0], dtype=int)  # 모든 프레임이 정상(0)
            X_data.append(embeddings)
            y_data.append(labels)
            
        except Exception as e:
            print(f"  정상 데이터 처리 중 오류 발생: {factory_path}, 오류: {e}")
            continue
    
    # 6) 데이터 배열화 및 전처리
    print(f"\n총 데이터 샘플 수: {len(X_data)}")
    
    if len(X_data) == 0:
        print("데이터가 없습니다. 프로그램을 종료합니다.")
        return
    
    try:
        # 동일한 길이로 맞추기 (패딩)
        if len(X_data) == 0:
            raise ValueError("X_data가 비어있습니다.")
            
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
                # y 길이를 x에 맞춤
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
        
        if len(X_data_padded) == 0:
            raise ValueError("유효한 데이터가 없습니다.")
        
        print(f"패딩 후 유효한 샘플 수: {len(X_data_padded)}")
        
        X_data = np.array(X_data_padded, dtype=np.float32)
        y_data = np.array(y_data_padded, dtype=np.int32)
        
        print(f"데이터 형태 - X: {X_data.shape}, y: {y_data.shape}")
        print(f"X 데이터 타입: {X_data.dtype}, y 데이터 타입: {y_data.dtype}")
        
        # One-hot 인코딩 (4-클래스)
        y_data_oh = tf.keras.utils.to_categorical(y_data, num_classes=num_classes)
        print(f"One-hot 인코딩 후 y 형태: {y_data_oh.shape}")
        
        # 클래스별 데이터 분포 확인
        unique, counts = np.unique(y_data, return_counts=True)
        print("클래스별 프레임 수:")
        class_names = ['정상', '화재', '가스누출', '비명']
        for cls, count in zip(unique, counts):
            print(f"  {class_names[cls]}: {count}개 프레임")
        
        # 클래스별 비율 확인
        total_frames = np.sum(counts)
        print("\n클래스별 비율:")
        for cls, count in zip(unique, counts):
            ratio = count / total_frames * 100
            print(f"  {class_names[cls]}: {ratio:.1f}%")
        
        print("\n참고: 시퀀스 모델에서는 class_weight를 사용하지 않고")
        print("데이터 샘플링과 모델 아키텍처로 불균형을 해결합니다.")
        
        # 7) 학습/검증 분리
        try:
            X_train, X_val, y_train, y_val = train_test_split(X_data, y_data_oh, test_size=0.2, random_state=42)
            print(f"\n훈련 데이터: {X_train.shape}, 레이블: {y_train.shape}")
            print(f"검증 데이터: {X_val.shape}, 레이블: {y_val.shape}")
        except Exception as e:
            print(f"데이터 분리 중 오류: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # 8) LSTM 모델 생성
        try:
            input_shape = X_train.shape[1:]  # (time_steps, 1024)
            model = create_lstm_model(input_shape, num_classes)
            model.summary()
        except Exception as e:
            print(f"모델 생성 중 오류: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # 콜백 설정
        try:
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=3,
                    min_lr=0.00001
                )
            ]
        except Exception as e:
            print(f"콜백 설정 중 오류: {e}")
            callbacks = []
        
        # 9) 학습 (가중치 적용 - class_weight 대신 다른 방법 사용)
        try:
            print(f"\n모델 학습 시작...")
            
            # 시퀀스 데이터의 경우 class_weight가 작동하지 않으므로 제거
            # 대신 모델 자체에서 불균형을 처리하도록 함
            history = model.fit(
                X_train, y_train, 
                epochs=15,  # 에폭 수 감소
                batch_size=8, 
                validation_data=(X_val, y_val),
                # class_weight=adjusted_weights,  # 제거: 시퀀스 데이터에서는 작동하지 않음
                callbacks=callbacks,  # 콜백 추가
                verbose=1
            )
        except Exception as e:
            print(f"모델 학습 중 오류: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # 10) 모델 평가
        try:
            print("\n=== 모델 평가 ===")
            val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
            print(f"검증 손실: {val_loss:.4f}")
            print(f"검증 정확도: {val_acc:.4f}")
        except Exception as e:
            print(f"모델 평가 중 오류: {e}")
            import traceback
            traceback.print_exc()
        
        # 클래스별 예측 성능 확인
        try:
            y_pred = model.predict(X_val, verbose=0)
            y_pred_classes = np.argmax(y_pred, axis=-1)  # (batch_size, time_steps)
            y_true_classes = np.argmax(y_val, axis=-1)   # (batch_size, time_steps)
            
            # 클래스별 정확도 계산 (프레임 단위로)
            print("\n클래스별 예측 성능 (프레임 단위):")
            
            # 모든 프레임을 1차원으로 평탄화
            y_pred_flat = y_pred_classes.flatten()
            y_true_flat = y_true_classes.flatten()
            
            for class_id in range(num_classes):
                # 해당 클래스에 속하는 프레임들 찾기
                class_mask = (y_true_flat == class_id)
                class_count = int(np.sum(class_mask))  # 명시적으로 int로 변환
                
                if class_count > 0:
                    # 해당 클래스 프레임들의 예측 정확도
                    correct_predictions = int(np.sum(y_pred_flat[class_mask] == class_id))  # 명시적으로 int로 변환
                    class_acc = float(correct_predictions) / float(class_count)  # 명시적으로 float로 변환
                    print(f"  {class_names[class_id]}: {class_acc:.3f} ({correct_predictions}/{class_count} 프레임)")
                else:
                    print(f"  {class_names[class_id]}: 데이터 없음")
        except Exception as e:
            print(f"클래스별 성능 계산 중 오류: {e}")
            import traceback
            traceback.print_exc()
        
        # 11) 모델 저장
        model.save('yamnet_lstm_model_multiclass_balanced.h5')
        print("\n균형 조정된 다중 클래스 모델 저장 완료: yamnet_lstm_model_multiclass_balanced.h5")
        
        # 클래스 매핑 및 정보 저장
        model_info = {
            'class_mapping': class_mapping,
            'class_names': class_names,
            'num_classes': num_classes
        }
        np.save('model_info_balanced.npy', model_info)
        print("모델 정보 저장 완료: model_info_balanced.npy")
        
    except Exception as e:
        print(f"데이터 처리 중 오류: {e}")
        import traceback
        print("전체 오류 스택:")
        traceback.print_exc()
        print("메모리 부족이거나 데이터 형태 문제일 수 있습니다.")

if __name__ == '__main__':
    main()
