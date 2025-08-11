import numpy as np
import librosa
import soundfile as sf
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split
import os
import glob

def main():
    print("=== 간단한 테스트 버전 시작 ===")
    
    sr = 16000
    num_classes = 4
    
    # 1) 데이터 경로 확인
    mixture_folder = 'mixture'
    envsound_folder = 'envsound'
    
    factory_paths = glob.glob(os.path.join(mixture_folder, '*.wav'))
    print(f"공장 소리 파일 수: {len(factory_paths)}")
    
    if len(factory_paths) == 0:
        print("공장 소리 파일이 없습니다!")
        return
    
    # 2) 위험 소리 파일 확인
    event_folders = ['fire', 'gas', 'scream']
    event_data = {}
    
    for folder in event_folders:
        folder_path = os.path.join(envsound_folder, folder)
        wav_files = glob.glob(os.path.join(folder_path, '*.wav'))[:2]  # 각각 2개씩만
        event_data[folder] = wav_files
        print(f"{folder}: {len(event_data[folder])}개 파일")
    
    # 3) YAMNet 모델 로드
    print("\nYAMNet 모델 로딩 중...")
    try:
        yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
        yamnet_model = hub.load(yamnet_model_handle)
        print("YAMNet 모델 로딩 완료!")
    except Exception as e:
        print(f"YAMNet 로딩 실패: {e}")
        return
    
    # 4) 간단한 데이터 생성 (소량)
    X_data = []
    y_data = []
    
    class_mapping = {'fire': 1, 'gas': 2, 'scream': 3}
    
    print("\n데이터 생성 중...")
    sample_count = 0
    
    # 각 클래스별로 1개씩만 생성
    for class_name, event_paths in event_data.items():
        if len(event_paths) == 0:
            continue
            
        class_id = class_mapping[class_name]
        event_path = event_paths[0]  # 첫 번째 파일만 사용
        
        try:
            print(f"  처리 중: {class_name} - {os.path.basename(event_path)}")
            
            # 오디오 로드
            event_audio, _ = librosa.load(event_path, sr=sr, duration=5.0)  # 5초만 로드
            factory_audio, _ = librosa.load(factory_paths[0], sr=sr, duration=10.0)  # 10초만 로드
            
            # 간단한 합성 (덧셈)
            min_len = min(len(event_audio), len(factory_audio))
            mixed_audio = factory_audio[:min_len] + 0.3 * event_audio[:min_len]
            
            # YAMNet 임베딩
            waveform = tf.convert_to_tensor(mixed_audio, dtype=tf.float32)
            yamnet_fn = yamnet_model.signatures['serving_default']
            yamnet_output = yamnet_fn(waveform=waveform)
            embeddings = yamnet_output['output_1'].numpy()
            
            # 간단한 라벨 (모든 프레임을 같은 클래스로)
            labels = np.full(embeddings.shape[0], class_id, dtype=np.int32)
            
            X_data.append(embeddings)
            y_data.append(labels)
            sample_count += 1
            
            print(f"    임베딩 형태: {embeddings.shape}, 라벨 형태: {labels.shape}")
            
        except Exception as e:
            print(f"    오류 발생: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 정상 데이터 1개 추가
    try:
        print("  정상 데이터 생성 중...")
        factory_audio, _ = librosa.load(factory_paths[1], sr=sr, duration=10.0)
        
        waveform = tf.convert_to_tensor(factory_audio, dtype=tf.float32)
        yamnet_fn = yamnet_model.signatures['serving_default']
        yamnet_output = yamnet_fn(waveform=waveform)
        embeddings = yamnet_output['output_1'].numpy()
        
        labels = np.zeros(embeddings.shape[0], dtype=np.int32)  # 정상 클래스
        
        X_data.append(embeddings)
        y_data.append(labels)
        sample_count += 1
        
        print(f"    임베딩 형태: {embeddings.shape}, 라벨 형태: {labels.shape}")
        
    except Exception as e:
        print(f"    정상 데이터 생성 오류: {e}")
    
    print(f"\n총 생성된 샘플 수: {sample_count}")
    
    if sample_count == 0:
        print("생성된 데이터가 없습니다!")
        return
    
    # 5) 데이터 배열화
    try:
        print("\n데이터 배열화 중...")
        
        # 최대 길이 확인
        max_len = max([x.shape[0] for x in X_data])
        print(f"최대 프레임 길이: {max_len}")
        
        # 패딩
        X_data_padded = []
        y_data_padded = []
        
        for i, (x, y) in enumerate(zip(X_data, y_data)):
            if x.shape[0] < max_len:
                x_padded = np.pad(x, ((0, max_len - x.shape[0]), (0, 0)), mode='constant')
                y_padded = np.pad(y, (0, max_len - len(y)), mode='constant')
            else:
                x_padded = x[:max_len]
                y_padded = y[:max_len]
            
            X_data_padded.append(x_padded)
            y_data_padded.append(y_padded)
        
        X_data = np.array(X_data_padded, dtype=np.float32)
        y_data = np.array(y_data_padded, dtype=np.int32)
        
        print(f"배열화 완료 - X: {X_data.shape}, y: {y_data.shape}")
        
        # One-hot 인코딩
        y_data_oh = tf.keras.utils.to_categorical(y_data, num_classes=num_classes)
        print(f"One-hot 인코딩 완료: {y_data_oh.shape}")
        
        # 클래스 분포 확인
        unique, counts = np.unique(y_data, return_counts=True)
        class_names = ['정상', '화재', '가스누출', '비명']
        print("\n클래스별 프레임 수:")
        for cls, count in zip(unique, counts):
            if cls < len(class_names):
                print(f"  {class_names[cls]}: {count}개")
        
        print("\n=== 테스트 성공! 데이터 생성 완료 ===")
        print("이제 전체 버전을 실행해보세요.")
        
    except Exception as e:
        print(f"데이터 배열화 중 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
