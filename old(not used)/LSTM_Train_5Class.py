"""
5클래스 LSTM 모델 학습 스크립트 v1.3
=====================================

이 스크립트는 YAMNet + LSTM을 사용하여 5개 클래스(무음, 정상, 화재, 가스누출, 비명)를  
분류하는 모델을 학습합니다.

데이터 가중치 설정 방법:
- VERSION: 모델 버전 (결과물 폴더명에 사용)
- SILENCE_SAMPLES: 무음 데이터 샘플 수
- NORMAL_SAMPLES: 정상(공장소음) 데이터 샘플 수  
- TRANSITION_SAMPLES: 무음→공장소리 전환 데이터 샘플 수
- DANGER_TRANSITION_SAMPLES: 무음→위험소리 전환 데이터 샘플 수
- FACTORY_TRANSITION_SAMPLES: 공장소리→다른공장소리 전환 데이터 샘플 수 (v1.3 신규)
- SILENCE_VARIATION_SAMPLES: 무음→다른무음 변화 데이터 샘플 수 (v1.3 신규)
- AUTO_WEIGHT_CALCULATION: 자동 가중치 계산 여부
  - True: envsound 폴더의 파일 개수를 기반으로 자동 계산
  - False: MANUAL_DANGER_WEIGHTS 사용
- MANUAL_DANGER_WEIGHTS: 수동 설정 가중치 딕셔너리
  - 'fire': 화재 소음 가중치
  - 'gas': 가스누출 소음 가중치  
  - 'scream': 비명 소음 가중치

v1.3 새로 추가된 훈련 데이터:
1. 공장소리→다른공장소리 전환:
   - 서로 다른 공장소리 파일 간의 전환 시나리오
   - 30~70% 지점에서 0.5초 페이드 전환
   - 모든 프레임이 정상(공장) 클래스로 라벨링
   - 목적: 공장소리 변화 시에도 안정적으로 "정상" 판단

2. 무음 변화 (다양한 배경소음):
   - 순수무음 ↔ 저잡음 ↔ 화이트노이즈 간 변화
   - 20~80% 지점에서 0.2초 페이드 전환
   - 모든 프레임이 무음 클래스로 라벨링
   - 목적: 다양한 배경소음 패턴에서도 안정적으로 "무음" 판단

기존 전환 데이터:
- 무음 상태에서 갑자기 공장 소리가 시작되는 현실적인 시나리오
- 무음 상태에서 갑자기 위험 소리가 시작되는 긴급 상황 시나리오
- 전체 길이의 20~80% 지점에서 무음→공장소리/위험소리 전환
- 0.5초 페이드인 효과로 자연스러운 전환 구현
- 무음 구간은 클래스 0, 전환 후 구간은 해당 클래스로 라벨링

자동 가중치 계산:
- 각 클래스당 목표 샘플 수를 250개로 설정 (조정됨)
- 파일 개수가 많은 클래스는 낮은 가중치, 적은 클래스는 높은 가중치
- 가중치가 1 미만이면 확률적 샘플링 적용

출력 파일 (버전별 폴더에 저장):
- yamnet_lstm_model_{VERSION}.h5: 학습된 모델
- model_info_{VERSION}.pkl: 모델 정보 (클래스 매핑, 이름 등)
- model_performance_{VERSION}.txt: 상세한 학습 결과 보고서
- dataset_info_{VERSION}.json: 훈련/검증/테스트 데이터셋 파일 정보
- summary_{VERSION}.json: 요약 정보 (성능, 설정값 등)

폴더 구조:
model_results_{VERSION}/
├── yamnet_lstm_model_{VERSION}.h5
├── model_info_{VERSION}.pkl  
├── model_performance_{VERSION}.txt
├── dataset_info_{VERSION}.json
└── summary_{VERSION}.json
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
from sklearn.metrics import classification_report, confusion_matrix
import os
import glob
import pickle
import json

# ================================
# 데이터 생성 가중치 설정 (수정 가능)
# ================================
# v1.4: 스마트 세그멘테이션 기본 설정
TARGET_SAMPLES_PER_CLASS = 180                  # 각 클래스당 목표 샘플 수 (균등 분포)

# v1.4: 모든 클래스 균등 분포를 위한 조정
SILENCE_SAMPLES = TARGET_SAMPLES_PER_CLASS      # 무음 데이터 샘플 수 (180개)
NORMAL_SAMPLES = TARGET_SAMPLES_PER_CLASS       # 정상(공장소음) 데이터 샘플 수 (180개)
TRANSITION_SAMPLES = 80                         # 무음→공장소리 전환 데이터 샘플 수 
DANGER_TRANSITION_SAMPLES = 60                  # 무음→위험소리 전환 데이터 샘플 수 

# v1.4 새로 추가된 데이터 타입 (v1.4에서 조정)
FACTORY_TRANSITION_SAMPLES = 60                 # 공장소리→다른공장소리 전환 데이터 샘플 수
SILENCE_VARIATION_SAMPLES = 40                  # 무음→다른무음 변화 데이터 샘플 수

# 버전 관리 설정
VERSION = "v1.4"  # 모델 버전 (결과물 폴더명에 사용)

# 위험 소음별 가중치 (파일당 생성할 샘플 수)
# 자동 계산 또는 수동 설정 가능
AUTO_WEIGHT_CALCULATION = True   # True: 스마트 세그멘테이션 기반 자동 계산, False: 수동 설정

# v1.35 새로 추가: 스마트 세그멘테이션 설정
ENABLE_SMART_SEGMENTATION = True  # 실제 소리 구간만 추출하여 학습
MIN_SEGMENT_DURATION = 2.0        # 최소 세그먼트 길이 (초)
MAX_SEGMENT_DURATION = 7.0        # 최대 세그먼트 길이 (초) 
SILENCE_THRESHOLD = 0.01          # 무음 판정 임계값 (RMS 기준)
MIN_SOUND_RATIO = 0.4             # 세그먼트 내 최소 소리 비율 (40% 이상이 소리여야 함)
OVERLAP_RATIO = 0.3               # 긴 오디오 분할시 겹침 비율

# 수동 설정시 사용되는 가중치 (ENABLE_SMART_SEGMENTATION=False일 때만 사용)
MANUAL_DANGER_WEIGHTS = {
    'fire': 1.2,      # 화재: 파일당 1.2개 샘플 
    'gas': 50,        # 가스누출: 파일당 50개 샘플  
    'scream': 22      # 비명: 파일당 22개 샘플
}

def calculate_auto_weights(envsound_folder, target_samples_per_class=250):
    """
    각 클래스의 파일 개수를 기반으로 자동으로 가중치를 계산합니다.
    
    Args:
        envsound_folder: 위험 소음 폴더 경로
        target_samples_per_class: 각 클래스당 목표 샘플 수 (250으로 조정)
    
    Returns:
        dict: 각 클래스별 파일당 샘플 수
    """
    event_folders = ['fire', 'gas', 'scream']  # spark는 제외 (클래스 매핑에 없음)
    file_counts = {}
    weights = {}
    
    print("📊 위험 소음 파일 개수 분석:")
    print("-" * 40)
    
    # 각 폴더의 파일 개수 계산
    for folder in event_folders:
        folder_path = os.path.join(envsound_folder, folder)
        if os.path.exists(folder_path):
            wav_files = glob.glob(os.path.join(folder_path, '*.wav'))
            mp3_files = glob.glob(os.path.join(folder_path, '*.mp3'))
            total_files = len(wav_files) + len(mp3_files)
            file_counts[folder] = total_files
            print(f"  {folder}: {total_files}개 파일")
        else:
            file_counts[folder] = 0
            print(f"  {folder}: 폴더 없음")
    
    print(f"\n🎯 목표: 각 클래스당 {target_samples_per_class}개 샘플 생성")
    print("⚖️ 자동 계산된 가중치:")
    print("-" * 40)
    
    # 가중치 계산
    for folder, file_count in file_counts.items():
        if file_count > 0:
            samples_per_file = target_samples_per_class / file_count
            
            # 너무 작은 값은 확률적 샘플링으로 처리
            if samples_per_file < 1:
                weights[folder] = round(samples_per_file, 2)
                print(f"  {folder}: {samples_per_file:.2f} (확률적 샘플링: {samples_per_file*100:.1f}%)")
            else:
                weights[folder] = max(1, round(samples_per_file))
                print(f"  {folder}: {weights[folder]} (파일당 샘플 수)")
        else:
            weights[folder] = 0
            print(f"  {folder}: 0 (파일 없음)")
    
    print(f"\n예상 총 샘플 수:")
    total_expected = 0
    for folder, weight in weights.items():
        expected = file_counts[folder] * weight if weight >= 1 else file_counts[folder] * weight
        total_expected += expected
        print(f"  {folder}: {expected:.0f}개")
    print(f"  총합: {total_expected:.0f}개")
    
    return weights

def create_version_folder(version):
    """
    버전별 결과물 저장 폴더를 생성합니다.
    
    Args:
        version: 버전 문자열 (예: "v1.0")
    
    Returns:
        str: 생성된 폴더 경로
    """
    folder_name = f"model_results_{version}"
    
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"📁 결과물 폴더 생성: {folder_name}")
    
    return folder_name

# ---------------------------
# 1) 스마트 세그멘테이션 함수들 (v1.35 새로 추가)
# ---------------------------
def detect_sound_activity(audio, sr, threshold=0.01, frame_length=2048, hop_length=512):
    """
    오디오에서 실제 소리가 있는 구간을 감지합니다.
    
    Args:
        audio: 오디오 데이터
        sr: 샘플링 레이트
        threshold: RMS 임계값
        frame_length: 프레임 길이
        hop_length: 홉 길이
    
    Returns:
        numpy.array: 각 프레임별 소리 활동 여부 (True/False)
    """
    # RMS 에너지 계산
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    
    # 임계값보다 큰 구간을 활성 구간으로 판정
    activity = rms > threshold
    
    return activity

def extract_sound_segments(audio, sr, min_duration=2.0, max_duration=7.0, 
                          threshold=0.01, min_sound_ratio=0.4):
    """
    오디오에서 실제 소리가 있는 의미있는 세그먼트들을 추출합니다.
    간헐적 소리 패턴(비명-무음-비명)도 하나의 세그먼트로 처리합니다.
    
    Args:
        audio: 입력 오디오
        sr: 샘플링 레이트
        min_duration: 최소 세그먼트 길이 (초)
        max_duration: 최대 세그먼트 길이 (초)
        threshold: 소리 감지 임계값
        min_sound_ratio: 세그먼트 내 최소 소리 비율
    
    Returns:
        list: 추출된 오디오 세그먼트 리스트
    """
    if len(audio) == 0:
        return []
    
    # 소리 활동 감지
    hop_length = 512
    frame_length = 2048
    activity = detect_sound_activity(audio, sr, threshold, frame_length, hop_length)
    
    # 프레임을 샘플 인덱스로 변환
    frame_to_sample = lambda frame_idx: frame_idx * hop_length
    
    # 짧은 무음 구간을 메워서 연결 (간헐적 소리 패턴 처리)
    max_gap_frames = int(2.0 * sr / hop_length)  # 2초 이하의 무음은 메움
    filled_activity = activity.copy()
    
    # 활성 구간 사이의 짧은 무음을 메우기
    active_indices = np.where(activity)[0]
    if len(active_indices) > 1:
        for i in range(len(active_indices) - 1):
            start_gap = active_indices[i] + 1
            end_gap = active_indices[i + 1]
            gap_length = end_gap - start_gap
            
            # 2초 이하의 무음 구간은 활성으로 변경 (비명-무음-비명 패턴 처리)
            if gap_length <= max_gap_frames:
                filled_activity[start_gap:end_gap] = True
    
    # 연속된 활성 구간 찾기 (개선된 버전)
    active_regions = []
    start_frame = None
    
    for i, is_active in enumerate(filled_activity):
        if is_active and start_frame is None:
            start_frame = i
        elif not is_active and start_frame is not None:
            end_frame = i
            duration = (end_frame - start_frame) * hop_length / sr
            
            if duration >= min_duration:
                start_sample = frame_to_sample(start_frame)
                end_sample = frame_to_sample(end_frame)
                active_regions.append((start_sample, end_sample))
            
            start_frame = None
    
    # 마지막 구간 처리
    if start_frame is not None:
        end_frame = len(filled_activity)
        duration = (end_frame - start_frame) * hop_length / sr
        if duration >= min_duration:
            start_sample = frame_to_sample(start_frame)
            end_sample = min(frame_to_sample(end_frame), len(audio))
            active_regions.append((start_sample, end_sample))
    
    # 세그먼트 추출
    segments = []
    
    for start_sample, end_sample in active_regions:
        segment_audio = audio[start_sample:end_sample]
        segment_duration = len(segment_audio) / sr
        
        # 너무 긴 세그먼트는 분할
        if segment_duration > max_duration:
            # 겹침을 고려한 분할
            segment_samples = int(max_duration * sr)
            overlap_samples = int(OVERLAP_RATIO * segment_samples)
            step_samples = segment_samples - overlap_samples
            
            for i in range(0, len(segment_audio) - segment_samples + 1, step_samples):
                sub_segment = segment_audio[i:i + segment_samples]
                
                # 소리 비율 확인 (원본 activity 사용)
                sub_start_frame = int((start_sample + i) / hop_length)
                sub_end_frame = int((start_sample + i + segment_samples) / hop_length)
                sub_end_frame = min(sub_end_frame, len(activity))
                
                if sub_end_frame > sub_start_frame:
                    sub_activity = activity[sub_start_frame:sub_end_frame]
                    sound_ratio = np.sum(sub_activity) / len(sub_activity) if len(sub_activity) > 0 else 0
                    
                    # 간헐적 소리의 경우 기준을 낮춤 (30%)
                    adjusted_min_ratio = min_sound_ratio * 0.75
                    if sound_ratio >= adjusted_min_ratio:
                        segments.append(sub_segment)
            
            # 마지막 남은 부분 처리
            remaining_length = len(segment_audio) % step_samples
            if remaining_length > min_duration * sr:
                last_segment = segment_audio[-int(max_duration * sr):]
                last_start_frame = int((end_sample - len(last_segment)) / hop_length)
                last_end_frame = int(end_sample / hop_length)
                last_end_frame = min(last_end_frame, len(activity))
                
                if last_end_frame > last_start_frame:
                    last_activity = activity[last_start_frame:last_end_frame]
                    last_sound_ratio = np.sum(last_activity) / len(last_activity) if len(last_activity) > 0 else 0
                    
                    adjusted_min_ratio = min_sound_ratio * 0.75
                    if last_sound_ratio >= adjusted_min_ratio:
                        segments.append(last_segment)
        else:
            # 적절한 길이의 세그먼트
            start_frame = int(start_sample / hop_length)
            end_frame = int(end_sample / hop_length)
            end_frame = min(end_frame, len(activity))
            
            if end_frame > start_frame:
                segment_activity = activity[start_frame:end_frame]
                sound_ratio = np.sum(segment_activity) / len(segment_activity) if len(segment_activity) > 0 else 0
                
                # 간헐적 소리의 경우 기준을 낮춤 (30%)
                adjusted_min_ratio = min_sound_ratio * 0.75
                if sound_ratio >= adjusted_min_ratio:
                    segments.append(segment_audio)
    
    return segments

def calculate_smart_weights(envsound_folder, target_samples_per_class=180):
    """
    스마트 세그멘테이션을 기반으로 각 클래스의 가중치를 계산합니다.
    
    Args:
        envsound_folder: 위험 소음 폴더 경로
        target_samples_per_class: 각 클래스당 목표 샘플 수
    
    Returns:
        dict: 각 클래스별 정보 및 가중치
    """
    event_folders = ['fire', 'gas', 'scream']
    class_info = {}
    
    print("🔍 스마트 세그멘테이션 분석 시작...")
    print("=" * 60)
    
    for folder in event_folders:
        folder_path = os.path.join(envsound_folder, folder)
        total_segments = 0
        file_count = 0
        
        if os.path.exists(folder_path):
            # 모든 오디오 파일 분석
            audio_files = glob.glob(os.path.join(folder_path, '*.wav')) + \
                         glob.glob(os.path.join(folder_path, '*.mp3'))
            
            print(f"\n📂 {folder.upper()} 클래스 분석:")
            print("-" * 40)
            
            for audio_file in audio_files:
                try:
                    # 오디오 로드
                    audio, sr = librosa.load(audio_file, sr=16000)
                    
                    # 스마트 세그멘테이션 적용
                    segments = extract_sound_segments(
                        audio, sr, 
                        min_duration=MIN_SEGMENT_DURATION,
                        max_duration=MAX_SEGMENT_DURATION,
                        threshold=SILENCE_THRESHOLD,
                        min_sound_ratio=MIN_SOUND_RATIO
                    )
                    
                    file_segments = len(segments)
                    total_segments += file_segments
                    file_count += 1
                    
                    # 파일 정보 출력 (너무 많으면 일부만)
                    if file_count <= 5 or file_count % 10 == 0:
                        duration = len(audio) / sr
                        print(f"  📄 {os.path.basename(audio_file)}: "
                              f"{duration:.1f}초 → {file_segments}개 세그먼트")
                    
                except Exception as e:
                    print(f"  ❌ {os.path.basename(audio_file)}: 오류 - {e}")
                    continue
            
            avg_segments_per_file = total_segments / file_count if file_count > 0 else 0
            
            print(f"\n📊 {folder.upper()} 클래스 요약:")
            print(f"  • 파일 수: {file_count}개")
            print(f"  • 추출된 총 세그먼트: {total_segments}개")
            print(f"  • 파일당 평균 세그먼트: {avg_segments_per_file:.1f}개")
            
            # 목표 샘플 수에 맞는 선택 비율 계산
            if total_segments > 0:
                selection_ratio = min(1.0, target_samples_per_class / total_segments)
                expected_samples = int(total_segments * selection_ratio)
            else:
                selection_ratio = 0
                expected_samples = 0
            
            print(f"  • 선택 비율: {selection_ratio:.3f} ({expected_samples}개 샘플 예상)")
            
            class_info[folder] = {
                'file_count': file_count,
                'total_segments': total_segments,
                'avg_segments_per_file': avg_segments_per_file,
                'selection_ratio': selection_ratio,
                'expected_samples': expected_samples
            }
        else:
            print(f"\n❌ {folder.upper()} 폴더를 찾을 수 없습니다: {folder_path}")
            class_info[folder] = {
                'file_count': 0,
                'total_segments': 0,
                'avg_segments_per_file': 0,
                'selection_ratio': 0,
                'expected_samples': 0
            }
    
    # 전체 요약
    print("\n🎯 스마트 세그멘테이션 요약:")
    print("=" * 60)
    total_expected = sum(info['expected_samples'] for info in class_info.values())
    
    for class_name, info in class_info.items():
        ratio = (info['expected_samples'] / total_expected * 100) if total_expected > 0 else 0
        print(f"{class_name.upper():>8}: {info['expected_samples']:>3}개 샘플 ({ratio:>5.1f}%)")
    
    print(f"{'총합':>8}: {total_expected:>3}개 샘플")
    print("=" * 60)
    
    return class_info

# ---------------------------
# 2) 무음 제거 함수 (기존)
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
    window_size = min(50, length)  # length보다 작거나 같게 제한
    if window_size > 0:
        noise_filtered = np.convolve(noise, np.ones(window_size)/window_size, mode='same')
    else:
        noise_filtered = noise
    
    # 정확한 길이로 조정
    if len(noise_filtered) != length:
        if len(noise_filtered) > length:
            noise_filtered = noise_filtered[:length]
        else:
            noise_filtered = np.pad(noise_filtered, (0, length - len(noise_filtered)), mode='constant')
    
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
# 9) 무음에서 공장소리로 전환되는 데이터 생성 함수
# ---------------------------
def create_silence_to_factory_transition(factory_audio, sr, total_duration=10.0):
    """
    무음 상태에서 공장 소리가 시작되는 전환 데이터 생성
    
    Args:
        factory_audio: 공장 소리 오디오
        sr: 샘플링 레이트
        total_duration: 총 오디오 길이 (초)
    
    Returns:
        tuple: (전환_오디오, 공장소리_시작시간)
    """
    target_len = int(sr * total_duration)
    
    # 공장 소리 준비
    if len(factory_audio) > target_len:
        start_pos = random.randint(0, len(factory_audio) - target_len)
        factory_audio = factory_audio[start_pos:start_pos + target_len]
    else:
        factory_audio = np.pad(factory_audio, (0, max(0, target_len - len(factory_audio))))
    
    # 전환 시점 결정 (전체 길이의 20%~80% 지점에서 시작)
    transition_start_ratio = random.uniform(0.2, 0.8)
    transition_frame = int(target_len * transition_start_ratio)
    
    # 무음 구간과 공장소리 구간으로 나누기
    mixed_audio = np.zeros(target_len, dtype=np.float32)
    
    # 앞부분은 무음 (매우 작은 배경 노이즈)
    silence_part = generate_background_noise(transition_frame / sr, sr)
    # 정확한 길이 보장
    if len(silence_part) != transition_frame:
        if len(silence_part) > transition_frame:
            silence_part = silence_part[:transition_frame]
        else:
            silence_part = np.pad(silence_part, (0, transition_frame - len(silence_part)), mode='constant')
    mixed_audio[:transition_frame] = silence_part
    
    # 뒷부분은 공장 소리
    factory_part = factory_audio[transition_frame:]
    remaining_samples = target_len - transition_frame
    # 정확한 길이 보장
    if len(factory_part) != remaining_samples:
        if len(factory_part) > remaining_samples:
            factory_part = factory_part[:remaining_samples]
        else:
            factory_part = np.pad(factory_part, (0, remaining_samples - len(factory_part)), mode='constant')
    mixed_audio[transition_frame:] = factory_part
    
    # 전환 지점에서 부드러운 fade-in 효과 (더 현실적)
    fade_duration = int(sr * 0.5)  # 0.5초 페이드인
    if transition_frame + fade_duration < target_len:
        fade_samples = min(fade_duration, len(factory_part))
        if fade_samples > 0:
            fade_curve = np.linspace(0, 1, fade_samples)
            mixed_audio[transition_frame:transition_frame + fade_samples] *= fade_curve
    
    # 볼륨 정규화
    max_val = np.max(np.abs(mixed_audio))
    if max_val > 1.0:
        mixed_audio = mixed_audio / max_val
    
    transition_start_sec = transition_frame / sr
    
    return mixed_audio, transition_start_sec

# ---------------------------
# 10) 무음→공장소리 전환 라벨 생성 함수
# ---------------------------
def generate_transition_labels(transition_start_sec, total_duration=10.0, frame_length=0.48):
    """
    무음→공장소리 전환에 대한 프레임별 라벨 생성
    
    Args:
        transition_start_sec: 공장소리 시작 시간 (초)
        total_duration: 총 오디오 길이 (초)
        frame_length: 프레임 길이 (초)
    
    Returns:
        numpy.array: 프레임별 클래스 라벨 (0=무음, 1=정상)
    """
    num_frames = int(total_duration / frame_length)
    labels = np.zeros(num_frames, dtype=int)  # 기본값: 무음(0)
    
    # 전환 시점 계산
    transition_frame = int(transition_start_sec / frame_length)
    
    # 전환 이후는 정상(공장소리)로 설정
    labels[transition_frame:] = 1
    
    return labels

# ---------------------------
# 11) 무음에서 위험소리로 전환되는 데이터 생성 함수
# ---------------------------
def create_silence_to_danger_transition(event_audio, sr, class_id, total_duration=10.0):
    """
    무음 상태에서 위험 소리가 시작되는 전환 데이터 생성
    
    Args:
        event_audio: 위험 소리 오디오
        sr: 샘플링 레이트
        class_id: 위험소리 클래스 ID (2=화재, 3=가스, 4=비명)
        total_duration: 총 오디오 길이 (초)
    
    Returns:
        tuple: (전환_오디오, 위험소리_시작시간, 위험소리_끝시간)
    """
    target_len = int(sr * total_duration)
    
    # 위험 소리 길이 제한 (너무 길면 잘라내기)
    max_event_len = int(sr * 6.0)  # 최대 6초
    if len(event_audio) > max_event_len:
        start_pos = random.randint(0, len(event_audio) - max_event_len)
        event_audio = event_audio[start_pos:start_pos + max_event_len]
    
    event_len = len(event_audio)
    
    # 전환 시점 결정 (전체 길이의 20%~60% 지점에서 시작, 위험소리를 위해 더 일찍)
    transition_start_ratio = random.uniform(0.2, 0.6)
    transition_frame = int(target_len * transition_start_ratio)
    
    # 위험소리가 끝나는 지점 계산
    event_end_frame = min(transition_frame + event_len, target_len)
    actual_event_len = event_end_frame - transition_frame
    
    # 전체 오디오 생성
    mixed_audio = np.zeros(target_len, dtype=np.float32)
    
    # 앞부분은 무음 (매우 작은 배경 노이즈)
    silence_part = generate_background_noise(transition_frame / sr, sr)
    # 정확한 길이 보장
    if len(silence_part) != transition_frame:
        if len(silence_part) > transition_frame:
            silence_part = silence_part[:transition_frame]
        else:
            silence_part = np.pad(silence_part, (0, transition_frame - len(silence_part)), mode='constant')
    mixed_audio[:transition_frame] = silence_part
    
    # 중간 부분은 위험 소리
    if actual_event_len > 0:
        event_part = event_audio[:actual_event_len]
        
        # 위험소리 볼륨 조정 (더 현실적인 볼륨)
        event_rms = np.sqrt(np.mean(event_part**2))
        if event_rms > 0:
            # 위험소리는 배경보다 충분히 크게 (하지만 클리핑 방지)
            target_rms = random.uniform(0.1, 0.3)
            volume_factor = target_rms / event_rms
            event_part = event_part * volume_factor
        
        mixed_audio[transition_frame:event_end_frame] = event_part
        
        # 전환 지점에서 부드러운 fade-in 효과
        fade_duration = int(sr * 0.3)  # 0.3초 페이드인 (위험소리는 빠르게)
        if transition_frame + fade_duration < event_end_frame:
            fade_samples = min(fade_duration, actual_event_len)
            fade_curve = np.linspace(0, 1, fade_samples)
            mixed_audio[transition_frame:transition_frame + fade_samples] *= fade_curve
    
    # 뒷부분은 다시 무음 (위험소리 후 조용해짐)
    if event_end_frame < target_len:
        remaining_samples = target_len - event_end_frame
        remaining_silence = generate_background_noise((target_len - event_end_frame) / sr, sr)
        # 정확한 길이 보장
        if len(remaining_silence) != remaining_samples:
            if len(remaining_silence) > remaining_samples:
                remaining_silence = remaining_silence[:remaining_samples]
            else:
                remaining_silence = np.pad(remaining_silence, (0, remaining_samples - len(remaining_silence)), mode='constant')
        mixed_audio[event_end_frame:] = remaining_silence
    
    # 볼륨 정규화
    max_val = np.max(np.abs(mixed_audio))
    if max_val > 1.0:
        mixed_audio = mixed_audio / max_val
    
    transition_start_sec = transition_frame / sr
    transition_end_sec = event_end_frame / sr
    
    return mixed_audio, transition_start_sec, transition_end_sec

# ---------------------------
# 12) 무음→위험소리 전환 라벨 생성 함수
# ---------------------------
def generate_danger_transition_labels(transition_start_sec, transition_end_sec, class_id, total_duration=10.0, frame_length=0.48):
    """
    무음→위험소리 전환에 대한 프레임별 라벨 생성
    
    Args:
        transition_start_sec: 위험소리 시작 시간 (초)
        transition_end_sec: 위험소리 끝 시간 (초)
        class_id: 위험소리 클래스 ID (2=화재, 3=가스, 4=비명)
        total_duration: 총 오디오 길이 (초)
        frame_length: 프레임 길이 (초)
    
    Returns:
        numpy.array: 프레임별 클래스 라벨 (0=무음, class_id=위험소리)
    """
    num_frames = int(total_duration / frame_length)
    labels = np.zeros(num_frames, dtype=int)  # 기본값: 무음(0)
    
    # 전환 시점 계산
    start_frame = int(transition_start_sec / frame_length)
    end_frame = int(transition_end_sec / frame_length)
    
    # 위험소리 구간은 해당 클래스로 설정
    labels[start_frame:end_frame] = class_id
    
    # 나머지는 무음(0)으로 유지
    
    return labels

def create_factory_to_factory_transition(factory_audio1, factory_audio2, sr, total_duration=10.0):
    """
    공장소리에서 다른 공장소리로 전환되는 오디오 생성
    
    Args:
        factory_audio1: 첫 번째 공장소리 오디오
        factory_audio2: 두 번째 공장소리 오디오
        sr: 샘플링 레이트
        total_duration: 총 오디오 길이 (초)
    
    Returns:
        tuple: (전환 오디오, 전환 시점(초))
    """
    total_samples = int(total_duration * sr)
    
    # 전환 시점을 30~70% 사이에서 랜덤 선택
    transition_ratio = random.uniform(0.3, 0.7)
    transition_start_sec = total_duration * transition_ratio
    transition_samples = int(transition_start_sec * sr)
    
    # 페이드 전환 길이 (0.1초)
    fade_duration = 0.1
    fade_samples = int(fade_duration * sr)
    
    # 결과 오디오 초기화
    result_audio = np.zeros(total_samples)
    
    # 첫 번째 공장소리로 시작 부분 채우기
    if len(factory_audio1) > 0:
        factory1_repeated = np.tile(factory_audio1, (transition_samples // len(factory_audio1)) + 1)
        result_audio[:transition_samples] = factory1_repeated[:transition_samples]
    
    # 두 번째 공장소리로 나머지 부분 채우기
    if len(factory_audio2) > 0:
        remaining_samples = total_samples - transition_samples
        factory2_repeated = np.tile(factory_audio2, (remaining_samples // len(factory_audio2)) + 1)
        second_part = factory2_repeated[:remaining_samples]
        
        # 정확한 길이 보장
        if len(second_part) != remaining_samples:
            if len(second_part) > remaining_samples:
                second_part = second_part[:remaining_samples]
            else:
                second_part = np.pad(second_part, (0, remaining_samples - len(second_part)), mode='constant')
        
        # 페이드 전환 적용
        if transition_samples + fade_samples <= total_samples:
            # 첫 번째 소리 페이드 아웃
            fade_start = max(0, transition_samples - fade_samples)
            fade_end = transition_samples
            fade_length = fade_end - fade_start
            if fade_length > 0:
                fade_out = np.linspace(1, 0, fade_length)
                result_audio[fade_start:fade_end] *= fade_out
            
            # 두 번째 소리 페이드 인
            second_fade_start = transition_samples
            second_fade_end = min(total_samples, transition_samples + fade_samples)
            second_fade_length = second_fade_end - second_fade_start
            if second_fade_length > 0 and second_fade_length <= len(second_part):
                fade_in = np.linspace(0, 1, second_fade_length)
                second_part[:second_fade_length] *= fade_in
        
        # 최종 결합 전 길이 재확인
        end_index = min(len(result_audio), transition_samples + len(second_part))
        copy_length = end_index - transition_samples
        if copy_length > 0:
            result_audio[transition_samples:end_index] = second_part[:copy_length]
    
    return result_audio, transition_start_sec

def create_silence_variation_transition(sr, total_duration=10.0):
    """
    다양한 무음/배경소음 변화 오디오 생성
    
    Args:
        sr: 샘플링 레이트
        total_duration: 총 오디오 길이 (초)
    
    Returns:
        tuple: (변화 오디오, 변화 시점(초))
    """
    total_samples = int(total_duration * sr)
    
    # 변화 시점을 20~80% 사이에서 랜덤 선택
    transition_ratio = random.uniform(0.2, 0.8)
    transition_start_sec = total_duration * transition_ratio
    transition_samples = int(transition_start_sec * sr)
    
    # 결과 오디오 초기화
    result_audio = np.zeros(total_samples)
    
    # 첫 번째 무음/배경소음 타입
    silence_type1 = random.choice(['pure_silence', 'low_noise', 'white_noise'])
    if silence_type1 == 'pure_silence':
        first_part = np.zeros(transition_samples)
    elif silence_type1 == 'low_noise':
        first_part = generate_background_noise(transition_start_sec, sr)
        # 정확한 길이 보장
        if len(first_part) != transition_samples:
            if len(first_part) > transition_samples:
                first_part = first_part[:transition_samples]
            else:
                first_part = np.pad(first_part, (0, transition_samples - len(first_part)), mode='constant')
    else:  # white_noise
        first_part = np.random.normal(0, 0.001, transition_samples)
    
    # 두 번째 무음/배경소음 타입 (첫 번째와 다르게)
    remaining_duration = total_duration - transition_start_sec
    remaining_samples = total_samples - transition_samples
    
    silence_types = ['pure_silence', 'low_noise', 'white_noise']
    silence_types.remove(silence_type1)  # 첫 번째와 다른 타입 선택
    silence_type2 = random.choice(silence_types)
    
    if silence_type2 == 'pure_silence':
        second_part = np.zeros(remaining_samples)
    elif silence_type2 == 'low_noise':
        second_part = generate_background_noise(remaining_duration, sr)
        # 정확한 길이 보장
        if len(second_part) != remaining_samples:
            if len(second_part) > remaining_samples:
                second_part = second_part[:remaining_samples]
            else:
                second_part = np.pad(second_part, (0, remaining_samples - len(second_part)), mode='constant')
    else:  # white_noise
        second_part = np.random.normal(0, 0.001, remaining_samples)
    
    # 페이드 전환 적용 (0.2초)
    fade_duration = 0.2
    fade_samples = int(fade_duration * sr)
    
    if transition_samples >= fade_samples and remaining_samples >= fade_samples:
        # 첫 번째 부분 페이드 아웃
        fade_start = transition_samples - fade_samples
        fade_out = np.linspace(1, 0, fade_samples)
        if len(first_part) >= fade_samples:
            first_part[fade_start:] *= fade_out
        
        # 두 번째 부분 페이드 인
        fade_in = np.linspace(0, 1, fade_samples)
        if len(second_part) >= fade_samples:
            second_part[:fade_samples] *= fade_in
    
    # 결합 (길이 재확인)
    copy_length1 = min(len(first_part), transition_samples)
    if copy_length1 > 0:
        result_audio[:copy_length1] = first_part[:copy_length1]
    
    copy_start = transition_samples
    copy_end = min(len(result_audio), transition_samples + len(second_part))
    copy_length2 = copy_end - copy_start
    if copy_length2 > 0:
        result_audio[copy_start:copy_end] = second_part[:copy_length2]
    
    return result_audio, transition_start_sec

def generate_factory_transition_labels(transition_start_sec, total_duration=10.0, frame_length=0.48):
    """
    공장소리→다른공장소리 전환에 대한 프레임별 라벨 생성
    (전체 구간이 모두 정상 공장소리이므로 모든 프레임이 클래스 1)
    
    Args:
        transition_start_sec: 전환 시작 시점 (초)
        total_duration: 총 오디오 길이 (초)
        frame_length: 프레임 길이 (초)
    
    Returns:
        numpy.array: 프레임별 클래스 라벨 (모두 1=정상공장소리)
    """
    num_frames = int(total_duration / frame_length)
    labels = np.ones(num_frames, dtype=int)  # 모든 프레임이 정상 공장소리 (1)
    return labels

def generate_silence_variation_labels(transition_start_sec, total_duration=10.0, frame_length=0.48):
    """
    무음 변화에 대한 프레임별 라벨 생성
    (전체 구간이 모두 무음이므로 모든 프레임이 클래스 0)
    
    Args:
        transition_start_sec: 변화 시작 시점 (초)
        total_duration: 총 오디오 길이 (초)
        frame_length: 프레임 길이 (초)
    
    Returns:
        numpy.array: 프레임별 클래스 라벨 (모두 0=무음)
    """
    num_frames = int(total_duration / frame_length)
    labels = np.zeros(num_frames, dtype=int)  # 모든 프레임이 무음 (0)
    return labels

# ---------------------------
# 13) 메인 실행 코드
# ---------------------------
def main():
    sr = 16000
    frame_length = 0.48
    total_duration = 10.0
    num_classes = 5  # 무음(0), 정상(1), 화재(2), 가스누출(3), 비명(4)
    
    # 버전별 결과물 폴더 생성
    output_folder = create_version_folder(VERSION)
    
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
    
    # 가중치 설정 (자동 또는 수동)
    if AUTO_WEIGHT_CALCULATION and ENABLE_SMART_SEGMENTATION:
        print(f"\n🔄 스마트 세그멘테이션 기반 자동 가중치 계산 모드")
        smart_weights = calculate_smart_weights(envsound_folder, target_samples_per_class=TARGET_SAMPLES_PER_CLASS)
        DANGER_WEIGHTS = {class_name: info['selection_ratio'] for class_name, info in smart_weights.items()}
    elif AUTO_WEIGHT_CALCULATION:
        print(f"\n🔄 기존 자동 가중치 계산 모드")
        DANGER_WEIGHTS = calculate_auto_weights(envsound_folder, target_samples_per_class=250)
    else:
        print(f"\n⚙️ 수동 가중치 설정 모드")
        DANGER_WEIGHTS = MANUAL_DANGER_WEIGHTS.copy()
        print("설정된 가중치:")
        for class_name, weight in DANGER_WEIGHTS.items():
            print(f"  {class_name}: {weight}")
    
    print(f"\n최종 사용될 가중치: {DANGER_WEIGHTS}")
    
    # YAMNet 모델을 한 번만 로드
    print("YAMNet 모델 로딩 중...")
    yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
    yamnet_model = hub.load(yamnet_model_handle)
    print("YAMNet 모델 로딩 완료!")
    
    X_data = []
    y_data = []
    data_info = []  # 각 샘플의 메타데이터를 저장
    
    # 4) 무음 데이터 생성 (증가)
    print(f"\n무음 데이터 생성 중... (총 {SILENCE_SAMPLES}개 샘플)")
    
    for i in range(SILENCE_SAMPLES):
        try:
            if (i + 1) % 30 == 0:  # 진행률 표시 간격 조정
                print(f"  무음 데이터 진행률: {i+1}/{SILENCE_SAMPLES}")
            
            # 완전 무음과 저음량 배경 소음을 섞어서 생성
            if i < SILENCE_SAMPLES // 2:
                audio = generate_silence(total_duration, sr)
                silence_type = "완전무음"
            else:
                audio = generate_background_noise(total_duration, sr)
                silence_type = "배경소음"
            
            embeddings = extract_yamnet_embeddings(audio, sr, yamnet_model)
            labels = generate_labels(None, None, 0, total_duration=total_duration, frame_length=frame_length)  # 무음 클래스
            
            X_data.append(embeddings)
            y_data.append(labels)
            data_info.append({
                'class': 'silence',
                'class_id': 0,
                'type': silence_type,
                'factory_file': None,
                'event_file': None,
                'sample_index': i
            })
            
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
            data_info.append({
                'class': 'normal',
                'class_id': 1,
                'type': '공장소음',
                'factory_file': os.path.basename(factory_path),
                'event_file': None,
                'sample_index': i
            })
            
        except Exception as e:
            print(f"  정상 데이터 처리 중 오류: {e}")
            continue

    # 6) 무음→공장소리 전환 데이터 생성 (새로 추가)
    print(f"\n무음→공장소리 전환 데이터 생성 중... (총 {TRANSITION_SAMPLES}개 샘플)")
    
    for i in range(TRANSITION_SAMPLES):
        try:
            if (i + 1) % 20 == 0:  # 진행률 표시 간격 조정
                print(f"  전환 데이터 진행률: {i+1}/{TRANSITION_SAMPLES}")
                
            factory_path = random.choice(factory_paths)
            factory_audio, _ = librosa.load(factory_path, sr=sr)
            
            # 무음→공장소리 전환 오디오 생성
            transition_audio, transition_start_sec = create_silence_to_factory_transition(
                factory_audio, sr, total_duration=total_duration
            )
            
            embeddings = extract_yamnet_embeddings(transition_audio, sr, yamnet_model)
            labels = generate_transition_labels(transition_start_sec, total_duration=total_duration, frame_length=frame_length)
            
            X_data.append(embeddings)
            y_data.append(labels)
            data_info.append({
                'class': 'transition',
                'class_id': 1,  # 최종적으로 정상(공장소리)로 분류
                'type': '무음→공장소리 전환',
                'factory_file': os.path.basename(factory_path),
                'event_file': None,
                'transition_start_sec': transition_start_sec,
                'sample_index': i
            })
            
        except Exception as e:
            print(f"  전환 데이터 처리 중 오류: {e}")
            continue

    # 7) 무음→위험소리 전환 데이터 생성 (새로 추가)
    print(f"\n무음→위험소리 전환 데이터 생성 중... (총 {DANGER_TRANSITION_SAMPLES}개 샘플)")
    
    # 클래스별로 균등하게 분배
    samples_per_danger_class = DANGER_TRANSITION_SAMPLES // len(event_folders)
    remaining_samples = DANGER_TRANSITION_SAMPLES % len(event_folders)
    
    current_sample_count = 0
    
    for class_idx, class_name in enumerate(event_folders):
        if class_name not in event_data or len(event_data[class_name]) == 0:
            print(f"  ⚠️ {class_name} 클래스 파일이 없어 건너뜁니다.")
            continue
            
        class_id = class_mapping[class_name]
        
        # 마지막 클래스에 남은 샘플 추가
        class_samples = samples_per_danger_class
        if class_idx < remaining_samples:
            class_samples += 1
            
        print(f"  {class_name} 클래스 전환 데이터: {class_samples}개 샘플")
        
        for i in range(class_samples):
            try:
                current_sample_count += 1
                if current_sample_count % 10 == 0:
                    print(f"    위험 전환 데이터 진행률: {current_sample_count}/{DANGER_TRANSITION_SAMPLES}")
                
                # 랜덤하게 위험소리 파일 선택
                event_path = random.choice(event_data[class_name])
                event_audio, _ = librosa.load(event_path, sr=sr)
                event_audio_ns = remove_silence(event_audio, sr, top_db=20)
                
                # 무음→위험소리 전환 오디오 생성
                transition_audio, transition_start_sec, transition_end_sec = create_silence_to_danger_transition(
                    event_audio_ns, sr, class_id, total_duration=total_duration
                )
                
                embeddings = extract_yamnet_embeddings(transition_audio, sr, yamnet_model)
                labels = generate_danger_transition_labels(
                    transition_start_sec, transition_end_sec, class_id, 
                    total_duration=total_duration, frame_length=frame_length
                )
                
                X_data.append(embeddings)
                y_data.append(labels)
                data_info.append({
                    'class': f'danger_transition_{class_name}',
                    'class_id': class_id,
                    'type': f'무음→{class_name} 전환',
                    'factory_file': None,
                    'event_file': os.path.basename(event_path),
                    'transition_start_sec': transition_start_sec,
                    'transition_end_sec': transition_end_sec,
                    'sample_index': i
                })
                
            except Exception as e:
                print(f"    위험 전환 데이터 처리 중 오류: {e}")
                continue
    
    # 8) 위험 소리 데이터 생성 - 스마트 세그멘테이션 적용
    if ENABLE_SMART_SEGMENTATION:
        print(f"\n🎯 스마트 세그멘테이션을 사용한 위험 소리 데이터 생성")
        print("=" * 60)
        
        for class_name, event_paths in event_data.items():
            class_id = class_mapping[class_name]
            selection_ratio = DANGER_WEIGHTS[class_name]
            
            print(f"\n{class_name.upper()} 클래스 데이터 생성 중... (클래스 ID: {class_id})")
            print(f"선택 비율: {selection_ratio:.3f}")
            
            class_segments = []
            
            # 각 파일에서 세그먼트 추출
            for idx, event_path in enumerate(event_paths):
                try:
                    print(f"  📄 [{idx+1}/{len(event_paths)}] {os.path.basename(event_path)} 분석 중...")
                    
                    # 오디오 로드
                    event_audio, _ = librosa.load(event_path, sr=sr)
                    
                    # 스마트 세그멘테이션으로 의미있는 구간 추출
                    segments = extract_sound_segments(
                        event_audio, sr,
                        min_duration=MIN_SEGMENT_DURATION,
                        max_duration=MAX_SEGMENT_DURATION,
                        threshold=SILENCE_THRESHOLD,
                        min_sound_ratio=MIN_SOUND_RATIO
                    )
                    
                    print(f"    → {len(segments)}개 세그먼트 추출됨")
                    
                    # 추출된 세그먼트들을 리스트에 추가
                    for seg_idx, segment in enumerate(segments):
                        class_segments.append({
                            'audio': segment,
                            'file_path': event_path,
                            'file_name': os.path.basename(event_path),
                            'segment_index': seg_idx,
                            'duration': len(segment) / sr
                        })
                        
                except Exception as e:
                    print(f"    ❌ 오류: {e}")
                    continue
            
            # 목표 샘플 수에 맞게 세그먼트 선택
            total_segments = len(class_segments)
            target_samples = int(total_segments * selection_ratio)
            
            print(f"\n📊 {class_name.upper()} 세그먼트 선택:")
            print(f"  • 총 추출된 세그먼트: {total_segments}개")
            print(f"  • 선택할 샘플: {target_samples}개")
            
            if target_samples > 0 and total_segments > 0:
                # 랜덤하게 세그먼트 선택
                selected_segments = random.sample(class_segments, min(target_samples, total_segments))
                
                for sample_idx, segment_info in enumerate(selected_segments):
                    try:
                        if (sample_idx + 1) % 20 == 0:
                            print(f"    처리 진행률: {sample_idx+1}/{len(selected_segments)}")
                        
                        segment_audio = segment_info['audio']
                        
                        # 세그먼트를 10초로 조정 (패딩 또는 자르기)
                        target_length = int(total_duration * sr)
                        
                        if len(segment_audio) > target_length:
                            # 랜덤 시작점에서 10초 추출
                            start_idx = random.randint(0, len(segment_audio) - target_length)
                            segment_audio = segment_audio[start_idx:start_idx + target_length]
                        else:
                            # 앞뒤로 무음 패딩 (중앙 정렬)
                            padding_total = target_length - len(segment_audio)
                            padding_start = padding_total // 2
                            padding_end = padding_total - padding_start
                            segment_audio = np.pad(segment_audio, (padding_start, padding_end), mode='constant')
                        
                        # 공장 소리와 믹싱
                        factory_path = random.choice(factory_paths)
                        factory_audio, _ = librosa.load(factory_path, sr=sr)
                        
                        mixed_audio, start_sec, end_sec = mix_factory_and_event(
                            factory_audio, segment_audio, sr, desired_length=total_duration
                        )
                        
                        # YAMNet 임베딩 추출
                        embeddings = extract_yamnet_embeddings(mixed_audio, sr, yamnet_model)
                        labels = generate_labels(start_sec, end_sec, class_id, 
                                               total_duration=total_duration, frame_length=frame_length)
                        
                        X_data.append(embeddings)
                        y_data.append(labels)
                        data_info.append({
                            'class': class_name,
                            'class_id': class_id,
                            'type': '스마트세그먼트_위험소음',
                            'factory_file': os.path.basename(factory_path),
                            'event_file': segment_info['file_name'],
                            'segment_index': segment_info['segment_index'],
                            'segment_duration': segment_info['duration'],
                            'event_start_sec': start_sec,
                            'event_end_sec': end_sec,
                            'sample_index': sample_idx
                        })
                        
                    except Exception as e:
                        print(f"    세그먼트 처리 중 오류: {e}")
                        continue
                
                print(f"  ✅ {len(selected_segments)}개 샘플 생성 완료")
            else:
                print(f"  ⚠️ 생성할 샘플이 없습니다.")
    
    # 9) v1.3 새로 추가: 공장소리→다른공장소리 전환 데이터 생성
    print(f"\n공장소리→다른공장소리 전환 데이터 생성 중... (총 {FACTORY_TRANSITION_SAMPLES}개 샘플)")
    if len(factory_paths) >= 2:
        for i in range(FACTORY_TRANSITION_SAMPLES):
            try:
                # 두 개의 서로 다른 공장소리 파일 선택
                factory_files = random.sample(factory_paths, 2)
                factory_audio1, _ = librosa.load(factory_files[0], sr=sr)
                factory_audio2, _ = librosa.load(factory_files[1], sr=sr)
                
                # 전환 오디오 생성
                transition_audio, transition_start_sec = create_factory_to_factory_transition(
                    factory_audio1, factory_audio2, sr, total_duration
                )
                
                # YAMNet 임베딩 추출
                embeddings = extract_yamnet_embeddings(transition_audio, sr, yamnet_model)
                labels = generate_factory_transition_labels(transition_start_sec, total_duration, frame_length)
                
                X_data.append(embeddings)
                y_data.append(labels)
                data_info.append({
                    'class': 'factory_transition',
                    'class_id': 1,  # 정상 공장소리
                    'type': '공장소리전환',
                    'factory_file1': os.path.basename(factory_files[0]),
                    'factory_file2': os.path.basename(factory_files[1]),
                    'transition_start_sec': transition_start_sec,
                    'sample_index': i
                })
                
                if (i + 1) % 10 == 0:
                    print(f"    공장 전환 데이터 진행률: {i+1}/{FACTORY_TRANSITION_SAMPLES}")
                    
            except Exception as e:
                print(f"    공장 전환 데이터 처리 중 오류: {e}")
                continue
    else:
        print("    경고: 공장소리 파일이 2개 미만입니다. 공장소리 전환 데이터를 생성할 수 없습니다.")
    
    # 10) v1.3 새로 추가: 무음 변화 데이터 생성
    print(f"\n무음 변화 데이터 생성 중... (총 {SILENCE_VARIATION_SAMPLES}개 샘플)")
    for i in range(SILENCE_VARIATION_SAMPLES):
        try:
            # 다양한 무음/배경소음 변화 오디오 생성
            variation_audio, transition_start_sec = create_silence_variation_transition(sr, total_duration)
            
            # YAMNet 임베딩 추출
            embeddings = extract_yamnet_embeddings(variation_audio, sr, yamnet_model)
            labels = generate_silence_variation_labels(transition_start_sec, total_duration, frame_length)
            
            X_data.append(embeddings)
            y_data.append(labels)
            data_info.append({
                'class': 'silence_variation',
                'class_id': 0,  # 무음
                'type': '무음변화',
                'transition_start_sec': transition_start_sec,
                'sample_index': i
            })
            
            if (i + 1) % 10 == 0:
                print(f"    무음 변화 데이터 진행률: {i+1}/{SILENCE_VARIATION_SAMPLES}")
                
        except Exception as e:
            print(f"    무음 변화 데이터 처리 중 오류: {e}")
            continue

    # 11) 데이터 배열화 및 전처리
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
        
        # 10) 학습/검증/테스트 분리 (80% / 10% / 10%)
        X_temp, X_test, y_temp, y_test, info_temp, info_test = train_test_split(
            X_data, y_data_oh, data_info, test_size=0.1, random_state=42, stratify=[info['class_id'] for info in data_info]
        )
        
        X_train, X_val, y_train, y_val, info_train, info_val = train_test_split(
            X_temp, y_temp, info_temp, test_size=0.111, random_state=42, stratify=[info['class_id'] for info in info_temp]  # 0.111 * 0.9 ≈ 0.1 (전체의 10%)
        )
        
        print(f"\n훈련 데이터: {X_train.shape}, 레이블: {y_train.shape}")
        print(f"검증 데이터: {X_val.shape}, 레이블: {y_val.shape}")
        print(f"테스트 데이터: {X_test.shape}, 레이블: {y_test.shape}")
        
        # 테스트 데이터를 별도 폴더에 저장
        test_folder = f"test_{VERSION}"
        if not os.path.exists(test_folder):
            os.makedirs(test_folder)
            print(f"📁 테스트 데이터 폴더 생성: {test_folder}")
        
        # 테스트 데이터 저장
        test_data_path = os.path.join(test_folder, f"test_data_{VERSION}.npz")
        np.savez_compressed(test_data_path, 
                          X_test=X_test, 
                          y_test=y_test, 
                          y_test_labels=np.argmax(y_test, axis=2))  # 원핫 -> 라벨 변환
        print(f"💾 테스트 데이터 저장: {test_data_path}")
        
        # 테스트 데이터 정보 저장
        test_info_path = os.path.join(test_folder, f"test_info_{VERSION}.json")
        with open(test_info_path, 'w', encoding='utf-8') as f:
            json.dump(info_test, f, ensure_ascii=False, indent=2)
        print(f"💾 테스트 정보 저장: {test_info_path}")
        
        # 데이터셋 정보 저장
        dataset_info = {
            'train': info_train,
            'validation': info_val,
            'test': info_test
        }
        
        dataset_file_path = os.path.join(output_folder, f'dataset_info_{VERSION}.json')
        with open(dataset_file_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)
        print(f"데이터셋 정보 저장 완료: {dataset_file_path}")
        
        # 각 데이터셋의 클래스별 분포 확인
        def print_dataset_distribution(y_data, dataset_name, class_names):
            y_labels = np.argmax(y_data, axis=2).flatten()  # one-hot을 클래스 인덱스로 변환
            unique, counts = np.unique(y_labels, return_counts=True)
            total = np.sum(counts)
            print(f"\n{dataset_name} 데이터셋 클래스 분포:")
            for cls, count in zip(unique, counts):
                if cls < len(class_names):
                    ratio = count / total * 100
                    print(f"  {class_names[cls]}: {count:,}개 프레임 ({ratio:.1f}%)")
        
        print_dataset_distribution(y_train, "훈련", class_names)
        print_dataset_distribution(y_val, "검증", class_names)
        print_dataset_distribution(y_test, "테스트", class_names)
        
        # 11) LSTM 모델 생성
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
        
        # 12) 학습
        print(f"\n모델 학습 시작...")
        history = model.fit(
            X_train, y_train, 
            epochs=20,
            batch_size=8, 
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # 13) 모델 평가
        print("\n=== 검증 데이터 평가 ===")
        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        print(f"검증 손실: {val_loss:.4f}")
        print(f"검증 정확도: {val_acc:.4f}")
        
        # 테스트 데이터 평가
        print("\n=== 테스트 데이터 평가 ===")
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"테스트 손실: {test_loss:.4f}")
        print(f"테스트 정확도: {test_acc:.4f}")
        
        # 상세한 테스트 성능 분석
        y_test_pred = model.predict(X_test, verbose=0)
        y_test_pred_classes = np.argmax(y_test_pred, axis=2)
        y_test_true_classes = np.argmax(y_test, axis=2)
        
        # 프레임 단위 평가를 위해 flatten
        y_test_pred_flat = y_test_pred_classes.flatten()
        y_test_true_flat = y_test_true_classes.flatten()
        
        print("\n=== 상세 테스트 성능 (프레임 단위) ===")
        print("분류 보고서:")
        print(classification_report(y_test_true_flat, y_test_pred_flat, 
                                  target_names=class_names, 
                                  zero_division=0))
        
        print("\n혼동 행렬:")
        cm = confusion_matrix(y_test_true_flat, y_test_pred_flat)
        print("실제\\예측", end="")
        for name in class_names:
            print(f"\t{name[:6]}", end="")
        print()
        for i, name in enumerate(class_names):
            print(f"{name[:8]}", end="")
            for j in range(len(class_names)):
                print(f"\t{cm[i][j]}", end="")
            print()
        
        # 14) 모델 저장
        model_file_path = os.path.join(output_folder, f'yamnet_lstm_model_{VERSION}.h5')
        model.save(model_file_path)
        print(f"\n5클래스 모델 저장 완료: {model_file_path}")
        
        # 클래스 매핑 정보 저장
        model_info = {
            'class_mapping': {**class_mapping, 'silence': 0, 'normal': 1},
            'class_names': class_names,
            'num_classes': num_classes,
            'version': VERSION,
            'model_file': f'yamnet_lstm_model_{VERSION}.h5'
        }
        
        # pickle을 사용하여 딕셔너리 저장
        model_info_path = os.path.join(output_folder, f'model_info_{VERSION}.pkl')
        with open(model_info_path, 'wb') as f:
            pickle.dump(model_info, f)
        print(f"모델 정보 저장 완료: {model_info_path}")
        
        # 15) 모델 성능 및 정보 텍스트 파일로 저장
        from datetime import datetime
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        performance_log = f"""
========================================
5클래스 LSTM 모델 학습 보고서 ({VERSION})
========================================
학습 완료 시간: {current_time}
모델 버전: {VERSION}
결과물 폴더: {output_folder}

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
전환 데이터: {TRANSITION_SAMPLES}개 샘플 (무음→공장소리)
위험 전환 데이터: {DANGER_TRANSITION_SAMPLES}개 샘플 (무음→위험소리)
가중치 계산 모드: {'자동' if AUTO_WEIGHT_CALCULATION else '수동'}
위험 소음 가중치:
  - 화재: {DANGER_WEIGHTS.get('fire', 0)} (파일당 샘플 수)
  - 가스누출: {DANGER_WEIGHTS.get('gas', 0)} (파일당 샘플 수)
  - 비명: {DANGER_WEIGHTS.get('scream', 0)} (파일당 샘플 수)

=== 데이터 분포 ===
총 샘플 수: {len(X_data)}개
훈련 데이터: {X_train.shape[0]}개 ({X_train.shape[0]/len(X_data)*100:.1f}%)
검증 데이터: {X_val.shape[0]}개 ({X_val.shape[0]/len(X_data)*100:.1f}%)
테스트 데이터: {X_test.shape[0]}개 ({X_test.shape[0]/len(X_data)*100:.1f}%)

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
검증 손실: {val_loss:.6f}
검증 정확도: {val_acc:.6f} ({val_acc*100:.2f}%)

테스트 손실: {test_loss:.6f}
테스트 정확도: {test_acc:.6f} ({test_acc*100:.2f}%)

=== 상세 테스트 성능 분석 ===
{classification_report(y_test_true_flat, y_test_pred_flat, target_names=class_names, zero_division=0)}

=== 데이터셋 파일 정보 ===
- dataset_info_{VERSION}.json: 훈련/검증/테스트 데이터셋에 사용된 오디오 파일 정보

=== 학습 설정 ===
에포크 수: 20
배치 크기: 8
최적화 알고리즘: Adam (learning_rate=0.001)
콜백:
  - EarlyStopping (patience=5)
  - ReduceLROnPlateau (patience=3, factor=0.5)

=== 저장된 파일 ===
- 모델 파일: yamnet_lstm_model_{VERSION}.h5
- 모델 정보: model_info_{VERSION}.pkl
- 성능 보고서: model_performance_{VERSION}.txt
- 데이터셋 정보: dataset_info_{VERSION}.json

========================================
"""
        
        # 텍스트 파일로 저장
        performance_report_path = os.path.join(output_folder, f'model_performance_{VERSION}.txt')
        with open(performance_report_path, 'w', encoding='utf-8') as f:
            f.write(performance_log)
        
        print(f"모델 성능 보고서 저장 완료: {performance_report_path}")
        
        # 요약 정보 파일 생성
        summary_info = {
            'version': VERSION,
            'timestamp': current_time,
            'folder': output_folder,
            'files': {
                'model': f'yamnet_lstm_model_{VERSION}.h5',
                'model_info': f'model_info_{VERSION}.pkl',
                'performance_report': f'model_performance_{VERSION}.txt',
                'dataset_info': f'dataset_info_{VERSION}.json'
            },
            'performance': {
                'validation_accuracy': float(val_acc),
                'validation_loss': float(val_loss),
                'test_accuracy': float(test_acc),
                'test_loss': float(test_loss)
            },
            'data_summary': {
                'total_samples': len(X_data),
                'train_samples': X_train.shape[0],
                'validation_samples': X_val.shape[0],
                'test_samples': X_test.shape[0],
                'silence_samples': SILENCE_SAMPLES,
                'normal_samples': NORMAL_SAMPLES,
                'transition_samples': TRANSITION_SAMPLES,
                'danger_transition_samples': DANGER_TRANSITION_SAMPLES,
                'auto_weight_calculation': AUTO_WEIGHT_CALCULATION,
                'danger_weights': DANGER_WEIGHTS
            }
        }
        
        summary_path = os.path.join(output_folder, f'summary_{VERSION}.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_info, f, ensure_ascii=False, indent=2)
        
        print(f"요약 정보 저장 완료: {summary_path}")
        
        # 결과물 폴더 정보 출력
        print(f"\n🎉 모든 결과물이 저장되었습니다!")
        print(f"📁 결과물 폴더: {output_folder}")
        print(f"📋 포함된 파일:")
        print(f"  - yamnet_lstm_model_{VERSION}.h5 (학습된 모델)")
        print(f"  - model_info_{VERSION}.pkl (모델 정보)")
        print(f"  - model_performance_{VERSION}.txt (상세 성능 보고서)")
        print(f"  - dataset_info_{VERSION}.json (데이터셋 정보)")
        print(f"  - summary_{VERSION}.json (요약 정보)")
        
    except Exception as e:
        print(f"데이터 처리 중 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
