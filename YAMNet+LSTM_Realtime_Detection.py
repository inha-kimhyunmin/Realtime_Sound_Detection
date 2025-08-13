"""
실시간 위험 소리 감지 시스템

마이크 감도 및 무음 감지 설정 가이드:
========================================

1. RECORD_DURATION: 녹음 시간 (초)
   - 10.0: 기본값 (10초 녹음)
   - 5.0: 빠른 반응 (5초 녹음)
   - 15.0: 더 긴 분석 (15초 녹음)
   
2. ANALYSIS_WAIT_TIME: 분석 완료 후 대기 시간 (초)
   - 5.0: 기본값 (분석 후 5초 대기)
   - 2.0: 더 빈번한 감지 (2초 대기)
   - 10.0: 덜 빈번한 감지 (10초 대기)

3. MIC_GAIN: 마이크 입력 감도
   - 1.0: 기본 감도 (변경 없음)
   - 2.0: 2배 증폭 (작은 소리도 잘 들림)
   - 0.5: 절반으로 감소 (큰 소리만 감지)
   
4. SILENCE_RMS_THRESHOLD: 무음 판단 RMS 기준
   - 0.005: 기본값 (매우 조용한 소리까지 감지)
   - 0.001: 더 민감 (더 작은 소리도 감지)
   - 0.01: 덜 민감 (어느 정도 큰 소리만 감지)
   
5. SILENCE_MAX_THRESHOLD: 무음 판단 최대값 기준
   - 0.01: 기본값

6. SILENCE_PROCESSING_MODE: 무음 처리 모드
   - True: 작은 소리를 무음으로 강제 처리 (권장)
   - False: 비활성화
   
7. SILENCE_FORCE_RMS_THRESHOLD / SILENCE_FORCE_MAX_THRESHOLD: 강제 무음 임계값
   - 0.02 / 0.05: 기본값 (이보다 작으면 무음으로 강제 분류)

무음 처리 모드 기능:
- 공장 소리 → 무음 전환 시 오인식 방지
- 작은 소리로 인한 위험 소리 오탐지 방지
- AI 예측보다 물리적 볼륨을 우선하여 무음 판단

사용 시나리오:
- 빠른 반응이 필요한 경우: RECORD_DURATION=5.0, ANALYSIS_WAIT_TIME=2.0
- 정확한 분석이 필요한 경우: RECORD_DURATION=15.0, ANALYSIS_WAIT_TIME=10.0
- 마이크가 작게 녹음되는 경우: MIC_GAIN을 2.0~5.0으로 증가
- 너무 민감하게 반응하는 경우: SILENCE_RMS_THRESHOLD를 0.01~0.02로 증가
"""

import numpy as np
import sounddevice as sd
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import load_model
import time
import os
import json
from scipy import signal

# --- 설정 ---
SAMPLE_RATE = 16000
DURATION = 10.0  # LSTM 모델 입력 길이 (10초)
THRESHOLD = 0.7  # 위험 소리 감지 임계값

# --- 모델 경로 설정 ---
MODEL_PATH = 'results/version_v2.25/models/yamnet_lstm_model_20250812_173732.h5'  # 기본 모델 경로
# MODEL_PATH = 'model_results_v1.2_20250808_170648/yamnet_lstm_model_v1.2.h5'
# MODEL_PATH = 'model_results_v1.0_20250808_123555/yamnet_lstm_model_v1.0.h5'  # 버전별 모델 경로 예시

# --- 녹음 및 분석 주기 설정 ---
RECORD_DURATION = 5.0      # 녹음 시간 (초) - 모델 입력 길이와 동일하게 설정 권장
ANALYSIS_WAIT_TIME = 1.0    # 분석 완료 후 다음 녹음까지 대기 시간 (초)

# --- 마이크 캘리브레이션 설정 ---
AUTO_CALIBRATION_MODE = True    # True: 자동 캘리브레이션, False: 수동 설정 사용
CALIBRATION_SILENCE_DURATION = 5.0      # 무음 캘리브레이션 시간 (초)
CALIBRATION_FACTORY_DURATION = 3.0      # 공장소리 캘리브레이션 녹음 시간 (초)
CALIBRATION_MAX_ATTEMPTS = 10           # 공장소리 인식 최대 시도 횟수
CALIBRATION_MIN_GAIN = 1.0              # 최소 마이크 감도
CALIBRATION_MAX_GAIN = 10.0             # 최대 마이크 감도
CALIBRATION_GAIN_STEP = 0.5             # 감도 증가 단계

# --- 마이크 및 무음 감지 설정 (수동 모드용) ---
MIC_GAIN = 3.0              # 마이크 입력 감도 (1.0 = 기본, 2.0 = 2배 증폭)
SILENCE_RMS_THRESHOLD = 0.005   # 무음 판단 RMS 임계값 (낮을수록 더 작은 소리도 감지)
SILENCE_MAX_THRESHOLD = 0.01    # 무음 판단 최대값 임계값

# --- 무음 처리 모드 설정 ---
SILENCE_PROCESSING_MODE = False  # True: 작은 소리를 무음으로 강제 처리, False: 비활성화
SILENCE_FORCE_RMS_THRESHOLD = 0.02  # 이 값보다 작으면 무음으로 강제 분류 (캘리브레이션 시 동적 변경됨)
SILENCE_FORCE_MAX_THRESHOLD = 0.05  # 이 값보다 작으면 무음으로 강제 분류 (캘리브레이션 시 동적 변경됨)

# --- 공장 소리 기준 무음 처리 설정 ---
FACTORY_BASED_SILENCE_MODE = False   # True: 공장 소리 크기 기준 무음 처리, False: 비활성화
FACTORY_SILENCE_RATIO = 0.5         # 공장 소리의 몇 % 이하를 무음으로 처리할지 (0.5 = 50%)
FACTORY_SILENCE_RMS_THRESHOLD = 0.0 # 공장 소리 RMS 기준값 (캘리브레이션 시 동적 설정됨)
FACTORY_SILENCE_MAX_THRESHOLD = 0.0 # 공장 소리 Max 기준값 (캘리브레이션 시 동적 설정됨)

# --- 클래스 정보 (5-클래스) ---
CLASS_NAMES = ['무음', '정상(공장)', '화재', '가스누출', '비명']
CLASS_COLORS = {
    0: '🔇',  # 무음
    1: '🟢',  # 정상(공장)
    2: '🔥',  # 화재
    3: '⚠️',  # 가스누출
    4: '😱'   # 비명
}

# --- YAMNet 모델 로드 ---
print("YAMNet 모델 로딩 중...")
YAMNET_MODEL_HANDLE = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(YAMNET_MODEL_HANDLE)
yamnet_fn = yamnet_model.signatures['serving_default']
print("YAMNet 모델 로딩 완료!")

# --- 학습된 LSTM 모델 로드 ---
if os.path.exists(MODEL_PATH):
    print(f"LSTM 모델 로딩 중: {MODEL_PATH}")
    lstm_model = load_model(MODEL_PATH)
    print("LSTM 모델 로딩 완료!")
else:
    print(f"오류: {MODEL_PATH} 파일이 없습니다.")
    print("먼저 LSTM_Train_5Class.py를 실행하거나 MODEL_PATH 설정을 확인해주세요.")
    exit(1)

def get_yamnet_embedding(audio):
    waveform = tf.convert_to_tensor(audio, dtype=tf.float32)
    waveform = tf.squeeze(waveform)
    yamnet_output = yamnet_fn(waveform=waveform)
    embeddings = yamnet_output['output_1'].numpy()  # (frames, 1024)
    return embeddings

def get_audio_volume(audio):
    """오디오의 RMS 볼륨과 최대 절댓값 계산"""
    rms = np.sqrt(np.mean(audio**2))
    max_val = np.max(np.abs(audio))
    return rms, max_val

def detect_clipping(audio, threshold=0.95):
    """오디오 클리핑 감지"""
    clipped_samples = np.sum(np.abs(audio) >= threshold)
    clipping_ratio = clipped_samples / len(audio)
    return clipping_ratio > 0.01  # 1% 이상 클리핑되면 True

def apply_compressor(audio, threshold=0.7, ratio=4.0, attack=0.003, release=0.1, sample_rate=16000):
    """간단한 컴프레서 적용 (과도한 볼륨 제어)"""
    # 간단한 피크 제한 컴프레서
    compressed = audio.copy()
    
    # 임계값을 넘는 부분을 압축
    mask = np.abs(audio) > threshold
    if np.any(mask):
        # 압축 비율 적용
        compressed[mask] = np.sign(audio[mask]) * (
            threshold + (np.abs(audio[mask]) - threshold) / ratio
        )
    
    return compressed

def normalize_audio_adaptive(audio, target_rms=0.1, max_gain=3.0):
    """적응형 오디오 정규화"""
    current_rms = np.sqrt(np.mean(audio**2))
    
    if current_rms < 1e-6:  # 거의 무음인 경우
        return audio
    
    # 목표 RMS에 맞춰 게인 계산
    gain = target_rms / current_rms
    
    # 최대 게인 제한
    gain = min(gain, max_gain)
    
    normalized = audio * gain
    
    # 최종 클리핑 방지
    max_val = np.max(np.abs(normalized))
    if max_val > 0.95:
        normalized = normalized * (0.95 / max_val)
    
    return normalized

def analyze_frequency_content(audio, sample_rate=16000):
    """주파수 분석으로 소리 특성 파악"""
    # FFT 수행
    fft = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(len(audio), 1/sample_rate)
    magnitude = np.abs(fft)
    
    # 주요 주파수 대역별 에너지 계산
    low_freq_energy = np.sum(magnitude[(freqs >= 50) & (freqs <= 500)])    # 저주파 (기계음)
    mid_freq_energy = np.sum(magnitude[(freqs >= 500) & (freqs <= 2000)])   # 중주파 (일반 소음)
    high_freq_energy = np.sum(magnitude[(freqs >= 2000) & (freqs <= 8000)]) # 고주파 (비명 등)
    
    total_energy = low_freq_energy + mid_freq_energy + high_freq_energy
    
    if total_energy < 1e-6:
        return 0.33, 0.33, 0.34  # 균등 분배
    
    # 비율 계산
    low_ratio = low_freq_energy / total_energy
    mid_ratio = mid_freq_energy / total_energy
    high_ratio = high_freq_energy / total_energy
    
    return low_ratio, mid_ratio, high_ratio

def preprocess_audio(audio, sample_rate=16000):
    """통합 오디오 전처리 함수"""
    original_rms, original_max = get_audio_volume(audio)
    
    # 1. 클리핑 감지
    is_clipped = detect_clipping(audio)
    
    # 2. 컴프레서 적용 (과도한 볼륨 제어)
    if original_max > 0.8 or is_clipped:
        audio = apply_compressor(audio, threshold=0.6, ratio=6.0)
    
    # 3. 적응형 정규화
    if original_rms > 0.001:  # 무음이 아닌 경우만
        # 볼륨이 매우 높은 경우 더 보수적으로 정규화
        if original_rms > 0.3:
            target_rms = 0.08  # 더 낮은 목표
            max_gain = 1.5     # 게인 제한
        else:
            target_rms = 0.1
            max_gain = 3.0
            
        audio = normalize_audio_adaptive(audio, target_rms, max_gain)
    
    # 4. 최종 안전장치 (하드 리미터)
    audio = np.clip(audio, -0.95, 0.95)
    
    # 전처리 정보 반환
    processed_rms, processed_max = get_audio_volume(audio)
    preprocessing_info = {
        'original_rms': original_rms,
        'original_max': original_max,
        'processed_rms': processed_rms,
        'processed_max': processed_max,
        'was_clipped': is_clipped,
        'volume_reduced': original_rms > processed_rms * 1.5
    }
    
    return audio, preprocessing_info

def is_silence(audio, rms_threshold=None, max_threshold=None):
    """오디오가 무음인지 판단"""
    if rms_threshold is None:
        rms_threshold = SILENCE_RMS_THRESHOLD
    if max_threshold is None:
        max_threshold = SILENCE_MAX_THRESHOLD
        
    rms, max_val = get_audio_volume(audio)
    return rms < rms_threshold and max_val < max_threshold

def record_audio(duration, sample_rate, mic_gain=None):
    """마이크로부터 오디오 녹음 (감도 조절 포함)"""
    if mic_gain is None:
        mic_gain = MIC_GAIN
        
    print(f"{duration}초 동안 녹음 중... (마이크 감도: {mic_gain}x)")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    audio = audio.flatten()
    
    # 마이크 감도 적용 (증폭)
    audio = audio * mic_gain
    
    # 클리핑 방지 (감도 증폭 후)
    max_val = np.max(np.abs(audio))
    if max_val > 1.0:
        audio = audio / max_val
        print(f"⚠️ 마이크 감도로 인한 클리핑 방지: {max_val:.3f} → 1.0")
    
    # 오디오 전처리 적용
    processed_audio, preprocessing_info = preprocess_audio(audio, sample_rate)
    
    # 전처리 정보에 마이크 감도 정보 추가
    preprocessing_info['mic_gain'] = mic_gain
    preprocessing_info['was_mic_amplified'] = mic_gain > 1.0
    
    return processed_audio, preprocessing_info

def calibrate_microphone():
    """
    마이크 자동 캘리브레이션 수행
    1. 공장 소리를 감지할 수 있는 최적 감도 찾기
    2. 최적 감도로 무음 상태에서 배경 노이즈 레벨 측정
    
    Returns:
        dict: 캘리브레이션 결과 (optimal_gain, silence_baseline, etc.)
    """
    print("🔧 마이크 자동 캘리브레이션 시작")
    print("=" * 50)
    
    # 1단계: 공장 소리 감지 캘리브레이션
    print(f"📍 1단계: 공장 소리 감지 캘리브레이션")
    print("💡 정상적인 공장 소리(기계 소음)를 내주세요...")
    print("   시스템이 공장 소리를 인식할 때까지 마이크 감도를 자동 조정합니다.")
    
    time.sleep(3)  # 준비 시간
    
    optimal_gain = CALIBRATION_MIN_GAIN
    factory_detected = False
    attempt = 0
    calibration_results = {
        'optimal_gain': optimal_gain,
        'factory_detected': False,
        'attempts': 0,
        'calibration_history': []
    }
    
    while attempt < CALIBRATION_MAX_ATTEMPTS and not factory_detected:
        attempt += 1
        current_gain = CALIBRATION_MIN_GAIN + (attempt - 1) * CALIBRATION_GAIN_STEP
        
        if current_gain > CALIBRATION_MAX_GAIN:
            break
            
        print(f"\n🎤 시도 {attempt}/{CALIBRATION_MAX_ATTEMPTS} - 마이크 감도: {current_gain:.1f}x")
        
        # 공장 소리 녹음
        factory_audio = sd.rec(int(CALIBRATION_FACTORY_DURATION * SAMPLE_RATE), 
                             samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait()
        factory_audio = factory_audio.flatten()
        
        # 감도 적용
        factory_audio = factory_audio * current_gain
        
        # 클리핑 방지
        max_val = np.max(np.abs(factory_audio))
        if max_val > 1.0:
            factory_audio = factory_audio / max_val
        
        # 전처리
        processed_audio, preprocessing_info = preprocess_audio(factory_audio, SAMPLE_RATE)
        
        # 오디오 분석
        rms, max_audio = get_audio_volume(processed_audio)
        
        # 기본 무음이 아닌지 확인 (임시 기준값 사용)
        is_not_silence = not is_silence(processed_audio, 
                                       rms_threshold=0.005,  # 임시 기준값
                                       max_threshold=0.01)   # 임시 기준값
        
        attempt_result = {
            'attempt': attempt,
            'gain': current_gain,
            'rms': rms,
            'max_val': max_audio,
            'is_not_silence': is_not_silence,
            'preprocessing_info': preprocessing_info
        }
        
        print(f"   📊 오디오 분석: RMS={rms:.4f}, Max={max_audio:.4f}")
        print(f"   🔍 무음 여부: {'아니오' if is_not_silence else '예'}")
        
        if is_not_silence:
            # AI 모델로 공장 소리인지 확인
            try:
                predicted_class, max_prob, frame_predictions, _, _, _ = predict_risk(processed_audio, preprocessing_info)
                
                # 정상(공장) 소리(클래스 1)로 분류되는지 확인
                if predicted_class == 1 and max_prob > 0.5:  # 공장 소리로 인식
                    factory_detected = True
                    optimal_gain = current_gain
                    attempt_result['ai_prediction'] = {
                        'class': predicted_class,
                        'probability': max_prob,
                        'class_name': CLASS_NAMES[predicted_class]
                    }
                    # 공장 소리 크기 기록
                    attempt_result['factory_audio_levels'] = {
                        'rms': rms,
                        'max_val': max_audio
                    }
                    print(f"   🎯 AI 예측: {CLASS_NAMES[predicted_class]} (확률: {max_prob:.3f})")
                    print(f"   📊 공장 소리 크기: RMS={rms:.4f}, Max={max_audio:.4f}")
                    print(f"   ✅ 공장 소리 인식 성공!")
                    break
                else:
                    attempt_result['ai_prediction'] = {
                        'class': predicted_class,
                        'probability': max_prob,
                        'class_name': CLASS_NAMES[predicted_class] if predicted_class < len(CLASS_NAMES) else f"클래스{predicted_class}"
                    }
                    print(f"   🎯 AI 예측: {attempt_result['ai_prediction']['class_name']} (확률: {max_prob:.3f})")
                    print(f"   ⚠️ 공장 소리로 인식되지 않음. 감도를 높입니다...")
                    
            except Exception as e:
                print(f"   ❌ AI 예측 오류: {e}")
                attempt_result['ai_prediction'] = None
        else:
            print(f"   ⚠️ 여전히 무음으로 감지됨. 감도를 높입니다...")
            attempt_result['ai_prediction'] = None
        
        calibration_results['calibration_history'].append(attempt_result)
        
        time.sleep(1)  # 다음 시도 전 잠시 대기
    
    calibration_results['attempts'] = attempt
    calibration_results['factory_detected'] = factory_detected
    calibration_results['optimal_gain'] = optimal_gain
    
    if not factory_detected:
        print(f"\n❌ 1단계 실패: 공장 소리를 인식하지 못했습니다.")
        print(f"💡 다음을 확인해주세요:")
        print(f"   - 마이크가 올바르게 연결되어 있는지")
        print(f"   - 공장 소음이 충분히 크게 들리는지")
        print(f"   - 최대 감도({CALIBRATION_MAX_GAIN}x)로도 인식되지 않았습니다.")
        print(f"📝 수동 설정을 사용하거나 환경을 확인 후 재시도하세요.")
        
        # 실패 시 기본값 사용
        calibration_results.update({
            'optimal_gain': MIC_GAIN,
            'silence_rms': SILENCE_RMS_THRESHOLD,
            'silence_max': SILENCE_MAX_THRESHOLD,
            'dynamic_silence_rms_threshold': SILENCE_RMS_THRESHOLD,
            'dynamic_silence_max_threshold': SILENCE_MAX_THRESHOLD
        })
        return calibration_results
    
    # 2단계: 최적 감도로 무음 상태 측정
    print(f"\n📍 2단계: 무음 상태 측정 (감도: {optimal_gain:.1f}x, {CALIBRATION_SILENCE_DURATION}초)")
    print("💡 이제 공장 소리를 완전히 끄고 주변을 최대한 조용하게 유지해주세요...")
    print("⏱️ 공장 소리를 끌 시간을 드립니다...")
    
    # 공장 소리를 끌 시간을 충분히 제공
    for i in range(5, 0, -1):
        print(f"   {i}초 후 무음 측정을 시작합니다...")
        time.sleep(1)
    
    print("🔇 무음 측정을 시작합니다!")
    
    # 최적 감도로 무음 녹음
    silence_audio = sd.rec(int(CALIBRATION_SILENCE_DURATION * SAMPLE_RATE), 
                          samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    silence_audio = silence_audio.flatten()
    
    # 최적 감도 적용
    silence_audio = silence_audio * optimal_gain
    
    # 클리핑 방지
    max_val = np.max(np.abs(silence_audio))
    if max_val > 1.0:
        silence_audio = silence_audio / max_val
    
    # 전처리
    processed_silence, _ = preprocess_audio(silence_audio, SAMPLE_RATE)
    
    # 무음 기준값 계산
    silence_rms = np.sqrt(np.mean(processed_silence**2))
    silence_max = np.max(np.abs(processed_silence))
    
    print(f"✅ 무음 기준값 측정 완료:")
    print(f"   - RMS: {silence_rms:.6f}")
    print(f"   - Max: {silence_max:.6f}")
    
    # 캘리브레이션 결과 출력
    print(f"\n🎉 캘리브레이션 완료!")
    print("=" * 50)
    print(f"✅ 성공: 최적 마이크 감도 = {optimal_gain:.1f}x")
    print(f"📊 무음 기준값: RMS={silence_rms:.6f}, Max={silence_max:.6f}")
    print(f"🏭 공장 소리 인식됨 ({attempt}번째 시도)")
    
    # 동적 임계값 계산
    dynamic_silence_rms = silence_rms * 2.0  # 무음 기준의 2배
    dynamic_silence_max = silence_max * 2.0
    
    # 무음 처리 모드용 강제 임계값 계산 (무음 기준의 3~4배)
    force_silence_rms = silence_rms * 3.0
    force_silence_max = silence_max * 4.0
    
    # 공장 소리 기준 무음 처리 임계값 계산
    factory_audio_levels = None
    for result in calibration_results['calibration_history']:
        if result.get('factory_audio_levels'):
            factory_audio_levels = result['factory_audio_levels']
            break
    
    if factory_audio_levels:
        factory_rms_threshold = factory_audio_levels['rms'] * FACTORY_SILENCE_RATIO
        factory_max_threshold = factory_audio_levels['max_val'] * FACTORY_SILENCE_RATIO
    else:
        # 공장 소리 크기를 찾을 수 없는 경우 기본값 사용
        factory_rms_threshold = dynamic_silence_rms
        factory_max_threshold = dynamic_silence_max
    
    calibration_results.update({
        'silence_rms': silence_rms,
        'silence_max': silence_max,
        'dynamic_silence_rms_threshold': dynamic_silence_rms,
        'dynamic_silence_max_threshold': dynamic_silence_max, 
        'force_silence_rms_threshold': force_silence_rms,
        'force_silence_max_threshold': force_silence_max,
        'factory_rms_threshold': factory_rms_threshold,
        'factory_max_threshold': factory_max_threshold,
        'factory_audio_levels': factory_audio_levels
    })
    
    print(f"🔧 동적 임계값:")
    print(f"   - 무음 RMS 임계값: {dynamic_silence_rms:.6f}")
    print(f"   - 무음 Max 임계값: {dynamic_silence_max:.6f}")
    print(f"   - 강제 무음 RMS 임계값: {force_silence_rms:.6f}")
    print(f"   - 강제 무음 Max 임계값: {force_silence_max:.6f}")
    if factory_audio_levels:
        print(f"   - 공장 기준 무음 RMS 임계값: {factory_rms_threshold:.6f} (공장 소리의 {FACTORY_SILENCE_RATIO*100:.0f}%)")
        print(f"   - 공장 기준 무음 Max 임계값: {factory_max_threshold:.6f} (공장 소리의 {FACTORY_SILENCE_RATIO*100:.0f}%)")
    else:
        print(f"   - 공장 기준 무음 임계값: 공장 소리 크기를 찾을 수 없어 기본값 사용")
    
    return calibration_results

def predict_risk(audio, preprocessing_info):
    # 먼저 실제 볼륨 체크
    rms, max_val = get_audio_volume(audio)
    
    # 1. 기본 무음 감지
    if is_silence(audio):
        # 무음으로 직접 판단
        return 0, 0.95, np.array([0.95, 0.01, 0.01, 0.01, 0.01]), 0, rms, max_val
    
    # 2. 무음 처리 모드 - 작은 소리를 무음으로 강제 처리
    if SILENCE_PROCESSING_MODE:
        if rms < SILENCE_FORCE_RMS_THRESHOLD and max_val < SILENCE_FORCE_MAX_THRESHOLD:
            print(f"🔇 무음 처리 모드: 작은 소리를 무음으로 강제 처리 (RMS: {rms:.4f}, Max: {max_val:.4f})")
            return 0, 0.90, np.array([0.90, 0.05, 0.02, 0.02, 0.01]), 0, rms, max_val
    
    # 3. 공장 기준 무음 처리 모드 - 공장 소리의 일정 비율 이하 무음 처리
    if FACTORY_BASED_SILENCE_MODE:
        if rms < FACTORY_SILENCE_RMS_THRESHOLD and max_val < FACTORY_SILENCE_MAX_THRESHOLD:
            print(f"🏭 공장 기준 무음 처리: 공장 소리의 {FACTORY_SILENCE_RATIO*100:.0f}% 이하로 무음 처리")
            print(f"   현재: RMS={rms:.4f}, Max={max_val:.4f}")
            print(f"   임계값: RMS<{FACTORY_SILENCE_RMS_THRESHOLD:.4f}, Max<{FACTORY_SILENCE_MAX_THRESHOLD:.4f}")
            return 0, 0.88, np.array([0.88, 0.07, 0.02, 0.02, 0.01]), 0, rms, max_val
    
    embeddings = get_yamnet_embedding(audio)  # (time_steps, 1024)
    
    # 모델 입력 형태 확인 및 차원 조정
    print(f"🔍 YAMNet 임베딩 형태: {embeddings.shape}")
    print(f"🔍 LSTM 모델 입력 형태: {lstm_model.input_shape}")
    
    # LSTM 모델의 입력 차원에 따라 처리 방식 결정
    if len(lstm_model.input_shape) == 2:  # Dense 레이어 기반 모델 (batch, features)
        # 시간축 평균으로 단일 벡터 생성
        embeddings_avg = np.mean(embeddings, axis=0)  # (1024,)
        embeddings_input = np.expand_dims(embeddings_avg, axis=0)  # (1, 1024)
        print(f"📏 Dense 모델용 임베딩: {embeddings_input.shape}")
        
    elif len(lstm_model.input_shape) == 3:  # LSTM 기반 모델 (batch, time_steps, features)
        # 모델 입력에 맞게 패딩/자르기
        target_length = lstm_model.input_shape[1]  # 모델의 time_steps 차원
        current_length = embeddings.shape[0]
        
        if current_length < target_length:
            # 패딩
            pad_length = target_length - current_length
            embeddings = np.pad(embeddings, ((0, pad_length), (0, 0)), mode='constant')
            print(f"📏 임베딩 패딩: {current_length} → {target_length} 프레임")
        elif current_length > target_length:
            # 자르기
            embeddings = embeddings[:target_length]
            print(f"📏 임베딩 자르기: {current_length} → {target_length} 프레임")
        
        embeddings_input = np.expand_dims(embeddings, axis=0)  # (1, time_steps, 1024)
        print(f"📏 LSTM 모델용 임베딩: {embeddings_input.shape}")
    else:
        raise ValueError(f"지원하지 않는 모델 입력 형태: {lstm_model.input_shape}")
    
    # 예측 수행
    preds = lstm_model.predict(embeddings_input, verbose=0)
    
    # 출력 형태에 따라 처리
    if len(preds.shape) == 3:  # LSTM 출력: (batch, time_steps, num_classes)
        preds = preds[0]  # (time_steps, num_classes)
        
        # 각 클래스의 최대 확률과 위치 찾기
        for i in range(len(preds)):
            print(f"{i+1}번 프레임", round(preds[i][0],2), round(preds[i][1],2), round(preds[i][2],2), round(preds[i][3],2), round(preds[i][4],2))

        max_probs = np.max(preds, axis=0)  # 각 클래스별 최대 확률
        overall_max_prob = np.max(max_probs)
        predicted_class = np.argmax(max_probs)
        
        # 프레임별 예측에서 가장 높은 확률을 가진 프레임 찾기
        max_frame_idx = np.argmax(np.max(preds, axis=1))
        frame_predictions = preds[max_frame_idx]  # 해당 프레임의 클래스별 확률
        
    elif len(preds.shape) == 2:  # Dense 출력: (batch, num_classes)
        preds = preds[0]  # (num_classes,)
        
        print(f"예측 확률:", round(preds[0],2), round(preds[1],2), round(preds[2],2), round(preds[3],2), round(preds[4],2))
        
        overall_max_prob = np.max(preds)
        predicted_class = np.argmax(preds)
        frame_predictions = preds
        max_frame_idx = 0  # Dense 모델은 단일 예측
    else:
        raise ValueError(f"지원하지 않는 모델 출력 형태: {preds.shape}")
    
    # 3. 무음 처리 모드 - AI 예측 후 추가 검증
    if SILENCE_PROCESSING_MODE and predicted_class >= 2:  # 위험 소리로 예측된 경우
        # 실제 오디오 볼륨이 매우 작다면 무음으로 재분류
        if rms < SILENCE_FORCE_RMS_THRESHOLD * 1.5 and max_val < SILENCE_FORCE_MAX_THRESHOLD * 1.5:
            print(f"🔇 무음 처리 모드: 위험 소리 예측이지만 볼륨이 너무 작아 무음으로 재분류")
            print(f"   원래 예측: {CLASS_NAMES[predicted_class] if predicted_class < len(CLASS_NAMES) else f'클래스{predicted_class}'} (확률: {overall_max_prob:.3f})")
            return 0, 0.85, np.array([0.85, 0.10, 0.02, 0.02, 0.01]), 0, rms, max_val
    
    # 4. 공장 기준 무음 처리 모드 - AI 예측 후 추가 검증
    if FACTORY_BASED_SILENCE_MODE and predicted_class >= 2:  # 위험 소리로 예측된 경우
        # 공장 소리 기준으로 볼륨이 매우 작다면 무음으로 재분류
        if rms < FACTORY_SILENCE_RMS_THRESHOLD * 1.2 and max_val < FACTORY_SILENCE_MAX_THRESHOLD * 1.2:
            print(f"🏭 공장 기준 무음 처리: 위험 소리 예측이지만 공장 소리 대비 볼륨이 너무 작아 무음으로 재분류")
            print(f"   원래 예측: {CLASS_NAMES[predicted_class] if predicted_class < len(CLASS_NAMES) else f'클래스{predicted_class}'} (확률: {overall_max_prob:.3f})")
            print(f"   현재 볼륨: RMS={rms:.4f}, Max={max_val:.4f}")
            return 0, 0.83, np.array([0.83, 0.12, 0.02, 0.02, 0.01]), 0, rms, max_val
    
    return predicted_class, overall_max_prob, frame_predictions, max_frame_idx, rms, max_val

def main():
    global MIC_GAIN, SILENCE_RMS_THRESHOLD, SILENCE_MAX_THRESHOLD
    global SILENCE_FORCE_RMS_THRESHOLD, SILENCE_FORCE_MAX_THRESHOLD
    global FACTORY_SILENCE_RMS_THRESHOLD, FACTORY_SILENCE_MAX_THRESHOLD
    
    window_length = RECORD_DURATION
    wait_time_between_recordings = ANALYSIS_WAIT_TIME
    
    print("🔊 실시간 위험 소리 감지 시스템")
    print("=" * 50)
    print("모델 정보: 5개 클래스 (5-클래스: 무음 포함)")
    print(f"모델 경로: {MODEL_PATH}")
    print(f"녹음 시간: {RECORD_DURATION}초")
    print(f"분석 후 대기시간: {ANALYSIS_WAIT_TIME}초")
    
    # 원본 설정값 저장
    original_mic_gain = MIC_GAIN
    original_silence_rms = SILENCE_RMS_THRESHOLD
    original_silence_max = SILENCE_MAX_THRESHOLD
    original_force_rms = SILENCE_FORCE_RMS_THRESHOLD
    original_force_max = SILENCE_FORCE_MAX_THRESHOLD
    original_factory_rms = FACTORY_SILENCE_RMS_THRESHOLD
    original_factory_max = FACTORY_SILENCE_MAX_THRESHOLD
    
    # 캘리브레이션 모드 처리
    if AUTO_CALIBRATION_MODE:
        print(f"\n🔧 자동 캘리브레이션 모드")
        print("💡 마이크 감도와 무음 기준값을 자동으로 설정합니다.")
        print("✅ 자동 캘리브레이션을 시작합니다.")
        
        try:
            calibration_results = calibrate_microphone()
        except KeyboardInterrupt:
            print("\n❌ 캘리브레이션이 취소되었습니다. 수동 설정값을 사용합니다.")
            calibration_results = None
    else:
        print(f"\n⚙️ 수동 설정 모드")
        calibration_results = None
    
    # 설정값 업데이트
    if calibration_results and calibration_results['factory_detected']:
        # 캘리브레이션 성공 - 전역 변수 업데이트
        MIC_GAIN = calibration_results['optimal_gain']
        SILENCE_RMS_THRESHOLD = calibration_results['dynamic_silence_rms_threshold']
        SILENCE_MAX_THRESHOLD = calibration_results['dynamic_silence_max_threshold']
        SILENCE_FORCE_RMS_THRESHOLD = calibration_results['force_silence_rms_threshold']
        SILENCE_FORCE_MAX_THRESHOLD = calibration_results['force_silence_max_threshold']
        FACTORY_SILENCE_RMS_THRESHOLD = calibration_results['factory_rms_threshold']
        FACTORY_SILENCE_MAX_THRESHOLD = calibration_results['factory_max_threshold']
        mode_name = "자동 캘리브레이션"
    else:
        # 캘리브레이션 실패 또는 수동 모드 - 원본 설정값 유지
        mode_name = "수동 설정"
    
    print(f"\n📊 사용 중인 설정값 ({mode_name}):")
    print(f"   - 마이크 감도: {MIC_GAIN:.1f}x")
    print(f"   - 무음 RMS 임계값: {SILENCE_RMS_THRESHOLD:.6f}")
    print(f"   - 무음 Max 임계값: {SILENCE_MAX_THRESHOLD:.6f}")
    print(f"   - 무음 처리 모드: {'켜짐' if SILENCE_PROCESSING_MODE else '꺼짐'}")
    if SILENCE_PROCESSING_MODE:
        print(f"     └ 강제 무음 RMS 임계값: {SILENCE_FORCE_RMS_THRESHOLD:.6f}")
        print(f"     └ 강제 무음 Max 임계값: {SILENCE_FORCE_MAX_THRESHOLD:.6f}")
    print(f"   - 공장 기준 무음 처리: {'켜짐' if FACTORY_BASED_SILENCE_MODE else '꺼짐'}")
    if FACTORY_BASED_SILENCE_MODE:
        print(f"     └ 공장 소리 비율 설정: {FACTORY_SILENCE_RATIO*100:.0f}% 이하 무음 처리")
        print(f"     └ 공장 기준 RMS 임계값: {FACTORY_SILENCE_RMS_THRESHOLD:.6f}")
        print(f"     └ 공장 기준 Max 임계값: {FACTORY_SILENCE_MAX_THRESHOLD:.6f}")
    
    print(f"\n🚀 실시간 위험 소리 감지 시작 (Ctrl+C로 종료)")
    print("=" * 50)
    
    try:
        while True:
            # 업데이트된 전역 변수를 사용하여 녹음
            audio, preprocessing_info = record_audio(window_length, SAMPLE_RATE)
            
            print("🔍 분석 중...")
            analysis_start_time = time.time()
            
            # 예측 수행
            predicted_class, max_prob, frame_predictions, max_frame_idx, rms, max_val = predict_risk(audio, preprocessing_info)
            
            # 주파수 분석 정보 추가
            low_ratio, mid_ratio, high_ratio = analyze_frequency_content(audio)
            
            analysis_end_time = time.time()
            analysis_duration = analysis_end_time - analysis_start_time
            
            # 결과 출력
            class_name = CLASS_NAMES[predicted_class] if predicted_class < len(CLASS_NAMES) else f"클래스{predicted_class}"
            class_icon = CLASS_COLORS.get(int(predicted_class), '❓')
            
            print(f"✅ 분석 완료! (소요시간: {analysis_duration:.2f}초)")
            print(f"시간: {time.strftime('%H:%M:%S')}")
            print(f"설정 모드: {mode_name}")
            print(f"오디오 볼륨: RMS={rms:.4f}, Max={max_val:.4f}")
            
            # 전처리 정보 출력
            if preprocessing_info['was_clipped']:
                print("⚠️ 클리핑 감지됨 - 신뢰도 조정")
            if preprocessing_info['volume_reduced']:
                print(f"🔧 볼륨 조정: {preprocessing_info['original_rms']:.3f} → {preprocessing_info['processed_rms']:.3f}")
            if preprocessing_info.get('was_mic_amplified', False):
                print(f"🎤 마이크 감도 적용: {preprocessing_info['mic_gain']:.1f}x")
            
            print(f"주파수 분석: 저음{low_ratio:.2f} | 중음{mid_ratio:.2f} | 고음{high_ratio:.2f}")
            print(f"예측 결과: {class_icon} {class_name} (확률: {max_prob:.3f})")
            print(f"감지 프레임: {max_frame_idx}")
            
            # 모든 클래스별 확률 출력
            print("클래스별 확률:")
            for i, prob in enumerate(frame_predictions):
                if i < len(CLASS_NAMES):
                    name = CLASS_NAMES[i]
                    icon = CLASS_COLORS.get(i, '❓')
                    print(f"  {icon} {name}: {prob:.3f}")
            
            # 위험 소리 감지 여부 판단 (5클래스: 무음(0), 정상(1)이 아닌 경우가 위험)
            is_dangerous = predicted_class >= 2 and max_prob > THRESHOLD
            
            if is_dangerous:
                print(f"\n🚨 위험 감지! {class_icon} {class_name} - 확률: {max_prob:.3f}")
                print("🔔 알림: 즉시 확인이 필요합니다!")
            elif predicted_class == 0:
                print(f"\n🔇 상태: 무음")
            elif predicted_class == 1:
                print(f"\n✅ 상태: 정상")
            else:
                print(f"\n⚠️ 상태: {class_name} 감지됨 (임계값 미만)")
            
            print("-" * 50)
            
            # 분석 완료 후 대기 시간
            if wait_time_between_recordings > 0:
                print(f"⏱️ {wait_time_between_recordings:.1f}초 대기 후 다음 녹음 시작...")
                time.sleep(wait_time_between_recordings)
            else:
                print("⏱️ 즉시 다음 녹음 시작...")
                
    except KeyboardInterrupt:
        print("\n프로그램을 종료합니다.")
        
        # 설정값 복원
        MIC_GAIN = original_mic_gain
        SILENCE_RMS_THRESHOLD = original_silence_rms
        SILENCE_MAX_THRESHOLD = original_silence_max
        SILENCE_FORCE_RMS_THRESHOLD = original_force_rms
        SILENCE_FORCE_MAX_THRESHOLD = original_force_max
        FACTORY_SILENCE_RMS_THRESHOLD = original_factory_rms
        FACTORY_SILENCE_MAX_THRESHOLD = original_factory_max

if __name__ == '__main__':
    main()