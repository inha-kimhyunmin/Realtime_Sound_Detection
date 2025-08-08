import numpy as np
import sounddevice as sd
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import load_model
import time
import os
from scipy import signal

# --- 설정 ---
SAMPLE_RATE = 16000
DURATION = 10.0  # LSTM 모델 입력 길이 (10초)
THRESHOLD = 0.7  # 위험 소리 감지 임계값

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
model_path = 'yamnet_lstm_model_5class_with_silence.h5'
if os.path.exists(model_path):
    print("5클래스 LSTM 모델(무음 포함) 로딩 중...")
    lstm_model = load_model(model_path)
    print("5클래스 LSTM 모델 로딩 완료!")
else:
    print(f"오류: {model_path} 파일이 없습니다.")
    print("먼저 LSTM_Train_5Class.py를 실행해주세요.")
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

def is_silence(audio, rms_threshold=0.005, max_threshold=0.01):
    """오디오가 무음인지 판단"""
    rms, max_val = get_audio_volume(audio)
    return rms < rms_threshold and max_val < max_threshold

def record_audio(duration, sample_rate):
    print(f"{duration}초 동안 녹음 중...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    audio = audio.flatten()
    
    # 오디오 전처리 적용
    processed_audio, preprocessing_info = preprocess_audio(audio, sample_rate)
    
    return processed_audio, preprocessing_info

def predict_risk(audio, preprocessing_info):
    # 먼저 실제 볼륨 체크
    rms, max_val = get_audio_volume(audio)
    if is_silence(audio):
        # 무음으로 직접 판단
        return 0, 0.95, np.array([0.95, 0.01, 0.01, 0.01, 0.01]), 0, rms, max_val
    
    # 주파수 분석으로 소리 특성 파악
    low_ratio, mid_ratio, high_ratio = analyze_frequency_content(audio)
    
    embeddings = get_yamnet_embedding(audio)  # (time_steps, 1024)
    
    # 모델 입력에 맞게 패딩 (훈련 시와 동일한 길이로)
    # 만약 임베딩이 훈련 시보다 짧으면 패딩, 길면 자름
    target_length = lstm_model.input_shape[1]  # 모델의 time_steps 차원
    current_length = embeddings.shape[0]
    
    if current_length < target_length:
        # 패딩
        pad_length = target_length - current_length
        embeddings = np.pad(embeddings, ((0, pad_length), (0, 0)), mode='constant')
    elif current_length > target_length:
        # 자르기
        embeddings = embeddings[:target_length]
    
    embeddings = np.expand_dims(embeddings, axis=0)  # 배치 차원 추가 (1, time_steps, 1024)
    preds = lstm_model.predict(embeddings, verbose=0)
    preds = preds[0]  # (time_steps, num_classes)
    
    # 각 클래스의 최대 확률과 위치 찾기
    max_probs = np.max(preds, axis=0)  # 각 클래스별 최대 확률
    overall_max_prob = np.max(max_probs)
    predicted_class = np.argmax(max_probs)
    
    # 프레임별 예측에서 가장 높은 확률을 가진 프레임 찾기
    max_frame_idx = np.argmax(np.max(preds, axis=1))
    frame_predictions = preds[max_frame_idx]  # 해당 프레임의 클래스별 확률
    
    # 전처리 정보 기반 보정
    confidence_adjustment = 1.0
    
    # 1. 무음 상황인데 모델이 다른 클래스로 예측한 경우 보정
    if rms < 0.01 and predicted_class != 0:
        frame_predictions = np.array([0.8, 0.15, 0.02, 0.02, 0.01])
        predicted_class = 0
        overall_max_prob = 0.8
    
    # 2. 클리핑이나 과도한 볼륨 감지 시 신뢰도 조정
    elif preprocessing_info['was_clipped'] or preprocessing_info['volume_reduced']:
        # 비명 클래스(4번)에 대한 신뢰도를 낮춤
        if predicted_class == 4:  # 비명으로 예측된 경우
            # 고주파 비율이 낮으면 (공장소리 특성) 비명 확률을 크게 낮춤
            if high_ratio < 0.3:
                frame_predictions[4] *= 0.3  # 비명 확률 70% 감소
                frame_predictions[1] *= 2.0  # 정상(공장) 확률 증가
                # 정규화
                frame_predictions = frame_predictions / np.sum(frame_predictions)
                
                # 재평가
                predicted_class = np.argmax(frame_predictions)
                overall_max_prob = np.max(frame_predictions)
        
        confidence_adjustment = 0.8  # 전반적인 신뢰도 낮춤
    
    # 3. 주파수 분석 기반 추가 검증
    if predicted_class == 4:  # 비명으로 예측된 경우
        # 공장소리 특성 (저-중주파 위주) 감지 시 보정
        if low_ratio + mid_ratio > 0.7 and high_ratio < 0.25:
            # 공장소리로 재분류
            frame_predictions = np.array([0.1, 0.7, 0.1, 0.05, 0.05])
            predicted_class = 1
            overall_max_prob = 0.7
            confidence_adjustment = 0.9
    
    overall_max_prob *= confidence_adjustment
    
    return predicted_class, overall_max_prob, frame_predictions, max_frame_idx, rms, max_val

def main():
    window_length = DURATION
    stride = 5.0  # 5초마다 윈도우 이동
    
    print("모델 정보: 5개 클래스 (5-클래스: 무음 포함)")
    print("실시간 위험 소리 감지 시작 (Ctrl+C로 종료)")
    print("=" * 50)
    
    try:
        while True:
            audio, preprocessing_info = record_audio(window_length, SAMPLE_RATE)
            
            # 5클래스 모델 예측
            predicted_class, max_prob, frame_predictions, max_frame_idx, rms, max_val = predict_risk(audio, preprocessing_info)
            
            # 주파수 분석 정보 추가
            low_ratio, mid_ratio, high_ratio = analyze_frequency_content(audio)
            
            # 결과 출력
            class_name = CLASS_NAMES[predicted_class] if predicted_class < len(CLASS_NAMES) else f"클래스{predicted_class}"
            class_icon = CLASS_COLORS.get(predicted_class, '❓')
            
            print(f"시간: {time.strftime('%H:%M:%S')}")
            print(f"오디오 볼륨: RMS={rms:.4f}, Max={max_val:.4f}")
            
            # 전처리 정보 출력
            if preprocessing_info['was_clipped']:
                print("⚠️ 클리핑 감지됨 - 신뢰도 조정")
            if preprocessing_info['volume_reduced']:
                print(f"🔧 볼륨 조정: {preprocessing_info['original_rms']:.3f} → {preprocessing_info['processed_rms']:.3f}")
            
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
            time.sleep(max(0, stride - window_length))
            
    except KeyboardInterrupt:
        print("\n프로그램을 종료합니다.")
            
    except KeyboardInterrupt:
        print("\n프로그램을 종료합니다.")

if __name__ == '__main__':
    main()
