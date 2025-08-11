import sounddevice as sd
import numpy as np
import librosa
import time

# === 사용자 설정값 ===
DURATION = 1.0          # 녹음 시간 (초)
SAMPLE_RATE = 16000     # 샘플링 레이트
REF_RMS = 0.1           # 기준 RMS (94 dB SPL에 해당하는 RMS)

# === 감지 임계값 ===
# 스크림 파일 분석 결과를 바탕으로 업데이트된 임계값
SCREAM_THRESHOLDS = {
    "rms": 0.1,           # 실제 스크림은 0.26 평균, 보수적으로 0.18
    "centroid": 1400,      # 실제 스크림은 1790Hz 평균, 보수적으로 1400Hz
    "flatness_max": 0.01,  # 실제 스크림은 0.0056 평균, 스크림은 매우 톤적
    "zcr": 0.12,           # 실제 스크림은 0.14 평균, 보수적으로 0.12
    "hnr": 5               # 실제 스크림은 12.84dB 평균, 다양성 고려해 5dB
}

EXPLOSION_THRESHOLDS = {
    "rms": 0.05,
    "peak": 0.3,
    "flatness_min": 0.4,
    "onset": 0.3,
    "hnr_max": 5
}

# === HNR 계산 함수 ===
def compute_hnr(y, sr):
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=100, fmax=2000, sr=sr)
    if f0 is not None and np.any(~np.isnan(f0)):
        harmonic_energy = np.nanmean(f0)
        noise_energy = np.nanstd(f0)
        if noise_energy == 0:
            return 100.0  # 완전 주기적이면 매우 높은 HNR
        return 10 * np.log10(harmonic_energy / noise_energy)
    else:
        return 0.0

# === 분석 함수 ===
def analyze_sound(y, sr):
    start_time = time.time()

    # RMS 및 dB SPL 계산
    rms = np.sqrt(np.mean(y ** 2))
    rms_db_spl = 94 + 20 * np.log10(rms / REF_RMS) if rms > 0 else 0

    # 오디오 특성
    peak = np.max(np.abs(y))
    zcr = librosa.feature.zero_crossing_rate(y)[0].mean()
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean()
    flatness = librosa.feature.spectral_flatness(y=y)[0].mean()
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0].mean()
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0].mean()
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset = np.mean(onset_env) if len(onset_env) > 0 else 0
    hnr = compute_hnr(y, sr)

    # 판별
    scream_detected = (
        rms > SCREAM_THRESHOLDS["rms"] and
        centroid > SCREAM_THRESHOLDS["centroid"] and
        flatness < SCREAM_THRESHOLDS["flatness_max"] and
        zcr > SCREAM_THRESHOLDS["zcr"] and
        hnr > SCREAM_THRESHOLDS["hnr"]
    )

    explosion_detected = (
        rms > EXPLOSION_THRESHOLDS["rms"] and
        peak > EXPLOSION_THRESHOLDS["peak"] and
        flatness > EXPLOSION_THRESHOLDS["flatness_min"] and
        onset > EXPLOSION_THRESHOLDS["onset"] and
        hnr < EXPLOSION_THRESHOLDS["hnr_max"]
    )

    delay = time.time() - start_time

    # 결과 출력
    print("\n🎧 분석 결과")
    print(f"🕒 녹음 시각: {time.strftime('%H:%M:%S')}")
    print(f"🎚️ RMS(dB SPL): {rms_db_spl:.1f} dB")
    print(f"🔁 분석 지연: {delay:.2f}초")
    print(f"📊 RMS : {rms:.3f} ,ZCR: {zcr:.3f}, Centroid: {centroid:.1f} Hz, Flatness: {flatness:.3f}, HNR: {hnr:.2f}")
    print(f"📈 Peak: {peak:.3f}, Onset Strength: {onset:.3f}")

    if scream_detected:
        print("🚨 비명 소리 감지됨!")
    elif explosion_detected:
        print("💥 폭발 소리 감지됨!")
    else:
        print("✅ 특이 소리 없음")

# === 실시간 루프 ===
def main():
    print("🎙️ 실시간 비명/폭발 감지 시작 (Ctrl+C로 종료)")
    try:
        while True:
            print("\n🎤 녹음 중...")
            recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
            sd.wait()
            y = recording.flatten()
            analyze_sound(y, SAMPLE_RATE)
    except KeyboardInterrupt:
        print("\n🛑 프로그램 종료됨.")

if __name__ == "__main__":
    main()
