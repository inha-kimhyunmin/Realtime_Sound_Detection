import librosa
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import time

# === 기본 설정 ===
SAMPLE_RATE = 16000
WINDOW_SIZE = 1024  # 분석 윈도우 크기 (약 0.064초)
HOP_LENGTH = 512    # 홉 길이

# === HNR 계산 함수 (기존 코드에서 가져옴) ===
def compute_hnr(y, sr):
    """Harmonic-to-Noise Ratio 계산"""
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=100, fmax=2000, sr=sr)
        if f0 is not None and np.any(~np.isnan(f0)):
            harmonic_energy = np.nanmean(f0)
            noise_energy = np.nanstd(f0)
            if noise_energy == 0:
                return 100.0  # 완전 주기적이면 매우 높은 HNR
            return 10 * np.log10(harmonic_energy / noise_energy)
        else:
            return 0.0
    except:
        return 0.0

def analyze_audio_segments(file_path, rms_threshold_percentile=70):
    """
    오디오 파일을 분석하여 RMS가 큰 구간에서 다른 특성들을 계산
    
    Args:
        file_path: 오디오 파일 경로
        rms_threshold_percentile: RMS 임계값 백분위수 (기본값: 70%)
    """
    print(f"\n🎵 분석 중: {os.path.basename(file_path)}")
    
    try:
        # 오디오 로드
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        print(f"📊 파일 길이: {len(y)/sr:.2f}초, 샘플링 레이트: {sr}Hz")
        
        # 프레임 단위로 RMS 계산
        frame_length = WINDOW_SIZE
        hop_length = HOP_LENGTH
        
        # RMS 계산 (프레임별)
        rms_frames = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        
        # 시간 축 생성
        times = librosa.frames_to_time(np.arange(len(rms_frames)), sr=sr, hop_length=hop_length)
        
        # RMS 임계값 설정 (상위 30% 구간)
        rms_threshold = np.percentile(rms_frames, rms_threshold_percentile)
        print(f"🎚️ RMS 임계값 ({rms_threshold_percentile}%): {rms_threshold:.4f}")
        
        # 높은 RMS 구간 찾기
        high_rms_indices = np.where(rms_frames > rms_threshold)[0]
        
        if len(high_rms_indices) == 0:
            print("⚠️ 임계값을 넘는 RMS 구간이 없습니다.")
            return
        
        print(f"🔊 높은 RMS 구간 수: {len(high_rms_indices)} / {len(rms_frames)} 프레임")
        
        # 연속된 구간들을 그룹화
        groups = []
        current_group = [high_rms_indices[0]]
        
        for i in range(1, len(high_rms_indices)):
            if high_rms_indices[i] - high_rms_indices[i-1] <= 2:  # 2프레임 이내면 연속으로 간주
                current_group.append(high_rms_indices[i])
            else:
                if len(current_group) > 5:  # 최소 5프레임 이상인 구간만 고려
                    groups.append(current_group)
                current_group = [high_rms_indices[i]]
        
        if len(current_group) > 5:
            groups.append(current_group)
        
        print(f"📈 연속된 높은 RMS 구간 개수: {len(groups)}개")
        
        # 각 구간별로 특성 분석
        segment_results = []
        
        for i, group in enumerate(groups):
            start_frame = group[0]
            end_frame = group[-1]
            
            # 프레임을 시간/샘플로 변환
            start_time = times[start_frame]
            end_time = times[end_frame]
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            
            # 해당 구간의 오디오 추출
            segment = y[start_sample:end_sample]
            
            if len(segment) < 1024:  # 너무 짧은 구간은 건너뛰기
                continue
            
            # 특성 계산
            segment_rms = np.sqrt(np.mean(segment ** 2))
            centroid = librosa.feature.spectral_centroid(y=segment, sr=sr)[0].mean()
            flatness = librosa.feature.spectral_flatness(y=segment)[0].mean()
            zcr = librosa.feature.zero_crossing_rate(segment)[0].mean()
            hnr = compute_hnr(segment, sr)
            
            # 추가 특성들
            peak = np.max(np.abs(segment))
            bandwidth = librosa.feature.spectral_bandwidth(y=segment, sr=sr)[0].mean()
            rolloff = librosa.feature.spectral_rolloff(y=segment, sr=sr)[0].mean()
            
            result = {
                'segment': i+1,
                'start_time': start_time,
                'end_time': end_time,
                'duration': end_time - start_time,
                'rms': segment_rms,
                'centroid': centroid,
                'flatness': flatness,
                'zcr': zcr,
                'hnr': hnr,
                'peak': peak,
                'bandwidth': bandwidth,
                'rolloff': rolloff
            }
            
            segment_results.append(result)
            
            print(f"\n📍 구간 {i+1}: {start_time:.2f}s - {end_time:.2f}s (길이: {end_time-start_time:.2f}s)")
            print(f"   RMS: {segment_rms:.4f}")
            print(f"   Spectral Centroid: {centroid:.1f} Hz")
            print(f"   Spectral Flatness: {flatness:.4f}")
            print(f"   Zero Crossing Rate: {zcr:.4f}")
            print(f"   HNR: {hnr:.2f} dB")
            print(f"   Peak: {peak:.4f}")
            print(f"   Bandwidth: {bandwidth:.1f} Hz")
            print(f"   Rolloff: {rolloff:.1f} Hz")
        
        # 전체 통계
        if segment_results:
            print(f"\n📊 전체 높은 RMS 구간 통계 (총 {len(segment_results)}개 구간):")
            rms_values = [r['rms'] for r in segment_results]
            centroid_values = [r['centroid'] for r in segment_results]
            flatness_values = [r['flatness'] for r in segment_results]
            zcr_values = [r['zcr'] for r in segment_results]
            hnr_values = [r['hnr'] for r in segment_results]
            
            print(f"🎚️ RMS - 평균: {np.mean(rms_values):.4f}, 최대: {np.max(rms_values):.4f}, 최소: {np.min(rms_values):.4f}")
            print(f"🎼 Centroid - 평균: {np.mean(centroid_values):.1f}Hz, 최대: {np.max(centroid_values):.1f}Hz, 최소: {np.min(centroid_values):.1f}Hz")
            print(f"📏 Flatness - 평균: {np.mean(flatness_values):.4f}, 최대: {np.max(flatness_values):.4f}, 최소: {np.min(flatness_values):.4f}")
            print(f"〰️ ZCR - 평균: {np.mean(zcr_values):.4f}, 최대: {np.max(zcr_values):.4f}, 최소: {np.min(zcr_values):.4f}")
            print(f"🎵 HNR - 평균: {np.mean(hnr_values):.2f}dB, 최대: {np.max(hnr_values):.2f}dB, 최소: {np.min(hnr_values):.2f}dB")
            
            return segment_results
        else:
            print("⚠️ 분석할 수 있는 구간이 없습니다.")
            return []
            
    except Exception as e:
        print(f"❌ 파일 분석 중 오류 발생: {e}")
        return []

def analyze_scream_folder(folder_path="scream"):
    """스크림 폴더의 모든 오디오 파일 분석"""
    print("🎭 스크림 파일 분석 시작")
    print("="*50)
    
    # 지원하는 오디오 포맷
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
    
    # 스크림 폴더의 파일들 찾기
    scream_folder = os.path.join(os.getcwd(), folder_path)
    
    if not os.path.exists(scream_folder):
        print(f"❌ 폴더를 찾을 수 없습니다: {scream_folder}")
        return
    
    audio_files = []
    for file in os.listdir(scream_folder):
        if any(file.lower().endswith(ext) for ext in audio_extensions):
            audio_files.append(os.path.join(scream_folder, file))
    
    if not audio_files:
        print(f"❌ {scream_folder}에서 오디오 파일을 찾을 수 없습니다.")
        return
    
    print(f"📁 찾은 오디오 파일: {len(audio_files)}개")
    for file in audio_files:
        print(f"   - {os.path.basename(file)}")
    
    # 각 파일 분석
    all_results = {}
    
    for file_path in audio_files:
        file_name = os.path.basename(file_path)
        results = analyze_audio_segments(file_path)
        all_results[file_name] = results
    
    # 전체 결과 요약
    print("\n" + "="*50)
    print("📈 전체 분석 결과 요약")
    print("="*50)
    
    total_segments = 0
    all_rms = []
    all_centroid = []
    all_flatness = []
    all_zcr = []
    all_hnr = []
    
    for file_name, results in all_results.items():
        if results:
            print(f"\n🎵 {file_name}: {len(results)}개 구간")
            total_segments += len(results)
            
            for result in results:
                all_rms.append(result['rms'])
                all_centroid.append(result['centroid'])
                all_flatness.append(result['flatness'])
                all_zcr.append(result['zcr'])
                all_hnr.append(result['hnr'])
    
    if total_segments > 0:
        print(f"\n🎯 모든 파일의 높은 RMS 구간 통합 통계 (총 {total_segments}개 구간):")
        print(f"🎚️ RMS - 평균: {np.mean(all_rms):.4f} ± {np.std(all_rms):.4f}")
        print(f"🎼 Centroid - 평균: {np.mean(all_centroid):.1f} ± {np.std(all_centroid):.1f} Hz")
        print(f"📏 Flatness - 평균: {np.mean(all_flatness):.4f} ± {np.std(all_flatness):.4f}")
        print(f"〰️ ZCR - 평균: {np.mean(all_zcr):.4f} ± {np.std(all_zcr):.4f}")
        print(f"🎵 HNR - 평균: {np.mean(all_hnr):.2f} ± {np.std(all_hnr):.2f} dB")
        
        # 기존 임계값과 비교
        print(f"\n🔍 현재 설정된 임계값과 비교:")
        print(f"   RMS 임계값: 0.02 (분석된 평균: {np.mean(all_rms):.4f})")
        print(f"   Centroid 임계값: 3000Hz (분석된 평균: {np.mean(all_centroid):.1f}Hz)")
        print(f"   Flatness 최대: 0.3 (분석된 평균: {np.mean(all_flatness):.4f})")
        print(f"   ZCR 임계값: 0.15 (분석된 평균: {np.mean(all_zcr):.4f})")
        print(f"   HNR 임계값: 10dB (분석된 평균: {np.mean(all_hnr):.2f}dB)")
        
        # 권장 임계값 제안
        print(f"\n💡 스크림 소리 기반 권장 임계값:")
        print(f"   RMS: {np.mean(all_rms) - np.std(all_rms):.4f} (평균 - 1표준편차)")
        print(f"   Centroid: {np.mean(all_centroid) - np.std(all_centroid):.1f}Hz")
        print(f"   Flatness 최대: {np.mean(all_flatness) + np.std(all_flatness):.4f}")
        print(f"   ZCR: {np.mean(all_zcr) - np.std(all_zcr):.4f}")
        print(f"   HNR: {np.mean(all_hnr) - np.std(all_hnr):.2f}dB")

if __name__ == "__main__":
    analyze_scream_folder()
