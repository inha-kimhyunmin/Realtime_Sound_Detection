"""
실시간 위험소리 감지 시스템 (병렬 처리 버전)
================================================

오디오 입력 수집과 모델 추론을 병렬로 처리하여 더 효율적인 실시간 감지를 구현합니다.

주요 특징:
- 지속적인 오디오 스트림 수집 (별도 스레드)
- 세그먼트 길이 달성시 자동 모델 추론 (별도 스레드)
- 논블로킹 방식으로 끊김없는 실시간 처리
- 순환 버퍼 방식으로 메모리 효율적 관리

사용법:
1. 먼저 LSTM_Train_5Class.py로 모델 훈련
2. 생성된 모델 파일 경로를 MODEL_PATH에 설정
3. python YAMNet_LSTM_Realtime_Parallel.py 실행
"""

import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub
import sounddevice as sd
import threading
import queue
import time
import pickle
import os
import glob
from datetime import datetime
from collections import deque
from scipy import signal

# ================================
# 설정 파라미터
# ================================
SAMPLE_RATE = 16000           # 샘플링 주파수 (YAMNet 기본값)
SEGMENT_DURATION = 10.0       # 분석할 오디오 세그먼트 길이 (초)
CHUNK_DURATION = 0.5          # 오디오 청크 수집 간격 (초)
DANGER_THRESHOLD = 0.7        # 위험 감지 확률 임계값
OVERLAP_RATIO = 0.5           # 세그먼트 간 겹침 비율 (0.5 = 50% 겹침)

# 모델 파일 경로 (자동 탐색 또는 수동 설정)
MODEL_PATH = None  # None이면 자동으로 최신 모델 탐색
MODEL_INFO_PATH = None  # None이면 자동으로 탐색

# 마이크 및 오디오 처리 설정
AUTO_CALIBRATION_MODE = True    # True: 자동 캘리브레이션, False: 수동 설정 사용
CALIBRATION_SILENCE_DURATION = 5.0      # 무음 캘리브레이션 시간 (초)
CALIBRATION_FACTORY_DURATION = 3.0      # 공장소리 캘리브레이션 녹음 시간 (초)
CALIBRATION_MAX_ATTEMPTS = 10           # 공장소리 인식 최대 시도 횟수
CALIBRATION_MIN_GAIN = 1.0              # 최소 마이크 감도
CALIBRATION_MAX_GAIN = 10.0             # 최대 마이크 감도
CALIBRATION_GAIN_STEP = 0.5             # 감도 증가 단계

# 마이크 및 무음 감지 설정 (수동 모드용)
MIC_GAIN = 3.0              # 마이크 입력 감도 (1.0 = 기본, 2.0 = 2배 증폭)
SILENCE_RMS_THRESHOLD = 0.005   # 무음 판단 RMS 임계값 (낮을수록 더 작은 소리도 감지)
SILENCE_MAX_THRESHOLD = 0.01    # 무음 판단 최대값 임계값

# 무음 처리 모드 설정
SILENCE_PROCESSING_MODE = True  # True: 작은 소리를 무음으로 강제 처리, False: 비활성화
SILENCE_FORCE_RMS_THRESHOLD = 0.02  # 이 값보다 작으면 무음으로 강제 분류 (캘리브레이션 시 동적 변경됨)
SILENCE_FORCE_MAX_THRESHOLD = 0.05  # 이 값보다 작으면 무음으로 강제 분류 (캘리브레이션 시 동적 변경됨)

# 공장 소리 기준 무음 처리 설정
FACTORY_BASED_SILENCE_MODE = True   # True: 공장 소리 크기 기준 무음 처리, False: 비활성화
FACTORY_SILENCE_RATIO = 0.5         # 공장 소리의 몇 % 이하를 무음으로 처리할지 (0.5 = 50%)
FACTORY_SILENCE_RMS_THRESHOLD = 0.0 # 공장 소리 RMS 기준값 (캘리브레이션 시 동적 설정됨)
FACTORY_SILENCE_MAX_THRESHOLD = 0.0 # 공장 소리 Max 기준값 (캘리브레이션 시 동적 설정됨)

# 클래스 정보
CLASS_NAMES = ['무음', '정상(공장)', '화재', '가스누출', '비명']
DANGER_CLASSES = [2, 3, 4]  # 화재, 가스누출, 비명
CLASS_COLORS = {
    0: '🔇',  # 무음
    1: '🟢',  # 정상(공장)
    2: '🔥',  # 화재
    3: '⚠️',  # 가스누출
    4: '😱'   # 비명
}

class RealTimeAudioDetector:
    def __init__(self):
        """실시간 오디오 감지기 초기화"""
        self.sample_rate = SAMPLE_RATE
        self.segment_length = int(SAMPLE_RATE * SEGMENT_DURATION)
        self.chunk_length = int(SAMPLE_RATE * CHUNK_DURATION)
        self.overlap_length = int(self.segment_length * OVERLAP_RATIO)
        self.step_length = self.segment_length - self.overlap_length
        
        # 오디오 버퍼 (순환 큐 방식)
        self.audio_buffer = deque(maxlen=int(SAMPLE_RATE * SEGMENT_DURATION * 2))  # 2배 버퍼
        self.buffer_lock = threading.Lock()
        
        # 모델 추론 큐
        self.inference_queue = queue.Queue(maxsize=5)  # 최대 5개 세그먼트 대기
        
        # 제어 플래그
        self.is_running = False
        self.audio_thread = None
        self.inference_thread = None
        
        # 캘리브레이션 결과 저장
        self.calibration_results = None
        self.current_mic_gain = MIC_GAIN
        self.current_silence_rms_threshold = SILENCE_RMS_THRESHOLD
        self.current_silence_max_threshold = SILENCE_MAX_THRESHOLD
        self.current_force_rms_threshold = SILENCE_FORCE_RMS_THRESHOLD
        self.current_force_max_threshold = SILENCE_FORCE_MAX_THRESHOLD
        self.current_factory_rms_threshold = FACTORY_SILENCE_RMS_THRESHOLD
        self.current_factory_max_threshold = FACTORY_SILENCE_MAX_THRESHOLD
        
        # 모델 및 YAMNet 로드
        self.load_models()
        
        # 캘리브레이션 수행
        self.perform_calibration()
        
        print("🎧 실시간 위험소리 감지 시스템 (병렬 처리 버전)")
        print(f"📁 모델: {self.model_path}")
        print(f"🎯 클래스: {CLASS_NAMES}")
        print(f"⚙️ 설정: 세그먼트 {SEGMENT_DURATION}초, 청크 {CHUNK_DURATION}초, 겹침 {OVERLAP_RATIO*100}%")
        print(f"🚨 위험 임계값: {DANGER_THRESHOLD*100}%")
        print(f"🎤 마이크 감도: {self.current_mic_gain:.1f}x")
        print("-" * 60)
    
    def find_latest_model(self):
        """가장 최신 모델 파일을 자동으로 찾기"""
        model_folders = glob.glob("model_results_*")
        if not model_folders:
            raise FileNotFoundError("모델 폴더를 찾을 수 없습니다. 먼저 LSTM_Train_5Class.py를 실행하여 모델을 훈련하세요.")
        
        # 가장 최신 폴더 선택 (타임스탬프 기준)
        latest_folder = max(model_folders, key=lambda x: os.path.getctime(x))
        
        # 모델 파일 찾기
        model_files = glob.glob(os.path.join(latest_folder, "yamnet_lstm_model_*.h5"))
        info_files = glob.glob(os.path.join(latest_folder, "model_info_*.pkl"))
        
        if not model_files:
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {latest_folder}")
        if not info_files:
            raise FileNotFoundError(f"모델 정보 파일을 찾을 수 없습니다: {latest_folder}")
        
        return model_files[0], info_files[0]
    
    def load_models(self):
        """모델 및 YAMNet 로드"""
        global MODEL_PATH, MODEL_INFO_PATH
        
        # 모델 경로 자동 탐색 또는 수동 설정
        if MODEL_PATH is None or MODEL_INFO_PATH is None:
            print("🔍 최신 모델 자동 탐색 중...")
            MODEL_PATH, MODEL_INFO_PATH = self.find_latest_model()
        
        self.model_path = MODEL_PATH
        self.model_info_path = MODEL_INFO_PATH
        
        # LSTM 모델 로드
        print("🧠 LSTM 모델 로드 중...")
        from tensorflow.keras.models import load_model
        self.lstm_model = load_model(self.model_path)
        print(f"✅ LSTM 모델 로드 완료: {os.path.basename(self.model_path)}")
        
        # 모델 정보 로드
        with open(self.model_info_path, 'rb') as f:
            self.model_info = pickle.load(f)
        
        # YAMNet 모델 로드
        print("🎵 YAMNet 모델 로드 중...")
        yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
        self.yamnet_model = hub.load(yamnet_model_handle)
        print("✅ YAMNet 모델 로드 완료")
    
    def extract_yamnet_embeddings(self, audio):
        """YAMNet 임베딩 추출"""
        try:
            # 오디오를 텐서로 변환
            waveform = tf.convert_to_tensor(audio, dtype=tf.float32)
            waveform = tf.squeeze(waveform)
            
            # YAMNet 추론
            yamnet_fn = self.yamnet_model.signatures['serving_default']
            yamnet_output = yamnet_fn(waveform=waveform)
            embeddings = yamnet_output['output_1'].numpy()  # (frames, 1024)
            
            return embeddings
        except Exception as e:
            print(f"⚠️ YAMNet 임베딩 추출 오류: {e}")
            return None
    
    def get_audio_volume(self, audio):
        """오디오의 RMS 볼륨과 최대 절댓값 계산"""
        rms = np.sqrt(np.mean(audio**2))
        max_val = np.max(np.abs(audio))
        return rms, max_val
    
    def detect_clipping(self, audio, threshold=0.95):
        """오디오 클리핑 감지"""
        clipped_samples = np.sum(np.abs(audio) >= threshold)
        clipping_ratio = clipped_samples / len(audio)
        return clipping_ratio > 0.01  # 1% 이상 클리핑되면 True
    
    def apply_compressor(self, audio, threshold=0.7, ratio=4.0, attack=0.003, release=0.1, sample_rate=16000):
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
    
    def normalize_audio_adaptive(self, audio, target_rms=0.1, max_gain=3.0):
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
    
    def preprocess_audio(self, audio, sample_rate=16000):
        """통합 오디오 전처리 함수"""
        original_rms, original_max = self.get_audio_volume(audio)
        
        # 1. 클리핑 감지
        is_clipped = self.detect_clipping(audio)
        
        # 2. 컴프레서 적용 (과도한 볼륨 제어)
        if original_max > 0.8 or is_clipped:
            audio = self.apply_compressor(audio, threshold=0.6, ratio=6.0)
        
        # 3. 적응형 정규화
        if original_rms > 0.001:  # 무음이 아닌 경우만
            # 볼륨이 매우 높은 경우 더 보수적으로 정규화
            if original_rms > 0.3:
                target_rms = 0.05
                max_gain = 1.5
            else:
                target_rms = 0.1
                max_gain = 3.0
                
            audio = self.normalize_audio_adaptive(audio, target_rms, max_gain)
        
        # 4. 최종 안전장치 (하드 리미터)
        audio = np.clip(audio, -0.95, 0.95)
        
        # 전처리 정보 반환
        processed_rms, processed_max = self.get_audio_volume(audio)
        preprocessing_info = {
            'original_rms': original_rms,
            'original_max': original_max,
            'processed_rms': processed_rms,
            'processed_max': processed_max,
            'was_clipped': is_clipped,
            'volume_reduced': original_rms > processed_rms * 1.5
        }
        
        return audio, preprocessing_info
    
    def is_silence(self, audio, rms_threshold=None, max_threshold=None):
        """오디오가 무음인지 판단"""
        if rms_threshold is None:
            rms_threshold = self.current_silence_rms_threshold
        if max_threshold is None:
            max_threshold = self.current_silence_max_threshold
            
        rms, max_val = self.get_audio_volume(audio)
        return rms < rms_threshold and max_val < max_threshold
    
    def perform_calibration(self):
        """캘리브레이션 수행"""
        if AUTO_CALIBRATION_MODE:
            print("🔧 자동 캘리브레이션 모드")
            self.calibration_results = self.calibrate_microphone()
            
            if self.calibration_results and self.calibration_results['factory_detected']:
                # 캘리브레이션 성공 시 설정값 업데이트
                self.current_mic_gain = self.calibration_results['optimal_gain']
                self.current_silence_rms_threshold = self.calibration_results['dynamic_silence_rms_threshold']
                self.current_silence_max_threshold = self.calibration_results['dynamic_silence_max_threshold']
                self.current_force_rms_threshold = self.calibration_results['force_silence_rms_threshold']
                self.current_force_max_threshold = self.calibration_results['force_silence_max_threshold']
                self.current_factory_rms_threshold = self.calibration_results['factory_rms_threshold']
                self.current_factory_max_threshold = self.calibration_results['factory_max_threshold']
            else:
                print("⚠️ 캘리브레이션 실패, 기본 설정값 사용")
        else:
            print("⚙️ 수동 설정 모드")
            self.current_mic_gain = MIC_GAIN
            self.current_silence_rms_threshold = SILENCE_RMS_THRESHOLD
            self.current_silence_max_threshold = SILENCE_MAX_THRESHOLD
            self.current_force_rms_threshold = SILENCE_FORCE_RMS_THRESHOLD
            self.current_force_max_threshold = SILENCE_FORCE_MAX_THRESHOLD
            self.current_factory_rms_threshold = FACTORY_SILENCE_RMS_THRESHOLD
            self.current_factory_max_threshold = FACTORY_SILENCE_MAX_THRESHOLD
    
    def calibrate_microphone(self):
        """마이크 자동 캘리브레이션 수행"""
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
            processed_audio, preprocessing_info = self.preprocess_audio(factory_audio, SAMPLE_RATE)
            
            # 오디오 분석
            rms, max_audio = self.get_audio_volume(processed_audio)
            
            # 기본 무음이 아닌지 확인 (임시 기준값 사용)
            is_not_silence = not self.is_silence(processed_audio, 
                                               rms_threshold=0.005,
                                               max_threshold=0.01)
            
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
                print(f"   ✅ 공장 소리 인식됨!")
                optimal_gain = current_gain
                factory_detected = True
                attempt_result['factory_audio_levels'] = {'rms': rms, 'max_val': max_audio}
            else:
                print(f"   ❌ 소리가 너무 작습니다. 감도를 높입니다.")
            
            calibration_results['calibration_history'].append(attempt_result)
            
            time.sleep(1)  # 다음 시도 전 잠시 대기
        
        calibration_results['attempts'] = attempt
        calibration_results['factory_detected'] = factory_detected
        calibration_results['optimal_gain'] = optimal_gain
        
        if not factory_detected:
            print(f"\n❌ 1단계 실패: 공장 소리를 인식하지 못했습니다.")
            return calibration_results
        
        # 2단계: 최적 감도로 무음 상태 측정
        print(f"\n📍 2단계: 무음 상태 측정 (감도: {optimal_gain:.1f}x, {CALIBRATION_SILENCE_DURATION}초)")
        print("💡 이제 공장 소리를 완전히 끄고 주변을 최대한 조용하게 유지해주세요...")
        
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
        processed_silence, _ = self.preprocess_audio(silence_audio, SAMPLE_RATE)
        
        # 무음 기준값 계산
        silence_rms = np.sqrt(np.mean(processed_silence**2))
        silence_max = np.max(np.abs(processed_silence))
        
        print(f"✅ 무음 기준값 측정 완료:")
        print(f"   - RMS: {silence_rms:.6f}")
        print(f"   - Max: {silence_max:.6f}")
        
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
        
        print(f"\n🎉 캘리브레이션 완료!")
        print("=" * 50)
        print(f"✅ 성공: 최적 마이크 감도 = {optimal_gain:.1f}x")
        print(f"📊 무음 기준값: RMS={silence_rms:.6f}, Max={silence_max:.6f}")
        
        return calibration_results
    
    def audio_callback(self, indata, frames, time_info, status):
        """오디오 입력 콜백 함수 (별도 스레드에서 실행)"""
        if status:
            print(f"⚠️ 오디오 상태: {status}")
        
        # 오디오 데이터를 버퍼에 추가
        audio_chunk = indata[:, 0]  # 모노로 변환
        
        # 마이크 감도 적용
        audio_chunk = audio_chunk * self.current_mic_gain
        
        # 클리핑 방지
        max_val = np.max(np.abs(audio_chunk))
        if max_val > 1.0:
            audio_chunk = audio_chunk / max_val
        
        with self.buffer_lock:
            self.audio_buffer.extend(audio_chunk)
            
            # 세그먼트가 준비되었는지 확인
            if len(self.audio_buffer) >= self.segment_length:
                # 세그먼트 추출
                segment = np.array(list(self.audio_buffer)[-self.segment_length:])
                
                # 추론 큐에 추가 (큐가 가득 차면 가장 오래된 것 제거)
                try:
                    self.inference_queue.put_nowait(segment.copy())
                except queue.Full:
                    try:
                        self.inference_queue.get_nowait()  # 오래된 것 제거
                        self.inference_queue.put_nowait(segment.copy())  # 새로운 것 추가
                    except queue.Empty:
                        pass
                
                # 버퍼에서 step_length만큼 제거 (겹침 구현)
                for _ in range(min(self.step_length, len(self.audio_buffer))):
                    if self.audio_buffer:
                        self.audio_buffer.popleft()
    
    def inference_worker(self):
        """모델 추론 워커 스레드"""
        while self.is_running:
            try:
                # 큐에서 오디오 세그먼트 가져오기 (1초 타임아웃)
                segment = self.inference_queue.get(timeout=1.0)
                
                # 모델 추론 수행
                self.process_audio_segment(segment)
                
                # 큐 작업 완료 표시
                self.inference_queue.task_done()
                
            except queue.Empty:
                continue  # 타임아웃시 계속 진행
            except Exception as e:
                print(f"⚠️ 추론 오류: {e}")
                continue
    
    def process_audio_segment(self, audio_segment):
        """오디오 세그먼트 처리 및 위험 감지"""
        try:
            start_time = time.time()
            
            # 오디오 전처리
            processed_audio, preprocessing_info = self.preprocess_audio(audio_segment, self.sample_rate)
            
            # 먼저 실제 볼륨 체크
            rms, max_val = self.get_audio_volume(processed_audio)
            
            # 1. 기본 무음 감지
            if self.is_silence(processed_audio):
                processing_time = time.time() - start_time
                self.handle_prediction_result(
                    np.array([1.0, 0.0, 0.0, 0.0, 0.0]), processing_time, 
                    forced_class=0, force_reason="기본 무음 감지", 
                    audio_info={'rms': rms, 'max_val': max_val}
                )
                return
            
            # 2. 무음 처리 모드 - 작은 소리를 무음으로 강제 처리
            if SILENCE_PROCESSING_MODE:
                if rms < self.current_force_rms_threshold and max_val < self.current_force_max_threshold:
                    processing_time = time.time() - start_time
                    self.handle_prediction_result(
                        np.array([1.0, 0.0, 0.0, 0.0, 0.0]), processing_time,
                        forced_class=0, force_reason="강제 무음 처리", 
                        audio_info={'rms': rms, 'max_val': max_val}
                    )
                    return
            
            # 3. 공장 기준 무음 처리 모드
            if FACTORY_BASED_SILENCE_MODE:
                if rms < self.current_factory_rms_threshold and max_val < self.current_factory_max_threshold:
                    processing_time = time.time() - start_time
                    self.handle_prediction_result(
                        np.array([1.0, 0.0, 0.0, 0.0, 0.0]), processing_time,
                        forced_class=0, force_reason="공장 기준 무음 처리", 
                        audio_info={'rms': rms, 'max_val': max_val}
                    )
                    return
            
            # YAMNet 임베딩 추출
            embeddings = self.extract_yamnet_embeddings(processed_audio)
            if embeddings is None:
                return
            
            # LSTM 모델 입력 형태로 변환
            # 패딩 또는 자르기
            expected_frames = self.lstm_model.input_shape[1]
            if embeddings.shape[0] < expected_frames:
                # 패딩
                padding_needed = expected_frames - embeddings.shape[0]
                embeddings = np.pad(embeddings, ((0, padding_needed), (0, 0)), mode='constant')
            elif embeddings.shape[0] > expected_frames:
                # 자르기
                embeddings = embeddings[:expected_frames]
            
            # 배치 차원 추가
            embeddings = np.expand_dims(embeddings, axis=0)  # (1, frames, 1024)
            
            # 모델 예측
            predictions = self.lstm_model.predict(embeddings, verbose=0)
            
            # 프레임별 예측을 평균내어 세그먼트 레벨 예측 생성
            segment_prediction = np.mean(predictions[0], axis=0)  # (num_classes,)
            
            predicted_class = np.argmax(segment_prediction)
            
            # 4. 무음 처리 모드 - AI 예측 후 추가 검증
            if SILENCE_PROCESSING_MODE and predicted_class >= 2:
                if rms < self.current_force_rms_threshold * 1.5 and max_val < self.current_force_max_threshold * 1.5:
                    processing_time = time.time() - start_time
                    self.handle_prediction_result(
                        np.array([1.0, 0.0, 0.0, 0.0, 0.0]), processing_time,
                        forced_class=0, force_reason="AI 후 강제 무음 처리", 
                        audio_info={'rms': rms, 'max_val': max_val}
                    )
                    return
            
            # 5. 공장 기준 무음 처리 모드 - AI 예측 후 추가 검증
            if FACTORY_BASED_SILENCE_MODE and predicted_class >= 2:
                if rms < self.current_factory_rms_threshold * 1.2 and max_val < self.current_factory_max_threshold * 1.2:
                    processing_time = time.time() - start_time
                    self.handle_prediction_result(
                        np.array([1.0, 0.0, 0.0, 0.0, 0.0]), processing_time,
                        forced_class=0, force_reason="AI 후 공장 기준 무음 처리", 
                        audio_info={'rms': rms, 'max_val': max_val}
                    )
                    return
            
            # 결과 처리
            processing_time = time.time() - start_time
            self.handle_prediction_result(segment_prediction, processing_time,
                                        audio_info={'rms': rms, 'max_val': max_val},
                                        preprocessing_info=preprocessing_info)
            
        except Exception as e:
            print(f"⚠️ 세그먼트 처리 오류: {e}")
    
    def handle_prediction_result(self, prediction, processing_time, forced_class=None, force_reason=None, 
                               audio_info=None, preprocessing_info=None):
        """예측 결과 처리 및 출력"""
        current_time = datetime.now().strftime("%H:%M:%S")
        
        if forced_class is not None:
            predicted_class = forced_class
            confidence = 1.0
            class_icon = CLASS_COLORS.get(int(predicted_class), '❓')
            
            # 강제 분류 정보 출력
            audio_str = ""
            if audio_info:
                audio_str = f" | 🔊 RMS={audio_info['rms']:.4f}, Max={audio_info['max_val']:.4f}"
            
            prob_str = " ".join([f"{p:.2f}" for p in prediction])
            print(f"🕒 {current_time} | 📊 확률: [{prob_str}]{audio_str} | ⚡ 처리시간: {processing_time:.3f}초")
            print(f"🔧 강제 분류: {class_icon} {CLASS_NAMES[predicted_class]} (사유: {force_reason})")
            
            if predicted_class == 0:  # 무음
                print(f"🔇 무음 상태")
        else:
            predicted_class = np.argmax(prediction)
            confidence = prediction[predicted_class]
            class_icon = CLASS_COLORS.get(int(predicted_class), '❓')
            
            # 오디오 정보 문자열 생성
            audio_str = ""
            if audio_info:
                audio_str = f" | 🔊 RMS={audio_info['rms']:.4f}, Max={audio_info['max_val']:.4f}"
            
            # 전처리 정보 문자열 생성
            preprocess_str = ""
            if preprocessing_info:
                if preprocessing_info.get('was_clipped'):
                    preprocess_str += " | 📎 클리핑감지"
                if preprocessing_info.get('volume_reduced'):
                    preprocess_str += " | 📉 볼륨조정"
            
            # 기본 정보 출력
            prob_str = " ".join([f"{p:.2f}" for p in prediction])
            print(f"🕒 {current_time} | 📊 확률: [{prob_str}]{audio_str}{preprocess_str} | ⚡ 처리시간: {processing_time:.3f}초")
            
            # 위험 감지 확인
            if predicted_class in DANGER_CLASSES and confidence >= DANGER_THRESHOLD:
                danger_type = CLASS_NAMES[predicted_class]
                print(f"🚨 위험 감지: {class_icon} {danger_type} ({confidence*100:.1f}% 확률)")
                print("=" * 80)
            elif predicted_class == 1 and confidence >= 0.8:  # 정상 소리도 높은 확률이면 표시
                print(f"✅ 정상: {class_icon} {CLASS_NAMES[predicted_class]} ({confidence*100:.1f}% 확률)")
            elif predicted_class == 0 and confidence >= 0.9:  # 무음도 높은 확률이면 표시
                print(f"🔇 무음 상태 ({confidence*100:.1f}% 확률)")
    
    def start_detection(self):
        """실시간 감지 시작"""
        if self.is_running:
            print("⚠️ 이미 실행 중입니다.")
            return
        
        self.is_running = True
        
        # 추론 워커 스레드 시작
        self.inference_thread = threading.Thread(target=self.inference_worker, daemon=True)
        self.inference_thread.start()
        
        print("🎤 오디오 입력 시작...")
        print("📡 실시간 감지 활성화")
        print("🛑 종료하려면 Ctrl+C를 누르세요")
        print("-" * 60)
        
        try:
            # 오디오 스트림 시작
            with sd.InputStream(
                channels=1,
                samplerate=self.sample_rate,
                blocksize=self.chunk_length,
                callback=self.audio_callback,
                dtype=np.float32
            ):
                # 메인 스레드는 종료 신호 대기
                while self.is_running:
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            print("\n🛑 사용자에 의해 중단됨")
        except Exception as e:
            print(f"❌ 오디오 스트림 오류: {e}")
        finally:
            self.stop_detection()
    
    def stop_detection(self):
        """실시간 감지 중지"""
        if not self.is_running:
            return
        
        print("\n🔄 시스템 종료 중...")
        self.is_running = False
        
        # 스레드 종료 대기
        if self.inference_thread and self.inference_thread.is_alive():
            self.inference_thread.join(timeout=2.0)
        
        # 큐 정리
        while not self.inference_queue.empty():
            try:
                self.inference_queue.get_nowait()
            except queue.Empty:
                break
        
        print("✅ 시스템 종료 완료")
    
    def get_system_info(self):
        """시스템 정보 출력"""
        mode_name = "자동 캘리브레이션" if AUTO_CALIBRATION_MODE else "수동 설정"
        
        print("\n📋 시스템 정보:")
        print(f"  - 샘플링 주파수: {self.sample_rate} Hz")
        print(f"  - 세그먼트 길이: {SEGMENT_DURATION}초 ({self.segment_length:,}샘플)")
        print(f"  - 청크 길이: {CHUNK_DURATION}초 ({self.chunk_length:,}샘플)")
        print(f"  - 겹침 비율: {OVERLAP_RATIO*100}%")
        print(f"  - 스텝 길이: {self.step_length:,}샘플")
        print(f"  - 버퍼 최대 크기: {self.audio_buffer.maxlen:,}샘플")
        print(f"  - 추론 큐 크기: {self.inference_queue.maxsize}")
        print(f"\n📊 사용 중인 설정값 ({mode_name}):")
        print(f"  - 마이크 감도: {self.current_mic_gain:.1f}x")
        print(f"  - 무음 RMS 임계값: {self.current_silence_rms_threshold:.6f}")
        print(f"  - 무음 Max 임계값: {self.current_silence_max_threshold:.6f}")
        print(f"  - 무음 처리 모드: {'켜짐' if SILENCE_PROCESSING_MODE else '꺼짐'}")
        if SILENCE_PROCESSING_MODE:
            print(f"    * 강제 무음 RMS 임계값: {self.current_force_rms_threshold:.6f}")
            print(f"    * 강제 무음 Max 임계값: {self.current_force_max_threshold:.6f}")
        print(f"  - 공장 기준 무음 처리: {'켜짐' if FACTORY_BASED_SILENCE_MODE else '꺼짐'}")
        if FACTORY_BASED_SILENCE_MODE:
            print(f"    * 공장 기준 RMS 임계값: {self.current_factory_rms_threshold:.6f}")
            print(f"    * 공장 기준 Max 임계값: {self.current_factory_max_threshold:.6f}")
            print(f"    * 공장 소리 비율: {FACTORY_SILENCE_RATIO*100:.0f}%")

def main():
    """메인 실행 함수"""
    try:
        # 감지기 초기화
        detector = RealTimeAudioDetector()
        
        # 시스템 정보 출력
        detector.get_system_info()
        
        # 실시간 감지 시작
        detector.start_detection()
        
    except FileNotFoundError as e:
        print(f"❌ 파일 오류: {e}")
        print("💡 해결 방법: 먼저 LSTM_Train_5Class.py를 실행하여 모델을 훈련하세요.")
    except Exception as e:
        print(f"❌ 시스템 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
