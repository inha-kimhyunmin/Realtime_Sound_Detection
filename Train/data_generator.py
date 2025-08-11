"""
데이터 생성 모듈
===============

이 모듈은 YAMNet + LSTM 훈련을 위한 데이터를 생성합니다.
- 각 클래스별 균등한 프레임 수 생성
- 데이터 증강을 통한 데이터 부족 해결
- 다양한 전환 시나리오 데이터 생성
- 프레임 레벨 라벨링
"""

import os
import json
import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from sklearn.utils import shuffle
import tensorflow as tf
import tensorflow_hub as hub
from tqdm import tqdm
import matplotlib.pyplot as plt
from config import *

class DataGenerator:
    def __init__(self):
        """데이터 생성기 초기화"""
        self.yamnet_model = None
        self.audio_files = {}
        self.audio_durations = {}
        self.total_frames_available = {}
        self.load_yamnet()
        self.scan_audio_files()
        
    def load_yamnet(self):
        """YAMNet 모델 로드"""
        print("🔄 YAMNet 모델 로딩 중...")
        yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
        self.yamnet_model = hub.load(yamnet_model_handle)
        print("✅ YAMNet 모델 로딩 완료")
        
    def scan_audio_files(self):
        """오디오 파일 스캔 및 프레임 수 계산"""
        print("🔍 오디오 파일 스캔 중...")
        
        # 위험 소리 파일 스캔
        for class_name in ACTIVE_DANGER_CLASSES:
            class_dir = os.path.join(ENVSOUND_DIR, class_name)
            if not os.path.exists(class_dir):
                print(f"⚠️ 경고: {class_dir} 폴더가 존재하지 않습니다.")
                continue
                
            files = [f for f in os.listdir(class_dir) if f.endswith(('.wav', '.mp3', '.flac'))]
            self.audio_files[class_name] = [os.path.join(class_dir, f) for f in files]
            
            # 각 파일의 길이 계산
            total_duration = 0
            durations = []
            for file_path in self.audio_files[class_name]:
                try:
                    duration = librosa.get_duration(filename=file_path)
                    durations.append(duration)
                    total_duration += duration
                except Exception as e:
                    print(f"⚠️ 파일 읽기 오류 {file_path}: {e}")
                    
            self.audio_durations[class_name] = durations
            # YAMNet 프레임 수 계산 (0.48초당 1프레임)
            self.total_frames_available[class_name] = int(total_duration / 0.48)
            
            print(f"  📁 {class_name}: {len(files)}개 파일, {total_duration:.1f}초, {self.total_frames_available[class_name]}개 프레임")
        
        # 공장 소리 파일 스캔
        if os.path.exists(MIXTURE_DIR):
            files = [f for f in os.listdir(MIXTURE_DIR) if f.endswith(('.wav', '.mp3', '.flac'))]
            self.audio_files['factory'] = [os.path.join(MIXTURE_DIR, f) for f in files]
            
            total_duration = 0
            durations = []
            for file_path in self.audio_files['factory']:
                try:
                    duration = librosa.get_duration(filename=file_path)
                    durations.append(duration)
                    total_duration += duration
                except Exception as e:
                    print(f"⚠️ 파일 읽기 오류 {file_path}: {e}")
                    
            self.audio_durations['factory'] = durations
            self.total_frames_available['factory'] = int(total_duration / 0.48)
            
            print(f"  📁 factory: {len(files)}개 파일, {total_duration:.1f}초, {self.total_frames_available['factory']}개 프레임")
        
        # 무음은 무제한으로 생성 가능
        self.total_frames_available['silence'] = 999999
        print(f"  📁 silence: 무제한 생성 가능")
        
    def calculate_optimal_samples(self):
        """클래스별 최적 샘플 수 계산"""
        target_frames = DATA_GENERATION_CONFIG['target_frames_per_class']
        frames_per_audio = get_audio_frames_count()
        
        print("\n📊 클래스별 데이터 분석:")
        print("-" * 60)
        
        recommendations = {}
        
        for class_name in ALL_CLASSES:
            available_frames = self.total_frames_available.get(class_name, 0)
            base_samples_possible = available_frames // frames_per_audio
            
            # 데이터 증강 고려
            if AUGMENTATION_CONFIG.get(class_name, {}).get('enabled', False):
                max_aug = AUGMENTATION_CONFIG[class_name]['max_augmentations']
                augmented_samples_possible = base_samples_possible * (1 + max_aug)
            else:
                augmented_samples_possible = base_samples_possible
            
            # 필요한 샘플 수 계산
            needed_samples = target_frames // frames_per_audio
            
            # 전환 데이터 기여분 계산
            transition_contribution = self._calculate_transition_contribution(class_name, frames_per_audio)
            actual_needed = max(0, needed_samples - transition_contribution)
            
            recommendations[class_name] = {
                'available_frames': available_frames,
                'base_samples_possible': base_samples_possible,
                'augmented_samples_possible': augmented_samples_possible,
                'needed_samples': needed_samples,
                'transition_contribution': transition_contribution,
                'actual_needed': actual_needed,
                'need_augmentation': actual_needed > base_samples_possible,
                'feasible': actual_needed <= augmented_samples_possible or class_name == 'silence'
            }
            
            status = "✅" if recommendations[class_name]['feasible'] else "❌"
            aug_status = "증강필요" if recommendations[class_name]['need_augmentation'] else "원본충분"
            
            print(f"{status} {class_name:10} | 가용: {base_samples_possible:4d} | 필요: {actual_needed:4d} | {aug_status}")
            
        print("-" * 60)
        
        # 전체 실현 가능성 확인
        all_feasible = all(rec['feasible'] for rec in recommendations.values())
        
        if all_feasible:
            print("✅ 모든 클래스의 목표 프레임 수 달성 가능")
        else:
            print("❌ 일부 클래스의 목표 프레임 수 달성 불가능")
            infeasible = [name for name, rec in recommendations.items() if not rec['feasible']]
            print(f"   문제 클래스: {', '.join(infeasible)}")
            
        return recommendations
    
    def _calculate_transition_contribution(self, class_name, frames_per_audio):
        """전환 데이터로부터 해당 클래스가 얻을 수 있는 프레임 수 계산"""
        if not TRANSITION_CONFIG['enabled']:
            return 0
            
        contribution = 0
        transition_ratio = DATA_GENERATION_CONFIG['transition_data_ratio']
        
        # 각 전환 타입에서 해당 클래스가 차지하는 프레임 계산
        for trans_type, config in TRANSITION_CONFIG['types'].items():
            if not config['enabled']:
                continue
                
            if trans_type == 'silence_to_silence' and class_name == 'silence':
                contribution += frames_per_audio * config['weight'] * transition_ratio
            elif trans_type == 'silence_to_factory' and class_name in ['silence', 'factory']:
                # 전환 데이터에서 각 클래스가 차지하는 비율 추정 (50:50)
                contribution += frames_per_audio * config['weight'] * transition_ratio * 0.5
            elif trans_type == 'silence_to_danger' and (class_name == 'silence' or class_name in ACTIVE_DANGER_CLASSES):
                contribution += frames_per_audio * config['weight'] * transition_ratio * 0.5
            elif trans_type == 'factory_to_factory' and class_name == 'factory':
                contribution += frames_per_audio * config['weight'] * transition_ratio
            elif trans_type == 'factory_to_danger' and (class_name == 'factory' or class_name in ACTIVE_DANGER_CLASSES):
                contribution += frames_per_audio * config['weight'] * transition_ratio * 0.5
                
        return int(contribution)
    
    def get_user_input_for_samples(self, recommendations):
        """사용자로부터 실제 생성할 샘플 수 입력받기"""
        if not DATA_GENERATION_CONFIG['allow_user_input']:
            # 자동으로 권장값 사용
            return {name: rec['actual_needed'] for name, rec in recommendations.items()}
        
        print(f"\n🎯 각 클래스별 생성할 샘플 수를 결정해주세요:")
        print(f"💡 목표: 클래스당 {DATA_GENERATION_CONFIG['target_frames_per_class']}개 프레임")
        print(f"📏 오디오당 프레임: {get_audio_frames_count()}개")
        print("-" * 60)
        
        user_samples = {}
        
        for class_name in ALL_CLASSES:
            rec = recommendations[class_name]
            
            print(f"\n📁 {class_name} ({CLASS_NAMES.get(class_name, class_name)}):")
            print(f"  - 가용 원본 샘플: {rec['base_samples_possible']}개")
            print(f"  - 증강 포함 최대: {rec['augmented_samples_possible']}개")
            print(f"  - 권장 생성 수: {rec['actual_needed']}개")
            print(f"  - 전환 데이터 기여: {rec['transition_contribution']:.0f}개 프레임")
            
            while True:
                try:
                    user_input = input(f"  👉 생성할 샘플 수 (권장: {rec['actual_needed']}): ").strip()
                    
                    if user_input == "":
                        samples = rec['actual_needed']
                    else:
                        samples = int(user_input)
                    
                    if samples < 0:
                        print("     ❌ 음수는 입력할 수 없습니다.")
                        continue
                    
                    if samples > rec['augmented_samples_possible'] and class_name != 'silence':
                        print(f"     ❌ 최대 {rec['augmented_samples_possible']}개까지 가능합니다.")
                        continue
                    
                    user_samples[class_name] = samples
                    print(f"     ✅ {samples}개 설정됨")
                    break
                    
                except ValueError:
                    print("     ❌ 숫자를 입력해주세요.")
                except KeyboardInterrupt:
                    print("\n\n👋 사용자 중단")
                    return None
        
        # 최종 확인
        print(f"\n📋 최종 데이터 생성 계획:")
        total_samples = sum(user_samples.values())
        for class_name, samples in user_samples.items():
            frames = samples * get_audio_frames_count() + self._calculate_transition_contribution(class_name, get_audio_frames_count())
            print(f"  - {class_name}: {samples}개 샘플 → 약 {frames:.0f}개 프레임")
        
        print(f"\n총 {total_samples}개 샘플 생성 예정")
        
        confirm = input("\n계속 진행하시겠습니까? (y/N): ").strip().lower()
        if confirm not in ['y', 'yes']:
            print("👋 중단됨")
            return None
            
        return user_samples
    
    def generate_silence_audio(self, duration):
        """무음 오디오 생성 (다양한 배경노이즈 포함)"""
        sr = MODEL_CONFIG['sample_rate']
        length = int(duration * sr)
        
        # 기본 저잡음
        noise_level = np.random.uniform(0.001, 0.005)
        audio = np.random.normal(0, noise_level, length)
        
        # 배경 노이즈 추가 (확률적)
        if np.random.random() < 0.3:  # 30% 확률로 배경노이즈 추가
            noise_type = np.random.choice(['white', 'pink', 'brown'])
            noise_level = np.random.uniform(0.005, 0.02)
            
            if noise_type == 'white':
                noise = np.random.normal(0, noise_level, length)
            elif noise_type == 'pink':
                # 핑크 노이즈 생성 (1/f 특성)
                freqs = np.fft.fftfreq(length, 1/sr)
                freqs[0] = 1  # DC 성분 방지
                pink_filter = 1 / np.sqrt(np.abs(freqs))
                white_noise = np.random.normal(0, 1, length)
                pink_fft = np.fft.fft(white_noise) * pink_filter
                noise = np.real(np.fft.ifft(pink_fft)) * noise_level
            else:  # brown
                # 브라운 노이즈 생성 (1/f^2 특성)
                freqs = np.fft.fftfreq(length, 1/sr)
                freqs[0] = 1
                brown_filter = 1 / np.abs(freqs)
                white_noise = np.random.normal(0, 1, length)
                brown_fft = np.fft.fft(white_noise) * brown_filter
                noise = np.real(np.fft.ifft(brown_fft)) * noise_level
            
            audio += noise
        
        # 클리핑 방지
        max_val = np.max(np.abs(audio))
        if max_val > 0.95:
            audio = audio * (0.95 / max_val)
            
        return audio.astype(np.float32)
    
    def load_audio_file(self, file_path):
        """오디오 파일 로드"""
        try:
            audio, sr = librosa.load(file_path, sr=MODEL_CONFIG['sample_rate'])
            return audio
        except Exception as e:
            print(f"⚠️ 오디오 로드 실패 {file_path}: {e}")
            return None
    
    def extract_audio_segment(self, audio, duration=None):
        """오디오에서 지정된 길이의 세그먼트 추출"""
        if duration is None:
            duration = MODEL_CONFIG['audio_duration']
            
        sr = MODEL_CONFIG['sample_rate']
        target_length = int(duration * sr)
        
        if len(audio) >= target_length:
            # 랜덤 위치에서 추출
            start_idx = np.random.randint(0, len(audio) - target_length + 1)
            return audio[start_idx:start_idx + target_length]
        else:
            # 패딩 또는 반복
            if len(audio) < target_length // 2:
                # 너무 짧으면 반복
                repeat_count = (target_length // len(audio)) + 1
                audio = np.tile(audio, repeat_count)
            
            # 부족한 부분은 제로 패딩
            padding = target_length - len(audio)
            if padding > 0:
                pad_left = padding // 2
                pad_right = padding - pad_left
                audio = np.pad(audio, (pad_left, pad_right), mode='constant')
            
            return audio[:target_length]
    
    def apply_augmentation(self, audio, class_name, method):
        """데이터 증강 적용"""
        config = AUGMENTATION_CONFIG.get(class_name, {})
        if not config.get('enabled', False):
            return audio
            
        sr = MODEL_CONFIG['sample_rate']
        augmented = audio.copy()
        
        if method == 'volume_change':
            vol_min, vol_max = config.get('volume_range', (0.7, 1.3))
            volume_factor = np.random.uniform(vol_min, vol_max)
            augmented = augmented * volume_factor
            
        elif method == 'noise_variation' and class_name == 'silence':
            # 무음 클래스의 노이즈 변화
            noise_types = config.get('noise_types', ['white'])
            noise_type = np.random.choice(noise_types)
            noise_level = np.random.uniform(0.001, 0.01)
            
            if noise_type == 'white':
                noise = np.random.normal(0, noise_level, len(augmented))
            elif noise_type == 'pink':
                freqs = np.fft.fftfreq(len(augmented), 1/sr)
                freqs[0] = 1
                pink_filter = 1 / np.sqrt(np.abs(freqs))
                white_noise = np.random.normal(0, 1, len(augmented))
                pink_fft = np.fft.fft(white_noise) * pink_filter
                noise = np.real(np.fft.ifft(pink_fft)) * noise_level
            else:  # brown
                freqs = np.fft.fftfreq(len(augmented), 1/sr)
                freqs[0] = 1
                brown_filter = 1 / np.abs(freqs)
                white_noise = np.random.normal(0, 1, len(augmented))
                brown_fft = np.fft.fft(white_noise) * brown_filter
                noise = np.real(np.fft.ifft(brown_fft)) * noise_level
            
            augmented = noise  # 무음의 경우 노이즈로 대체
            
        elif method == 'reverb':
            # 간단한 리버브 효과
            decay = np.random.uniform(*config.get('reverb_decay', (0.1, 0.5)))
            delay_samples = int(0.05 * sr)  # 50ms 지연
            
            reverb = np.zeros_like(augmented)
            for i in range(3):  # 3회 반복
                delay = delay_samples * (i + 1)
                amplitude = decay ** (i + 1)
                if delay < len(augmented):
                    reverb[delay:] += augmented[:-delay] * amplitude
            
            augmented = augmented + reverb * 0.3
            
        elif method == 'room_effect':
            # 룸 효과 (간단한 IIR 필터)
            room_size = np.random.uniform(*config.get('room_size', (0.1, 0.9)))
            try:
                b, a = signal.butter(2, 0.1 + room_size * 0.4, btype='low')
                augmented = signal.filtfilt(b, a, augmented)
            except Exception:
                # 필터 생성 실패시 원본 반환
                pass
            
        elif method == 'speed_change':
            # 속도 변화 (시간 스트레칭)
            speed_min, speed_max = config.get('speed_range', (0.9, 1.1))
            speed_factor = np.random.uniform(speed_min, speed_max)
            augmented = librosa.effects.time_stretch(augmented, rate=speed_factor)
            # 길이 조정
            augmented = self.extract_audio_segment(augmented, MODEL_CONFIG['audio_duration'])
            
        elif method == 'noise_add':
            # 노이즈 추가
            noise_level = np.random.uniform(*config.get('noise_level', (0.01, 0.05)))
            noise = np.random.normal(0, noise_level, len(augmented))
            augmented = augmented + noise
            
        elif method == 'factory_mix':
            # 공장소리와 혼합 (위험소리용)
            if 'factory' in self.audio_files and self.audio_files['factory']:
                factory_file = np.random.choice(self.audio_files['factory'])
                factory_audio = self.load_audio_file(factory_file)
                if factory_audio is not None:
                    factory_segment = self.extract_audio_segment(factory_audio, MODEL_CONFIG['audio_duration'])
                    
                    # SNR 계산
                    snr_min, snr_max = config.get('snr_range', (5, 20))
                    target_snr = np.random.uniform(snr_min, snr_max)
                    
                    # 신호와 노이즈(공장소리) 파워 계산
                    signal_power = np.mean(augmented ** 2)
                    noise_power = np.mean(factory_segment ** 2)
                    
                    if noise_power > 0:
                        # SNR에 맞는 스케일링 팩터 계산
                        snr_linear = 10 ** (target_snr / 10)
                        noise_scale = np.sqrt(signal_power / (noise_power * snr_linear))
                        factory_segment = factory_segment * noise_scale
                        
                        augmented = augmented + factory_segment
        
        # 클리핑 방지
        max_val = np.max(np.abs(augmented))
        if max_val > 0.95:
            augmented = augmented * (0.95 / max_val)
            
        return augmented
    
    def generate_transition_audio(self, trans_type, class1, class2=None):
        """전환 오디오 생성"""
        duration = MODEL_CONFIG['audio_duration']
        sr = MODEL_CONFIG['sample_rate']
        length = int(duration * sr)
        
        config = TRANSITION_CONFIG['types'][trans_type]
        trans_point_range = config['transition_point_range']
        transition_point = np.random.uniform(*trans_point_range)
        transition_frame = int(transition_point * length)
        
        fade_duration = TRANSITION_CONFIG['fade_duration']
        fade_samples = int(fade_duration * sr)
        
        # 기본 무음으로 초기화
        audio1 = self.generate_silence_audio(duration)
        audio2 = self.generate_silence_audio(duration)
        
        # 오디오 생성
        if trans_type == 'silence_to_silence':
            # 두 종류의 무음 생성
            audio1 = self.generate_silence_audio(duration)
            audio2 = self.generate_silence_audio(duration)
            
        elif trans_type == 'silence_to_factory':
            audio1 = self.generate_silence_audio(duration)
            if 'factory' in self.audio_files and self.audio_files['factory']:
                factory_file = np.random.choice(self.audio_files['factory'])
                factory_audio = self.load_audio_file(factory_file)
                audio2 = self.extract_audio_segment(factory_audio, duration)
            else:
                audio2 = self.generate_silence_audio(duration)
                
        elif trans_type == 'silence_to_danger':
            audio1 = self.generate_silence_audio(duration)
            danger_class = np.random.choice(ACTIVE_DANGER_CLASSES)
            if danger_class in self.audio_files and self.audio_files[danger_class]:
                danger_file = np.random.choice(self.audio_files[danger_class])
                danger_audio = self.load_audio_file(danger_file)
                audio2 = self.extract_audio_segment(danger_audio, duration)
            else:
                audio2 = self.generate_silence_audio(duration)
                
        elif trans_type == 'factory_to_factory':
            if 'factory' in self.audio_files and len(self.audio_files['factory']) >= 2:
                files = np.random.choice(self.audio_files['factory'], 2, replace=False)
                audio1 = self.extract_audio_segment(self.load_audio_file(files[0]), duration)
                audio2 = self.extract_audio_segment(self.load_audio_file(files[1]), duration)
            else:
                audio1 = self.generate_silence_audio(duration)
                audio2 = self.generate_silence_audio(duration)
                
        elif trans_type == 'factory_to_danger':
            if 'factory' in self.audio_files and self.audio_files['factory']:
                factory_file = np.random.choice(self.audio_files['factory'])
                factory_audio = self.load_audio_file(factory_file)
                audio1 = self.extract_audio_segment(factory_audio, duration)
            else:
                audio1 = self.generate_silence_audio(duration)
                
            danger_class = np.random.choice(ACTIVE_DANGER_CLASSES)
            if danger_class in self.audio_files and self.audio_files[danger_class]:
                danger_file = np.random.choice(self.audio_files[danger_class])
                danger_audio = self.load_audio_file(danger_file)
                danger_segment = self.extract_audio_segment(danger_audio, duration)
                
                # 위험소리 볼륨 조절
                vol_min, vol_max = config.get('danger_volume_ratio', (0.8, 1.5))
                volume_ratio = np.random.uniform(vol_min, vol_max)
                danger_segment = danger_segment * volume_ratio
                
                # 공장소리에 위험소리 추가 (믹싱)
                audio2 = audio1 + danger_segment
            else:
                audio2 = audio1
        
        # 페이드 전환 적용
        result = np.zeros(length, dtype=np.float32)
        
        # 전환 전 구간
        result[:transition_frame] = audio1[:transition_frame]
        
        # 전환 구간 (페이드)
        fade_start = max(0, transition_frame - fade_samples // 2)
        fade_end = min(length, transition_frame + fade_samples // 2)
        fade_length = fade_end - fade_start
        
        if fade_length > 0:
            fade_curve = np.linspace(0, 1, fade_length)
            result[fade_start:fade_end] = (
                audio1[fade_start:fade_end] * (1 - fade_curve) +
                audio2[fade_start:fade_end] * fade_curve
            )
        
        # 전환 후 구간
        if transition_frame < length:
            result[transition_frame:] = audio2[transition_frame:]
        
        # 프레임별 라벨 생성
        frames_per_audio = get_audio_frames_count()
        transition_frame_idx = int(transition_point * frames_per_audio)
        
        labels = np.zeros(frames_per_audio, dtype=int)
        
        if trans_type == 'silence_to_silence':
            labels[:] = ALL_CLASSES.index('silence')
        elif trans_type == 'silence_to_factory':
            labels[:transition_frame_idx] = ALL_CLASSES.index('silence')
            labels[transition_frame_idx:] = ALL_CLASSES.index('factory')
        elif trans_type == 'silence_to_danger':
            labels[:transition_frame_idx] = ALL_CLASSES.index('silence')
            danger_class = np.random.choice(ACTIVE_DANGER_CLASSES)
            labels[transition_frame_idx:] = ALL_CLASSES.index(danger_class)
        elif trans_type == 'factory_to_factory':
            labels[:] = ALL_CLASSES.index('factory')
        elif trans_type == 'factory_to_danger':
            labels[:transition_frame_idx] = ALL_CLASSES.index('factory')
            danger_class = np.random.choice(ACTIVE_DANGER_CLASSES)
            labels[transition_frame_idx:] = ALL_CLASSES.index(danger_class)
        
        return result, labels
    
    def extract_yamnet_embeddings(self, audio):
        """YAMNet 임베딩 추출"""
        try:
            if self.yamnet_model is None:
                print("⚠️ YAMNet 모델이 로드되지 않았습니다.")
                return None
                
            audio_tensor = tf.convert_to_tensor(audio, dtype=tf.float32)
            # YAMNet 모델 호출
            _, embeddings, _ = self.yamnet_model(audio_tensor)
            return embeddings.numpy()
        except Exception as e:
            print(f"⚠️ YAMNet 임베딩 추출 실패: {e}")
            return None
    
    def generate_dataset(self, sample_counts):
        """전체 데이터셋 생성"""
        print("\n🏭 데이터셋 생성 시작...")
        
        all_embeddings = []
        all_labels = []
        dataset_info = {
            'config': {
                'model_version': MODEL_CONFIG['version'],
                'audio_duration': MODEL_CONFIG['audio_duration'],
                'sample_rate': MODEL_CONFIG['sample_rate'],
                'num_classes': NUM_CLASSES,
                'class_names': ALL_CLASSES
            },
            'generation_stats': {},
            'files_used': {}
        }
        
        total_samples = sum(sample_counts.values())
        pbar = tqdm(total=total_samples, desc="데이터 생성")
        
        # 각 클래스별 기본 데이터 생성
        for class_name, target_count in sample_counts.items():
            if target_count == 0:
                continue
                
            print(f"\n📁 {class_name} 클래스 생성 중... (목표: {target_count}개)")
            
            class_embeddings = []
            class_labels = []
            files_used = []
            
            class_idx = ALL_CLASSES.index(class_name)
            frames_per_audio = get_audio_frames_count()
            
            if class_name == 'silence':
                # 무음 데이터 생성
                for i in range(target_count):
                    audio = self.generate_silence_audio(MODEL_CONFIG['audio_duration'])
                    
                    # 데이터 증강 적용 (확률적)
                    if AUGMENTATION_CONFIG.get(class_name, {}).get('enabled', False):
                        if np.random.random() < 0.5:  # 50% 확률로 증강
                            methods = AUGMENTATION_CONFIG[class_name]['methods']
                            method = np.random.choice(methods)
                            audio = self.apply_augmentation(audio, class_name, method)
                    
                    embeddings = self.extract_yamnet_embeddings(audio)
                    if embeddings is not None:
                        labels = np.full(embeddings.shape[0], class_idx, dtype=int)
                        class_embeddings.append(embeddings)
                        class_labels.append(labels)
                        files_used.append(f"generated_silence_{i}")
                    
                    pbar.update(1)
                    
            else:
                # 실제 오디오 파일 사용
                if class_name not in self.audio_files or not self.audio_files[class_name]:
                    print(f"⚠️ {class_name} 클래스의 오디오 파일이 없습니다.")
                    continue
                
                files = self.audio_files[class_name]
                generated_count = 0
                
                # 필요한 만큼 반복 생성
                while generated_count < target_count:
                    for file_path in files:
                        if generated_count >= target_count:
                            break
                            
                        audio = self.load_audio_file(file_path)
                        if audio is None:
                            continue
                        
                        # 파일이 충분히 길면 여러 세그먼트 추출 가능
                        duration = MODEL_CONFIG['audio_duration']
                        sr = MODEL_CONFIG['sample_rate']
                        target_length = int(duration * sr)
                        
                        num_segments = max(1, len(audio) // target_length)
                        
                        for seg_idx in range(num_segments):
                            if generated_count >= target_count:
                                break
                            
                            # 세그먼트 추출
                            if seg_idx == 0:
                                segment = self.extract_audio_segment(audio, duration)
                            else:
                                # 약간씩 겹치게 추출
                                start_idx = int(seg_idx * target_length * 0.8)
                                segment = audio[start_idx:start_idx + target_length]
                                if len(segment) < target_length:
                                    segment = self.extract_audio_segment(audio, duration)
                            
                            # 데이터 증강 적용 (필요시)
                            if AUGMENTATION_CONFIG.get(class_name, {}).get('enabled', False):
                                # 증강 필요성 판단
                                base_samples = len(files) * num_segments
                                if target_count > base_samples:
                                    methods = AUGMENTATION_CONFIG[class_name]['methods']
                                    method = np.random.choice(methods)
                                    segment = self.apply_augmentation(segment, class_name, method)
                            
                            embeddings = self.extract_yamnet_embeddings(segment)
                            if embeddings is not None:
                                labels = np.full(embeddings.shape[0], class_idx, dtype=int)
                                class_embeddings.append(embeddings)
                                class_labels.append(labels)
                                files_used.append(f"{os.path.basename(file_path)}_seg{seg_idx}")
                                generated_count += 1
                            
                            pbar.update(1)
            
            # 클래스별 통계 저장
            if class_embeddings:
                total_frames = sum(emb.shape[0] for emb in class_embeddings)
                dataset_info['generation_stats'][class_name] = {
                    'samples_generated': len(class_embeddings),
                    'total_frames': total_frames,
                    'target_samples': target_count
                }
                dataset_info['files_used'][class_name] = files_used
                
                all_embeddings.extend(class_embeddings)
                all_labels.extend(class_labels)
        
        # 전환 데이터 생성
        if TRANSITION_CONFIG['enabled']:
            print(f"\n🔄 전환 데이터 생성 중...")
            
            transition_ratio = DATA_GENERATION_CONFIG['transition_data_ratio']
            base_samples = len(all_embeddings)
            transition_samples_needed = int(base_samples * transition_ratio / (1 - transition_ratio))
            
            transition_embeddings = []
            transition_labels = []
            transition_files = []
            
            for trans_type, config in TRANSITION_CONFIG['types'].items():
                if not config['enabled']:
                    continue
                    
                type_samples = int(transition_samples_needed * config['weight'] / 
                                 sum(c['weight'] for c in TRANSITION_CONFIG['types'].values() if c['enabled']))
                
                for i in range(type_samples):
                    audio, labels = self.generate_transition_audio(trans_type, None, None)
                    embeddings = self.extract_yamnet_embeddings(audio)
                    
                    if embeddings is not None:
                        # 라벨 길이 조정
                        if len(labels) != embeddings.shape[0]:
                            if len(labels) < embeddings.shape[0]:
                                # 패딩
                                labels = np.pad(labels, (0, embeddings.shape[0] - len(labels)), 
                                              mode='edge')
                            else:
                                # 자르기
                                labels = labels[:embeddings.shape[0]]
                        
                        transition_embeddings.append(embeddings)
                        transition_labels.append(labels)
                        transition_files.append(f"{trans_type}_{i}")
                        
                        pbar.update(1)
            
            if transition_embeddings:
                all_embeddings.extend(transition_embeddings)
                all_labels.extend(transition_labels)
                
                dataset_info['generation_stats']['transitions'] = {
                    'total_samples': len(transition_embeddings),
                    'total_frames': sum(emb.shape[0] for emb in transition_embeddings),
                    'by_type': {}
                }
                
                for trans_type in TRANSITION_CONFIG['types']:
                    type_count = sum(1 for f in transition_files if f.startswith(trans_type))
                    if type_count > 0:
                        dataset_info['generation_stats']['transitions']['by_type'][trans_type] = type_count
        
        pbar.close()
        
        # 데이터 정리 및 셔플
        print("\n🔄 데이터 정리 중...")
        
        # 모든 임베딩과 라벨을 하나의 배열로 합치기
        if all_embeddings:
            # 프레임별 데이터로 변환
            frame_embeddings = []
            frame_labels = []
            
            for embeddings, labels in zip(all_embeddings, all_labels):
                for frame_idx in range(embeddings.shape[0]):
                    frame_embeddings.append(embeddings[frame_idx])
                    frame_labels.append(labels[frame_idx])
            
            X = np.array(frame_embeddings)
            y = np.array(frame_labels)
            
            # 셔플
            X, y = shuffle(X, y, random_state=42)
            
            # 최종 통계
            print(f"\n📊 최종 데이터셋 통계:")
            print(f"  - 총 프레임 수: {len(X):,}개")
            for class_idx, class_name in enumerate(ALL_CLASSES):
                count = np.sum(y == class_idx)
                percentage = count / len(y) * 100
                print(f"  - {class_name}: {count:,}개 ({percentage:.1f}%)")
            
            dataset_info['final_stats'] = {
                'total_frames': len(X),
                'class_distribution': {
                    class_name: int(np.sum(y == class_idx))
                    for class_idx, class_name in enumerate(ALL_CLASSES)
                }
            }
            
            return X, y, dataset_info
        else:
            print("❌ 생성된 데이터가 없습니다.")
            return None, None, None

def main():
    """메인 실행 함수"""
    print("🚀 YAMNet + LSTM 데이터 생성기")
    print("=" * 60)
    
    # 설정 검증
    errors = validate_config()
    if errors:
        print("❌ 설정 오류:")
        for error in errors:
            print(f"  - {error}")
        return
    
    # 출력 디렉토리 생성
    create_output_directories()
    
    # 데이터 생성기 초기화
    generator = DataGenerator()
    
    # 최적 샘플 수 계산
    recommendations = generator.calculate_optimal_samples()
    
    # 사용자 입력 받기
    sample_counts = generator.get_user_input_for_samples(recommendations)
    if sample_counts is None:
        return
    
    # 데이터셋 생성
    X, y, dataset_info = generator.generate_dataset(sample_counts)
    
    if X is not None:
        # 3-way 데이터 분할 (train/validation/test)
        print("\n🔄 데이터를 train/validation/test로 분할 중...")
        X_train, X_val, X_test, y_train, y_val, y_test = split_dataset_3way(X, y)
        
        # 분할된 데이터셋 저장
        split_paths = save_dataset_splits(X_train, X_val, X_test, y_train, y_val, y_test)
        
        # 데이터셋 정보 저장
        dataset_path = get_dataset_save_path()
        dataset_info['split_info'] = {
            'train_samples': len(X_train),
            'validation_samples': len(X_val),
            'test_samples': len(X_test),
            'train_ratio': TRAINING_CONFIG['data_split']['train_ratio'],
            'validation_ratio': TRAINING_CONFIG['data_split']['validation_ratio'],
            'test_ratio': TRAINING_CONFIG['data_split']['test_ratio'],
            'split_files': split_paths
        }
        
        with open(dataset_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 데이터셋 정보 저장: {dataset_path}")
        
        # 전체 데이터도 저장 (호환성 유지)
        data_path = dataset_path.replace('.json', '.npz')
        np.savez_compressed(data_path, X=X, y=y)
        print(f"💾 전체 데이터 저장: {data_path}")
        
        print("\n✅ 데이터 생성 및 분할 완료!")
        return data_path, dataset_path, split_paths
    else:
        print("\n❌ 데이터 생성 실패")
        return None, None, None

if __name__ == "__main__":
    main()
