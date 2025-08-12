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
from datetime import datetime
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
        """클래스별 최적 샘플 수 계산 및 상세 분석"""
        target_frames = DATA_GENERATION_CONFIG['target_frames_per_class']
        frames_per_audio = get_audio_frames_count()
        
        print("\n📊 클래스별 상세 데이터 분석:")
        print("=" * 80)
        print(f"🎯 목표: 클래스당 {target_frames:,}개 프레임")
        print(f"📏 오디오당 프레임 수: {frames_per_audio}개")
        print("=" * 80)
        
        recommendations = {}
        
        for class_name in ALL_CLASSES:
            available_frames = self.total_frames_available.get(class_name, 0)
            base_samples_possible = available_frames // frames_per_audio if frames_per_audio > 0 else 0
            
            # 필요한 샘플 수 계산
            needed_samples = target_frames // frames_per_audio if frames_per_audio > 0 else 0
            needed_frames_from_samples = needed_samples * frames_per_audio
            
            # 전환 데이터 기여분 계산
            transition_contribution = self._calculate_transition_contribution(class_name, frames_per_audio)
            
            # 실제 필요한 기본 샘플 수 (전환 데이터 고려)
            remaining_frames_needed = max(0, target_frames - transition_contribution)
            actual_needed_samples = remaining_frames_needed // frames_per_audio if frames_per_audio > 0 else 0
            
            # 증강 필요 여부 및 필요량 계산
            shortage = max(0, actual_needed_samples - base_samples_possible)
            
            # 무음은 무제한, 다른 클래스는 증강으로 부족분 해결
            if class_name == 'silence':
                augmentation_needed = 0  # 무음은 무제한 생성 가능
                can_achieve_target = True
            else:
                # 증강 활성화 확인
                aug_config = AUGMENTATION_CONFIG.get(class_name, {})
                if aug_config.get('enabled', False) and shortage > 0:
                    # 증강으로 부족분 해결 (제한 없음)
                    augmentation_needed = shortage
                    can_achieve_target = True
                else:
                    # 증강 비활성화 또는 부족분 없음
                    augmentation_needed = 0
                    can_achieve_target = base_samples_possible >= actual_needed_samples
            
            # 총 예상 프레임 수 계산 (기본 + 증강 + 전환)
            base_frames = min(base_samples_possible * frames_per_audio, remaining_frames_needed)
            augmented_frames = augmentation_needed * frames_per_audio
            total_estimated_frames = base_frames + augmented_frames + transition_contribution
            
            recommendations[class_name] = {
                'available_frames': available_frames,
                'target_frames': target_frames,
                'base_samples_possible': base_samples_possible,
                'needed_samples': needed_samples,
                'transition_contribution': transition_contribution,
                'actual_needed_samples': actual_needed_samples,
                'shortage_samples': shortage,
                'augmentation_needed': augmentation_needed,
                'estimated_total_frames': min(total_estimated_frames, target_frames),
                'balance_ratio': min(total_estimated_frames, target_frames) / target_frames if target_frames > 0 else 0,
                'need_augmentation': augmentation_needed > 0,
                'feasible': can_achieve_target  # 증강 가능 여부를 고려한 실현가능성
            }
            
            # 상세 정보 출력
            print(f"\n📁 {class_name} ({CLASS_NAMES.get(class_name, class_name)}):")
            print(f"  📊 가용 프레임: {available_frames:,}개")
            print(f"  🎯 목표 프레임: {target_frames:,}개")
            print(f"  📝 기본 샘플 가능: {base_samples_possible:,}개 → {base_frames:,}개 프레임")
            print(f"  🔄 전환 데이터 기여: {transition_contribution:.0f}개 프레임")
            print(f"  📊 실제 필요 샘플: {actual_needed_samples:,}개")
            
            if augmentation_needed > 0:
                print(f"  🔧 증강 필요: {augmentation_needed:,}개 샘플 → {augmented_frames:,}개 프레임")
                print(f"  ⚡ 증강 배율: {augmentation_needed/base_samples_possible:.1f}배" if base_samples_possible > 0 else "  ⚡ 증강 배율: 무한")
            else:
                print(f"  ✅ 증강 불필요")
            
            print(f"  📈 총 예상 프레임: {min(total_estimated_frames, target_frames):,.0f}개")
            print(f"  ⚖️ 균형도: {min(total_estimated_frames, target_frames)/target_frames*100:.1f}%")
            
            # 상태 표시
            if recommendations[class_name]['feasible']:
                if augmentation_needed > 0:
                    print(f"  🟡 상태: 증강 필요하지만 달성 가능")
                else:
                    print(f"  🟢 상태: 충분한 데이터 보유")
            else:
                aug_config = AUGMENTATION_CONFIG.get(class_name, {})
                if not aug_config.get('enabled', False):
                    print(f"  🔴 상태: 목표 달성 불가 (증강 비활성화)")
                else:
                    print(f"  🔴 상태: 목표 달성 불가 (데이터 부족)")  # 이 경우는 이제 발생하지 않아야 함
            
        print("\n" + "=" * 80)
        
        # 전체 균형 분석
        balance_ratios = [rec['balance_ratio'] for rec in recommendations.values()]
        min_balance = min(balance_ratios)
        max_balance = max(balance_ratios)
        balance_difference = max_balance - min_balance
        
        print(f"📊 전체 균형 분석:")
        print(f"  - 최소 균형도: {min_balance*100:.1f}%")
        print(f"  - 최대 균형도: {max_balance*100:.1f}%")
        print(f"  - 균형 차이: {balance_difference*100:.1f}%p")
        
        if balance_difference < 0.05:  # 5% 이내
            print(f"  ✅ 매우 균등한 분포 (차이 5% 이내)")
        elif balance_difference < 0.1:  # 10% 이내
            print(f"  🟡 양호한 분포 (차이 10% 이내)")
        else:
            print(f"  🔴 불균등한 분포 (차이 10% 초과)")
        
        # 실현 가능성 확인
        all_feasible = all(rec['feasible'] for rec in recommendations.values())
        
        if all_feasible:
            print(f"  ✅ 모든 클래스의 목표 프레임 수 달성 가능")
        else:
            infeasible = [name for name, rec in recommendations.items() if not rec['feasible']]
            print(f"  ❌ 일부 클래스의 목표 프레임 수 달성 불가능")
            print(f"     문제 클래스: {', '.join(infeasible)}")
            
        print("=" * 80)
        
        return recommendations
    
    def _calculate_transition_contribution(self, class_name, frames_per_audio):
        """전환 데이터로부터 해당 클래스가 얻을 수 있는 프레임 수 계산"""
        if not TRANSITION_CONFIG['enabled']:
            return 0
            
        contribution = 0
        transition_ratio = DATA_GENERATION_CONFIG['transition_data_ratio']
        
        # 디버그 출력을 위한 상세 계산
        details = []
        
        # 각 전환 타입에서 해당 클래스가 차지하는 프레임 계산
        for trans_type, config in TRANSITION_CONFIG['types'].items():
            if not config['enabled']:
                continue
                
            base_contribution = frames_per_audio * config['weight'] * transition_ratio
            
            if trans_type == 'silence_to_silence' and class_name == 'silence':
                contrib = base_contribution
                contribution += contrib
                details.append(f"    - {trans_type}: {contrib:.1f}프레임 (전체)")
                
            elif trans_type == 'silence_to_factory' and class_name in ['silence', 'factory']:
                # 전환 데이터에서 각 클래스가 차지하는 비율 추정 (50:50)
                contrib = base_contribution * 0.5
                contribution += contrib
                details.append(f"    - {trans_type}: {contrib:.1f}프레임 (50% 기여)")
                
            elif trans_type == 'silence_to_danger' and (class_name == 'silence' or class_name in ACTIVE_DANGER_CLASSES):
                contrib = base_contribution * 0.5
                contribution += contrib
                details.append(f"    - {trans_type}: {contrib:.1f}프레임 (50% 기여)")
                
            elif trans_type == 'factory_to_factory' and class_name == 'factory':
                contrib = base_contribution
                contribution += contrib
                details.append(f"    - {trans_type}: {contrib:.1f}프레임 (전체)")
                
            elif trans_type == 'factory_to_danger' and (class_name == 'factory' or class_name in ACTIVE_DANGER_CLASSES):
                contrib = base_contribution * 0.5
                contribution += contrib
                details.append(f"    - {trans_type}: {contrib:.1f}프레임 (50% 기여)")
        
        # 상세 계산 출력 (디버그용, 필요시 주석 해제)
        # if details:
        #     print(f"  🔄 {class_name} 전환 데이터 상세:")
        #     for detail in details:
        #         print(detail)
        #     print(f"    💡 총 기여도: {contribution:.1f}프레임")
                
        return int(contribution)
    
    def get_user_input_for_frames(self, recommendations):
        """사용자로부터 실제 생성할 프레임 수 입력받기"""
        if not DATA_GENERATION_CONFIG['allow_user_input']:
            # 자동으로 목표값 사용
            target_frames = DATA_GENERATION_CONFIG['target_frames_per_class']
            return {name: target_frames for name in ALL_CLASSES}
        
        print(f"\n🎯 각 클래스별 생성할 프레임 수를 결정해주세요:")
        print(f"💡 권장: 클래스당 {DATA_GENERATION_CONFIG['target_frames_per_class']}개 프레임 (균등 분배)")
        print(f"📏 참고: 오디오당 평균 {get_audio_frames_count()}개 프레임")
        print("-" * 60)
        
        user_frames = {}
        
        for class_name in ALL_CLASSES:
            rec = recommendations[class_name]
            target_frames = DATA_GENERATION_CONFIG['target_frames_per_class']
            
            # 증강을 고려한 최대 가능 프레임 수 계산
            if class_name == 'silence':
                max_possible_frames = 999999  # 무음은 무제한
            else:
                # 증강이 활성화된 경우 이론적으로 무제한
                aug_config = AUGMENTATION_CONFIG.get(class_name, {})
                if aug_config.get('enabled', False):
                    max_possible_frames = 999999  # 증강으로 무제한 생성 가능
                else:
                    max_possible_frames = rec['available_frames']  # 증강 비활성화시만 원본 제한
            
            print(f"\n📁 {class_name} ({CLASS_NAMES.get(class_name, class_name)}):")
            if max_possible_frames >= 999999:
                print(f"  - 가용 최대 프레임: 무제한 (증강 활성화)")
            else:
                print(f"  - 가용 최대 프레임: {max_possible_frames:,}개")
            print(f"  - 권장 프레임 수: {target_frames:,}개")
            print(f"  - 전환 데이터 기여: {rec['transition_contribution']:.0f}개 프레임")
            
            while True:
                try:
                    user_input = input(f"  👉 생성할 프레임 수 (권장: {target_frames:,}): ").strip()
                    
                    if user_input == "":
                        frames = target_frames
                    else:
                        frames = int(user_input.replace(',', ''))
                    
                    if frames < 0:
                        print("     ❌ 음수는 입력할 수 없습니다.")
                        continue
                    
                    # 무제한 클래스가 아닌 경우에만 제한 검사
                    if max_possible_frames < 999999 and frames > max_possible_frames:
                        print(f"     ❌ 최대 {max_possible_frames:,}개까지 가능합니다.")
                        continue
                    
                    user_frames[class_name] = frames
                    print(f"     ✅ {frames:,}개 프레임 설정됨")
                    break
                    
                except ValueError:
                    print("     ❌ 숫자를 입력해주세요.")
                except KeyboardInterrupt:
                    print("\n\n👋 사용자 중단")
                    return None
        
        # 최종 확인
        print(f"\n📋 최종 데이터 생성 계획:")
        total_frames = sum(user_frames.values())
        for class_name, frames in user_frames.items():
            percentage = (frames / total_frames) * 100 if total_frames > 0 else 0
            print(f"  - {class_name}: {frames:,}개 프레임 ({percentage:.1f}%)")
        
        print(f"\n총 {total_frames:,}개 프레임 생성 예정")
        
        # 균등성 확인
        frame_values = list(user_frames.values())
        if len(set(frame_values)) == 1:
            print("✅ 모든 클래스가 동일한 프레임 수로 설정됨 (완벽한 균형)")
        else:
            max_diff = max(frame_values) - min(frame_values)
            print(f"⚠️  클래스간 프레임 수 차이: {max_diff:,}개")
        
        confirm = input("\n계속 진행하시겠습니까? (y/N): ").strip().lower()
        if confirm not in ['y', 'yes']:
            print("👋 중단됨")
            return None
            
        return user_frames
    
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
    
    def load_audio_segment(self, file_path, duration=None):
        """오디오 파일에서 지정된 길이의 세그먼트 로드"""
        audio = self.load_audio_file(file_path)
        if audio is not None:
            return self.extract_audio_segment(audio, duration)
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
    
    def generate_transition_sequences(self, total_base_sequences):
        """전환 데이터 시퀀스 생성 - 설정 비율에 맞게"""
        transition_sequences = []
        transition_labels = []
        transition_frame_counts = []
        
        # 전환 데이터 총 개수 계산 (설정 비율 적용)
        transition_ratio = DATA_GENERATION_CONFIG['transition_data_ratio']
        total_transition_sequences = int(total_base_sequences * transition_ratio / (1 - transition_ratio))
        
        print(f"🔄 전환 데이터 생성:")
        print(f"  - 기본 시퀀스: {total_base_sequences}개")
        print(f"  - 전환 비율: {transition_ratio:.1%}")
        print(f"  - 전환 시퀀스 목표: {total_transition_sequences}개")
        
        # 활성화된 전환 타입들의 총 가중치 계산
        enabled_types = {k: v for k, v in TRANSITION_CONFIG['types'].items() if v.get('enabled', True)}
        total_weight = sum(config['weight'] for config in enabled_types.values())
        
        # 각 전환 타입별 생성 개수 계산
        for trans_type, config in enabled_types.items():
            type_weight = config['weight']
            samples_for_type = int(total_transition_sequences * type_weight / total_weight)
            
            print(f"  🔄 {trans_type}: 가중치 {type_weight:.1f} → {samples_for_type}개 생성")
            
            for i in range(samples_for_type):
                try:
                    # 전환 오디오 생성
                    if 'silence_to_' in trans_type:
                        # 무음에서 다른 소리로 전환
                        target_class = trans_type.replace('silence_to_', '')
                        if target_class == 'danger':
                            target_class = np.random.choice(ACTIVE_DANGER_CLASSES)
                        
                        transition_audio, labels = self.generate_transition_audio(trans_type, 'silence', target_class)
                        
                    elif '_to_silence' in trans_type:
                        # 다른 소리에서 무음으로 전환
                        source_class = trans_type.replace('_to_silence', '')
                        if source_class == 'danger':
                            source_class = np.random.choice(ACTIVE_DANGER_CLASSES)
                            
                        transition_audio, labels = self.generate_transition_audio(trans_type, source_class, 'silence')
                        
                    elif 'danger_to_danger' in trans_type:
                        # 위험 소리 간 전환
                        source_class = np.random.choice(ACTIVE_DANGER_CLASSES)
                        target_class = np.random.choice(ACTIVE_DANGER_CLASSES)
                        while target_class == source_class:
                            target_class = np.random.choice(ACTIVE_DANGER_CLASSES)
                            
                        transition_audio, labels = self.generate_transition_audio(trans_type, source_class, target_class)
                    
                    elif 'factory_to_factory' in trans_type:
                        # 공장 소리 내 전환
                        transition_audio, labels = self.generate_transition_audio(trans_type, 'factory', 'factory')
                        
                    elif 'factory_to_danger' in trans_type:
                        # 공장에서 위험 소리로 전환
                        target_class = np.random.choice(ACTIVE_DANGER_CLASSES)
                        transition_audio, labels = self.generate_transition_audio(trans_type, 'factory', target_class)
                    
                    else:
                        continue
                    
                    if transition_audio is not None:
                        embeddings = self.extract_yamnet_embeddings(transition_audio)
                        if embeddings is not None:
                            transition_sequences.append(embeddings)
                            
                            # 전환 지점 기준으로 라벨 결정
                            transition_point = len(labels) // 2
                            if len(labels) > transition_point:
                                # 전환 후 클래스를 주 라벨로 사용
                                main_label = labels[transition_point:]
                                if len(main_label) > 0:
                                    label_counts = np.bincount(main_label)
                                    majority_label = np.argmax(label_counts)
                                else:
                                    majority_label = labels[-1] if len(labels) > 0 else 0
                            else:
                                majority_label = labels[-1] if len(labels) > 0 else 0
                            
                            transition_labels.append(majority_label)
                            transition_frame_counts.append(embeddings.shape[0])
                            
                except Exception as e:
                    print(f"    ⚠️ 전환 데이터 생성 실패: {e}")
                    continue
            
            print(f"    ✅ {trans_type}: 생성 완료")
        
        actual_generated = len(transition_sequences)
        print(f"  📊 실제 생성된 전환 시퀀스: {actual_generated}개")
        print(f"  📊 목표 대비 달성률: {actual_generated/total_transition_sequences*100:.1f}%")
        
        return transition_sequences, transition_labels, transition_frame_counts
    
    def generate_sequence_dataset(self, samples_per_class):
        """시퀀스 기반 데이터셋 생성 (오디오별 완전한 시퀀스)"""
        print("\n🏭 시퀀스 기반 데이터셋 생성 시작...")
        
        all_sequences = []
        all_labels = []
        all_frame_counts = []  # 각 시퀀스의 유효 프레임 수 저장
        sequence_lengths = []  # 패딩 전 실제 길이 저장
        
        dataset_info = {
            'config': {
                'model_version': MODEL_CONFIG['version'],
                'audio_duration': MODEL_CONFIG['audio_duration'],
                'sample_rate': MODEL_CONFIG['sample_rate'],
                'num_classes': NUM_CLASSES,
                'class_names': ALL_CLASSES,
                'generation_mode': 'full_sequence'
            },
            'generation_stats': {},
            'files_used': {},
            'frame_statistics': {}
        }
        
        # 각 클래스별 시퀀스 생성
        for class_name, target_samples in samples_per_class.items():
            if target_samples == 0:
                continue
                
            print(f"\n📁 {class_name} 클래스 생성 중... (목표: {target_samples:,}개 시퀀스)")
            
            class_idx = ALL_CLASSES.index(class_name)
            collected_samples = 0
            files_used = []
            class_frame_counts = []
            
            while collected_samples < target_samples:
                if class_name == 'silence':
                    # 무음 데이터 생성
                    audio = self.generate_silence_audio(MODEL_CONFIG['audio_duration'])
                    
                    # 데이터 증강 적용
                    if AUGMENTATION_CONFIG.get(class_name, {}).get('enabled', False):
                        if np.random.random() < 0.3:
                            methods = AUGMENTATION_CONFIG[class_name]['methods']
                            method = np.random.choice(methods)
                            audio = self.apply_augmentation(audio, class_name, method)
                    
                    embeddings = self.extract_yamnet_embeddings(audio)
                    if embeddings is not None:
                        # 전체 시퀀스를 하나의 샘플로 저장
                        all_sequences.append(embeddings)
                        all_labels.append(class_idx)
                        class_frame_counts.append(embeddings.shape[0])
                        sequence_lengths.append(embeddings.shape[0])  # 실제 길이 저장
                        collected_samples += 1
                        files_used.append(f"silence_{collected_samples}")
                
                elif class_name == 'factory':
                    # 공장 소리 처리
                    if not self.audio_files.get(class_name):
                        print(f"⚠️ {class_name} 오디오 파일이 없습니다.")
                        break
                    
                    audio_file = np.random.choice(self.audio_files[class_name])
                    audio = self.load_audio_segment(audio_file, duration=MODEL_CONFIG['audio_duration'])
                    
                    if audio is not None:
                        if AUGMENTATION_CONFIG.get(class_name, {}).get('enabled', False):
                            if np.random.random() < 0.4:
                                methods = AUGMENTATION_CONFIG[class_name]['methods']
                                method = np.random.choice(methods)
                                audio = self.apply_augmentation(audio, class_name, method)
                        
                        embeddings = self.extract_yamnet_embeddings(audio)
                        if embeddings is not None:
                            all_sequences.append(embeddings)
                            all_labels.append(class_idx)
                            class_frame_counts.append(embeddings.shape[0])
                            sequence_lengths.append(embeddings.shape[0])  # 실제 길이 저장
                            collected_samples += 1
                            files_used.append(os.path.basename(audio_file))
                
                else:
                    # 위험 소리 클래스들
                    if not self.audio_files.get(class_name):
                        print(f"⚠️ {class_name} 오디오 파일이 없습니다.")
                        break
                    
                    audio_file = np.random.choice(self.audio_files[class_name])
                    audio = self.load_audio_segment(audio_file, duration=MODEL_CONFIG['audio_duration'])
                    
                    if audio is not None:
                        if AUGMENTATION_CONFIG.get(class_name, {}).get('enabled', False):
                            if np.random.random() < 0.6:
                                methods = AUGMENTATION_CONFIG[class_name]['methods']
                                method = np.random.choice(methods)
                                audio = self.apply_augmentation(audio, class_name, method)
                        
                        embeddings = self.extract_yamnet_embeddings(audio)
                        if embeddings is not None:
                            all_sequences.append(embeddings)
                            all_labels.append(class_idx)
                            class_frame_counts.append(embeddings.shape[0])
                            sequence_lengths.append(embeddings.shape[0])  # 실제 길이 저장
                            collected_samples += 1
                            files_used.append(os.path.basename(audio_file))
                
                # 무한 루프 방지
                if len(files_used) > target_samples * 3:
                    print(f"⚠️ {class_name}: 너무 많은 시도 후 중단")
                    break
            
            print(f"  ✅ {class_name}: {collected_samples:,}개 시퀀스 생성")
            
            # 클래스별 프레임 통계
            if class_frame_counts:
                total_frames = sum(class_frame_counts)
                avg_frames = np.mean(class_frame_counts)
                dataset_info['frame_statistics'][class_name] = {
                    'total_frames': total_frames,
                    'avg_frames_per_sequence': avg_frames,
                    'sequences_count': len(class_frame_counts),
                    'min_frames': min(class_frame_counts),
                    'max_frames': max(class_frame_counts)
                }
                all_frame_counts.extend(class_frame_counts)
            
            dataset_info['generation_stats'][class_name] = {
                'target_samples': target_samples,
                'actual_samples': collected_samples,
                'files_used': len(set(files_used))
            }
            dataset_info['files_used'][class_name] = list(set(files_used))
        
        # 전환 데이터 생성
        if TRANSITION_CONFIG['enabled']:
            print(f"\n🔄 전환 데이터 생성 중...")
            
            # 기본 시퀀스 총 개수 계산
            total_base_sequences = sum(samples_per_class.values())
            
            # 설정 비율에 맞게 전환 데이터 생성
            transition_sequences, transition_labels, transition_frame_counts = self.generate_transition_sequences(total_base_sequences)
            
            if transition_sequences:
                all_sequences.extend(transition_sequences)
                all_labels.extend(transition_labels)
                all_frame_counts.extend(transition_frame_counts)
                sequence_lengths.extend(transition_frame_counts)  # 전환 데이터의 실제 길이도 저장
                
                # 전환 데이터 프레임 통계
                dataset_info['frame_statistics']['transitions'] = {
                    'total_frames': sum(transition_frame_counts),
                    'avg_frames_per_sequence': np.mean(transition_frame_counts),
                    'sequences_count': len(transition_frame_counts),
                    'min_frames': min(transition_frame_counts),
                    'max_frames': max(transition_frame_counts)
                }
                
                print(f"  ✅ 전환 데이터: {len(transition_sequences):,}개 시퀀스 생성")
        
        # 시퀀스 길이 통일 (패딩/자르기)
        if all_sequences:
            print("\n🔄 시퀀스 길이 통일 중...")
            
            # 시퀀스 길이 분석
            seq_lengths = [seq.shape[0] for seq in all_sequences]
            max_length = max(seq_lengths)
            target_length = max_length  # 최대 길이로 통일
            
            print(f"  📏 시퀀스 길이 통계:")
            print(f"    - 최소: {min(seq_lengths)}, 최대: {max_length}, 평균: {np.mean(seq_lengths):.1f}")
            print(f"    - 목표 길이: {target_length} 프레임")
            
            # 시퀀스 길이 통일 (제로 패딩 또는 자르기)
            unified_sequences = []
            for seq in all_sequences:
                if seq.shape[0] < target_length:
                    # 제로 패딩
                    pad_length = target_length - seq.shape[0]
                    padded_seq = np.pad(seq, ((0, pad_length), (0, 0)), mode='constant', constant_values=0)
                    unified_sequences.append(padded_seq)
                elif seq.shape[0] > target_length:
                    # 자르기 (앞부분 사용)
                    unified_sequences.append(seq[:target_length])
                else:
                    unified_sequences.append(seq)
            
            X = np.array(unified_sequences)  # (samples, time_steps, features)
            y = np.array(all_labels)         # (samples,)
            original_lengths = np.array(sequence_lengths)  # (samples,) - 패딩 전 실제 길이
            
            # 셔플 (모든 배열을 동일하게)
            shuffle_indices = np.random.permutation(len(X))
            X = X[shuffle_indices]
            y = y[shuffle_indices]
            original_lengths = original_lengths[shuffle_indices]
            
            print(f"\n📊 최종 시퀀스 데이터셋 통계:")
            print(f"  - 총 시퀀스 수: {len(X):,}개")
            print(f"  - 시퀀스 형태: {X.shape}")
            print(f"  - 시간 스텝: {X.shape[1]}")
            print(f"  - 특성 수: {X.shape[2]}")
            
            # 클래스별 분포 확인
            actual_class_samples = {}
            class_frame_totals = {}
            
            for class_idx, class_name in enumerate(ALL_CLASSES):
                count = np.sum(y == class_idx)
                actual_class_samples[class_name] = count
                percentage = count / len(y) * 100
                
                # 해당 클래스의 총 프레임 수 계산
                if class_name in dataset_info['frame_statistics']:
                    total_frames = dataset_info['frame_statistics'][class_name]['total_frames']
                    class_frame_totals[class_name] = total_frames
                else:
                    class_frame_totals[class_name] = 0
                
                print(f"  - {class_name}: {count:,}개 시퀀스 ({percentage:.1f}%), {class_frame_totals[class_name]:,} 프레임")
            
            # 전환 데이터 비율 통계 추가
            total_sequences = len(X)
            base_sequences = sum(samples_per_class.values())
            actual_transition_sequences = total_sequences - base_sequences
            actual_transition_ratio = actual_transition_sequences / total_sequences if total_sequences > 0 else 0
            target_transition_ratio = DATA_GENERATION_CONFIG['transition_data_ratio']
            
            print(f"\n📊 최종 데이터 구성 비율:")
            print(f"  - 기본 시퀀스: {base_sequences:,}개 ({(1-actual_transition_ratio):.1%})")
            print(f"  - 전환 시퀀스: {actual_transition_sequences:,}개 ({actual_transition_ratio:.1%})")
            print(f"  - 목표 전환 비율: {target_transition_ratio:.1%}")
            print(f"  - 실제 전환 비율: {actual_transition_ratio:.1%}")
            
            dataset_info['final_stats'] = {
                'total_sequences': len(X),
                'sequence_shape': list(X.shape),
                'target_length': target_length,
                'class_distribution': actual_class_samples,
                'class_frame_totals': class_frame_totals,
                'sequence_length_stats': {
                    'min_length': int(np.min(original_lengths)),
                    'max_length': int(np.max(original_lengths)),
                    'mean_length': float(np.mean(original_lengths)),
                    'padded_length': target_length
                },
                'transition_data_stats': {
                    'target_ratio': target_transition_ratio,
                    'actual_ratio': actual_transition_ratio,
                    'base_sequences': base_sequences,
                    'transition_sequences': actual_transition_sequences,
                    'total_sequences': total_sequences
                }
            }
            
            return X, y, dataset_info, original_lengths
        else:
            print("❌ 생성된 데이터가 없습니다.")
            return None, None, None
        
        all_sequences = []
        all_labels = []
        dataset_info = {
            'config': {
                'model_version': MODEL_CONFIG['version'],
                'audio_duration': MODEL_CONFIG['audio_duration'],
                'sample_rate': MODEL_CONFIG['sample_rate'],
                'num_classes': NUM_CLASSES,
                'class_names': ALL_CLASSES,
                'generation_mode': 'sequence_based'
            },
            'generation_stats': {},
            'files_used': {}
        }
        
        # 각 클래스별 시퀀스 생성
        for class_name, target_samples in target_samples_per_class.items():
            if target_samples == 0:
                continue
                
            print(f"\n📁 {class_name} 클래스 생성 중... (목표: {target_samples:,}개 시퀀스)")
            
            class_idx = ALL_CLASSES.index(class_name)
            collected_samples = 0
            files_used = []
            
            while collected_samples < target_samples:
                if class_name == 'silence':
                    # 무음 데이터 생성
                    audio = self.generate_silence_audio(MODEL_CONFIG['audio_duration'])
                    
                    # 데이터 증강 적용
                    if AUGMENTATION_CONFIG.get(class_name, {}).get('enabled', False):
                        if np.random.random() < 0.3:
                            methods = AUGMENTATION_CONFIG[class_name]['methods']
                            method = np.random.choice(methods)
                            audio = self.apply_augmentation(audio, class_name, method)
                    
                    embeddings = self.extract_yamnet_embeddings(audio)
                    if embeddings is not None:
                        # 전체 시퀀스를 하나의 샘플로 저장
                        all_sequences.append(embeddings)
                        all_labels.append(class_idx)
                        collected_samples += 1
                        files_used.append(f"silence_{collected_samples}")
                
                elif class_name == 'factory':
                    # 공장 소리 처리
                    if not self.audio_files.get(class_name):
                        print(f"⚠️ {class_name} 오디오 파일이 없습니다.")
                        break
                    
                    audio_file = np.random.choice(self.audio_files[class_name])
                    audio = self.load_audio_segment(audio_file, duration=MODEL_CONFIG['audio_duration'])
                    
                    if audio is not None:
                        if AUGMENTATION_CONFIG.get(class_name, {}).get('enabled', False):
                            if np.random.random() < 0.4:
                                methods = AUGMENTATION_CONFIG[class_name]['methods']
                                method = np.random.choice(methods)
                                audio = self.apply_augmentation(audio, class_name, method)
                        
                        embeddings = self.extract_yamnet_embeddings(audio)
                        if embeddings is not None:
                            all_sequences.append(embeddings)
                            all_labels.append(class_idx)
                            collected_samples += 1
                            files_used.append(os.path.basename(audio_file))
                
                else:
                    # 위험 소리 클래스들
                    if not self.audio_files.get(class_name):
                        print(f"⚠️ {class_name} 오디오 파일이 없습니다.")
                        break
                    
                    audio_file = np.random.choice(self.audio_files[class_name])
                    audio = self.load_audio_segment(audio_file, duration=MODEL_CONFIG['audio_duration'])
                    
                    if audio is not None:
                        if AUGMENTATION_CONFIG.get(class_name, {}).get('enabled', False):
                            if np.random.random() < 0.6:
                                methods = AUGMENTATION_CONFIG[class_name]['methods']
                                method = np.random.choice(methods)
                                audio = self.apply_augmentation(audio, class_name, method)
                        
                        embeddings = self.extract_yamnet_embeddings(audio)
                        if embeddings is not None:
                            all_sequences.append(embeddings)
                            all_labels.append(class_idx)
                            collected_samples += 1
                            files_used.append(os.path.basename(audio_file))
                
                # 무한 루프 방지
                if len(files_used) > target_samples * 3:
                    print(f"⚠️ {class_name}: 너무 많은 시도 후 중단")
                    break
            
            print(f"  ✅ {class_name}: {collected_samples:,}개 시퀀스 생성")
            dataset_info['generation_stats'][class_name] = {
                'target_samples': target_samples,
                'actual_samples': collected_samples,
                'files_used': len(set(files_used))
            }
            dataset_info['files_used'][class_name] = list(set(files_used))
        
        # 시퀀스 길이 통일 (패딩/자르기)
        if all_sequences:
            print("\n🔄 시퀀스 길이 통일 중...")
            
            # 시퀀스 길이 분석
            seq_lengths = [seq.shape[0] for seq in all_sequences]
            max_length = max(seq_lengths)
            min_length = min(seq_lengths)
            avg_length = np.mean(seq_lengths)
            
            print(f"  📏 시퀀스 길이 - 최소: {min_length}, 최대: {max_length}, 평균: {avg_length:.1f}")
            
            # 목표 길이 설정 (평균 또는 가장 일반적인 길이)
            target_length = int(np.percentile(seq_lengths, 75))  # 75% 지점 사용
            print(f"  🎯 목표 길이: {target_length} 프레임")
            
            # 시퀀스 길이 통일
            unified_sequences = []
            for seq in all_sequences:
                if seq.shape[0] < target_length:
                    # 패딩 (제로 패딩)
                    pad_length = target_length - seq.shape[0]
                    padded_seq = np.pad(seq, ((0, pad_length), (0, 0)), mode='constant')
                    unified_sequences.append(padded_seq)
                elif seq.shape[0] > target_length:
                    # 자르기 (앞부분 사용)
                    unified_sequences.append(seq[:target_length])
                else:
                    unified_sequences.append(seq)
            
            X = np.array(unified_sequences)  # (samples, time_steps, features)
            y = np.array(all_labels)         # (samples,)
            
            # 셔플
            X, y = shuffle(X, y, random_state=42)
            
            print(f"\n📊 최종 시퀀스 데이터셋 통계:")
            print(f"  - 총 시퀀스 수: {len(X):,}개")
            print(f"  - 시퀀스 형태: {X.shape}")
            print(f"  - 시간 스텝: {X.shape[1]}")
            print(f"  - 특성 수: {X.shape[2]}")
            
            # 클래스별 분포 확인
            actual_class_samples = {}
            for class_idx, class_name in enumerate(ALL_CLASSES):
                count = np.sum(y == class_idx)
                actual_class_samples[class_name] = count
                percentage = count / len(y) * 100
                print(f"  - {class_name}: {count:,}개 ({percentage:.1f}%)")
            
            dataset_info['final_stats'] = {
                'total_sequences': len(X),
                'sequence_shape': list(X.shape),
                'target_length': target_length,
                'class_distribution': actual_class_samples
            }
            
            return X, y, dataset_info
        else:
            print("❌ 생성된 데이터가 없습니다.")
            return None, None, None
        """목표 프레임 수에 맞춰 정확한 데이터셋 생성"""
        print("\n🏭 프레임 기반 데이터셋 생성 시작...")
        
        all_embeddings = []
        all_labels = []
        dataset_info = {
            'config': {
                'model_version': MODEL_CONFIG['version'],
                'audio_duration': MODEL_CONFIG['audio_duration'],
                'sample_rate': MODEL_CONFIG['sample_rate'],
                'num_classes': NUM_CLASSES,
                'class_names': ALL_CLASSES,
                'generation_mode': 'frame_based'
            },
            'generation_stats': {},
            'files_used': {}
        }
        
        total_target_frames = sum(target_frames_per_class.values())
        
        # 각 클래스별 정확한 프레임 수 생성
        for class_name, target_frames in target_frames_per_class.items():
            if target_frames == 0:
                continue
                
            print(f"\n📁 {class_name} 클래스 생성 중... (목표: {target_frames:,}개 프레임)")
            
            class_idx = ALL_CLASSES.index(class_name)
            collected_frames = 0
            generated_samples = 0
            files_used = []
            
            # 프레임 수집 루프
            while collected_frames < target_frames:
                frames_needed = target_frames - collected_frames
                
                if class_name == 'silence':
                    # 무음 데이터 생성
                    audio = self.generate_silence_audio(MODEL_CONFIG['audio_duration'])
                    
                    # 데이터 증강 적용 (확률적)
                    if AUGMENTATION_CONFIG.get(class_name, {}).get('enabled', False):
                        if np.random.random() < 0.5:  # 50% 확률로 증강
                            methods = AUGMENTATION_CONFIG[class_name]['methods']
                            method = np.random.choice(methods)
                            audio = self.apply_augmentation(audio, class_name, method)
                    
                    embeddings = self.extract_yamnet_embeddings(audio)
                    if embeddings is not None:
                        available_frames = embeddings.shape[0]
                        frames_to_use = min(available_frames, frames_needed)
                        
                        # 필요한 프레임만 선택
                        selected_embeddings = embeddings[:frames_to_use]
                        selected_labels = np.full(frames_to_use, class_idx, dtype=int)
                        
                        all_embeddings.extend(selected_embeddings)
                        all_labels.extend(selected_labels)
                        
                        collected_frames += frames_to_use
                        generated_samples += 1
                        files_used.append(f"generated_silence_{generated_samples}")
                        
                        print(f"\r  진행률: {collected_frames:,}/{target_frames:,} "
                              f"({collected_frames/target_frames*100:.1f}%)", end='')
                        
                else:
                    # 실제 오디오 파일 사용
                    if class_name not in self.audio_files or not self.audio_files[class_name]:
                        print(f"⚠️ {class_name} 클래스의 오디오 파일이 없습니다.")
                        break
                    
                    # 랜덤하게 파일 선택
                    file_path = np.random.choice(self.audio_files[class_name])
                    audio = self.load_audio_file(file_path)
                    
                    if audio is None:
                        continue
                    
                    # 세그먼트 추출
                    segment = self.extract_audio_segment(audio, MODEL_CONFIG['audio_duration'])
                    
                    # 데이터 증강 적용 (필요시)
                    if AUGMENTATION_CONFIG.get(class_name, {}).get('enabled', False):
                        # 수집된 프레임이 목표의 50%를 넘으면 증강 적용
                        if collected_frames > target_frames * 0.5:
                            methods = AUGMENTATION_CONFIG[class_name]['methods']
                            method = np.random.choice(methods)
                            segment = self.apply_augmentation(segment, class_name, method)
                    
                    embeddings = self.extract_yamnet_embeddings(segment)
                    if embeddings is not None:
                        available_frames = embeddings.shape[0]
                        frames_to_use = min(available_frames, frames_needed)
                        
                        # 필요한 프레임만 선택
                        selected_embeddings = embeddings[:frames_to_use]
                        selected_labels = np.full(frames_to_use, class_idx, dtype=int)
                        
                        all_embeddings.extend(selected_embeddings)
                        all_labels.extend(selected_labels)
                        
                        collected_frames += frames_to_use
                        generated_samples += 1
                        files_used.append(f"{os.path.basename(file_path)}_seg{generated_samples}")
                        
                        print(f"\r  진행률: {collected_frames:,}/{target_frames:,} "
                              f"({collected_frames/target_frames*100:.1f}%)", end='')
            
            print()  # 줄바꿈
            
            # 클래스별 통계 저장
            dataset_info['generation_stats'][class_name] = {
                'target_frames': target_frames,
                'actual_frames': collected_frames,
                'samples_generated': generated_samples,
                'accuracy': collected_frames / target_frames if target_frames > 0 else 0
            }
            dataset_info['files_used'][class_name] = files_used
            
            print(f"  ✅ {class_name}: {collected_frames:,}/{target_frames:,} 프레임 "
                  f"({collected_frames/target_frames*100:.1f}%)")
        
        # 전환 데이터 생성 (선택적)
        if TRANSITION_CONFIG['enabled']:
            print(f"\n🔄 전환 데이터 생성 중...")
            
            transition_ratio = DATA_GENERATION_CONFIG['transition_data_ratio']
            base_frames = len(all_embeddings)
            transition_frames_needed = int(base_frames * transition_ratio / (1 - transition_ratio))
            
            transition_collected = 0
            transition_files = []
            
            for trans_type, config in TRANSITION_CONFIG['types'].items():
                if not config['enabled']:
                    continue
                    
                type_frames_needed = int(transition_frames_needed * config['weight'] / 
                                       sum(c['weight'] for c in TRANSITION_CONFIG['types'].values() if c['enabled']))
                
                type_collected = 0
                sample_count = 0
                
                while type_collected < type_frames_needed:
                    frames_needed = type_frames_needed - type_collected
                    
                    audio, labels = self.generate_transition_audio(trans_type, None, None)
                    embeddings = self.extract_yamnet_embeddings(audio)
                    
                    if embeddings is not None:
                        available_frames = embeddings.shape[0]
                        frames_to_use = min(available_frames, frames_needed)
                        
                        # 라벨 길이 조정
                        if len(labels) != embeddings.shape[0]:
                            if len(labels) < embeddings.shape[0]:
                                labels = np.pad(labels, (0, embeddings.shape[0] - len(labels)), 
                                              mode='edge')
                            else:
                                labels = labels[:embeddings.shape[0]]
                        
                        # 필요한 프레임만 선택
                        selected_embeddings = embeddings[:frames_to_use]
                        selected_labels = labels[:frames_to_use]
                        
                        all_embeddings.extend(selected_embeddings)
                        all_labels.extend(selected_labels)
                        
                        type_collected += frames_to_use
                        transition_collected += frames_to_use
                        sample_count += 1
                        transition_files.append(f"{trans_type}_{sample_count}")
                
                print(f"  - {trans_type}: {type_collected:,}개 프레임")
            
            dataset_info['generation_stats']['transitions'] = {
                'target_frames': transition_frames_needed,
                'actual_frames': transition_collected,
                'samples_generated': len(transition_files),
                'by_type': {}
            }
            
            # 전환 타입별 통계
            for trans_type in TRANSITION_CONFIG['types']:
                type_count = sum(1 for f in transition_files if f.startswith(trans_type))
                if type_count > 0:
                    dataset_info['generation_stats']['transitions']['by_type'][trans_type] = type_count
        
        # 데이터 정리
        print("\n🔄 데이터 정리 중...")
        
        if all_embeddings:
            X = np.array(all_embeddings)
            y = np.array(all_labels)
            
            # 셔플
            X, y = shuffle(X, y, random_state=42)
            
            # 최종 통계
            print(f"\n📊 최종 데이터셋 통계:")
            print(f"  - 총 프레임 수: {len(X):,}개")
            
            # 클래스별 실제 프레임 수 확인
            actual_class_frames = {}
            for class_idx, class_name in enumerate(ALL_CLASSES):
                count = np.sum(y == class_idx)
                percentage = count / len(y) * 100
                actual_class_frames[class_name] = count
                target = target_frames_per_class.get(class_name, 0)
                accuracy = (count / target * 100) if target > 0 else 0
                print(f"  - {class_name}: {count:,}개 ({percentage:.1f}%) "
                      f"[목표: {target:,}, 달성률: {accuracy:.1f}%]")
            
            # 균등성 검사
            class_frame_counts = [actual_class_frames[name] for name in ALL_CLASSES 
                                if target_frames_per_class.get(name, 0) > 0]
            if class_frame_counts:
                max_diff = max(class_frame_counts) - min(class_frame_counts)
                if max_diff == 0:
                    print("✅ 완벽한 클래스 균형 달성!")
                else:
                    print(f"📊 클래스간 최대 프레임 차이: {max_diff:,}개")
            
            dataset_info['final_stats'] = {
                'total_frames': len(X),
                'class_distribution': actual_class_frames,
                'target_vs_actual': {
                    name: {
                        'target': target_frames_per_class.get(name, 0),
                        'actual': actual_class_frames.get(name, 0),
                        'accuracy': (actual_class_frames.get(name, 0) / target_frames_per_class.get(name, 1) * 100) 
                                  if target_frames_per_class.get(name, 0) > 0 else 0
                    }
                    for name in ALL_CLASSES
                }
            }
            
            return X, y, dataset_info
        else:
            print("❌ 생성된 데이터가 없습니다.")
            return None, None, None

def main():
    """메인 실행 함수"""
    print("🚀 YAMNet + LSTM 시퀀스 기반 데이터 생성기")
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
    
    # 시퀀스 기반 데이터 생성을 위한 샘플 수 입력
    print("\n📊 시퀀스 기반 데이터 생성 설정")
    print("각 클래스별로 생성할 시퀀스 수를 입력하세요.")
    print("(한 시퀀스는 하나의 완전한 오디오 파일입니다)")
    
    samples_per_class = {}
    
    # 기본 시퀀스 수 설정
    default_samples = {
        'silence': 500,
        'factory': 300,
        'fire': 200,
        'gas': 200,
        'scream': 200
    }
    
    print(f"\n🎯 권장 시퀀스 수:")
    for class_name in ALL_CLASSES:
        recommended = default_samples.get(class_name, 200)
        print(f"  - {class_name}: {recommended}개")
    
    print(f"\n⚙️ 시퀀스 수 설정:")
    use_defaults = input("권장 설정을 사용하시겠습니까? (Y/n): ").strip().lower()
    
    if use_defaults in ['', 'y', 'yes']:
        samples_per_class = default_samples.copy()
        print("✅ 권장 설정 적용됨")
    else:
        for class_name in ALL_CLASSES:
            while True:
                try:
                    default_val = default_samples.get(class_name, 200)
                    user_input = input(f"{class_name} 시퀀스 수 (기본값: {default_val}): ").strip()
                    if not user_input:
                        samples_per_class[class_name] = default_val
                    else:
                        samples_per_class[class_name] = max(0, int(user_input))
                    break
                except ValueError:
                    print("⚠️ 숫자를 입력해주세요.")
    
    # 총 시퀀스 수 확인
    total_sequences = sum(samples_per_class.values())
    print(f"\n📊 설정된 시퀀스 수:")
    for class_name, count in samples_per_class.items():
        percentage = (count / total_sequences) * 100 if total_sequences > 0 else 0
        print(f"  - {class_name}: {count:,}개 ({percentage:.1f}%)")
    print(f"  총 {total_sequences:,}개 시퀀스 생성 예정")
    
    confirm = input("\n계속 진행하시겠습니까? (Y/n): ").strip().lower()
    if confirm not in ['', 'y', 'yes']:
        print("👋 중단됨")
        return
    
    # 시퀀스 기반 데이터셋 생성
    result = generator.generate_sequence_dataset(samples_per_class)
    
    if len(result) == 4:
        X, y, dataset_info, original_lengths = result
    else:
        # 이전 버전 호환성
        X, y, dataset_info = result
        original_lengths = None
    
    if X is not None:
        # 클래스 가중치 계산
        print("\n⚖️ 클래스 가중치 계산 중...")
        class_weights = {}
        
        # 각 클래스별 실제 프레임 수 계산 (패딩 제외)
        class_frame_counts = {class_name: 0 for class_name in ALL_CLASSES}
        
        for i, (label, length) in enumerate(zip(y, original_lengths)):
            class_name = ALL_CLASSES[label]
            class_frame_counts[class_name] += length  # 실제 길이만 더함
        
        # 전체 실제 프레임 수
        total_frames = sum(class_frame_counts.values())
        
        for class_idx, class_name in enumerate(ALL_CLASSES):
            frames = class_frame_counts[class_name]
            if frames > 0:
                # 실제 프레임 수에 반비례하는 가중치
                weight = total_frames / (len(ALL_CLASSES) * frames)
            else:
                weight = 1.0
            
            class_weights[class_idx] = weight
            print(f"  📊 {class_name}: {frames:,} 프레임 (실제), 가중치: {weight:.3f}")
        
        # 클래스 가중치 정보를 데이터셋 정보에 추가
        dataset_info['class_weights'] = {str(k): float(v) for k, v in class_weights.items()}
        dataset_info['actual_frame_counts'] = {class_name: int(count) for class_name, count in class_frame_counts.items()}
        
        # 3-way 데이터 분할 (train/validation/test)
        print("\n🔄 데이터를 train/validation/test로 분할 중...")
        if original_lengths is not None:
            X_train, X_val, X_test, y_train, y_val, y_test, lengths_train, lengths_val, lengths_test = split_dataset_3way_with_lengths(X, y, original_lengths)
        else:
            X_train, X_val, X_test, y_train, y_val, y_test = split_dataset_3way(X, y)
            lengths_train = lengths_val = lengths_test = None
        
        # 분할된 데이터셋 저장
        if lengths_train is not None:
            split_paths = save_dataset_splits_with_lengths(X_train, X_val, X_test, y_train, y_val, y_test, 
                                                          lengths_train, lengths_val, lengths_test)
        else:
            split_paths = save_dataset_splits(X_train, X_val, X_test, y_train, y_val, y_test)
        
        # 분할 후 클래스별 통계
        print(f"\n📊 분할된 데이터셋 클래스별 통계:")
        for split_name, split_data in [('Train', y_train), ('Validation', y_val), ('Test', y_test)]:
            print(f"\n{split_name} 셋:")
            for class_idx, class_name in enumerate(ALL_CLASSES):
                count = np.sum(split_data == class_idx)
                percentage = count / len(split_data) * 100 if len(split_data) > 0 else 0
                print(f"  - {class_name}: {count:,}개 ({percentage:.1f}%)")
        
        # 데이터셋 정보 저장 (JSON 직렬화 가능하도록 변환)
        dataset_path = get_dataset_save_path()
        
        # NumPy 타입들을 Python 기본 타입으로 변환
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        dataset_info = convert_numpy_types(dataset_info)
        
        # 클래스 가중치 정보 추가
        dataset_info['class_weights'] = {str(k): float(v) for k, v in class_weights.items()}
        
        dataset_info['split_info'] = {
            'train_samples': len(X_train),
            'validation_samples': len(X_val),
            'test_samples': len(X_test),
            'train_ratio': TRAINING_CONFIG['data_split']['train_ratio'],
            'validation_ratio': TRAINING_CONFIG['data_split']['validation_ratio'],
            'test_ratio': TRAINING_CONFIG['data_split']['test_ratio'],
            'split_files': split_paths,
            'split_class_distribution': {
                'train': {class_name: int(np.sum(y_train == class_idx)) 
                         for class_idx, class_name in enumerate(ALL_CLASSES)},
                'validation': {class_name: int(np.sum(y_val == class_idx)) 
                              for class_idx, class_name in enumerate(ALL_CLASSES)},
                'test': {class_name: int(np.sum(y_test == class_idx)) 
                        for class_idx, class_name in enumerate(ALL_CLASSES)}
            }
        }
        
        with open(dataset_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 데이터셋 정보 저장: {dataset_path}")
        
        # 분할된 데이터셋 통계를 별도 파일로 저장
        split_stats_path = dataset_path.replace('dataset_info_', 'split_statistics_')
        
        # 분할 통계 상세 정보 생성
        split_statistics = {
            'metadata': {
                'creation_time': datetime.now().isoformat(),
                'model_version': MODEL_CONFIG['version'],
                'total_samples': len(X),
                'split_ratios': {
                    'train': TRAINING_CONFIG['data_split']['train_ratio'],
                    'validation': TRAINING_CONFIG['data_split']['validation_ratio'],
                    'test': TRAINING_CONFIG['data_split']['test_ratio']
                }
            },
            'split_summary': {
                'train': {
                    'total_samples': len(X_train),
                    'percentage': len(X_train) / len(X) * 100
                },
                'validation': {
                    'total_samples': len(X_val),
                    'percentage': len(X_val) / len(X) * 100
                },
                'test': {
                    'total_samples': len(X_test),
                    'percentage': len(X_test) / len(X) * 100
                }
            },
            'class_distribution': {
                'train': {},
                'validation': {},
                'test': {}
            },
            'class_statistics': {}
        }
        
        # 각 분할별 클래스 분포 계산
        for split_name, y_split in [('train', y_train), ('validation', y_val), ('test', y_test)]:
            total_split_samples = len(y_split)
            
            for class_idx, class_name in enumerate(ALL_CLASSES):
                class_count = int(np.sum(y_split == class_idx))
                class_percentage = (class_count / total_split_samples) * 100 if total_split_samples > 0 else 0
                
                split_statistics['class_distribution'][split_name][class_name] = {
                    'count': class_count,
                    'percentage': round(class_percentage, 1)
                }
        
        # 클래스별 전체 통계
        for class_idx, class_name in enumerate(ALL_CLASSES):
            train_count = int(np.sum(y_train == class_idx))
            val_count = int(np.sum(y_val == class_idx))
            test_count = int(np.sum(y_test == class_idx))
            total_class_count = train_count + val_count + test_count
            
            split_statistics['class_statistics'][class_name] = {
                'total_samples': total_class_count,
                'train': {
                    'count': train_count,
                    'percentage_of_class': round((train_count / total_class_count) * 100, 1) if total_class_count > 0 else 0,
                    'percentage_of_split': round((train_count / len(X_train)) * 100, 1) if len(X_train) > 0 else 0
                },
                'validation': {
                    'count': val_count,
                    'percentage_of_class': round((val_count / total_class_count) * 100, 1) if total_class_count > 0 else 0,
                    'percentage_of_split': round((val_count / len(X_val)) * 100, 1) if len(X_val) > 0 else 0
                },
                'test': {
                    'count': test_count,
                    'percentage_of_class': round((test_count / total_class_count) * 100, 1) if total_class_count > 0 else 0,
                    'percentage_of_split': round((test_count / len(X_test)) * 100, 1) if len(X_test) > 0 else 0
                }
            }
        
        # 분할 통계 파일 저장
        with open(split_stats_path, 'w', encoding='utf-8') as f:
            json.dump(split_statistics, f, indent=2, ensure_ascii=False)
        
        print(f"📊 분할 통계 저장: {split_stats_path}")
        
        # 전체 데이터도 저장 (호환성 유지)
        data_path = dataset_path.replace('.json', '.npz')
        np.savez_compressed(data_path, X=X, y=y)
        print(f"💾 전체 데이터 저장: {data_path}")
        
        # 최종 요약
        print(f"\n✅ 프레임 기반 데이터 생성 및 분할 완료!")
        print(f"📊 총 프레임: {len(X):,}개")
        print(f"📂 Train: {len(X_train):,}개 ({len(X_train)/len(X)*100:.1f}%)")
        print(f"📂 Validation: {len(X_val):,}개 ({len(X_val)/len(X)*100:.1f}%)")
        print(f"📂 Test: {len(X_test):,}개 ({len(X_test)/len(X)*100:.1f}%)")
        
        return data_path, dataset_path, split_paths
    else:
        print("\n❌ 데이터 생성 실패")
        return None, None, None

if __name__ == "__main__":
    main()
