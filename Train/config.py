"""
YAMNet + LSTM 훈련 설정 파일
============================

모든 훈련 관련 설정을 이 파일에서 관리합니다.
각 단계별로 필요한 설정들을 정의하고, 데이터 생성부터 모델 훈련, 평가까지의 
모든 파라미터를 중앙에서 제어할 수 있습니다.
"""

import os
from datetime import datetime
import json
import numpy as np

# ================================
# 기본 경로 설정
# ================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENVSOUND_DIR = os.path.join(BASE_DIR, 'envsound')
MIXTURE_DIR = os.path.join(BASE_DIR, 'mixture')

# 호환성을 위한 AUDIO_DATA_DIR (evaluation.py에서 사용)
AUDIO_DATA_DIR = ENVSOUND_DIR

# ================================
# 평가용 오디오 파일 경로 설정
# ================================
EVALUATION_AUDIO_PATHS = {
    'silence': None,  # 무음은 스킵
    'factory': MIXTURE_DIR,  # 공장소리는 mixture 폴더에서
    'fire': os.path.join(ENVSOUND_DIR, 'fire'),
    'gas': os.path.join(ENVSOUND_DIR, 'gas'), 
    'scream': os.path.join(ENVSOUND_DIR, 'scream'),
    'spark': os.path.join(ENVSOUND_DIR, 'spark'),
}

# 결과 저장 경로 - 버전별 체계적 관리
RESULTS_BASE_DIR = os.path.join(BASE_DIR, 'results')

# 버전별 폴더 구조
def get_version_dir():
    """현재 모델 버전에 따른 디렉토리 경로"""
    try:
        version = MODEL_CONFIG['version']
        return os.path.join(RESULTS_BASE_DIR, f"version_{version}")
    except NameError:
        # MODEL_CONFIG가 아직 정의되지 않은 경우 기본값 사용
        return os.path.join(RESULTS_BASE_DIR, 'version_default')

def get_experiment_dir():
    """실험별 디렉토리 경로 (버전 + 타임스탬프)"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"exp_{timestamp}"
    return os.path.join(get_version_dir(), experiment_name)

# 모든 설정이 정의된 후 동적 경로 초기화
def initialize_paths():
    """모든 설정이 정의된 후 경로들을 초기화"""
    global VERSION_DIR, TRAINING_RESULTS_DIR, EVALUATION_RESULTS_DIR
    global MODEL_SAVE_DIR, REPORT_SAVE_DIR, DATASET_SAVE_DIR
    
    VERSION_DIR = get_version_dir()
    TRAINING_RESULTS_DIR = os.path.join(VERSION_DIR, 'training')
    EVALUATION_RESULTS_DIR = os.path.join(VERSION_DIR, 'evaluation')
    MODEL_SAVE_DIR = os.path.join(VERSION_DIR, 'models')
    REPORT_SAVE_DIR = os.path.join(VERSION_DIR, 'reports')
    DATASET_SAVE_DIR = os.path.join(VERSION_DIR, 'datasets')

# ================================
# 모델 및 훈련 설정
# ================================
MODEL_CONFIG = {
    'version': 'v2.26',                    # 모델 버전
    'audio_duration': 5.0,                # 오디오 입력 길이 (초) [5.0 ~ 10.0]
    'sample_rate': 16000,                 # 샘플링 주파수
}

TRAINING_CONFIG = {
    'epochs': 50,                         # 훈련 에포크 수
    'batch_size': 8,                     # 배치 크기
    'learning_rate': 0.001,               # 학습률
    'validation_split': 0.2,              # 검증 데이터 비율 (레거시 - train/val만 사용시)
    'random_seed': 42,                    # 랜덤 시드
    'normalize_input': True,              # 입력 정규화
    'use_class_weights': True,            # 클래스 가중치 사용
    
    # 3-way 데이터 분할 설정 (train/validation/test)
    'data_split': {
        'train_ratio': 0.7,               # 훈련 데이터 비율 (70%)
        'validation_ratio': 0.15,         # 검증 데이터 비율 (15%)  
        'test_ratio': 0.15,               # 테스트 데이터 비율 (15%)
        'stratify': True,                 # 클래스별 균등 분할
        'shuffle': True,                  # 데이터 셔플
    },
    'dense_units': 256,                   # Dense 레이어 유닛 수
    'lstm_units': 128,                    # LSTM 레이어 유닛 수
    'dropout_rate': 0.6,                  # 드롭아웃 비율
    'save_checkpoints': True,             # 체크포인트 저장
    'early_stopping': {
        'enabled': True,
        'patience': 5
    },
    'learning_rate_schedule': {
        'enabled': True,
        'factor': 0.5,
        'patience': 5,
        'min_lr': 1e-7
    }
}

# ================================
# 사용할 위험 소리 클래스 설정
# ================================
DANGER_CLASSES = {
    'fire': True,        # 화재 소리 사용 여부
    'gas': True,         # 가스누출 소리 사용 여부
    'scream': True,      # 비명 소리 사용 여부
}

# 클래스 정보 자동 생성
BASE_CLASSES = ['silence', 'factory']
ACTIVE_DANGER_CLASSES = [name for name, enabled in DANGER_CLASSES.items() if enabled]
ALL_CLASSES = BASE_CLASSES + ACTIVE_DANGER_CLASSES
NUM_CLASSES = len(ALL_CLASSES)

CLASS_NAMES = {
    'silence': '무음',
    'factory': '정상(공장)',
    'fire': '화재',
    'gas': '가스누출',
    'scream': '비명'
}

# 영어 클래스 이름 (차트 및 그래프용)
CLASS_NAMES_EN = {
    'silence': 'Silence',
    'factory': 'Factory',
    'fire': 'Fire',
    'gas': 'Gas Leak',
    'scream': 'Scream'
}

# ================================
# 데이터 증강 설정
# ================================
AUGMENTATION_CONFIG = {
    'silence': {
        'enabled': True,
        'methods': ['noise_variation', 'volume_change'],
        'noise_types': ['white', 'pink', 'brown'],        # 노이즈 종류
        'volume_range': (0.1, 0.8),                       # 볼륨 범위
    },
    'factory': {
        'enabled': True,
        'methods': ['volume_change','noise_add'],
        'volume_range': (0.7, 1.3),                       # 볼륨 범위
        'reverb_decay': (0.1, 0.5),                       # 리버브 감쇠 시간
        'room_size': (0.1, 0.9),                           # 룸 크기
        'speed_range': (0.9, 1.1),                        # 속도 변화 범위
        'noise_level': (0.01, 0.05),  
    },
    'fire': {
        'enabled': True,
        'methods': ['factory_mix', 'volume_change','noise_add'],
        'snr_range': (10, 30),                             # SNR 범위 (dB)
        'volume_range': (0.8, 1.2),                       # 볼륨 범위
        'noise_level': (0.01, 0.05),                      # 노이즈 레벨
    },
    'gas': {
        'enabled': True,
        'methods': ['factory_mix', 'volume_change','noise_add'],
        'snr_range': (10, 25),                             # SNR 범위 (dB)
        'volume_range': (0.8, 1.2),                       # 볼륨 범위
        'noise_level': (0.01, 0.05),                      # 노이즈 레벨
    },
    'scream': {
        'enabled': True,
        'methods': ['factory_mix', 'volume_change', 'reverb','noise_add'],
        'snr_range': (10, 30),                            # SNR 범위 (dB)
        'volume_range': (0.7, 1.3),                       # 볼륨 범위
        'room_size': (0.1, 0.9),                          # 룸 크기
        'reverb_decay': (0.1, 0.3),                       # 리버브 감쇠 시간
        'noise_level': (0.01, 0.05), 
    }
}

# ================================
# 전환 데이터 설정
# ================================
TRANSITION_CONFIG = {
    'enabled': True,
    'fade_duration': 0.2,                                 # 페이드 전환 시간 (초)
    
    # 전환 타입별 설정
    'types': {
        'silence_to_silence': {
            'enabled': True,
            'description': '무음 → 다른 무음 (배경노이즈 변화)',
            'transition_point_range': (0.2, 0.8),         # 전환 시점 범위 (비율)
            'weight': 1.0,                                 # 생성 가중치
        },
        'silence_to_factory': {
            'enabled': True,
            'description': '무음 → 공장소리 (기계 시작)',
            'transition_point_range': (0.2, 0.7),         # 전환 시점 범위
            'weight': 1.5,                                 # 생성 가중치
        },
        'silence_to_danger': {
            'enabled': True,
            'description': '무음 → 위험소리 (긴급상황)',
            'transition_point_range': (0.1, 0.6),         # 전환 시점 범위
            'weight': 2.0,                                 # 생성 가중치
        },
        'factory_to_factory': {
            'enabled': True,
            'description': '공장소리 → 다른 공장소리 (기계 변경)',
            'transition_point_range': (0.3, 0.7),         # 전환 시점 범위
            'weight': 1.0,                                 # 생성 가중치
        },
        'factory_to_danger': {
            'enabled': True,
            'description': '공장소리 → 위험소리 (사고 발생)',
            'transition_point_range': (0.2, 0.8),         # 전환 시점 범위
            'danger_volume_ratio': (0.8, 1.5),            # 위험소리 볼륨 비율
            'weight': 1.0,                                 # 생성 가중치 (가장 중요)
        }
    }
}

# ================================
# 데이터 생성 목표 설정
# ================================
DATA_GENERATION_CONFIG = {
    'target_frames_per_class': 1000,                      # 클래스당 목표 프레임 수 (수정: 2000 → 1000)
    'min_frames_for_augmentation': 500,                   # 데이터 증강 시작 임계값
    'transition_data_ratio': 0.2,                         # 전환 데이터 비율 (전체의 20%)
    'auto_balance': True,                                  # 자동 클래스 균형 조정
    'allow_user_input': True,                             # 사용자 입력 허용 (수정: True → False)
}

# ================================
# 출력 및 로깅 설정
# ================================
OUTPUT_CONFIG = {
    'verbose': True,                                       # 상세 출력 여부
    'save_dataset_info': True,                            # 데이터셋 정보 저장
    'save_model_summary': True,                           # 모델 요약 저장
    'save_training_history': True,                        # 훈련 히스토리 저장
    'save_confusion_matrix': True,                        # 혼동행렬 저장
    'save_classification_report': True,                   # 분류 보고서 저장
    'plot_training_curves': True,                         # 훈련 곡선 그래프 저장
    'plot_confusion_matrix': True,                        # 혼동행렬 시각화 저장
}

# ================================
# 계산된 설정값들 (자동 생성)
# ================================
def get_audio_frames_count(duration=None):
    """오디오 길이에 따른 YAMNet 프레임 수 계산"""
    if duration is None:
        duration = MODEL_CONFIG['audio_duration']
    # YAMNet은 0.48초마다 하나의 프레임 생성
    return int(duration / 0.48)

def get_transition_frame_counts():
    """전환 데이터로 생성될 프레임 수 계산"""
    if not TRANSITION_CONFIG['enabled']:
        return {}
    
    total_frames = get_audio_frames_count()
    transition_counts = {}
    
    for trans_type, config in TRANSITION_CONFIG['types'].items():
        if config['enabled']:
            # 전환 데이터는 전체 길이를 사용하므로 full frame count
            transition_counts[trans_type] = total_frames * config['weight']
    
    return transition_counts

def get_recommended_samples():
    """권장 샘플 수 계산"""
    target_frames = DATA_GENERATION_CONFIG['target_frames_per_class']
    frames_per_audio = get_audio_frames_count()
    
    # 기본 데이터 필요 샘플 수
    base_samples_needed = target_frames // frames_per_audio
    
    # 전환 데이터로 얻을 수 있는 프레임 수 고려
    transition_frames = get_transition_frame_counts()
    total_transition_frames = sum(transition_frames.values()) if transition_frames else 0
    
    # 전환 데이터 비율 고려
    transition_ratio = DATA_GENERATION_CONFIG['transition_data_ratio']
    transition_contribution = total_transition_frames * transition_ratio
    
    # 실제 필요한 기본 샘플 수
    actual_samples_needed = max(1, int((target_frames - transition_contribution) // frames_per_audio))
    
    return {
        'frames_per_audio': frames_per_audio,
        'target_frames_per_class': target_frames,
        'base_samples_needed': base_samples_needed,
        'transition_contribution': transition_contribution,
        'recommended_samples': actual_samples_needed,
        'total_classes': NUM_CLASSES,
        'active_classes': ALL_CLASSES
    }

def create_output_directories():
    """출력 디렉토리 생성 - 버전별 체계적 구조"""
    directories = [
        RESULTS_BASE_DIR,
        VERSION_DIR,
        TRAINING_RESULTS_DIR,
        EVALUATION_RESULTS_DIR,
        MODEL_SAVE_DIR,
        REPORT_SAVE_DIR,
        DATASET_SAVE_DIR
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    # 실험 기록 파일 생성
    create_experiment_log()

def create_experiment_log():
    """실험 로그 파일 생성 및 업데이트"""
    log_file = os.path.join(VERSION_DIR, 'experiment_log.txt')
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 실험 정보 수집
    experiment_info = f"""
========================================
실험 시작: {timestamp}
모델 버전: {MODEL_CONFIG['version']}
오디오 길이: {MODEL_CONFIG['audio_duration']}초
훈련 에포크: {TRAINING_CONFIG['epochs']}
배치 크기: {TRAINING_CONFIG['batch_size']}
학습률: {TRAINING_CONFIG['learning_rate']}
목표 프레임/클래스: {DATA_GENERATION_CONFIG['target_frames_per_class']}
활성 클래스: {', '.join(ALL_CLASSES)}
========================================
"""
    
    # 로그 파일에 추가
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(experiment_info)

def get_experiment_save_path(file_type='model', custom_name=None):
    """실험별 저장 경로 생성"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version = MODEL_CONFIG['version']
    
    if custom_name:
        base_name = custom_name
    else:
        base_name = f"yamnet_lstm_{version}_{timestamp}"
    
    if file_type == 'model':
        return os.path.join(MODEL_SAVE_DIR, f"{base_name}.h5")
    elif file_type == 'dataset':
        return os.path.join(DATASET_SAVE_DIR, f"dataset_{base_name}.npz")
    elif file_type == 'dataset_info':
        return os.path.join(DATASET_SAVE_DIR, f"dataset_info_{base_name}.json")
    elif file_type == 'training_report':
        return os.path.join(REPORT_SAVE_DIR, f"training_report_{base_name}.json")
    elif file_type == 'evaluation_report':
        return os.path.join(REPORT_SAVE_DIR, f"evaluation_report_{base_name}.md")
    elif file_type == 'experiment_summary':
        return os.path.join(VERSION_DIR, f"experiment_summary_{timestamp}.json")
    else:
        return os.path.join(REPORT_SAVE_DIR, f"{file_type}_{base_name}")

def save_experiment_summary(results_dict):
    """실험 결과 요약 저장"""
    summary_path = get_experiment_save_path('experiment_summary')
    
    summary = {
        'experiment_info': {
            'timestamp': datetime.now().isoformat(),
            'version': MODEL_CONFIG['version'],
            'config': {
                'model_config': MODEL_CONFIG,
                'training_config': TRAINING_CONFIG,
                'data_generation_config': DATA_GENERATION_CONFIG
            }
        },
        'results': results_dict
    }
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"📋 실험 요약 저장: {summary_path}")
    return summary_path

def create_training_report(results, model_config=None, best_epoch=1, dataset_stats=None, class_weights=None, user_samples=None):
    """상세 훈련 보고서 생성 (텍스트 + 시각화)"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version = MODEL_CONFIG['version']
    
    # 텍스트 보고서 경로
    report_path = os.path.join(TRAINING_RESULTS_DIR, f"training_report_{version}_{timestamp}.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("🎵 YAMNet + LSTM 모델 훈련 보고서\n")
        f.write("=" * 80 + "\n\n")
        
        # 1. 기본 정보
        f.write("📋 실험 정보\n")
        f.write("-" * 40 + "\n")
        f.write(f"실험 일시: {datetime.now().strftime('%Y년 %m월 %d일 %H시 %M분')}\n")
        f.write(f"모델 버전: {MODEL_CONFIG['version']}\n")
        f.write(f"오디오 길이: {MODEL_CONFIG['audio_duration']}초\n")
        f.write(f"샘플링 레이트: {MODEL_CONFIG['sample_rate']:,} Hz\n\n")
        
        # 2. 데이터셋 구성 정보
        f.write("📊 데이터셋 구성\n")
        f.write("-" * 40 + "\n")
        
        if dataset_stats and 'final_stats' in dataset_stats:
            stats = dataset_stats['final_stats']
            f.write(f"총 시퀀스 수: {stats['total_sequences']:,}개\n")
            f.write(f"시퀀스 형태: {stats['sequence_shape']}\n")
            f.write(f"패딩 길이: {stats['target_length']}개 프레임\n\n")
            
            # 시퀀스 길이 통계
            if 'sequence_length_stats' in stats:
                length_stats = stats['sequence_length_stats']
                f.write("시퀀스 길이 통계 (패딩 전):\n")
                f.write(f"  최소: {length_stats['min_length']}프레임\n")
                f.write(f"  최대: {length_stats['max_length']}프레임\n")
                f.write(f"  평균: {length_stats['mean_length']:.1f}프레임\n")
                f.write(f"  패딩: {length_stats['padded_length']}프레임\n\n")
            
            # 클래스별 분포
            f.write("클래스별 시퀀스 분포:\n")
            total_sequences = stats['total_sequences']
            for class_name, count in stats['class_distribution'].items():
                percentage = (count / total_sequences) * 100
                frame_total = stats.get('class_frame_totals', {}).get(class_name, 0)
                f.write(f"  {class_name:8}: {count:4,}개 ({percentage:5.1f}%) - {frame_total:,} 실제프레임\n")
            
            f.write("\n")
        
        # 3. 사용자 설정 샘플 수 (있는 경우)
        if user_samples:
            f.write("🎯 사용자 설정 샘플 수\n")
            f.write("-" * 40 + "\n")
            total_user_samples = sum(user_samples.values())
            for class_name, sample_count in user_samples.items():
                percentage = (sample_count / total_user_samples) * 100
                f.write(f"  {class_name:8}: {sample_count:4,}개 ({percentage:5.1f}%)\n")
            f.write(f"  총합:      {total_user_samples:4,}개\n\n")
        
        # 4. 클래스 가중치 정보
        if class_weights:
            f.write("⚖️ 클래스 가중치\n")
            f.write("-" * 40 + "\n")
            for class_idx, weight in class_weights.items():
                class_name = ALL_CLASSES[int(class_idx)]
                f.write(f"  {class_name:8}: {weight:.3f}\n")
            f.write("\n")
        
        # 5. 훈련 설정
        f.write("🧠 훈련 설정\n")
        f.write("-" * 40 + "\n")
        f.write(f"에포크 수: {TRAINING_CONFIG['epochs']}\n")
        f.write(f"배치 크기: {TRAINING_CONFIG['batch_size']}\n")
        f.write(f"학습률: {TRAINING_CONFIG['learning_rate']}\n")
        f.write(f"LSTM 유닛: {TRAINING_CONFIG['lstm_units']}\n")
        f.write(f"드롭아웃: {TRAINING_CONFIG['dropout_rate']}\n")
        f.write(f"클래스 가중치: {'사용' if TRAINING_CONFIG['use_class_weights'] else '미사용'}\n")
        f.write(f"조기 종료: {'사용' if TRAINING_CONFIG['early_stopping']['enabled'] else '미사용'}\n")
        f.write(f"학습률 스케줄링: {'사용' if TRAINING_CONFIG['learning_rate_schedule']['enabled'] else '미사용'}\n\n")
        
        # 6. 데이터 분할 정보
        split_config = TRAINING_CONFIG['data_split']
        f.write("📊 데이터 분할 설정\n")
        f.write("-" * 40 + "\n")
        f.write(f"훈련: {split_config['train_ratio']:.1%}\n")
        f.write(f"검증: {split_config['validation_ratio']:.1%}\n")
        f.write(f"테스트: {split_config['test_ratio']:.1%}\n")
        f.write(f"계층 분할: {'사용' if split_config['stratify'] else '미사용'}\n")
        f.write(f"데이터 셔플: {'사용' if split_config['shuffle'] else '미사용'}\n\n")
        
        # 7. 데이터 증강 설정
        f.write("🎨 데이터 증강 설정\n")
        f.write("-" * 40 + "\n")
        for class_name in ALL_CLASSES:
            if class_name in AUGMENTATION_CONFIG:
                config = AUGMENTATION_CONFIG[class_name]
                enabled = config.get('enabled', False)
                f.write(f"  {class_name:8}: {'사용' if enabled else '미사용'}")
                if enabled and 'methods' in config:
                    f.write(f" - {', '.join(config['methods'])}")
                f.write("\n")
        f.write("\n")
        
        # 8. 전환 데이터 설정
        f.write("🔄 전환 데이터 설정\n")
        f.write("-" * 40 + "\n")
        f.write(f"전환 데이터: {'사용' if TRANSITION_CONFIG['enabled'] else '미사용'}\n")
        if TRANSITION_CONFIG['enabled']:
            f.write(f"페이드 시간: {TRANSITION_CONFIG['fade_duration']}초\n")
            f.write("전환 타입별 설정:\n")
            for trans_type, config in TRANSITION_CONFIG['types'].items():
                enabled = config.get('enabled', False)
                weight = config.get('weight', 1.0)
                f.write(f"  {trans_type:20}: {'사용' if enabled else '미사용'} (가중치: {weight})\n")
        f.write("\n")
        
        # 9. 훈련 결과
        if 'evaluation' in results:
            eval_results = results['evaluation']
            f.write("🎯 훈련 결과\n")
            f.write("-" * 40 + "\n")
            f.write(f"검증 정확도: {eval_results.get('accuracy', 0):.4f} ({eval_results.get('accuracy', 0)*100:.2f}%)\n")
            f.write(f"검증 손실: {eval_results.get('loss', 0):.4f}\n")
            
            if 'precision' in eval_results:
                f.write(f"정밀도: {eval_results['precision']:.4f}\n")
            if 'recall' in eval_results:
                f.write(f"재현율: {eval_results['recall']:.4f}\n")
            
            # 클래스별 성능
            if 'class_report' in results:
                f.write("\n클래스별 성능:\n")
                class_report = results['class_report']
                for class_name in ALL_CLASSES:
                    if class_name in class_report:
                        report = class_report[class_name]
                        f.write(f"  {class_name:8}: P={report['precision']:.3f}, R={report['recall']:.3f}, F1={report['f1-score']:.3f}\n")
            f.write("\n")
        
        # 10. 모델 파일 정보
        if 'model_paths' in results:
            f.write("💾 생성된 파일\n")
            f.write("-" * 40 + "\n")
            for path in results['model_paths']:
                f.write(f"모델: {os.path.basename(path)}\n")
        
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("보고서 생성 완료\n")
        f.write("=" * 80 + "\n")
    
    print(f"📋 훈련 보고서 저장: {report_path}")
    return {'training_report': report_path}

def create_dataset_visualization(dataset_info, save_dir=None):
    """데이터셋 구성 시각화"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    if save_dir is None:
        save_dir = TRAINING_RESULTS_DIR
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 한글 폰트 설정
    plt.rcParams['font.family'] = ['Malgun Gothic', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    if 'final_stats' not in dataset_info:
        return None
    
    stats = dataset_info['final_stats']
    
    # 1. 클래스별 시퀀스 분포 파이 차트
    if 'class_distribution' in stats:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 파이 차트
        class_names = list(stats['class_distribution'].keys())
        class_counts = list(stats['class_distribution'].values())
        
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']
        
        wedges, texts, autotexts = ax1.pie(class_counts, labels=class_names, autopct='%1.1f%%',
                                          colors=colors[:len(class_names)], startangle=90)
        ax1.set_title('클래스별 시퀀스 분포', fontsize=14, fontweight='bold')
        
        # 막대 그래프
        bars = ax2.bar(class_names, class_counts, color=colors[:len(class_names)])
        ax2.set_title('클래스별 시퀀스 수', fontsize=14, fontweight='bold')
        ax2.set_ylabel('시퀀스 수')
        ax2.tick_params(axis='x', rotation=45)
        
        # 막대 위에 값 표시
        for bar, count in zip(bars, class_counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(class_counts)*0.01,
                    f'{count:,}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # 저장
        pie_chart_path = os.path.join(save_dir, f'dataset_distribution_{timestamp}.png')
        plt.savefig(pie_chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 데이터셋 분포 차트 저장: {pie_chart_path}")
    
    # 2. 클래스별 실제 프레임 수 비교
    if 'class_frame_totals' in stats:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        frame_totals = stats['class_frame_totals']
        class_names = list(frame_totals.keys())
        frame_counts = list(frame_totals.values())
        
        bars = ax.bar(class_names, frame_counts, color=colors[:len(class_names)])
        ax.set_title('클래스별 실제 프레임 수 (패딩 제외)', fontsize=14, fontweight='bold')
        ax.set_ylabel('프레임 수')
        ax.tick_params(axis='x', rotation=45)
        
        # 막대 위에 값 표시
        for bar, count in zip(bars, frame_counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(frame_counts)*0.01,
                   f'{count:,}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # 저장
        frame_chart_path = os.path.join(save_dir, f'frame_distribution_{timestamp}.png')
        plt.savefig(frame_chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 프레임 분포 차트 저장: {frame_chart_path}")
    
    # 3. 클래스 가중치 시각화
    if 'class_weights' in dataset_info:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        weights = dataset_info['class_weights']
        class_indices = [int(k) for k in weights.keys()]
        class_names = [ALL_CLASSES[i] for i in class_indices]
        weight_values = [float(weights[str(i)]) for i in class_indices]
        
        bars = ax.bar(class_names, weight_values, color=colors[:len(class_names)])
        ax.set_title('클래스별 가중치', fontsize=14, fontweight='bold')
        ax.set_ylabel('가중치')
        ax.tick_params(axis='x', rotation=45)
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='기준선 (1.0)')
        
        # 막대 위에 값 표시
        for bar, weight in zip(bars, weight_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(weight_values)*0.01,
                   f'{weight:.3f}', ha='center', va='bottom')
        
        ax.legend()
        plt.tight_layout()
        
        # 저장
        weight_chart_path = os.path.join(save_dir, f'class_weights_{timestamp}.png')
        plt.savefig(weight_chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 클래스 가중치 차트 저장: {weight_chart_path}")
    
    return True

def get_model_save_path():
    """모델 저장 경로 생성"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"yamnet_lstm_{MODEL_CONFIG['version']}_{timestamp}"
    return os.path.join(MODEL_SAVE_DIR, f"{model_name}.h5")

def get_report_save_path():
    """보고서 저장 경로 생성"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_name = f"training_report_{MODEL_CONFIG['version']}_{timestamp}"
    return os.path.join(REPORT_SAVE_DIR, f"{report_name}.txt")

def get_dataset_save_path():
    """데이터셋 저장 경로 생성"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = f"dataset_info_{MODEL_CONFIG['version']}_{timestamp}"
    return os.path.join(DATASET_SAVE_DIR, f"{dataset_name}.json")

# ================================
# 설정 검증 함수
# ================================
def validate_config():
    """설정값 검증"""
    errors = []
    
    # 오디오 길이 검증
    if not (5.0 <= MODEL_CONFIG['audio_duration'] <= 10.0):
        errors.append("audio_duration은 5.0~10.0 사이여야 합니다.")
    
    # 위험 클래스 최소 하나 활성화 검증
    if not any(DANGER_CLASSES.values()):
        errors.append("최소 하나의 위험 클래스는 활성화되어야 합니다.")
    
    # 경로 존재 여부 검증
    if not os.path.exists(ENVSOUND_DIR):
        errors.append(f"envsound 폴더가 존재하지 않습니다: {ENVSOUND_DIR}")
    
    if not os.path.exists(MIXTURE_DIR):
        errors.append(f"mixture 폴더가 존재하지 않습니다: {MIXTURE_DIR}")
    
    # 전환 데이터 가중치 검증
    if TRANSITION_CONFIG['enabled']:
        total_weight = sum(config['weight'] for config in TRANSITION_CONFIG['types'].values() if config['enabled'])
        if total_weight == 0:
            errors.append("활성화된 전환 타입이 없습니다.")
    
    # 데이터 분할 비율 검증
    split_config = TRAINING_CONFIG['data_split']
    total_ratio = split_config['train_ratio'] + split_config['validation_ratio'] + split_config['test_ratio']
    if abs(total_ratio - 1.0) > 0.001:
        errors.append(f"데이터 분할 비율의 합이 1.0이 아닙니다: {total_ratio:.3f}")
    
    if split_config['train_ratio'] <= 0 or split_config['validation_ratio'] <= 0 or split_config['test_ratio'] <= 0:
        errors.append("모든 데이터 분할 비율은 0보다 커야 합니다.")
    
    return errors

def split_dataset_3way(X, y, random_seed=None):
    """데이터를 train/validation/test로 3-way 분할 (중복 없음)"""
    from sklearn.model_selection import train_test_split
    
    if random_seed is None:
        random_seed = TRAINING_CONFIG['random_seed']
    
    split_config = TRAINING_CONFIG['data_split']
    train_ratio = split_config['train_ratio']
    val_ratio = split_config['validation_ratio'] 
    test_ratio = split_config['test_ratio']
    stratify = split_config['stratify']
    shuffle = split_config['shuffle']
    
    print(f"📊 데이터 3-way 분할 (Train:{train_ratio:.1%} / Val:{val_ratio:.1%} / Test:{test_ratio:.1%})")
    print(f"  - 총 데이터: {len(X):,}개")
    
    # 1단계: Train + Validation vs Test 분할
    train_val_ratio = train_ratio + val_ratio  # 남은 비율
    test_size_step1 = test_ratio
    
    stratify_step1 = y if stratify else None
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=test_size_step1,
        random_state=random_seed,
        stratify=stratify_step1,
        shuffle=shuffle
    )
    
    # 2단계: Train vs Validation 분할
    val_size_step2 = val_ratio / train_val_ratio  # train_val 내에서의 validation 비율
    
    stratify_step2 = y_train_val if stratify else None
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_size_step2,
        random_state=random_seed + 1,  # 다른 시드 사용
        stratify=stratify_step2,
        shuffle=shuffle
    )
    
    # 결과 출력
    print(f"  - Train: {len(X_train):,}개 ({len(X_train)/len(X):.1%})")
    print(f"  - Validation: {len(X_val):,}개 ({len(X_val)/len(X):.1%})")
    print(f"  - Test: {len(X_test):,}개 ({len(X_test)/len(X):.1%})")
    
    # 클래스별 분포 확인
    if hasattr(y, '__iter__'):
        print(f"\n📈 클래스별 분포 확인:")
        unique_classes = np.unique(y)
        for class_idx in unique_classes:
            train_count = np.sum(y_train == class_idx)
            val_count = np.sum(y_val == class_idx)
            test_count = np.sum(y_test == class_idx)
            total_count = train_count + val_count + test_count
            
            class_name = ALL_CLASSES[class_idx] if class_idx < len(ALL_CLASSES) else f"Class_{class_idx}"
            print(f"  - {class_name}: Train {train_count}, Val {val_count}, Test {test_count} (총 {total_count})")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def split_dataset_3way_with_lengths(X, y, lengths, random_seed=None):
    """데이터를 train/validation/test로 3-way 분할 (길이 정보 포함)"""
    from sklearn.model_selection import train_test_split
    
    if random_seed is None:
        random_seed = TRAINING_CONFIG['random_seed']
    
    split_config = TRAINING_CONFIG['data_split']
    train_ratio = split_config['train_ratio']
    val_ratio = split_config['validation_ratio'] 
    test_ratio = split_config['test_ratio']
    stratify = split_config['stratify']
    shuffle = split_config['shuffle']
    
    print(f"📊 데이터 3-way 분할 (Train:{train_ratio:.1%} / Val:{val_ratio:.1%} / Test:{test_ratio:.1%})")
    print(f"  - 총 데이터: {len(X):,}개")
    
    # 1단계: Train + Validation vs Test 분할
    train_val_ratio = train_ratio + val_ratio  # 남은 비율
    test_size_step1 = test_ratio
    
    stratify_step1 = y if stratify else None
    
    X_train_val, X_test, y_train_val, y_test, lengths_train_val, lengths_test = train_test_split(
        X, y, lengths,
        test_size=test_size_step1,
        random_state=random_seed,
        stratify=stratify_step1,
        shuffle=shuffle
    )
    
    # 2단계: Train vs Validation 분할
    val_size_step2 = val_ratio / train_val_ratio  # train_val 내에서의 validation 비율
    
    stratify_step2 = y_train_val if stratify else None
    
    X_train, X_val, y_train, y_val, lengths_train, lengths_val = train_test_split(
        X_train_val, y_train_val, lengths_train_val,
        test_size=val_size_step2,
        random_state=random_seed + 1,  # 다른 시드 사용
        stratify=stratify_step2,
        shuffle=shuffle
    )
    
    # 결과 출력
    print(f"  - Train: {len(X_train):,}개 ({len(X_train)/len(X):.1%})")
    print(f"  - Validation: {len(X_val):,}개 ({len(X_val)/len(X):.1%})")
    print(f"  - Test: {len(X_test):,}개 ({len(X_test)/len(X):.1%})")
    
    # 클래스별 분포 확인
    if hasattr(y, '__iter__'):
        print(f"\n📈 클래스별 분포 확인:")
        unique_classes = np.unique(y)
        for class_idx in unique_classes:
            train_count = np.sum(y_train == class_idx)
            val_count = np.sum(y_val == class_idx)
            test_count = np.sum(y_test == class_idx)
            total_count = train_count + val_count + test_count
            
            class_name = ALL_CLASSES[class_idx] if class_idx < len(ALL_CLASSES) else f"Class_{class_idx}"
            print(f"  - {class_name}: Train {train_count}, Val {val_count}, Test {test_count} (총 {total_count})")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, lengths_train, lengths_val, lengths_test

def save_dataset_splits(X_train, X_val, X_test, y_train, y_val, y_test, base_name=None):
    """분할된 데이터셋을 개별 파일로 저장"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if base_name is None:
        base_name = f"split_dataset_{MODEL_CONFIG['version']}_{timestamp}"
    
    # 각 분할을 개별 파일로 저장
    train_path = os.path.join(DATASET_SAVE_DIR, f"{base_name}_train.npz")
    val_path = os.path.join(DATASET_SAVE_DIR, f"{base_name}_val.npz")
    test_path = os.path.join(DATASET_SAVE_DIR, f"{base_name}_test.npz")
    
    np.savez_compressed(train_path, X=X_train, y=y_train)
    np.savez_compressed(val_path, X=X_val, y=y_val)
    np.savez_compressed(test_path, X=X_test, y=y_test)
    
    # 메타데이터 저장
    metadata = {
        'split_info': {
            'train_ratio': TRAINING_CONFIG['data_split']['train_ratio'],
            'validation_ratio': TRAINING_CONFIG['data_split']['validation_ratio'],
            'test_ratio': TRAINING_CONFIG['data_split']['test_ratio'],
            'stratify': TRAINING_CONFIG['data_split']['stratify'],
            'random_seed': TRAINING_CONFIG['random_seed']
        },
        'data_counts': {
            'train': len(X_train),
            'validation': len(X_val),
            'test': len(X_test),
            'total': len(X_train) + len(X_val) + len(X_test)
        },
        'files': {
            'train': train_path,
            'validation': val_path,
            'test': test_path
        },
        'creation_time': datetime.now().isoformat(),
        'model_version': MODEL_CONFIG['version']
    }
    
    metadata_path = os.path.join(DATASET_SAVE_DIR, f"{base_name}_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"💾 분할된 데이터셋 저장:")
    print(f"  - Train: {train_path}")
    print(f"  - Validation: {val_path}")
    print(f"  - Test: {test_path}")
    print(f"  - Metadata: {metadata_path}")
    
    return {
        'train': train_path,
        'validation': val_path,
        'test': test_path,
        'metadata': metadata_path
    }

def save_dataset_splits_with_lengths(X_train, X_val, X_test, y_train, y_val, y_test, 
                                   lengths_train, lengths_val, lengths_test, base_name=None):
    """분할된 데이터셋을 개별 파일로 저장 (길이 정보 포함)"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if base_name is None:
        base_name = f"split_dataset_{MODEL_CONFIG['version']}_{timestamp}"
    
    # 각 분할을 개별 파일로 저장 (길이 정보 포함)
    train_path = os.path.join(DATASET_SAVE_DIR, f"{base_name}_train.npz")
    val_path = os.path.join(DATASET_SAVE_DIR, f"{base_name}_val.npz")
    test_path = os.path.join(DATASET_SAVE_DIR, f"{base_name}_test.npz")
    
    np.savez_compressed(train_path, X=X_train, y=y_train, lengths=lengths_train)
    np.savez_compressed(val_path, X=X_val, y=y_val, lengths=lengths_val)
    np.savez_compressed(test_path, X=X_test, y=y_test, lengths=lengths_test)
    
    # 메타데이터 저장 (길이 정보 포함)
    metadata = {
        'split_info': {
            'train_ratio': TRAINING_CONFIG['data_split']['train_ratio'],
            'validation_ratio': TRAINING_CONFIG['data_split']['validation_ratio'],
            'test_ratio': TRAINING_CONFIG['data_split']['test_ratio'],
            'stratify': TRAINING_CONFIG['data_split']['stratify'],
            'random_seed': TRAINING_CONFIG['random_seed']
        },
        'data_counts': {
            'train': len(X_train),
            'validation': len(X_val),
            'test': len(X_test),
            'total': len(X_train) + len(X_val) + len(X_test)
        },
        'sequence_length_stats': {
            'train': {
                'min': int(np.min(lengths_train)),
                'max': int(np.max(lengths_train)),
                'mean': float(np.mean(lengths_train)),
                'std': float(np.std(lengths_train))
            },
            'validation': {
                'min': int(np.min(lengths_val)),
                'max': int(np.max(lengths_val)),
                'mean': float(np.mean(lengths_val)),
                'std': float(np.std(lengths_val))
            },
            'test': {
                'min': int(np.min(lengths_test)),
                'max': int(np.max(lengths_test)),
                'mean': float(np.mean(lengths_test)),
                'std': float(np.std(lengths_test))
            }
        },
        'files': {
            'train': train_path,
            'validation': val_path,
            'test': test_path
        },
        'creation_time': datetime.now().isoformat(),
        'model_version': MODEL_CONFIG['version'],
        'includes_sequence_lengths': True
    }
    
    metadata_path = os.path.join(DATASET_SAVE_DIR, f"{base_name}_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"💾 분할된 데이터셋 저장 (길이 정보 포함):")
    print(f"  - Train: {train_path}")
    print(f"  - Validation: {val_path}")
    print(f"  - Test: {test_path}")
    print(f"  - Metadata: {metadata_path}")
    
    return {
        'train': train_path,
        'validation': val_path,
        'test': test_path,
        'metadata': metadata_path
    }

def print_config_summary():
    """설정 요약 출력"""
    # 경로 초기화 확인
    if 'MODEL_SAVE_DIR' not in globals():
        initialize_paths()
    
    print("=" * 60)
    print("🔧 YAMNet + LSTM 훈련 설정 요약")
    print("=" * 60)
    
    print(f"📱 모델 버전: {MODEL_CONFIG['version']}")
    print(f"🎵 오디오 길이: {MODEL_CONFIG['audio_duration']}초")
    print(f"📊 총 클래스 수: {NUM_CLASSES}개")
    print(f"🎯 활성 클래스: {', '.join(ALL_CLASSES)}")
    
    print(f"\n📈 훈련 설정:")
    print(f"  - 에포크: {TRAINING_CONFIG['epochs']}")
    print(f"  - 배치 크기: {TRAINING_CONFIG['batch_size']}")
    print(f"  - 학습률: {TRAINING_CONFIG['learning_rate']}")
    
    # 데이터 분할 비율 정보 추가
    split_config = TRAINING_CONFIG['data_split']
    print(f"\n📊 데이터 분할 비율:")
    print(f"  - Train: {split_config['train_ratio']:.1%}")
    print(f"  - Validation: {split_config['validation_ratio']:.1%}")
    print(f"  - Test: {split_config['test_ratio']:.1%}")
    print(f"  - 클래스별 균등분할: {'✅' if split_config['stratify'] else '❌'}")
    print(f"  - 데이터 셔플: {'✅' if split_config['shuffle'] else '❌'}")
    
    print(f"\n🔄 데이터 증강 활성화:")
    for class_name, config in AUGMENTATION_CONFIG.items():
        if class_name in ALL_CLASSES and config['enabled']:
            methods = ', '.join(config['methods'])
            print(f"  - {class_name}: {methods}")
    
    print(f"\n↔️ 전환 데이터 설정:")
    if TRANSITION_CONFIG['enabled']:
        for trans_type, config in TRANSITION_CONFIG['types'].items():
            if config['enabled']:
                print(f"  - {config['description']}: 가중치 {config['weight']}")
    else:
        print("  - 비활성화")
    
    print(f"\n💾 출력 경로:")
    print(f"  - 모델: {MODEL_SAVE_DIR}")
    print(f"  - 보고서: {REPORT_SAVE_DIR}")
    print(f"  - 데이터셋: {DATASET_SAVE_DIR}")
    
    print(f"\n🎵 평가용 오디오 경로:")
    for class_name, path in EVALUATION_AUDIO_PATHS.items():
        if path is not None:
            status = "✅" if os.path.exists(path) else "❌"
            print(f"  - {class_name}: {status} {path}")
        else:
            print(f"  - {class_name}: ⏭️ 스킵")
    
    print("=" * 60)

def save_config_info(user_samples=None):
    """설정 정보를 텍스트 파일로 저장"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version = MODEL_CONFIG['version']
    
    config_path = os.path.join(TRAINING_RESULTS_DIR, f"config_{version}_{timestamp}.txt")
    
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("⚙️ YAMNet + LSTM 모델 설정 정보\n")
        f.write("=" * 80 + "\n\n")
        
        # 1. 기본 모델 설정
        f.write("📱 모델 기본 설정\n")
        f.write("-" * 40 + "\n")
        f.write(f"모델 버전: {MODEL_CONFIG['version']}\n")
        f.write(f"오디오 길이: {MODEL_CONFIG['audio_duration']}초\n")
        f.write(f"샘플링 레이트: {MODEL_CONFIG['sample_rate']:,} Hz\n")
        f.write(f"YAMNet 프레임 수: {get_audio_frames_count()}개\n")
        f.write(f"YAMNet 특성 수: 1024차원\n\n")
        
        # 2. 클래스 설정
        f.write("🎯 클래스 설정\n")
        f.write("-" * 40 + "\n")
        f.write(f"총 클래스 수: {NUM_CLASSES}개\n")
        f.write("활성 클래스:\n")
        for i, class_name in enumerate(ALL_CLASSES):
            korean_name = CLASS_NAMES.get(class_name, class_name)
            f.write(f"  {i}. {class_name} ({korean_name})\n")
        
        f.write("\n위험 클래스 활성화:\n")
        for danger_class, enabled in DANGER_CLASSES.items():
            f.write(f"  {danger_class}: {'✅' if enabled else '❌'}\n")
        f.write("\n")
        
        # 3. 훈련 설정
        f.write("🧠 훈련 설정\n")
        f.write("-" * 40 + "\n")
        f.write(f"에포크 수: {TRAINING_CONFIG['epochs']}\n")
        f.write(f"배치 크기: {TRAINING_CONFIG['batch_size']}\n")
        f.write(f"학습률: {TRAINING_CONFIG['learning_rate']}\n")
        f.write(f"랜덤 시드: {TRAINING_CONFIG['random_seed']}\n")
        f.write(f"입력 정규화: {'✅' if TRAINING_CONFIG['normalize_input'] else '❌'}\n")
        f.write(f"클래스 가중치 사용: {'✅' if TRAINING_CONFIG['use_class_weights'] else '❌'}\n\n")
        
        # 4. 모델 아키텍처 설정
        f.write("🏗️ 모델 아키텍처\n")
        f.write("-" * 40 + "\n")
        f.write(f"LSTM 유닛 수: {TRAINING_CONFIG['lstm_units']}\n")
        f.write(f"Dense 유닛 수: {TRAINING_CONFIG['dense_units']}\n")
        f.write(f"드롭아웃 비율: {TRAINING_CONFIG['dropout_rate']}\n")
        f.write("LSTM 구조:\n")
        f.write(f"  1층: {TRAINING_CONFIG['lstm_units']}유닛 (return_sequences=True)\n")
        f.write(f"  2층: {TRAINING_CONFIG['lstm_units']//2}유닛 (return_sequences=True)\n")
        f.write(f"  3층: {TRAINING_CONFIG['lstm_units']//4}유닛 (return_sequences=False)\n")
        f.write("Dense 구조:\n")
        f.write(f"  1층: {TRAINING_CONFIG['dense_units']}유닛 (ReLU)\n")
        f.write(f"  2층: {TRAINING_CONFIG['dense_units']//2}유닛 (ReLU)\n")
        f.write(f"  출력: {NUM_CLASSES}유닛 (Softmax)\n\n")
        
        # 5. 데이터 분할 설정
        split_config = TRAINING_CONFIG['data_split']
        f.write("📊 데이터 분할 설정\n")
        f.write("-" * 40 + "\n")
        f.write(f"훈련 비율: {split_config['train_ratio']:.1%}\n")
        f.write(f"검증 비율: {split_config['validation_ratio']:.1%}\n")
        f.write(f"테스트 비율: {split_config['test_ratio']:.1%}\n")
        f.write(f"계층 분할: {'✅' if split_config['stratify'] else '❌'}\n")
        f.write(f"데이터 셔플: {'✅' if split_config['shuffle'] else '❌'}\n\n")
        
        # 6. 콜백 설정
        f.write("📞 콜백 설정\n")
        f.write("-" * 40 + "\n")
        f.write(f"체크포인트 저장: {'✅' if TRAINING_CONFIG['save_checkpoints'] else '❌'}\n")
        
        if TRAINING_CONFIG['early_stopping']['enabled']:
            f.write(f"조기 종료: ✅ (patience: {TRAINING_CONFIG['early_stopping']['patience']})\n")
        else:
            f.write("조기 종료: ❌\n")
        
        if TRAINING_CONFIG['learning_rate_schedule']['enabled']:
            lr_config = TRAINING_CONFIG['learning_rate_schedule']
            f.write(f"학습률 스케줄링: ✅\n")
            f.write(f"  감소 비율: {lr_config['factor']}\n")
            f.write(f"  대기 에포크: {lr_config['patience']}\n")
            f.write(f"  최소 학습률: {lr_config['min_lr']}\n")
        else:
            f.write("학습률 스케줄링: ❌\n")
        f.write("\n")
        
        # 7. 데이터 생성 설정
        data_config = DATA_GENERATION_CONFIG
        f.write("🏭 데이터 생성 설정\n")
        f.write("-" * 40 + "\n")
        f.write(f"클래스당 목표 프레임: {data_config['target_frames_per_class']:,}개\n")
        f.write(f"증강 시작 임계값: {data_config['min_frames_for_augmentation']:,}개\n")
        f.write(f"전환 데이터 비율: {data_config['transition_data_ratio']:.1%}\n")
        f.write(f"자동 균형 조정: {'✅' if data_config['auto_balance'] else '❌'}\n")
        f.write(f"사용자 입력 허용: {'✅' if data_config['allow_user_input'] else '❌'}\n\n")
        
        # 8. 사용자 입력 샘플 수
        if user_samples:
            f.write("🎯 사용자 설정 샘플 수\n")
            f.write("-" * 40 + "\n")
            total_user_samples = sum(user_samples.values())
            for class_name, sample_count in user_samples.items():
                percentage = (sample_count / total_user_samples) * 100
                f.write(f"  {class_name:8}: {sample_count:4,}개 ({percentage:5.1f}%)\n")
            f.write(f"  총합:      {total_user_samples:4,}개\n\n")
        
        # 9. 데이터 증강 설정
        f.write("🎨 데이터 증강 설정\n")
        f.write("-" * 40 + "\n")
        for class_name in ALL_CLASSES:
            if class_name in AUGMENTATION_CONFIG:
                config = AUGMENTATION_CONFIG[class_name]
                enabled = config.get('enabled', False)
                f.write(f"{class_name:8}: {'✅' if enabled else '❌'}")
                
                if enabled:
                    methods = config.get('methods', [])
                    f.write(f" - 방법: {', '.join(methods)}")
                    
                    if 'volume_range' in config:
                        vol_range = config['volume_range']
                        f.write(f", 볼륨: {vol_range[0]}-{vol_range[1]}")
                    
                    if 'snr_range' in config:
                        snr_range = config['snr_range']
                        f.write(f", SNR: {snr_range[0]}-{snr_range[1]}dB")
                
                f.write("\n")
        f.write("\n")
        
        # 10. 전환 데이터 설정
        f.write("🔄 전환 데이터 설정\n")
        f.write("-" * 40 + "\n")
        f.write(f"전환 데이터 사용: {'✅' if TRANSITION_CONFIG['enabled'] else '❌'}\n")
        
        if TRANSITION_CONFIG['enabled']:
            f.write(f"페이드 시간: {TRANSITION_CONFIG['fade_duration']}초\n")
            f.write("전환 타입별 설정:\n")
            
            for trans_type, config in TRANSITION_CONFIG['types'].items():
                enabled = config.get('enabled', False)
                weight = config.get('weight', 1.0)
                description = config.get('description', trans_type)
                
                f.write(f"  {trans_type:20}: {'✅' if enabled else '❌'}")
                if enabled:
                    f.write(f" (가중치: {weight:.1f}) - {description}")
                f.write("\n")
        f.write("\n")
        
        # 11. 경로 설정
        f.write("📁 경로 설정\n")
        f.write("-" * 40 + "\n")
        f.write(f"환경음 디렉토리: {ENVSOUND_DIR}\n")
        f.write(f"공장음 디렉토리: {MIXTURE_DIR}\n")
        f.write(f"결과 저장 디렉토리: {VERSION_DIR}\n")
        f.write(f"모델 저장 디렉토리: {MODEL_SAVE_DIR}\n")
        f.write(f"데이터셋 저장 디렉토리: {DATASET_SAVE_DIR}\n\n")
        
        # 12. 시스템 정보
        f.write("💻 시스템 정보\n")
        f.write("-" * 40 + "\n")
        f.write(f"설정 생성 시간: {datetime.now().strftime('%Y년 %m월 %d일 %H시 %M분 %S초')}\n")
        
        try:
            import sys
            f.write(f"Python 버전: {sys.version.split()[0]}\n")
        except:
            f.write("Python 버전: 확인 불가\n")
        
        try:
            import tensorflow as tf
            f.write(f"TensorFlow 버전: {tf.__version__}\n")
        except:
            f.write("TensorFlow 버전: 확인 불가\n")
        
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("설정 정보 저장 완료\n")
        f.write("=" * 80 + "\n")
    
    print(f"⚙️ 설정 정보 저장: {config_path}")
    return config_path

if __name__ == "__main__":
    # 설정 검증
    errors = validate_config()
    if errors:
        print("❌ 설정 오류:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("✅ 설정 검증 완료")
        
    # 설정 요약 출력
    print_config_summary()
    
    # 권장 샘플 수 출력
    recommendations = get_recommended_samples()
    print(f"\n📊 권장 데이터 구성:")
    print(f"  - 오디오당 프레임 수: {recommendations['frames_per_audio']}")
    print(f"  - 클래스당 목표 프레임: {recommendations['target_frames_per_class']}")
    print(f"  - 권장 기본 샘플 수: {recommendations['recommended_samples']}개/클래스")

    print(f"  - 전환 데이터 기여분: {recommendations['transition_contribution']:.0f} 프레임")

# 모든 설정이 정의된 후 경로 초기화
initialize_paths()

def update_evaluation_path(class_name, new_path):
    """평가용 오디오 경로 업데이트"""
    global EVALUATION_AUDIO_PATHS
    if class_name in EVALUATION_AUDIO_PATHS:
        EVALUATION_AUDIO_PATHS[class_name] = new_path
        print(f"✅ {class_name} 클래스 경로 업데이트: {new_path}")
    else:
        print(f"⚠️ 알 수 없는 클래스: {class_name}")

def get_evaluation_path(class_name):
    """특정 클래스의 평가용 오디오 경로 반환"""
    return EVALUATION_AUDIO_PATHS.get(class_name, os.path.join(ENVSOUND_DIR, class_name))

def print_evaluation_paths():
    """평가용 경로 확인 및 출력"""
    print(f"\n🎵 평가용 오디오 경로 확인:")
    for class_name, path in EVALUATION_AUDIO_PATHS.items():
        if path is not None:
            exists = os.path.exists(path)
            status = "✅" if exists else "❌"
            if exists:
                file_count = len([f for f in os.listdir(path) 
                                if f.endswith(('.wav', '.mp3', '.flac'))]) if os.path.isdir(path) else 0
                print(f"  {status} {class_name}: {path} ({file_count}개 파일)")
            else:
                print(f"  {status} {class_name}: {path} (경로 없음)")
        else:
            print(f"  ⏭️ {class_name}: 스킵")
