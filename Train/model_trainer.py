"""
모델 훈련 모듈
=============

이 모듈은 YAMNet + LSTM 모델을 훈련합니다.
- 생성된 데이터를 로드하여 훈련
- 모델 아키텍처 정의 및 컴파일
- 훈련 과정 모니터링 및 체크포인트
- 모델 성능 평가 및 저장
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
from tqdm import tqdm
from datetime import datetime
from config import *

class ModelTrainer:
    def __init__(self):
        """모델 훈련기 초기화"""
        self.model = None
        self.history = None
        self.class_weights = None
        self.training_info = {
            'start_time': None,
            'end_time': None,
            'config': TRAINING_CONFIG.copy(),
            'model_config': MODEL_CONFIG.copy(),
            'performance': {}
        }
        
    def load_dataset(self, data_path, dataset_info_path=None):
        """생성된 데이터셋 로드"""
        print("📂 데이터셋 로딩 중...")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"데이터 파일이 없습니다: {data_path}")
        
        # 데이터 로드
        data = np.load(data_path)
        X = data['X']
        y = data['y']
        
        print(f"✅ 데이터 로드 완료:")
        print(f"  - 임베딩 형태: {X.shape}")
        print(f"  - 라벨 형태: {y.shape}")
        
        # 데이터셋 정보 로드 (있는 경우)
        if dataset_info_path and os.path.exists(dataset_info_path):
            with open(dataset_info_path, 'r', encoding='utf-8') as f:
                dataset_info = json.load(f)
                
            print(f"\n📊 데이터셋 정보:")
            if 'final_stats' in dataset_info:
                # 시퀀스 기반과 프레임 기반 모두 지원
                total_count = dataset_info['final_stats'].get('total_sequences', 
                             dataset_info['final_stats'].get('total_frames', 0))
                
                if total_count > 0:
                    for class_name, count in dataset_info['final_stats']['class_distribution'].items():
                        percentage = count / total_count * 100
                        print(f"  - {class_name}: {count:,}개 ({percentage:.1f}%)")
                else:
                    print("  - 데이터셋 통계 정보가 없습니다.")
                    
            self.training_info['dataset_info'] = dataset_info
        
        return X, y
    
    def load_presplit_data(self, data_base_name=None):
        """미리 분할된 데이터 로드"""
        print("\n📊 분할된 데이터 로딩 중...")
        
        if data_base_name is None:
            # 가장 최근 분할 데이터 찾기
            split_files = [f for f in os.listdir(DATASET_SAVE_DIR) 
                          if f.startswith('split_dataset_') and f.endswith('_train.npz')]
            
            if not split_files:
                return None, None, None, None, None, None
            
            # 가장 최근 파일 찾기
            latest_file = max(split_files, key=lambda f: os.path.getmtime(os.path.join(DATASET_SAVE_DIR, f)))
            data_base_name = latest_file.replace('_train.npz', '').replace('split_dataset_', '')
        
        # 분할된 파일 경로
        train_path = os.path.join(DATASET_SAVE_DIR, f"split_dataset_{data_base_name}_train.npz")
        val_path = os.path.join(DATASET_SAVE_DIR, f"split_dataset_{data_base_name}_val.npz")
        
        if not (os.path.exists(train_path) and os.path.exists(val_path)):
            print(f"⚠️ 분할된 데이터가 없습니다: {train_path}, {val_path}")
            return None, None, None, None, None, None
        
        print(f"📂 Train 데이터: {train_path}")
        print(f"📂 Validation 데이터: {val_path}")
        
        # 데이터 로드
        train_data = np.load(train_path)
        val_data = np.load(val_path)
        
        X_train, y_train = train_data['X'], train_data['y']
        X_val, y_val = val_data['X'], val_data['y']
        
        print(f"  - 훈련 데이터: {X_train.shape[0]:,}개")
        print(f"  - 검증 데이터: {X_val.shape[0]:,}개")
        
        # 클래스 분포 확인
        print(f"\n📊 클래스별 분포:")
        for class_idx in range(NUM_CLASSES):
            train_count = np.sum(y_train == class_idx)
            val_count = np.sum(y_val == class_idx)
            class_name = ALL_CLASSES[class_idx]
            print(f"  - {class_name}: Train {train_count}, Val {val_count}")
        
        return X_train, X_val, y_train, y_val, train_path, val_path

    def prepare_data(self, X, y, use_presplit=True):
        """데이터 전처리 및 분할"""
        print("\n🔄 데이터 전처리 중...")
        
        # 분할된 데이터 우선 사용
        if use_presplit:
            split_data = self.load_presplit_data()
            if split_data[0] is not None:  # 분할된 데이터가 있는 경우
                X_train, X_val, y_train, y_val, train_path, val_path = split_data
                print("✅ 기존 분할된 데이터 사용")
                
                # 클래스 가중치 계산 (훈련 데이터 기준)
                if TRAINING_CONFIG['use_class_weights']:
                    print("  - 클래스 가중치 계산")
                    unique_classes = np.unique(y_train)
                    class_weights_array = compute_class_weight(
                        'balanced', 
                        classes=unique_classes, 
                        y=y_train
                    )
                    self.class_weights = dict(zip(unique_classes, class_weights_array))
                    
                    print("    클래스 가중치:")
                    for class_idx, weight in self.class_weights.items():
                        class_name = ALL_CLASSES[class_idx]
                        print(f"      {class_name}: {weight:.3f}")
                
                # 원-핫 인코딩
                y_train_onehot = keras.utils.to_categorical(y_train, num_classes=NUM_CLASSES)
                y_val_onehot = keras.utils.to_categorical(y_val, num_classes=NUM_CLASSES)
                
                return X_train, X_val, y_train_onehot, y_val_onehot, y_train, y_val
        
        # 기존 방식: 전체 데이터에서 분할
        print("🔄 전체 데이터에서 분할 진행")
        
        # 입력 정규화
        if TRAINING_CONFIG['normalize_input']:
            print("  - 입력 데이터 정규화")
            X_mean = np.mean(X, axis=0)
            X_std = np.std(X, axis=0)
            X = (X - X_mean) / (X_std + 1e-8)
            
            # 정규화 파라미터 저장
            self.training_info['normalization'] = {
                'mean': X_mean.tolist(),
                'std': X_std.tolist()
            }
        
        # 클래스 가중치 계산
        if TRAINING_CONFIG['use_class_weights']:
            print("  - 클래스 가중치 계산")
            unique_classes = np.unique(y)
            class_weights_array = compute_class_weight(
                'balanced', 
                classes=unique_classes, 
                y=y
            )
            self.class_weights = dict(zip(unique_classes, class_weights_array))
            
            print("    클래스 가중치:")
            for class_idx, weight in self.class_weights.items():
                class_name = ALL_CLASSES[class_idx]
                print(f"      {class_name}: {weight:.3f}")
        
        # 훈련/검증 데이터 분할
        test_size = TRAINING_CONFIG['validation_split']
        random_state = TRAINING_CONFIG['random_seed']
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y  # 클래스 비율 유지
        )
        
        print(f"  - 훈련 데이터: {X_train.shape[0]:,}개")
        print(f"  - 검증 데이터: {X_val.shape[0]:,}개")
        
        # 원-핫 인코딩
        y_train_onehot = keras.utils.to_categorical(y_train, num_classes=NUM_CLASSES)
        y_val_onehot = keras.utils.to_categorical(y_val, num_classes=NUM_CLASSES)
        
        return X_train, X_val, y_train_onehot, y_val_onehot, y_train, y_val
    
    def create_model(self, input_shape):
        """YAMNet + LSTM 모델 생성"""
        print("\n🏗️ 모델 아키텍처 생성 중...")
        print(f"📏 입력 형태: {input_shape}")
        
        # 입력 형태가 3D(time_steps, features)인지 확인
        if len(input_shape) != 2:
            raise ValueError(f"LSTM 모델은 3D 입력 (batch, time_steps, features)이 필요합니다. 현재: {input_shape}")
        
        time_steps, features = input_shape
        print(f"⏱️ 시간 스텝: {time_steps}, 🎵 특성 수: {features}")
        
        model = keras.Sequential([
            layers.Input(shape=input_shape),
            
            # 첫 번째 LSTM 레이어 - 시퀀스를 유지하며 패턴 학습
            layers.LSTM(
                TRAINING_CONFIG['lstm_units'],
                return_sequences=True,
                dropout=TRAINING_CONFIG['dropout_rate'],
                recurrent_dropout=TRAINING_CONFIG['dropout_rate'],
                name='lstm_1'
            ),
            layers.BatchNormalization(name='batch_norm_1'),
            
            # 두 번째 LSTM 레이어 - 더 복잡한 시간적 패턴 학습
            layers.LSTM(
                TRAINING_CONFIG['lstm_units'] // 2,
                return_sequences=True,
                dropout=TRAINING_CONFIG['dropout_rate'],
                recurrent_dropout=TRAINING_CONFIG['dropout_rate'],
                name='lstm_2'
            ),
            layers.BatchNormalization(name='batch_norm_2'),
            
            # 세 번째 LSTM 레이어 - 최종 시퀀스 요약
            layers.LSTM(
                TRAINING_CONFIG['lstm_units'] // 4,
                return_sequences=False,  # 마지막 출력만 사용
                dropout=TRAINING_CONFIG['dropout_rate'],
                recurrent_dropout=TRAINING_CONFIG['dropout_rate'],
                name='lstm_3'
            ),
            
            # Dense 레이어들로 최종 분류
            layers.Dense(
                TRAINING_CONFIG['dense_units'], 
                activation='relu',
                name='dense_1'
            ),
            layers.Dropout(TRAINING_CONFIG['dropout_rate']),
            layers.BatchNormalization(name='batch_norm_3'),
            
            layers.Dense(
                TRAINING_CONFIG['dense_units'] // 2, 
                activation='relu',
                name='dense_2'
            ),
            layers.Dropout(TRAINING_CONFIG['dropout_rate']),
            
            # 출력 레이어
            layers.Dense(NUM_CLASSES, activation='softmax', name='output')
        ])
        
        # 모델 컴파일
        optimizer = keras.optimizers.Adam(
            learning_rate=TRAINING_CONFIG['learning_rate']
        )
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        # 모델 구조 출력
        model.summary()
        
        # 모델 구조를 이미지로 저장
        try:
            model_plot_path = os.path.join(TRAINING_RESULTS_DIR, 'model_architecture.png')
            keras.utils.plot_model(
                model, 
                to_file=model_plot_path, 
                show_shapes=True, 
                show_layer_names=True,
                dpi=150
            )
            print(f"📊 모델 구조 저장: {model_plot_path}")
        except ImportError as e:
            if 'pydot' in str(e):
                print(f"⚠️ 모델 구조 이미지 저장 실패: pydot 패키지가 필요합니다")
                print(f"💡 해결 방법: pip install pydot graphviz")
            else:
                print(f"⚠️ 모델 구조 이미지 저장 실패: {e}")
        except Exception as e:
            print(f"⚠️ 모델 구조 이미지 저장 실패: {e}")
            print(f"💡 Graphviz가 시스템에 설치되어 있는지 확인하세요")
        
        self.model = model
        return model
    
    def create_callbacks(self):
        """훈련 콜백 생성"""
        callbacks = []
        
        # 체크포인트
        if TRAINING_CONFIG['save_checkpoints']:
            checkpoint_path = os.path.join(
                TRAINING_RESULTS_DIR, 
                'checkpoints', 
                'model_epoch_{epoch:03d}_val_acc_{val_accuracy:.4f}.h5'
            )
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            
            checkpoint_callback = keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                mode='max',
                verbose=1
            )
            callbacks.append(checkpoint_callback)
        
        # 조기 종료
        if TRAINING_CONFIG['early_stopping']['enabled']:
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=TRAINING_CONFIG['early_stopping']['patience'],
                restore_best_weights=True,
                verbose=1
            )
            callbacks.append(early_stopping)
        
        # 학습률 스케줄링
        if TRAINING_CONFIG['learning_rate_schedule']['enabled']:
            lr_scheduler = keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=TRAINING_CONFIG['learning_rate_schedule']['factor'],
                patience=TRAINING_CONFIG['learning_rate_schedule']['patience'],
                min_lr=TRAINING_CONFIG['learning_rate_schedule']['min_lr'],
                verbose=1
            )
            callbacks.append(lr_scheduler)
        
        # 사용자 정의 로깅 콜백
        class TrainingLogger(keras.callbacks.Callback):
            def __init__(self, trainer):
                super().__init__()
                self.trainer = trainer
                
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                
                # 에포크별 성능 기록
                if 'epoch_logs' not in self.trainer.training_info:
                    self.trainer.training_info['epoch_logs'] = []
                    
                epoch_info = {
                    'epoch': epoch + 1,
                    'loss': logs.get('loss', 0),
                    'accuracy': logs.get('accuracy', 0),
                    'val_loss': logs.get('val_loss', 0),
                    'val_accuracy': logs.get('val_accuracy', 0),
                    'precision': logs.get('precision', 0),
                    'recall': logs.get('recall', 0),
                    'val_precision': logs.get('val_precision', 0),
                    'val_recall': logs.get('val_recall', 0)
                }
                
                self.trainer.training_info['epoch_logs'].append(epoch_info)
                
                # 진행 상황 출력
                print(f"에포크 {epoch + 1}/{TRAINING_CONFIG['epochs']} - "
                      f"정확도: {logs.get('accuracy', 0):.4f} - "
                      f"검증 정확도: {logs.get('val_accuracy', 0):.4f}")
        
        callbacks.append(TrainingLogger(self))
        
        return callbacks
    
    def convert_frames_to_sequences(self, X, y, sequence_length=None):
        """프레임 데이터를 시퀀스 데이터로 변환"""
        if sequence_length is None:
            sequence_length = TRAINING_CONFIG.get('sequence_length', 21)
            
        print(f"\n🔄 프레임 데이터를 시퀀스로 변환 중...")
        print(f"  📏 입력 형태: {X.shape}")
        print(f"  🎯 목표 시퀀스 길이: {sequence_length}")
        
        # 총 프레임 수가 시퀀스 길이로 나누어 떨어지는지 확인
        total_frames = X.shape[0]
        num_sequences = total_frames // sequence_length
        
        if num_sequences == 0:
            raise ValueError(f"프레임 수({total_frames})가 시퀀스 길이({sequence_length})보다 작습니다.")
        
        # 시퀀스로 재구성할 수 있는 프레임만 사용
        usable_frames = num_sequences * sequence_length
        X_usable = X[:usable_frames]
        y_usable = y[:usable_frames]
        
        print(f"  📊 사용 가능한 프레임: {usable_frames}/{total_frames}")
        print(f"  📦 생성될 시퀀스 수: {num_sequences}")
        
        # 프레임을 시퀀스로 재구성
        X_sequences = X_usable.reshape(num_sequences, sequence_length, X.shape[1])
        
        # 라벨은 각 시퀀스의 대표값 사용 (다수결 또는 첫 번째 프레임)
        y_sequences = []
        for i in range(num_sequences):
            start_idx = i * sequence_length
            end_idx = start_idx + sequence_length
            sequence_labels = y_usable[start_idx:end_idx]
            
            # 다수결로 시퀀스 라벨 결정
            unique_labels, counts = np.unique(sequence_labels, return_counts=True)
            majority_label = unique_labels[np.argmax(counts)]
            y_sequences.append(majority_label)
        
        y_sequences = np.array(y_sequences)
        
        print(f"  ✅ 변환 완료: {X_sequences.shape}")
        print(f"  📊 시퀀스별 라벨 분포:")
        
        for class_idx, class_name in enumerate(ALL_CLASSES):
            count = np.sum(y_sequences == class_idx)
            percentage = count / len(y_sequences) * 100 if len(y_sequences) > 0 else 0
            print(f"    - {class_name}: {count:,}개 ({percentage:.1f}%)")
        
        return X_sequences, y_sequences
    
    def train_model(self, X_train, X_val, y_train, y_val):
        """모델 훈련"""
        print("\n🚀 모델 훈련 시작...")
        
        self.training_info['start_time'] = datetime.now().isoformat()
        
        # 데이터 형태 확인
        print(f"📏 훈련 데이터 형태: {X_train.shape}")
        print(f"📏 검증 데이터 형태: {X_val.shape}")
        
        # 2D 데이터인 경우 3D로 변환
        if len(X_train.shape) == 2:
            print("🔄 2D 프레임 데이터를 3D 시퀀스로 변환 중...")
            
            # 프레임을 시퀀스로 변환
            X_train, y_train_seq = self.convert_frames_to_sequences(X_train, np.argmax(y_train, axis=1))
            X_val, y_val_seq = self.convert_frames_to_sequences(X_val, np.argmax(y_val, axis=1))
            
            # 원-핫 인코딩 다시 적용
            y_train = keras.utils.to_categorical(y_train_seq, num_classes=NUM_CLASSES)
            y_val = keras.utils.to_categorical(y_val_seq, num_classes=NUM_CLASSES)
            
            print(f"✅ 변환 완료:")
            print(f"  📏 훈련 데이터: {X_train.shape}")
            print(f"  📏 검증 데이터: {X_val.shape}")
        
        elif len(X_train.shape) == 3:
            print("✅ 이미 3D 시퀀스 데이터입니다.")
        
        else:
            raise ValueError(f"지원하지 않는 데이터 형태: {X_train.shape}")
        
        # 모델 생성 (시간 스텝과 특성 수를 입력 형태로 사용)
        model = self.create_model((X_train.shape[1], X_train.shape[2]))
        
        # 콜백 생성
        callbacks = self.create_callbacks()
        
        # 훈련 실행
        history = model.fit(
            X_train, y_train,
            batch_size=TRAINING_CONFIG['batch_size'],
            epochs=TRAINING_CONFIG['epochs'],
            validation_data=(X_val, y_val),
            class_weight=self.class_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        self.history = history
        self.training_info['end_time'] = datetime.now().isoformat()
        
        # 훈련 시간 계산
        start_time = datetime.fromisoformat(self.training_info['start_time'])
        end_time = datetime.fromisoformat(self.training_info['end_time'])
        training_duration = end_time - start_time
        
        self.training_info['training_duration'] = {
            'total_seconds': training_duration.total_seconds(),
            'formatted': str(training_duration)
        }
        
        print(f"\n✅ 훈련 완료! 소요 시간: {training_duration}")
        
        return history
    
    def evaluate_model(self, X_val, y_val_onehot, y_val):
        """모델 평가"""
        print("\n📊 모델 평가 중...")
        
        # 기본 평가
        eval_results = self.model.evaluate(X_val, y_val_onehot, verbose=0)
        
        # 평가 결과 저장
        metrics = self.model.metrics_names
        evaluation = dict(zip(metrics, eval_results))
        
        # 예측 수행
        y_pred_proba = self.model.predict(X_val, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # 클래스별 성능 계산
        from sklearn.metrics import classification_report, confusion_matrix
        
        # 분류 리포트
        class_report = classification_report(
            y_val, y_pred, 
            target_names=ALL_CLASSES,
            output_dict=True
        )
        
        # 혼동 행렬
        conf_matrix = confusion_matrix(y_val, y_pred)
        
        # 결과 저장
        self.training_info['performance'] = {
            'overall': evaluation,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist()
        }
        
        # 결과 출력
        print(f"\n📈 최종 성능:")
        print(f"  📊 평가 메트릭: {list(evaluation.keys())}")
        
        # 안전한 키 접근
        loss_value = evaluation.get('loss', evaluation.get('val_loss', 0))
        accuracy_value = evaluation.get('accuracy', evaluation.get('val_accuracy', 0))
        
        print(f"  - 손실: {loss_value:.4f}")
        print(f"  - 정확도: {accuracy_value:.4f}")
        
        # 전체 정확도를 계산 (만약 evaluation에 없다면)
        if 'accuracy' not in evaluation and 'val_accuracy' not in evaluation:
            overall_accuracy = np.mean(y_pred == y_val)
            evaluation['accuracy'] = overall_accuracy
            print(f"  - 계산된 정확도: {overall_accuracy:.4f}")
        
        # 가중 평균에서 정밀도와 재현율 가져오기
        try:
            weighted_avg = class_report.get('weighted avg', {})
            if isinstance(weighted_avg, dict):
                precision = evaluation.get('precision', weighted_avg.get('precision', 0))
                recall = evaluation.get('recall', weighted_avg.get('recall', 0))
            else:
                precision = evaluation.get('precision', 0)
                recall = evaluation.get('recall', 0)
        except:
            precision = evaluation.get('precision', 0)
            recall = evaluation.get('recall', 0)
        
        print(f"  - 정밀도: {precision:.4f}")
        print(f"  - 재현율: {recall:.4f}")
        
        print(f"\n📊 클래스별 성능:")
        for class_name in ALL_CLASSES:
            try:
                if isinstance(class_report, dict) and class_name in class_report:
                    metrics = class_report[class_name]
                    if isinstance(metrics, dict):
                        print(f"  - {class_name}:")
                        print(f"    정밀도: {metrics.get('precision', 0):.4f}, "
                              f"재현율: {metrics.get('recall', 0):.4f}, "
                              f"F1: {metrics.get('f1-score', 0):.4f}")
            except Exception as e:
                print(f"  - {class_name}: 메트릭 계산 오류 ({e})")
        
        return evaluation, class_report, conf_matrix
    
    def plot_training_history(self):
        """Training process visualization (English)"""
        if self.history is None:
            print("⚠️ No training history available.")
            return
        
        print("\n📊 Creating training progress visualization...")
        
        # Set matplotlib to use safe fonts
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        
        history = self.history.history
        epochs = range(1, len(history['loss']) + 1)
        
        # Plot configuration
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('YAMNet + LSTM Training Progress', fontsize=16, fontweight='bold')
        
        # Loss graph
        axes[0, 0].plot(epochs, history['loss'], 'b-', label='Training Loss', linewidth=2)
        axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy graph
        axes[0, 1].plot(epochs, history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
        axes[0, 1].plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
        axes[0, 1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Precision graph
        if 'precision' in history:
            axes[1, 0].plot(epochs, history['precision'], 'b-', label='Training Precision', linewidth=2)
            axes[1, 0].plot(epochs, history['val_precision'], 'r-', label='Validation Precision', linewidth=2)
            axes[1, 0].set_title('Model Precision', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Recall graph
        if 'recall' in history:
            axes[1, 1].plot(epochs, history['recall'], 'b-', label='Training Recall', linewidth=2)
            axes[1, 1].plot(epochs, history['val_recall'], 'r-', label='Validation Recall', linewidth=2)
            axes[1, 1].set_title('Model Recall', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save graph
        plot_path = os.path.join(TRAINING_RESULTS_DIR, 'training_history.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"📊 Training progress graph saved: {plot_path}")
    
    def plot_confusion_matrix(self, conf_matrix):
        """Confusion matrix visualization (English)"""
        print("\n📊 Creating confusion matrix visualization...")
        
        # Set matplotlib to use a safe font
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        
        plt.figure(figsize=(12, 10))
        
        # Normalized confusion matrix
        conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        
        # Get English class names
        from config import CLASS_NAMES_EN
        english_labels = [CLASS_NAMES_EN.get(cls, cls.title()) for cls in ALL_CLASSES]
        
        if HAS_SEABORN:
            # Using seaborn
            import seaborn as sns
            sns.heatmap(
                conf_matrix_norm, 
                annot=True, 
                fmt='.3f',
                cmap='Blues',
                xticklabels=english_labels,
                yticklabels=english_labels,
                cbar_kws={'label': 'Normalized Rate'},
                square=True,
                linewidths=0.5
            )
        else:
            # Using matplotlib only
            im = plt.imshow(conf_matrix_norm, interpolation='nearest', cmap='Blues')
            plt.colorbar(im, label='Normalized Rate')
            
            # Add text annotations
            for i in range(conf_matrix_norm.shape[0]):
                for j in range(conf_matrix_norm.shape[1]):
                    plt.text(j, i, f'{conf_matrix_norm[i, j]:.3f}',
                           ha="center", va="center", 
                           color="white" if conf_matrix_norm[i, j] > 0.5 else "black",
                           fontweight='bold')
            
            plt.xticks(range(len(ALL_CLASSES)), english_labels, rotation=45)
            plt.yticks(range(len(ALL_CLASSES)), english_labels, rotation=0)
        
        plt.title('Confusion Matrix - Normalized', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Predicted Class', fontsize=12, fontweight='bold')
        plt.ylabel('True Class', fontsize=12, fontweight='bold')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save with high quality
        cm_path = os.path.join(TRAINING_RESULTS_DIR, 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"📊 Confusion matrix saved: {cm_path}")
    
    def save_model(self, model_name=None):
        """모델 저장"""
        if self.model is None:
            print("⚠️ 저장할 모델이 없습니다.")
            return None
        
        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"yamnet_lstm_model_{timestamp}.h5"
        else:
            # 사용자가 입력한 모델명에 확장자가 없으면 .h5 추가
            if not model_name.endswith(('.h5', '.keras')):
                model_name = f"{model_name}.h5"
        
        model_path = os.path.join(TRAINING_RESULTS_DIR, model_name)
        
        # 모델 저장
        self.model.save(model_path)
        print(f"💾 모델 저장: {model_path}")
        
        # models 폴더에도 복사
        try:
            import shutil
            from config import MODEL_SAVE_DIR
            os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
            model_copy_path = os.path.join(MODEL_SAVE_DIR, os.path.basename(model_path))
            shutil.copy2(model_path, model_copy_path)
            print(f"📋 모델 복사: {model_copy_path}")
        except Exception as e:
            print(f"⚠️ 모델 복사 실패: {e}")
        
        # 훈련 정보 저장 (확장자에 따른 처리)
        base_name = model_path
        if model_path.endswith('.h5'):
            base_name = model_path[:-3]  # .h5 제거
        elif model_path.endswith('.keras'):
            base_name = model_path[:-6]  # .keras 제거
        
        info_path = f"{base_name}_info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(self.training_info, f, indent=2, ensure_ascii=False)
        print(f"📄 훈련 정보 저장: {info_path}")
        
        # 클래스 매핑 저장
        class_mapping_path = f"{base_name}_classes.npy"
        np.save(class_mapping_path, ALL_CLASSES)
        print(f"🏷️ 클래스 매핑 저장: {class_mapping_path}")
        
        # models 폴더에 추가 파일들도 복사
        try:
            import shutil
            from config import MODEL_SAVE_DIR
            if os.path.exists(info_path):
                info_copy_path = os.path.join(MODEL_SAVE_DIR, os.path.basename(info_path))
                shutil.copy2(info_path, info_copy_path)
                print(f"📋 훈련 정보 복사: {info_copy_path}")
            
            if os.path.exists(class_mapping_path):
                mapping_copy_path = os.path.join(MODEL_SAVE_DIR, os.path.basename(class_mapping_path))
                shutil.copy2(class_mapping_path, mapping_copy_path)
                print(f"📋 클래스 매핑 복사: {mapping_copy_path}")
        except Exception as e:
            print(f"⚠️ 추가 파일 복사 실패: {e}")
        
        return model_path, info_path, class_mapping_path
    
    def train_full_pipeline(self, data_path, dataset_info_path=None, model_name=None):
        """전체 훈련 파이프라인 실행"""
        try:
            # 1. 데이터 로드
            X, y = self.load_dataset(data_path, dataset_info_path)
            
            # 2. 데이터 전처리
            X_train, X_val, y_train_onehot, y_val_onehot, y_train_raw, y_val_raw = self.prepare_data(X, y)
            
            # 3. 모델 훈련
            history = self.train_model(X_train, X_val, y_train_onehot, y_val_onehot)
            
            # 4. 모델 평가
            evaluation, class_report, conf_matrix = self.evaluate_model(
                X_val, y_val_onehot, y_val_raw
            )
            
            # 5. 시각화
            self.plot_training_history()
            self.plot_confusion_matrix(conf_matrix)
            
            # 6. 모델 저장
            model_paths = self.save_model(model_name)
            
            print(f"\n🎉 모든 훈련 과정이 완료되었습니다!")
            print(f"📁 결과 폴더: {TRAINING_RESULTS_DIR}")
            
            return {
                'model_paths': model_paths,
                'evaluation': evaluation,
                'class_report': class_report,
                'confusion_matrix': conf_matrix,
                'training_info': self.training_info
            }
            
        except Exception as e:
            print(f"❌ 훈련 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """메인 실행 함수"""
    print("🚀 YAMNet + LSTM 모델 훈련기")
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
    
    # 데이터 경로 확인
    dataset_files = [f for f in os.listdir('.') if f.endswith('.npz')]
    
    if not dataset_files:
        print("❌ 데이터셋 파일(.npz)을 찾을 수 없습니다.")
        print("💡 먼저 data_generator.py를 실행하여 데이터를 생성해주세요.")
        return
    
    # 가장 최근 데이터셋 사용
    latest_dataset = max(dataset_files, key=os.path.getmtime)
    dataset_info_file = latest_dataset.replace('.npz', '.json')
    
    print(f"📂 사용할 데이터셋: {latest_dataset}")
    
    if os.path.exists(dataset_info_file):
        print(f"📄 데이터셋 정보: {dataset_info_file}")
    else:
        dataset_info_file = None
        print("⚠️ 데이터셋 정보 파일이 없습니다.")
    
    # 모델 이름 입력
    model_name = input(f"\n💾 모델 이름 (기본값: 자동생성): ").strip()
    if not model_name:
        model_name = None
    
    # 훈련 시작
    trainer = ModelTrainer()
    results = trainer.train_full_pipeline(
        data_path=latest_dataset,
        dataset_info_path=dataset_info_file,
        model_name=model_name
    )
    
    if results:
        print(f"\n✅ 훈련 성공!")
        # 안전한 정확도 접근
        evaluation = results['evaluation']
        accuracy = evaluation.get('accuracy', evaluation.get('val_accuracy', 0))
        print(f"🎯 최종 정확도: {accuracy:.4f}")
    else:
        print(f"\n❌ 훈련 실패")

if __name__ == "__main__":
    main()
