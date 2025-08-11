"""
환경음 분류 모델 평가 모듈 (복원된 버전)
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow import keras
import librosa
from tqdm import tqdm
import json
from datetime import datetime
import warnings
import glob

import config
from config import CLASS_NAMES, MODEL_CONFIG, initialize_paths

# 초기화
initialize_paths()

# 초기화 후 변수들 임포트
from config import MODEL_SAVE_DIR, DATASET_SAVE_DIR, AUDIO_DATA_DIR, EVALUATION_RESULTS_DIR

warnings.filterwarnings('ignore')

# 영어 클래스명 리스트
CLASS_NAMES_EN_LIST = ['Silence', 'Factory', 'Fire', 'Gas Leak', 'Scream']

class ModelEvaluator:
    """모델 평가를 위한 클래스"""
    
    def __init__(self, model_path=None):
        """
        초기화
        Args:
            model_path: 평가할 모델의 경로
        """
        self.model = None
        self.model_path = model_path
        self.evaluation_results = {}
        
        # 한글 폰트 문제 해결을 위해 영어 클래스명 사용
        plt.rcParams['font.family'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
    def load_model(self, model_path=None):
        """모델 로드"""
        if model_path:
            self.model_path = model_path
            
        try:
            # 모델 로드 (여러 방법 시도)
            try:
                self.model = tf.keras.models.load_model(self.model_path)
            except AttributeError:
                # TensorFlow 버전이 다른 경우
                from tensorflow.keras.models import load_model
                self.model = load_model(self.model_path)
            print(f"✅ 모델 로드 완료: {self.model_path}")
            return True
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            return False
    
    def preprocess_audio_for_yamnet(self, audio_file):
        """YAMNet을 위한 오디오 전처리 (훈련 시와 동일한 방식)"""
        import tensorflow_hub as hub
        
        if not hasattr(self, 'yamnet_model'):
            self.yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
        
        # 오디오 로드
        audio, sr = librosa.load(audio_file, sr=16000)
        
        # 올바른 길이로 자르거나 패딩
        target_length = int(MODEL_CONFIG['audio_duration'] * 16000)
        if len(audio) > target_length:
            audio = audio[:target_length]
        else:
            audio = np.pad(audio, (0, target_length - len(audio)))
        
        # TensorFlow 텐서로 변환
        audio_tensor = tf.convert_to_tensor(audio, dtype=tf.float32)
        
        # YAMNet 특징 추출
        _, embeddings, _ = self.yamnet_model(audio_tensor)
        
        # 훈련 시와 동일하게 모든 임베딩 프레임 반환
        return embeddings.numpy()
    
    def create_test_dataset(self):
        """테스트 데이터셋 생성 (3-way split된 데이터 사용)"""
        print("🔍 테스트 데이터셋 생성 중...")
        
        # split_dataset_버전명_타임스탬프_test.npz 형식의 파일 찾기
        test_files = glob.glob(os.path.join(DATASET_SAVE_DIR, 'split_dataset_*_test.npz'))
        
        if test_files:
            # 가장 최근 파일 선택
            latest_test_file = max(test_files, key=os.path.getctime)
            print(f"📂 테스트 데이터 로드 중: {latest_test_file}")
            
            # npz 파일 로드
            test_data = np.load(latest_test_file)
            X_test = test_data['X']
            y_test = test_data['y']
            
            print(f"✅ 테스트 데이터 로드 완료: {X_test.shape}")
            return X_test, y_test
        
        print("❌ 미리 분할된 테스트 데이터를 찾을 수 없습니다.")
        print("💡 data_generator.py를 통해 3-way split 데이터를 먼저 생성하세요.")
        return None, None
    
    def evaluate_model(self, X_test, y_test):
        """모델 평가 수행"""
        if self.model is None:
            print("❌ 모델이 로드되지 않았습니다.")
            return None
        
        if X_test is None or y_test is None:
            print("❌ 테스트 데이터가 없습니다.")
            return None
        
        print("🔍 모델 평가 수행 중...")
        
        # 예측 수행
        predictions = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(predictions, axis=1)
        
        # 정확도 계산
        accuracy = accuracy_score(y_test, y_pred)
        
        # 분류 리포트
        report = classification_report(
            y_test, y_pred, 
            target_names=CLASS_NAMES_EN_LIST,
            output_dict=True
        )
        
        # 혼동 행렬
        cm = confusion_matrix(y_test, y_pred)
        
        self.evaluation_results = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': predictions,
            'y_true': y_test,
            'y_pred': y_pred
        }
        
        print(f"✅ 모델 평가 완료 - 정확도: {accuracy:.4f}")
        return self.evaluation_results
    
    def plot_confusion_matrix(self, save_path=None):
        """혼동 행렬 시각화 (영어 버전)"""
        if 'confusion_matrix' not in self.evaluation_results:
            print("❌ 평가 결과가 없습니다. 먼저 evaluate_model을 실행하세요.")
            return
        
        cm = self.evaluation_results['confusion_matrix']
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=CLASS_NAMES_EN_LIST,
            yticklabels=CLASS_NAMES_EN_LIST
        )
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 혼동 행렬 저장 완료: {save_path}")
        
    
    def plot_class_accuracy(self, save_path=None):
        """클래스별 정확도 시각화 (영어 버전)"""
        if 'classification_report' not in self.evaluation_results:
            print("❌ 평가 결과가 없습니다.")
            return
        
        report = self.evaluation_results['classification_report']
        
        # 클래스별 정확도 추출
        class_accuracies = []
        for class_name in CLASS_NAMES_EN_LIST:
            if class_name in report:
                class_accuracies.append(report[class_name]['f1-score'])
            else:
                class_accuracies.append(0)
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(CLASS_NAMES_EN_LIST, class_accuracies, 
                      color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
        
        plt.title('F1-Score by Class', fontsize=16, fontweight='bold')
        plt.xlabel('Class', fontsize=12)
        plt.ylabel('F1-Score', fontsize=12)
        plt.ylim(0, 1)
        
        # 막대 위에 값 표시
        for bar, accuracy in zip(bars, class_accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{accuracy:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 클래스별 정확도 저장 완료: {save_path}")
        
    
    def save_evaluation_report(self, save_dir=None):
        """평가 결과 저장"""
        if not self.evaluation_results:
            print("❌ 저장할 평가 결과가 없습니다.")
            return
        
        if save_dir is None:
            save_dir = EVALUATION_RESULTS_DIR
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(save_dir, f'evaluation_report_{timestamp}.json')
        
        # JSON 직렬화 가능한 형태로 변환
        report_data = {}
        
        for key, value in self.evaluation_results.items():
            if key == 'confusion_matrix':
                report_data[key] = value.tolist()
            elif key == 'predictions':
                report_data[key] = value.tolist()
            elif key in ['y_true', 'y_pred']:
                report_data[key] = value.tolist()
            else:
                report_data[key] = value
        
        # 모델 경로와 평가 시간 추가
        report_data['model_path'] = self.model_path
        report_data['evaluation_time'] = timestamp
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            print(f"📄 평가 보고서 저장 완료: {report_path}")
        except Exception as e:
            print(f"❌ 보고서 저장 실패: {e}")
    
    def print_summary(self):
        """평가 결과 요약 출력"""
        if not self.evaluation_results:
            print("❌ 평가 결과가 없습니다.")
            return
        
        print("\n" + "="*60)
        print("📊 모델 평가 결과 요약")
        print("="*60)
        
        if 'accuracy' in self.evaluation_results:
            print(f"🎯 전체 정확도: {self.evaluation_results['accuracy']:.4f}")
        
        if 'classification_report' in self.evaluation_results:
            report = self.evaluation_results['classification_report']
            print(f"📈 매크로 평균 F1-Score: {report['macro avg']['f1-score']:.4f}")
            print(f"📈 가중 평균 F1-Score: {report['weighted avg']['f1-score']:.4f}")
        
        print("="*60)

def find_latest_model(model_dir=None):
    """가장 최근 모델 파일 찾기"""
    if model_dir is None:
        model_dir = MODEL_SAVE_DIR
    
    # .h5 파일 검색
    model_files = glob.glob(os.path.join(model_dir, "*.h5"))
    
    if not model_files:
        # training 폴더에서도 검색
        training_dir = os.path.join(os.path.dirname(model_dir), 'training')
        if os.path.exists(training_dir):
            # 버전별 폴더에서 검색
            version_dirs = []
            try:
                dirs = os.listdir(training_dir)
                version_dirs = [d for d in dirs if d.startswith('v')]
            except:
                pass
                
            for version_dir in sorted(version_dirs, reverse=True):
                version_path = os.path.join(training_dir, version_dir)
                if os.path.isdir(version_path):
                    version_models = glob.glob(os.path.join(version_path, "*.h5"))
                    if version_models:
                        model_files.extend(version_models)
                        break  # 가장 최신 버전에서 찾으면 중단
    
    if not model_files:
        print(f"❌ 모델 파일을 찾을 수 없습니다: {model_dir}")
        return None
    
    # 가장 최근 파일 반환
    latest_model = max(model_files, key=os.path.getctime)
    print(f"📂 발견된 모델 파일: {latest_model}")
    return latest_model

def main():
    """메인 실행 함수"""
    print("🚀 모델 평가 시작")
    
    # 모델 경로 찾기
    model_path = find_latest_model()
    
    if not model_path:
        print("❌ 평가할 모델을 찾을 수 없습니다.")
        return
    
    # 평가기 초기화
    evaluator = ModelEvaluator(model_path)
    
    # 모델 로드
    if not evaluator.load_model():
        return
    
    try:
        # 테스트 데이터셋 생성
        X_test, y_test = evaluator.create_test_dataset()
        
        # 모델 평가
        results = evaluator.evaluate_model(X_test, y_test)
        
        if results:
            # 결과 출력
            evaluator.print_summary()
            
            # 평가 결과 저장을 위한 타임스탬프 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 시각화 이미지 저장 경로 설정
            confusion_matrix_path = os.path.join(EVALUATION_RESULTS_DIR, f'confusion_matrix_{timestamp}.png')
            f1_score_path = os.path.join(EVALUATION_RESULTS_DIR, f'f1_score_by_class_{timestamp}.png')
            
            # 시각화 (이미지 저장)
            evaluator.plot_confusion_matrix(save_path=confusion_matrix_path)
            evaluator.plot_class_accuracy(save_path=f1_score_path)
            
            # 보고서 저장
            evaluator.save_evaluation_report()
            
            print("✅ 모델 평가 완료!")
        
    except Exception as e:
        print(f"❌ 평가 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
