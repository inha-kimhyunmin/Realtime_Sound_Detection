"""
YAMNet + LSTM 훈련 시스템 메인 실행기
====================================

이 스크립트는 모듈화된 훈련 시스템의 메인 인터페이스입니다.
- 데이터 생성, 모델 훈련, 평가를 통합 실행
- 각 단계별 개별 실행 가능
- 사용자 친화적 인터페이스 제공
"""

import os
import sys
import json
from datetime import datetime
from config import *

def print_header():
    """프로그램 헤더 출력"""
    print("=" * 80)
    print("🎵 YAMNet + LSTM 모듈형 훈련 시스템")
    print("=" * 80)
    print("📌 고급 데이터 생성 및 증강")
    print("🧠 딥러닝 모델 훈련 및 최적화")
    print("📊 종합적 성능 평가 및 분석")
    print("=" * 80)

def check_requirements():
    """시스템 요구사항 확인"""
    print("\n🔍 시스템 요구사항 확인 중...")
    
    # 필수 디렉토리 확인
    required_dirs = [ENVSOUND_DIR, MIXTURE_DIR]
    missing_dirs = []
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print("❌ 필수 디렉토리가 없습니다:")
        for dir_path in missing_dirs:
            print(f"  - {dir_path}")
        return False
    
    # 오디오 파일 확인
    total_files = 0
    for class_name in ACTIVE_DANGER_CLASSES:
        class_dir = os.path.join(ENVSOUND_DIR, class_name)
        if os.path.exists(class_dir):
            files = [f for f in os.listdir(class_dir) if f.endswith(('.wav', '.mp3', '.flac'))]
            total_files += len(files)
            print(f"  📁 {class_name}: {len(files)}개 파일")
    
    # 공장 소리 파일 확인
    if os.path.exists(MIXTURE_DIR):
        factory_files = [f for f in os.listdir(MIXTURE_DIR) if f.endswith(('.wav', '.mp3', '.flac'))]
        total_files += len(factory_files)
        print(f"  📁 factory: {len(factory_files)}개 파일")
    
    if total_files == 0:
        print("❌ 훈련용 오디오 파일이 없습니다.")
        return False
    
    print(f"✅ 총 {total_files}개 오디오 파일 확인됨")
    
    # 설정 검증
    config_errors = validate_config()
    if config_errors:
        print("❌ 설정 오류:")
        for error in config_errors:
            print(f"  - {error}")
        return False
    
    print("✅ 모든 요구사항 충족")
    return True

def show_menu():
    """메인 메뉴 표시"""
    print(f"\n📋 실행 옵션:")
    print(f"  1. 🏭 데이터 생성 (data_generator.py)")
    print(f"  2. 🧠 모델 훈련 (model_trainer.py)")
    print(f"  3. 📊 모델 평가 (evaluation.py)")
    print(f"  4. 🔄 전체 파이프라인 실행 (1→2→3)")
    print(f"  5. ⚙️ 설정 확인 및 수정")
    print(f"  6. 🎵 평가용 오디오 경로 설정")
    print(f"  7. 📁 결과 폴더 열기")
    print(f"  8. ❓ 도움말")
    print(f"  0. 🚪 종료")

def run_data_generation():
    """데이터 생성 실행"""
    print(f"\n🏭 데이터 생성 시작...")
    print("=" * 60)
    
    try:
        # data_generator 모듈 임포트 및 실행
        from data_generator import main as data_main
        result = data_main()
        
        if result and len(result) == 3:
            data_path, info_path, split_paths = result
            print(f"\n✅ 데이터 생성 완료!")
            print(f"📂 데이터 파일: {data_path}")
            print(f"📄 정보 파일: {info_path}")
            if split_paths:
                print(f"📊 분할 파일들: {len(split_paths)}개")
            return data_path, info_path
        else:
            print(f"\n❌ 데이터 생성 실패")
            return None, None
            
    except Exception as e:
        print(f"\n❌ 데이터 생성 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def run_model_training(data_path=None):
    """모델 훈련 실행"""
    print(f"\n🧠 모델 훈련 시작...")
    print("=" * 60)
    
    # 데이터 파일 확인
    if data_path is None:
        dataset_files = [f for f in os.listdir('.') if f.endswith('.npz')]
        if not dataset_files:
            print("❌ 훈련용 데이터셋이 없습니다.")
            print("💡 먼저 데이터 생성을 실행해주세요.")
            return None
        data_path = max(dataset_files, key=os.path.getmtime)
        print(f"📂 최신 데이터셋 사용: {data_path}")
    
    try:
        # model_trainer 모듈 임포트 및 실행
        from model_trainer import ModelTrainer
        
        trainer = ModelTrainer()
        
        # 데이터셋 정보 파일 찾기
        info_path = data_path.replace('.npz', '.json')
        if not os.path.exists(info_path):
            info_path = None
        
        # 모델 이름 입력받기
        print(f"\n💾 모델 저장 설정:")
        print(f"  💡 확장자(.h5 또는 .keras)가 없으면 자동으로 .h5가 추가됩니다")
        model_name = input(f"모델 이름 (기본값: 자동생성): ").strip()
        if not model_name:
            model_name = None
        
        # 훈련 실행
        results = trainer.train_full_pipeline(
            data_path=data_path,
            dataset_info_path=info_path,
            model_name=model_name
        )
        
        if results:
            model_paths = results['model_paths']
            accuracy = results['evaluation']['accuracy']
            
            print(f"\n✅ 모델 훈련 완료!")
            print(f"🎯 검증 정확도: {accuracy:.4f}")
            print(f"📂 모델 파일: {model_paths[0]}")
            
            # 실험 요약 저장
            from config import save_experiment_summary
            results_dict = {
                'model_paths': model_paths,
                'accuracy': accuracy,
                'evaluation_results': results['evaluation']
            }
            save_experiment_summary(results_dict)
            
            return model_paths[0]
        else:
            print(f"\n❌ 모델 훈련 실패")
            return None
            
    except Exception as e:
        print(f"\n❌ 모델 훈련 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_model_evaluation(model_path=None, custom_paths=None):
    """모델 평가 실행
    Args:
        model_path: 평가할 모델 파일 경로
        custom_paths: 사용자 정의 오디오 파일 경로 딕셔너리
    """
    print(f"\n📊 모델 평가 시작...")
    print("=" * 60)
    
    # 모델 파일 확인
    if model_path is None:
        model_files = []
        
        # 현재 디렉토리에서 모델 찾기
        for file in os.listdir('.'):
            if file.endswith('.h5'):
                model_files.append(file)
        
        # Training 결과 디렉토리에서도 찾기
        if os.path.exists(TRAINING_RESULTS_DIR):
            for file in os.listdir(TRAINING_RESULTS_DIR):
                if file.endswith('.h5'):
                    model_files.append(os.path.join(TRAINING_RESULTS_DIR, file))
        
        if not model_files:
            print("❌ 평가할 모델 파일이 없습니다.")
            print("💡 먼저 모델 훈련을 실행해주세요.")
            return None
        
        if len(model_files) == 1:
            model_path = model_files[0]
        else:
            print(f"\n📂 사용 가능한 모델:")
            for i, model_file in enumerate(model_files):
                print(f"  {i+1}. {os.path.basename(model_file)}")
            
            while True:
                try:
                    choice = input(f"\n모델 선택 (1-{len(model_files)}, 기본값: 1): ").strip()
                    if not choice:
                        model_path = model_files[0]
                        break
                    
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(model_files):
                        model_path = model_files[choice_idx]
                        break
                    else:
                        print(f"❌ 1-{len(model_files)} 범위의 숫자를 입력해주세요.")
                except ValueError:
                    print("❌ 숫자를 입력해주세요.")
    
    # 절대 경로로 변환
    if not os.path.isabs(model_path):
        model_path = os.path.abspath(model_path)
    
    print(f"📂 평가할 모델: {os.path.basename(model_path)}")
    
    try:
        # evaluation 모듈 임포트 및 실행
        from evaluation import ModelEvaluator
        
        evaluator = ModelEvaluator(model_path)
        
        # 사용자 정의 경로가 있으면 실제 오디오 테스트에서 사용
        if custom_paths:
            print(f"🎵 사용자 정의 경로로 실제 오디오 테스트 진행...")
            # custom_paths를 test_audio_dir 형태로 변환하거나 나중에 사용
        
        # 모델 로드
        if not evaluator.load_model():
            print(f"❌ 모델 로드 실패")
            return None
        
        # 테스트 데이터셋 생성
        X_test, y_test = evaluator.create_test_dataset()
        
        # 모델 평가
        results = evaluator.evaluate_model(X_test, y_test)
        
        if results:
            # 결과 출력
            evaluator.print_summary()
            
            # 시각화
            evaluator.plot_confusion_matrix()
            evaluator.plot_class_accuracy()
            
            # 실제 오디오 테스트
            evaluator.test_real_audio_files(num_samples=3)
            
            # 보고서 저장
            evaluator.save_evaluation_report()
            
            accuracy = results['accuracy']
            print(f"\n✅ 모델 평가 완료!")
            print(f"🎯 테스트 정확도: {accuracy:.4f}")
            
            if 'real_audio_test' in evaluator.evaluation_results:
                real_acc = evaluator.evaluation_results['real_audio_test']['overall_accuracy']
                print(f"🎵 실제 오디오 정확도: {real_acc:.4f}")
            
            print(f"📁 결과 폴더: {EVALUATION_RESULTS_DIR}")
            return results
        else:
            print(f"\n❌ 모델 평가 실패")
            return None
            
    except Exception as e:
        print(f"\n❌ 모델 평가 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_full_pipeline():
    """전체 파이프라인 실행"""
    print(f"\n🔄 전체 파이프라인 실행...")
    print("=" * 60)
    
    pipeline_start = datetime.now()
    results = {
        'data_generation': None,
        'model_training': None,
        'model_evaluation': None,
        'start_time': pipeline_start.isoformat(),
        'success': False
    }
    
    try:
        # 1. 데이터 생성
        print(f"\n📍 1단계: 데이터 생성")
        data_path, info_path = run_data_generation()
        
        if data_path is None:
            print(f"❌ 데이터 생성 실패 - 파이프라인 중단")
            return results
        
        results['data_generation'] = {
            'data_path': data_path,
            'info_path': info_path,
            'success': True
        }
        
        # 2. 모델 훈련
        print(f"\n📍 2단계: 모델 훈련")
        model_path = run_model_training(data_path)
        
        if model_path is None:
            print(f"❌ 모델 훈련 실패 - 파이프라인 중단")
            return results
        
        results['model_training'] = {
            'model_path': model_path,
            'success': True
        }
        
        # 3. 모델 평가
        print(f"\n📍 3단계: 모델 평가")
        evaluation_results = run_model_evaluation(model_path)
        
        if evaluation_results is None:
            print(f"❌ 모델 평가 실패")
            print(f"💡 해결 방법:")
            print(f"   1. 먼저 '1. 데이터 생성'을 실행하세요")
            print(f"   2. 그 다음 '2. 모델 훈련'을 실행하세요")
            print(f"   3. 마지막으로 '3. 모델 평가'를 실행하세요")
            
            # 평가 실패 정보 기록
            results['model_evaluation'] = {
                'success': False,
                'error': 'Dataset not found or evaluation failed'
            }
        else:
            results['model_evaluation'] = {
                'evaluation_results': evaluation_results,
                'success': True
            }
        
        # 파이프라인 완료
        pipeline_end = datetime.now()
        pipeline_duration = pipeline_end - pipeline_start
        
        results['end_time'] = pipeline_end.isoformat()
        results['duration'] = {
            'total_seconds': pipeline_duration.total_seconds(),
            'formatted': str(pipeline_duration)
        }
        results['success'] = True
        
        # 결과 저장
        from config import save_experiment_summary
        # datetime 객체는 JSON 직렬화 불가능하므로 복사본에서 처리
        results_copy = results.copy()
        
        # 안전한 model_evaluation 처리
        if ('model_evaluation' in results_copy and 
            results_copy['model_evaluation'] is not None and 
            isinstance(results_copy['model_evaluation'], dict) and
            results_copy['model_evaluation'].get('evaluation_results')):
            
            # 복잡한 numpy 배열 등은 제거
            eval_results = results_copy['model_evaluation']['evaluation_results']
            if isinstance(eval_results, dict) and 'test_results' in eval_results:
                eval_copy = eval_results.copy()
                if 'test_results' in eval_copy and isinstance(eval_copy['test_results'], dict):
                    eval_copy['test_results'] = {
                        'overall_metrics': eval_copy['test_results'].get('overall_metrics', {})
                    }
                results_copy['model_evaluation']['evaluation_results'] = eval_copy
        
        save_experiment_summary(results_copy)
        
        print(f"\n🎉 전체 파이프라인 완료!")
        print(f"⏱️ 총 소요 시간: {pipeline_duration}")
        print(f"� 실험 요약이 저장되었습니다.")
        
        # 최종 성능 요약
        model_eval = results.get('model_evaluation')
        if (model_eval and 
            isinstance(model_eval, dict) and 
            model_eval.get('success') and 
            model_eval.get('evaluation_results')):
            
            eval_results = model_eval['evaluation_results']
            if (isinstance(eval_results, dict) and 
                'test_results' in eval_results and 
                isinstance(eval_results['test_results'], dict)):
                
                test_results = eval_results['test_results']
                overall_metrics = test_results.get('overall_metrics', {})
                
                if 'accuracy' in overall_metrics:
                    test_acc = overall_metrics['accuracy']
                    print(f"\n📊 최종 성능 요약:")
                    print(f"🎯 테스트 정확도: {test_acc:.4f}")
                
                # 실제 오디오 테스트 결과 확인
                evaluation_results = eval_results.get('evaluation_results', {})
                if (isinstance(evaluation_results, dict) and 
                    'real_audio_test' in evaluation_results):
                    
                    real_audio_test = evaluation_results['real_audio_test']
                    if isinstance(real_audio_test, dict) and 'overall_accuracy' in real_audio_test:
                        real_acc = real_audio_test['overall_accuracy']
                        print(f"🎵 실제 오디오 정확도: {real_acc:.4f}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ 파이프라인 실행 중 오류: {e}")
        import traceback
        traceback.print_exc()
        
        pipeline_end = datetime.now()
        results['end_time'] = pipeline_end.isoformat()
        results['error'] = str(e)
        return results

def show_config():
    """설정 정보 표시 및 수정"""
    global ACTIVE_DANGER_CLASSES, ALL_CLASSES, NUM_CLASSES, DATA_GENERATION_CONFIG, TRAINING_CONFIG
    
    print(f"\n⚙️ 현재 설정:")
    print("=" * 60)
    
    print(f"📁 데이터 경로:")
    print(f"  - 환경음 디렉토리: {ENVSOUND_DIR}")
    print(f"  - 공장음 디렉토리: {MIXTURE_DIR}")
    
    print(f"\n🎵 오디오 설정:")
    print(f"  - 샘플링 레이트: {MODEL_CONFIG['sample_rate']} Hz")
    print(f"  - 오디오 길이: {MODEL_CONFIG['audio_duration']}초")
    print(f"  - 프레임당 시간: 0.48초 (YAMNet 고정)")
    
    print(f"\n🧠 모델 설정:")
    print(f"  - 클래스 수: {NUM_CLASSES}개")
    print(f"  - 활성 위험 클래스: {', '.join(ACTIVE_DANGER_CLASSES)}")
    
    print(f"\n🏭 데이터 생성 설정:")
    print(f"  - 클래스당 목표 프레임: {DATA_GENERATION_CONFIG['target_frames_per_class']:,}개")
    print(f"  - 전환 데이터 비율: {DATA_GENERATION_CONFIG['transition_data_ratio']:.1%}")
    
    print(f"\n🎯 훈련 설정:")
    print(f"  - 에포크: {TRAINING_CONFIG['epochs']}")
    print(f"  - 배치 크기: {TRAINING_CONFIG['batch_size']}")
    print(f"  - 학습률: {TRAINING_CONFIG['learning_rate']}")
    
    # 평가용 경로 설정 출력 추가
    print(f"\n📂 평가용 오디오 경로:")
    try:
        import config
        config.print_evaluation_paths()
    except Exception as e:
        print(f"⚠️ 평가 경로 정보를 가져올 수 없습니다: {e}")

def configure_evaluation_paths():
    """평가용 오디오 경로 설정"""
    
    print(f"\n🎵 평가용 오디오 경로 설정")
    
    try:
        import config
        print(f"📋 현재 클래스: {config.CLASS_NAMES}")
        
        # 현재 설정 출력
        config.print_evaluation_paths()
        
        print(f"\n설정 옵션:")
        print(f"  1. 현재 설정 그대로 사용")
        print(f"  2. 특정 클래스 경로 변경")
        print(f"  3. 모든 경로를 같은 폴더로 설정")
        
        choice = input("\n선택 (1-3): ").strip()
        
        custom_paths = None
        
        if choice == '1':
            print("✅ 현재 설정을 사용합니다.")
            
        elif choice == '2':
            custom_paths = config.EVALUATION_AUDIO_PATHS.copy()
            
            while True:
                print(f"\n변경할 클래스를 선택하세요:")
                class_names_list = list(config.CLASS_NAMES.keys())
                for i, class_name in enumerate(class_names_list, 1):
                    if class_name == 'silence':
                        continue
                    current_path = custom_paths.get(class_name, 'None')
                    print(f"  {i}. {class_name}: {current_path}")
                print(f"  0. 설정 완료")
                
                class_choice = input("\n클래스 번호: ").strip()
                
                if class_choice == '0':
                    break
                    
                try:
                    class_idx = int(class_choice) - 1
                    class_names_list = list(config.CLASS_NAMES.keys())
                    if 0 <= class_idx < len(class_names_list):
                        selected_class = class_names_list[class_idx]
                        if selected_class == 'silence':
                            print("⚠️ silence 클래스는 설정할 수 없습니다.")
                            continue
                            
                        new_path = input(f"\n{selected_class} 클래스의 새 경로: ").strip()
                        if os.path.exists(new_path):
                            custom_paths[selected_class] = new_path
                            print(f"✅ {selected_class} 경로 업데이트: {new_path}")
                        else:
                            print(f"❌ 경로가 존재하지 않습니다: {new_path}")
                    else:
                        print("❌ 잘못된 번호입니다.")
                except ValueError:
                    print("❌ 숫자를 입력하세요.")
                    
        elif choice == '3':
            folder_path = input("\n모든 클래스가 들어있는 폴더 경로: ").strip()
            if os.path.exists(folder_path):
                custom_paths = {}
                for class_name in config.CLASS_NAMES:
                    if class_name == 'silence':
                        continue
                    class_folder = os.path.join(folder_path, class_name)
                    if os.path.exists(class_folder):
                        custom_paths[class_name] = class_folder
                        print(f"✅ {class_name}: {class_folder}")
                    else:
                        print(f"⚠️ {class_name} 폴더가 없습니다: {class_folder}")
            else:
                print(f"❌ 폴더가 존재하지 않습니다: {folder_path}")
        else:
            print("❌ 잘못된 선택입니다.")
        
        return custom_paths
        
    except Exception as e:
        print(f"❌ 설정 중 오류 발생: {e}")
        return None
    print(f"  - 검증 분할: {TRAINING_CONFIG['validation_split']:.1%}")
    
    # 설정 수정 옵션
    print(f"\n🔧 설정 수정 옵션:")
    print(f"  1. 목표 프레임 수 변경")
    print(f"  2. 훈련 에포크 변경")
    print(f"  3. 배치 크기 변경")
    print(f"  4. 활성 위험 클래스 변경")
    print(f"  0. 돌아가기")
    
    while True:
        try:
            choice = input(f"\n선택 (0-4): ").strip()
            
            if choice == '0':
                break
            elif choice == '1':
                new_frames = int(input(f"새로운 목표 프레임 수 (현재: {DATA_GENERATION_CONFIG['target_frames_per_class']}): "))
                if new_frames > 0:
                    DATA_GENERATION_CONFIG['target_frames_per_class'] = new_frames
                    print(f"✅ 목표 프레임 수가 {new_frames}개로 변경되었습니다.")
            elif choice == '2':
                new_epochs = int(input(f"새로운 에포크 수 (현재: {TRAINING_CONFIG['epochs']}): "))
                if new_epochs > 0:
                    TRAINING_CONFIG['epochs'] = new_epochs
                    print(f"✅ 에포크 수가 {new_epochs}개로 변경되었습니다.")
            elif choice == '3':
                new_batch = int(input(f"새로운 배치 크기 (현재: {TRAINING_CONFIG['batch_size']}): "))
                if new_batch > 0:
                    TRAINING_CONFIG['batch_size'] = new_batch
                    print(f"✅ 배치 크기가 {new_batch}개로 변경되었습니다.")
            elif choice == '4':
                print(f"사용 가능한 위험 클래스: {', '.join(['fire', 'gas', 'scream', 'spark'])}")
                new_classes = input(f"활성 위험 클래스 (쉼표로 구분): ").strip().split(',')
                new_classes = [cls.strip() for cls in new_classes if cls.strip()]
                
                valid_classes = [cls for cls in new_classes if cls in ['fire', 'gas', 'scream', 'spark']]
                if valid_classes:
                    # 전역 변수 수정 (이미 함수 시작에서 global 선언됨)
                    ACTIVE_DANGER_CLASSES = valid_classes
                    ALL_CLASSES = ['silence', 'factory'] + ACTIVE_DANGER_CLASSES
                    NUM_CLASSES = len(ALL_CLASSES)
                    print(f"✅ 활성 위험 클래스가 변경되었습니다: {', '.join(valid_classes)}")
                else:
                    print(f"❌ 유효한 클래스가 없습니다.")
            else:
                print(f"❌ 0-4 범위의 숫자를 입력해주세요.")
                
        except ValueError:
            print(f"❌ 유효한 숫자를 입력해주세요.")
        except KeyboardInterrupt:
            break

def open_results_folder():
    """결과 폴더 열기"""
    import platform
    import subprocess
    
    folders = [TRAINING_RESULTS_DIR, EVALUATION_RESULTS_DIR]
    existing_folders = [folder for folder in folders if os.path.exists(folder)]
    
    if not existing_folders:
        print(f"❌ 결과 폴더가 없습니다.")
        print(f"💡 먼저 훈련이나 평가를 실행해주세요.")
        return
    
    print(f"\n📁 사용 가능한 결과 폴더:")
    for i, folder in enumerate(existing_folders):
        print(f"  {i+1}. {folder}")
    
    if len(existing_folders) == 1:
        folder_to_open = existing_folders[0]
    else:
        while True:
            try:
                choice = input(f"\n폴더 선택 (1-{len(existing_folders)}, 기본값: 1): ").strip()
                if not choice:
                    folder_to_open = existing_folders[0]
                    break
                
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(existing_folders):
                    folder_to_open = existing_folders[choice_idx]
                    break
                else:
                    print(f"❌ 1-{len(existing_folders)} 범위의 숫자를 입력해주세요.")
            except ValueError:
                print(f"❌ 숫자를 입력해주세요.")
    
    try:
        system = platform.system()
        if system == "Windows":
            os.startfile(folder_to_open)
        elif system == "Darwin":  # macOS
            subprocess.run(["open", folder_to_open])
        else:  # Linux
            subprocess.run(["xdg-open", folder_to_open])
        
        print(f"✅ 폴더 열기: {folder_to_open}")
        
    except Exception as e:
        print(f"❌ 폴더 열기 실패: {e}")
        print(f"📁 수동으로 열어주세요: {folder_to_open}")

def show_help():
    """도움말 표시"""
    print(f"\n❓ 도움말")
    print("=" * 60)
    
    print(f"🎯 시스템 개요:")
    print(f"이 시스템은 YAMNet + LSTM을 사용한 환경음 분류 모델을 훈련합니다.")
    print(f"모듈화된 구조로 데이터 생성, 훈련, 평가가 분리되어 있습니다.")
    
    print(f"\n📋 실행 단계:")
    print(f"1. 🏭 데이터 생성:")
    print(f"   - 각 클래스별 균등한 프레임 수 생성")
    print(f"   - 데이터 증강을 통한 데이터 부족 해결")
    print(f"   - 다양한 전환 시나리오 데이터 생성")
    
    print(f"\n2. 🧠 모델 훈련:")
    print(f"   - YAMNet 임베딩 + LSTM 모델 훈련")
    print(f"   - 클래스 가중치 적용으로 불균형 해결")
    print(f"   - 체크포인트 저장 및 조기 종료")
    
    print(f"\n3. 📊 모델 평가:")
    print(f"   - 테스트 데이터셋 자동 생성")
    print(f"   - 상세한 성능 분석 및 시각화")
    print(f"   - 실제 오디오 파일 테스트")
    
    print(f"\n📁 필수 폴더 구조:")
    print(f"  {ENVSOUND_DIR}/")
    for class_name in ACTIVE_DANGER_CLASSES:
        print(f"    {class_name}/    # {CLASS_NAMES.get(class_name, class_name)} 오디오 파일들")
    print(f"  {MIXTURE_DIR}/       # 공장소리 오디오 파일들")
    
    print(f"\n🎵 지원 오디오 형식:")
    print(f"  - WAV, MP3, FLAC")
    print(f"  - 권장: 16kHz, 모노")
    
    print(f"\n⚙️ 주요 설정:")
    print(f"  - 오디오 길이: {MODEL_CONFIG['audio_duration']}초")
    print(f"  - 목표 프레임/클래스: {DATA_GENERATION_CONFIG['target_frames_per_class']:,}개")
    print(f"  - 훈련 에포크: {TRAINING_CONFIG['epochs']}개")
    
    print(f"\n🔧 문제 해결:")
    print(f"  - 메모리 부족: 배치 크기나 목표 프레임 수 감소")
    print(f"  - 훈련 시간 과다: 에포크 수 감소")
    print(f"  - 낮은 정확도: 데이터 증강 강화, 더 많은 데이터")

def main():
    """메인 함수"""
    print_header()
    
    # 시스템 요구사항 확인
    if not check_requirements():
        print(f"\n❌ 시스템 요구사항을 충족하지 않습니다.")
        print(f"💡 필요한 파일과 폴더를 준비한 후 다시 실행해주세요.")
        return
    
    # 출력 디렉토리 생성
    create_output_directories()
    
    # 메인 루프
    while True:
        try:
            show_menu()
            choice = input(f"\n선택하세요 (0-8): ").strip()
            
            if choice == '0':
                print(f"\n👋 YAMNet + LSTM 훈련 시스템을 종료합니다.")
                break
                
            elif choice == '1':
                run_data_generation()
                
            elif choice == '2':
                run_model_training()
                
            elif choice == '3':
                run_model_evaluation()
                
            elif choice == '4':
                run_full_pipeline()
                
            elif choice == '5':
                show_config()
                
            elif choice == '6':
                custom_paths = configure_evaluation_paths()
                if custom_paths:
                    print(f"✅ 평가용 경로가 설정되었습니다.")
                
            elif choice == '7':
                open_results_folder()
                
            elif choice == '8':
                show_help()
                
            else:
                print(f"❌ 0-8 범위의 숫자를 입력해주세요.")
            
            # 계속 진행
            if choice in ['1', '2', '3', '4', '6']:
                input(f"\n⏸️ 계속하려면 Enter를 누르세요...")
                
        except KeyboardInterrupt:
            print(f"\n\n👋 사용자가 중단했습니다.")
            break
        except Exception as e:
            print(f"\n❌ 예상치 못한 오류: {e}")
            import traceback
            traceback.print_exc()
            
            input(f"\n⏸️ 계속하려면 Enter를 누르세요...")

if __name__ == "__main__":
    main()
