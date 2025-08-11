#!/usr/bin/env python3
"""
폴더 구조 및 파일 저장 테스트 스크립트
"""

import os
import sys
sys.path.append('.')

def test_folder_structure():
    """결과 폴더 구조 확인"""
    print("📁 폴더 구조 테스트")
    print("=" * 50)
    
    try:
        from config import (
            VERSION_DIR, TRAINING_RESULTS_DIR, EVALUATION_RESULTS_DIR,
            MODEL_SAVE_DIR, REPORT_SAVE_DIR, DATASET_SAVE_DIR
        )
        
        folders = {
            'VERSION_DIR': VERSION_DIR,
            'TRAINING_RESULTS_DIR': TRAINING_RESULTS_DIR,
            'EVALUATION_RESULTS_DIR': EVALUATION_RESULTS_DIR,
            'MODEL_SAVE_DIR': MODEL_SAVE_DIR,
            'REPORT_SAVE_DIR': REPORT_SAVE_DIR,
            'DATASET_SAVE_DIR': DATASET_SAVE_DIR
        }
        
        print("📋 설정된 폴더 경로:")
        for name, path in folders.items():
            exists = os.path.exists(path)
            status = "✅" if exists else "❌"
            print(f"  {status} {name}: {path}")
            
            if exists:
                try:
                    files = os.listdir(path)
                    print(f"      📄 파일 수: {len(files)}개")
                    if files:
                        print(f"      📝 예시: {', '.join(files[:3])}")
                        if len(files) > 3:
                            print(f"           ... 외 {len(files)-3}개")
                except PermissionError:
                    print(f"      ⚠️ 권한 없음")
                except Exception as e:
                    print(f"      ❌ 오류: {e}")
            else:
                print(f"      💡 폴더가 없습니다 (훈련/평가 실행 후 생성됨)")
                
        return True
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_path_creation():
    """경로 생성 테스트"""
    print(f"\n🔧 경로 생성 테스트")
    print("=" * 50)
    
    try:
        from config import create_output_directories
        
        print("📁 출력 디렉토리 생성 중...")
        create_output_directories()
        
        print("✅ 디렉토리 생성 완료!")
        
        # 다시 폴더 구조 확인
        test_folder_structure()
        
        return True
        
    except Exception as e:
        print(f"❌ 경로 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_existing_files():
    """기존 파일들 위치 확인"""
    print(f"\n🔍 기존 파일 위치 확인")
    print("=" * 50)
    
    try:
        from config import TRAINING_RESULTS_DIR, MODEL_SAVE_DIR
        
        # training 폴더의 모델 파일들 확인
        if os.path.exists(TRAINING_RESULTS_DIR):
            training_files = [f for f in os.listdir(TRAINING_RESULTS_DIR) 
                            if f.endswith(('.h5', '.keras', '.json', '.npy'))]
            
            print(f"📂 TRAINING 폴더 ({TRAINING_RESULTS_DIR}):")
            print(f"  📄 모델 관련 파일: {len(training_files)}개")
            for file in training_files:
                print(f"    - {file}")
        
        # models 폴더 확인
        if os.path.exists(MODEL_SAVE_DIR):
            model_files = os.listdir(MODEL_SAVE_DIR)
            print(f"\n📂 MODELS 폴더 ({MODEL_SAVE_DIR}):")
            print(f"  📄 파일: {len(model_files)}개")
            for file in model_files:
                print(f"    - {file}")
        else:
            print(f"\n📂 MODELS 폴더: 없음 ({MODEL_SAVE_DIR})")
            
        return True
        
    except Exception as e:
        print(f"❌ 파일 확인 실패: {e}")
        return False

def suggest_file_organization():
    """파일 정리 제안"""
    print(f"\n💡 파일 정리 제안")
    print("=" * 50)
    
    try:
        from config import TRAINING_RESULTS_DIR, MODEL_SAVE_DIR, EVALUATION_RESULTS_DIR, REPORT_SAVE_DIR
        import shutil
        
        moved_files = []
        
        # training 폴더의 모델 파일들을 models 폴더로 복사
        if os.path.exists(TRAINING_RESULTS_DIR):
            os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
            
            for file in os.listdir(TRAINING_RESULTS_DIR):
                if file.endswith(('.h5', '.keras', '.json', '.npy')):
                    src_path = os.path.join(TRAINING_RESULTS_DIR, file)
                    dst_path = os.path.join(MODEL_SAVE_DIR, file)
                    
                    if not os.path.exists(dst_path):
                        shutil.copy2(src_path, dst_path)
                        moved_files.append(f"📋 {file} → models/")
        
        # evaluation 폴더의 리포트를 reports 폴더로 복사
        if os.path.exists(EVALUATION_RESULTS_DIR):
            os.makedirs(REPORT_SAVE_DIR, exist_ok=True)
            
            for file in os.listdir(EVALUATION_RESULTS_DIR):
                if file.endswith(('.md', '.json', '.txt')):
                    src_path = os.path.join(EVALUATION_RESULTS_DIR, file)
                    dst_path = os.path.join(REPORT_SAVE_DIR, file)
                    
                    if not os.path.exists(dst_path):
                        shutil.copy2(src_path, dst_path)
                        moved_files.append(f"📋 {file} → reports/")
        
        if moved_files:
            print("✅ 파일 정리 완료:")
            for move in moved_files:
                print(f"  {move}")
        else:
            print("💡 정리할 파일이 없거나 이미 정리되어 있습니다.")
            
        return True
        
    except Exception as e:
        print(f"❌ 파일 정리 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """메인 테스트 함수"""
    print("🧪 Train 폴더 구조 및 파일 저장 테스트")
    print("=" * 60)
    
    # 1. 현재 폴더 구조 확인
    test_folder_structure()
    
    # 2. 경로 생성 테스트
    test_path_creation()
    
    # 3. 기존 파일 위치 확인
    check_existing_files()
    
    # 4. 파일 정리 제안
    suggest_file_organization()
    
    print(f"\n📋 요약:")
    print(f"  ✅ models/ 폴더: 모델 파일들의 복사본 저장")
    print(f"  ✅ reports/ 폴더: 평가 리포트들의 복사본 저장")
    print(f"  ✅ training/ 폴더: 원본 훈련 결과 (기존과 동일)")
    print(f"  ✅ evaluation/ 폴더: 원본 평가 결과 (기존과 동일)")
    
    print(f"\n🚀 이제 다음번 훈련/평가부터는 자동으로 models/와 reports/에도 저장됩니다!")

if __name__ == "__main__":
    main()
