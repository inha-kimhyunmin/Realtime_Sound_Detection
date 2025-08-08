"""
가중치 계산 확인 프로그램 (간단 버전)
=====================================

자동 가중치 계산 결과와 동일한 수동 가중치 설정을 출력합니다.
"""

import os
import glob

def calculate_auto_weights(envsound_folder, target_samples_per_class=300):
    """
    각 클래스의 파일 개수를 기반으로 자동으로 가중치를 계산합니다.
    """
    event_folders = ['fire', 'gas', 'scream']
    file_counts = {}
    weights = {}
    
    # 각 폴더의 파일 개수 계산
    for folder in event_folders:
        folder_path = os.path.join(envsound_folder, folder)
        if os.path.exists(folder_path):
            wav_files = glob.glob(os.path.join(folder_path, '*.wav'))
            mp3_files = glob.glob(os.path.join(folder_path, '*.mp3'))
            file_counts[folder] = len(wav_files) + len(mp3_files)
        else:
            file_counts[folder] = 0
    
    # 가중치 계산
    for folder, file_count in file_counts.items():
        if file_count > 0:
            samples_per_file = target_samples_per_class / file_count
            if samples_per_file < 1:
                weights[folder] = round(samples_per_file, 3)
            else:
                weights[folder] = max(1, round(samples_per_file))
        else:
            weights[folder] = 0
    
    return weights, file_counts

def main():
    print("🔍 자동 가중치 → 수동 가중치 변환")
    print("=" * 50)
    
    envsound_folder = 'envsound'
    
    if not os.path.exists(envsound_folder):
        print(f"❌ envsound 폴더를 찾을 수 없습니다: {envsound_folder}")
        return
    
    # 자동 가중치 계산
    auto_weights, file_counts = calculate_auto_weights(envsound_folder, target_samples_per_class=300)
    
    print(f"📁 현재 파일 개수:")
    for folder, count in file_counts.items():
        print(f"   {folder}: {count}개")
    
    print(f"\n🤖 자동 계산된 가중치를 수동으로 설정하려면:")
    print("-" * 50)
    print("MANUAL_DANGER_WEIGHTS = {")
    
    total_samples = 0
    for class_name in ['fire', 'gas', 'scream']:
        weight = auto_weights[class_name]
        file_count = file_counts[class_name]
        
        if file_count > 0:
            expected_samples = file_count * weight
            total_samples += expected_samples
            
            if weight < 1:
                print(f"    '{class_name}': {weight},      # {weight*100:.1f}% 확률, 예상 {expected_samples:.1f}개")
            else:
                print(f"    '{class_name}': {weight},        # 파일당 {weight}개, 예상 {expected_samples:.0f}개")
        else:
            print(f"    '{class_name}': 0,         # 파일 없음")
    
    print("}")
    print(f"\n📊 총 예상 샘플: {total_samples:.1f}개")

if __name__ == '__main__':
    main()
