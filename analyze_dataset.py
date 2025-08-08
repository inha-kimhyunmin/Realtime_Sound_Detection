"""
데이터셋 분석 스크립트
====================

이 스크립트는 dataset_info_5class.json 파일을 분석하여 
훈련/검증/테스트 데이터셋에 사용된 오디오 파일들의 상세 정보를 제공합니다.
"""

import json
import os
import glob
from collections import Counter, defaultdict

def load_dataset_info(filename=None, version=None, folder=None):
    """
    데이터셋 정보 파일 로드
    
    Args:
        filename: 직접 파일명 지정 (우선순위 1)
        version: 버전별 자동 파일명 생성 (우선순위 2)
        folder: 특정 폴더에서 검색 (우선순위 3)
    """
    # 1. 직접 파일명이 지정된 경우
    if filename:
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f), filename
        else:
            print(f"❌ {filename} 파일을 찾을 수 없습니다.")
            return None, None
    
    # 2. 버전이 지정된 경우 - 최신 결과물 폴더에서 검색
    if version:
        pattern = f"model_results_{version}_*"
        matching_folders = glob.glob(pattern)
        if matching_folders:
            # 가장 최신 폴더 선택 (타임스탬프 기준)
            latest_folder = sorted(matching_folders)[-1]
            dataset_file = os.path.join(latest_folder, f'dataset_info_{version}.json')
            if os.path.exists(dataset_file):
                with open(dataset_file, 'r', encoding='utf-8') as f:
                    return json.load(f), dataset_file
            else:
                print(f"❌ {dataset_file} 파일을 찾을 수 없습니다.")
        else:
            print(f"❌ 버전 {version}에 해당하는 결과물 폴더를 찾을 수 없습니다.")
    
    # 3. 특정 폴더가 지정된 경우
    if folder:
        json_files = glob.glob(os.path.join(folder, "dataset_info_*.json"))
        if json_files:
            dataset_file = json_files[0]  # 첫 번째 파일 사용
            with open(dataset_file, 'r', encoding='utf-8') as f:
                return json.load(f), dataset_file
        else:
            print(f"❌ {folder} 폴더에서 dataset_info 파일을 찾을 수 없습니다.")
    
    # 4. 기본값: 현재 디렉토리에서 검색
    default_files = ['dataset_info_5class.json']  # 이전 버전 호환성
    json_files = glob.glob("dataset_info_*.json")
    all_files = default_files + json_files
    
    for file in all_files:
        if os.path.exists(file):
            with open(file, 'r', encoding='utf-8') as f:
                return json.load(f), file
    
    print("❌ 데이터셋 정보 파일을 찾을 수 없습니다.")
    print("사용 가능한 옵션:")
    print("  - analyze_dataset.py --version v1.0")
    print("  - analyze_dataset.py --folder model_results_v1.0_20250808_123456")
    print("  - analyze_dataset.py --file dataset_info_v1.0.json")
    return None, None

def analyze_dataset_distribution(dataset_info):
    """데이터셋 분포 분석"""
    class_names = ['무음', '정상(공장)', '화재', '가스누출', '비명']
    
    print("=" * 60)
    print("📊 데이터셋 분포 분석")
    print("=" * 60)
    
    for dataset_type in ['train', 'validation', 'test']:
        data = dataset_info[dataset_type]
        print(f"\n🔍 {dataset_type.upper()} 데이터셋 ({len(data)}개 샘플)")
        print("-" * 40)
        
        # 클래스별 개수
        class_counts = Counter([item['class_id'] for item in data])
        total = len(data)
        
        for class_id in range(5):
            count = class_counts.get(class_id, 0)
            percentage = (count / total * 100) if total > 0 else 0
            print(f"  {class_names[class_id]}: {count}개 ({percentage:.1f}%)")

def analyze_audio_files(dataset_info):
    """사용된 오디오 파일 분석"""
    print("\n" + "=" * 60)
    print("🎵 사용된 오디오 파일 분석")
    print("=" * 60)
    
    for dataset_type in ['train', 'validation', 'test']:
        data = dataset_info[dataset_type]
        print(f"\n📁 {dataset_type.upper()} 데이터셋")
        print("-" * 40)
        
        # 공장 소리 파일 분석
        factory_files = [item['factory_file'] for item in data if item['factory_file']]
        factory_counts = Counter(factory_files)
        
        print(f"🏭 공장 소리 파일 ({len(factory_counts)}개 고유 파일):")
        for filename, count in factory_counts.most_common(5):
            print(f"  {filename}: {count}번 사용")
        if len(factory_counts) > 5:
            print(f"  ... 그 외 {len(factory_counts) - 5}개 파일")
        
        # 위험 소리 파일 분석
        event_files = [item['event_file'] for item in data if item['event_file']]
        event_counts = Counter(event_files)
        
        print(f"\n⚠️ 위험 소리 파일 ({len(event_counts)}개 고유 파일):")
        for filename, count in event_counts.most_common(5):
            print(f"  {filename}: {count}번 사용")
        if len(event_counts) > 5:
            print(f"  ... 그 외 {len(event_counts) - 5}개 파일")
        
        # 클래스별 위험 소리 파일
        class_event_files = defaultdict(list)
        for item in data:
            if item['event_file'] and item['class'] != 'normal':
                class_event_files[item['class']].append(item['event_file'])
        
        print(f"\n📋 클래스별 위험 소리 파일:")
        for class_name, files in class_event_files.items():
            unique_files = len(set(files))
            total_usage = len(files)
            print(f"  {class_name}: {unique_files}개 고유 파일 (총 {total_usage}번 사용)")

def analyze_data_overlap(dataset_info):
    """데이터셋 간 파일 중복 분석"""
    print("\n" + "=" * 60)
    print("🔄 데이터셋 간 파일 중복 분석")
    print("=" * 60)
    
    # 각 데이터셋에서 사용된 파일들 수집
    datasets = {}
    for dataset_type in ['train', 'validation', 'test']:
        data = dataset_info[dataset_type]
        factory_files = set([item['factory_file'] for item in data if item['factory_file']])
        event_files = set([item['event_file'] for item in data if item['event_file']])
        datasets[dataset_type] = {
            'factory': factory_files,
            'event': event_files
        }
    
    # 중복 검사
    print("🏭 공장 소리 파일 중복:")
    train_factory = datasets['train']['factory']
    val_factory = datasets['validation']['factory']
    test_factory = datasets['test']['factory']
    
    print(f"  훈련-검증 중복: {len(train_factory & val_factory)}개")
    print(f"  훈련-테스트 중복: {len(train_factory & test_factory)}개")
    print(f"  검증-테스트 중복: {len(val_factory & test_factory)}개")
    
    print("\n⚠️ 위험 소리 파일 중복:")
    train_event = datasets['train']['event']
    val_event = datasets['validation']['event']
    test_event = datasets['test']['event']
    
    print(f"  훈련-검증 중복: {len(train_event & val_event)}개")
    print(f"  훈련-테스트 중복: {len(train_event & test_event)}개")
    print(f"  검증-테스트 중복: {len(val_event & test_event)}개")
    
    # 중복된 파일 목록 표시 (처음 3개만)
    overlaps = {
        '훈련-검증 공장파일': train_factory & val_factory,
        '훈련-테스트 공장파일': train_factory & test_factory,
        '훈련-검증 위험파일': train_event & val_event,
        '훈련-테스트 위험파일': train_event & test_event
    }
    
    for overlap_type, files in overlaps.items():
        if files:
            print(f"\n🔍 {overlap_type} 중복 파일 예시:")
            for i, filename in enumerate(list(files)[:3]):
                print(f"  {i+1}. {filename}")
            if len(files) > 3:
                print(f"  ... 총 {len(files)}개")

def generate_summary_report(dataset_info):
    """요약 보고서 생성"""
    print("\n" + "=" * 60)
    print("📋 데이터셋 요약 보고서")
    print("=" * 60)
    
    total_samples = sum(len(dataset_info[dt]) for dt in ['train', 'validation', 'test'])
    
    print(f"📊 전체 통계:")
    print(f"  총 샘플 수: {total_samples:,}개")
    print(f"  훈련 샘플: {len(dataset_info['train']):,}개 ({len(dataset_info['train'])/total_samples*100:.1f}%)")
    print(f"  검증 샘플: {len(dataset_info['validation']):,}개 ({len(dataset_info['validation'])/total_samples*100:.1f}%)")
    print(f"  테스트 샘플: {len(dataset_info['test']):,}개 ({len(dataset_info['test'])/total_samples*100:.1f}%)")
    
    # 전체 사용된 파일 수
    all_factory_files = set()
    all_event_files = set()
    
    for dataset_type in ['train', 'validation', 'test']:
        data = dataset_info[dataset_type]
        all_factory_files.update([item['factory_file'] for item in data if item['factory_file']])
        all_event_files.update([item['event_file'] for item in data if item['event_file']])
    
    print(f"\n🎵 사용된 오디오 파일:")
    print(f"  공장 소리 파일: {len(all_factory_files)}개")
    print(f"  위험 소리 파일: {len(all_event_files)}개")
    print(f"  총 고유 파일: {len(all_factory_files) + len(all_event_files)}개")

def main():
    """메인 실행 함수"""
    import sys
    
    # 명령행 인수 처리
    version = None
    folder = None
    filename = None
    
    if len(sys.argv) > 1:
        for i, arg in enumerate(sys.argv[1:], 1):
            if arg == '--version' and i + 1 < len(sys.argv):
                version = sys.argv[i + 1]
            elif arg == '--folder' and i + 1 < len(sys.argv):
                folder = sys.argv[i + 1]
            elif arg == '--file' and i + 1 < len(sys.argv):
                filename = sys.argv[i + 1]
    
    print("🔍 데이터셋 분석 시작...")
    
    # 데이터셋 정보 로드
    dataset_info, used_file = load_dataset_info(filename=filename, version=version, folder=folder)
    if dataset_info is None:
        return
    
    print(f"📂 사용된 파일: {used_file}")
    
    # 각종 분석 실행
    analyze_dataset_distribution(dataset_info)
    analyze_audio_files(dataset_info)
    analyze_data_overlap(dataset_info)
    generate_summary_report(dataset_info)
    
    print("\n✅ 데이터셋 분석 완료!")
    
    # 사용법 안내
    if not any([filename, version, folder]):
        print("\n💡 사용법:")
        print("  python analyze_dataset.py --version v1.0")
        print("  python analyze_dataset.py --folder model_results_v1.0_20250808_123456")
        print("  python analyze_dataset.py --file dataset_info_v1.0.json")

if __name__ == '__main__':
    main()
