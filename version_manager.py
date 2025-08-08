"""
모델 버전 관리 도구
==================

이 스크립트는 생성된 모델 결과물들을 관리하고 비교할 수 있는 도구입니다.
"""

import os
import json
import glob
from datetime import datetime

def list_model_versions():
    """생성된 모든 모델 버전을 나열합니다."""
    print("🔍 생성된 모델 버전 목록:")
    print("=" * 60)
    
    # model_results_* 패턴의 폴더 검색
    pattern = "model_results_*"
    folders = glob.glob(pattern)
    
    if not folders:
        print("❌ 생성된 모델 결과물 폴더가 없습니다.")
        return []
    
    versions_info = []
    
    for folder in sorted(folders):
        # 폴더명에서 버전과 타임스탬프 추출
        parts = folder.split('_')
        if len(parts) >= 4:
            version = parts[2]
            timestamp_str = f"{parts[3]}_{parts[4]}"
            
            # summary 파일에서 상세 정보 로드
            summary_pattern = os.path.join(folder, "summary_*.json")
            summary_files = glob.glob(summary_pattern)
            
            if summary_files:
                try:
                    with open(summary_files[0], 'r', encoding='utf-8') as f:
                        summary = json.load(f)
                    
                    versions_info.append({
                        'folder': folder,
                        'version': version,
                        'timestamp': timestamp_str,
                        'summary': summary
                    })
                    
                    # 정보 출력
                    print(f"📁 {folder}")
                    print(f"   버전: {version}")
                    print(f"   생성시간: {summary.get('timestamp', timestamp_str)}")
                    print(f"   테스트 정확도: {summary.get('performance', {}).get('test_accuracy', 'N/A'):.4f}")
                    print(f"   검증 정확도: {summary.get('performance', {}).get('validation_accuracy', 'N/A'):.4f}")
                    print(f"   총 샘플 수: {summary.get('data_summary', {}).get('total_samples', 'N/A')}")
                    print()
                    
                except Exception as e:
                    print(f"❌ {folder} 요약 정보 로드 실패: {e}")
            else:
                print(f"📁 {folder} (요약 정보 없음)")
                versions_info.append({
                    'folder': folder,
                    'version': version,
                    'timestamp': timestamp_str,
                    'summary': None
                })
    
    return versions_info

def compare_versions(version1, version2):
    """두 버전의 모델 성능을 비교합니다."""
    print(f"⚖️ 버전 비교: {version1} vs {version2}")
    print("=" * 60)
    
    # 각 버전의 최신 폴더 찾기
    def find_latest_folder(version):
        pattern = f"model_results_{version}_*"
        folders = glob.glob(pattern)
        return sorted(folders)[-1] if folders else None
    
    folder1 = find_latest_folder(version1)
    folder2 = find_latest_folder(version2)
    
    if not folder1:
        print(f"❌ 버전 {version1}의 결과물 폴더를 찾을 수 없습니다.")
        return
    
    if not folder2:
        print(f"❌ 버전 {version2}의 결과물 폴더를 찾을 수 없습니다.")
        return
    
    # 요약 정보 로드
    def load_summary(folder):
        summary_files = glob.glob(os.path.join(folder, "summary_*.json"))
        if summary_files:
            with open(summary_files[0], 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    summary1 = load_summary(folder1)
    summary2 = load_summary(folder2)
    
    if not summary1 or not summary2:
        print("❌ 요약 정보를 로드할 수 없습니다.")
        return
    
    # 성능 비교
    print("🎯 성능 비교:")
    print("-" * 40)
    
    perf1 = summary1.get('performance', {})
    perf2 = summary2.get('performance', {})
    
    metrics = [
        ('테스트 정확도', 'test_accuracy'),
        ('검증 정확도', 'validation_accuracy'),
        ('테스트 손실', 'test_loss'),
        ('검증 손실', 'validation_loss')
    ]
    
    for metric_name, metric_key in metrics:
        val1 = perf1.get(metric_key, 0)
        val2 = perf2.get(metric_key, 0)
        diff = val2 - val1
        
        if 'accuracy' in metric_key:
            diff_str = f"(+{diff:.4f})" if diff > 0 else f"({diff:.4f})"
            better = "🔺" if diff > 0 else "🔻" if diff < 0 else "➖"
        else:  # loss
            diff_str = f"(+{diff:.4f})" if diff > 0 else f"({diff:.4f})"
            better = "🔻" if diff > 0 else "🔺" if diff < 0 else "➖"
        
        print(f"  {metric_name}:")
        print(f"    {version1}: {val1:.4f}")
        print(f"    {version2}: {val2:.4f} {diff_str} {better}")
    
    # 데이터 비교
    print(f"\n📊 데이터 설정 비교:")
    print("-" * 40)
    
    data1 = summary1.get('data_summary', {})
    data2 = summary2.get('data_summary', {})
    
    data_fields = [
        ('총 샘플 수', 'total_samples'),
        ('훈련 샘플', 'train_samples'),
        ('무음 샘플', 'silence_samples'),
        ('정상 샘플', 'normal_samples'),
        ('자동 가중치', 'auto_weight_calculation')
    ]
    
    for field_name, field_key in data_fields:
        val1 = data1.get(field_key, 'N/A')
        val2 = data2.get(field_key, 'N/A')
        print(f"  {field_name}: {val1} → {val2}")
    
    # 위험 소음 가중치 비교
    weights1 = data1.get('danger_weights', {})
    weights2 = data2.get('danger_weights', {})
    
    if weights1 or weights2:
        print(f"\n⚠️ 위험 소음 가중치 비교:")
        print("-" * 40)
        all_classes = set(weights1.keys()) | set(weights2.keys())
        for class_name in sorted(all_classes):
            w1 = weights1.get(class_name, 0)
            w2 = weights2.get(class_name, 0)
            print(f"  {class_name}: {w1} → {w2}")

def show_version_details(version):
    """특정 버전의 상세 정보를 표시합니다."""
    print(f"📋 버전 {version} 상세 정보:")
    print("=" * 60)
    
    # 최신 폴더 찾기
    pattern = f"model_results_{version}_*"
    folders = glob.glob(pattern)
    
    if not folders:
        print(f"❌ 버전 {version}의 결과물 폴더를 찾을 수 없습니다.")
        return
    
    latest_folder = sorted(folders)[-1]
    print(f"📁 폴더: {latest_folder}")
    
    # 요약 정보 로드
    summary_files = glob.glob(os.path.join(latest_folder, "summary_*.json"))
    if not summary_files:
        print("❌ 요약 정보 파일을 찾을 수 없습니다.")
        return
    
    with open(summary_files[0], 'r', encoding='utf-8') as f:
        summary = json.load(f)
    
    # 기본 정보
    print(f"\n🕐 생성 시간: {summary.get('timestamp', 'N/A')}")
    print(f"📦 버전: {summary.get('version', 'N/A')}")
    
    # 성능 정보
    perf = summary.get('performance', {})
    print(f"\n🎯 성능:")
    print(f"  테스트 정확도: {perf.get('test_accuracy', 0):.4f}")
    print(f"  검증 정확도: {perf.get('validation_accuracy', 0):.4f}")
    print(f"  테스트 손실: {perf.get('test_loss', 0):.4f}")
    print(f"  검증 손실: {perf.get('validation_loss', 0):.4f}")
    
    # 데이터 정보
    data = summary.get('data_summary', {})
    print(f"\n📊 데이터:")
    print(f"  총 샘플: {data.get('total_samples', 0):,}개")
    print(f"  훈련 샘플: {data.get('train_samples', 0):,}개")
    print(f"  검증 샘플: {data.get('validation_samples', 0):,}개")
    print(f"  테스트 샘플: {data.get('test_samples', 0):,}개")
    print(f"  무음 샘플: {data.get('silence_samples', 0):,}개")
    print(f"  정상 샘플: {data.get('normal_samples', 0):,}개")
    print(f"  자동 가중치: {data.get('auto_weight_calculation', False)}")
    
    # 위험 소음 가중치
    weights = data.get('danger_weights', {})
    if weights:
        print(f"\n⚠️ 위험 소음 가중치:")
        for class_name, weight in weights.items():
            print(f"  {class_name}: {weight}")
    
    # 파일 목록
    files = summary.get('files', {})
    print(f"\n📄 포함된 파일:")
    for file_type, filename in files.items():
        file_path = os.path.join(latest_folder, filename)
        size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        size_mb = size / (1024 * 1024)
        print(f"  {filename} ({size_mb:.1f} MB)")

def cleanup_old_versions(keep_count=3):
    """오래된 버전들을 정리합니다."""
    print(f"🧹 오래된 버전 정리 (최신 {keep_count}개 유지):")
    print("=" * 60)
    
    # 버전별로 그룹화
    version_groups = {}
    pattern = "model_results_*"
    folders = glob.glob(pattern)
    
    for folder in folders:
        parts = folder.split('_')
        if len(parts) >= 4:
            version = parts[2]
            if version not in version_groups:
                version_groups[version] = []
            version_groups[version].append(folder)
    
    # 각 버전별로 정리
    for version, version_folders in version_groups.items():
        sorted_folders = sorted(version_folders)
        
        if len(sorted_folders) > keep_count:
            to_remove = sorted_folders[:-keep_count]
            print(f"\n📦 버전 {version}:")
            print(f"  총 {len(sorted_folders)}개 폴더, {len(to_remove)}개 제거 예정")
            
            for folder in to_remove:
                print(f"  ❌ 제거 대상: {folder}")
            
            # 실제 제거 (주석 해제하여 사용)
            # for folder in to_remove:
            #     import shutil
            #     shutil.rmtree(folder)
            #     print(f"  🗑️ 제거 완료: {folder}")
        else:
            print(f"📦 버전 {version}: {len(sorted_folders)}개 폴더 (정리 불필요)")
    
    print(f"\n💡 실제 정리를 수행하려면 cleanup_old_versions 함수의 주석을 해제하세요.")

def main():
    """메인 실행 함수"""
    import sys
    
    if len(sys.argv) < 2:
        print("🔧 모델 버전 관리 도구")
        print("=" * 40)
        print("사용법:")
        print("  python version_manager.py list                    # 버전 목록")
        print("  python version_manager.py details v1.0          # 버전 상세 정보")
        print("  python version_manager.py compare v1.0 v1.1     # 버전 비교")
        print("  python version_manager.py cleanup               # 오래된 버전 정리")
        return
    
    command = sys.argv[1]
    
    if command == 'list':
        list_model_versions()
    
    elif command == 'details':
        if len(sys.argv) < 3:
            print("❌ 버전을 지정해주세요. 예: python version_manager.py details v1.0")
            return
        version = sys.argv[2]
        show_version_details(version)
    
    elif command == 'compare':
        if len(sys.argv) < 4:
            print("❌ 두 버전을 지정해주세요. 예: python version_manager.py compare v1.0 v1.1")
            return
        version1 = sys.argv[2]
        version2 = sys.argv[3]
        compare_versions(version1, version2)
    
    elif command == 'cleanup':
        keep_count = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        cleanup_old_versions(keep_count)
    
    else:
        print(f"❌ 알 수 없는 명령어: {command}")
        print("사용 가능한 명령어: list, details, compare, cleanup")

if __name__ == '__main__':
    main()
