"""
필수 패키지 설치 스크립트
=======================

YAMNet + LSTM 훈련 시스템에 필요한 모든 패키지를 설치합니다.
"""

import subprocess
import sys

# 필수 패키지 목록
REQUIRED_PACKAGES = [
    'tensorflow>=2.8.0',
    'tensorflow-hub>=0.12.0',
    'librosa>=0.9.0',
    'scikit-learn>=1.0.0',
    'matplotlib>=3.5.0',
    'seaborn>=0.11.0',
    'tqdm>=4.60.0',
    'pandas>=1.3.0',
    'numpy>=1.21.0',
    'scipy>=1.7.0',
    'soundfile>=0.10.0'
]

def install_package(package):
    """패키지 설치"""
    try:
        print(f"📦 {package} 설치 중...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        print(f"✅ {package} 설치 완료")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {package} 설치 실패: {e}")
        return False

def main():
    """메인 설치 함수"""
    print("🚀 YAMNet + LSTM 훈련 시스템 패키지 설치")
    print("=" * 60)
    
    success_count = 0
    failed_packages = []
    
    for package in REQUIRED_PACKAGES:
        if install_package(package):
            success_count += 1
        else:
            failed_packages.append(package)
    
    print("\n" + "=" * 60)
    print(f"📊 설치 결과:")
    print(f"✅ 성공: {success_count}/{len(REQUIRED_PACKAGES)}개")
    
    if failed_packages:
        print(f"❌ 실패: {len(failed_packages)}개")
        print("실패한 패키지:")
        for package in failed_packages:
            print(f"  - {package}")
    else:
        print("🎉 모든 패키지 설치 완료!")
    
    return len(failed_packages) == 0

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n💡 일부 패키지 설치에 실패했습니다.")
        print("   수동으로 설치하거나 권한을 확인해주세요.")
        sys.exit(1)
