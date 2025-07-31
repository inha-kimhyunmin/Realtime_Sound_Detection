🔍 소개
이 프로젝트는 실시간 오디오 녹음을 통해 비명 소리 또는 폭발 소리를 rule-based 방식으로 감지하는 Python 프로그램입니다.
딥러닝을 사용하지 않고, 사람의 목소리 특성 및 폭발 소리의 물리적 특성을 기반으로 여러 오디오 파라미터를 계산해 소리를 분류합니다.

📦 설치 방법
bash
복사
편집
pip install sounddevice librosa numpy
⚠️ Windows 사용자는 pyaudio 대신 sounddevice 사용을 권장합니다.

🚀 실행 방법
bash
복사
편집
python realtime_scream_explosion_detector.py
프로그램이 시작되면 1초 단위로 소리를 녹음하고, 실시간으로 분석 결과를 터미널에 출력합니다.

⚙️ 주요 기능
🎤 실시간 오디오 녹음

🎚️ RMS → dB SPL 변환

📊 다양한 오디오 특성 추출 (ZCR, centroid, flatness, onset 등)

🔁 분석 지연 시간 표시

🕒 분석 시점 시간 출력

🚨 비명 / 💥 폭발 구분 감지

🧠 분석 기준 (Rule-based)
비명 소리 감지 기준
파라미터	기준값	설명
rms	> 0.02	음량 크기
centroid	> 3000 Hz	스펙트럼 중심 (고음 여부)
flatness	< 0.3	소음 비율 (낮을수록 명확한 목소리)
zcr	> 0.15	고주파 성분
hnr	> 10	음성의 조화도 (사람 목소리는 높음)

폭발 소리 감지 기준
파라미터	기준값	설명
rms	> 0.05	큰 음압
peak	> 0.3	순간 최대 진폭
flatness	> 0.4	폭발 소리는 평탄도가 높음
onset strength	> 0.3	급격한 에너지 변화
hnr	< 5	조화도 낮음 (비음성적 소리)

🧪 출력 예시
text
복사
편집
🎧 분석 결과
🕒 녹음 시각: 14:22:31
🎚️ RMS(dB SPL): 91.2 dB
🔁 분석 지연: 0.47초
📊 ZCR: 0.187, Centroid: 4150.6 Hz, Flatness: 0.220, HNR: 13.45
📈 Peak: 0.652, Onset Strength: 0.135
🚨 비명 소리 감지됨!
✏️ 파라미터 조정
파일 상단에서 SCREAM_THRESHOLDS, EXPLOSION_THRESHOLDS를 조정해 감지 민감도를 변경할 수 있습니다.

python
복사
편집
SCREAM_THRESHOLDS = {
    "rms": 0.02,
    ...
}

EXPLOSION_THRESHOLDS = {
    "rms": 0.05,
    ...
}
🧰 사용 기술
Python 3

librosa

sounddevice

numpy

📌 참고
RMS → dB SPL 변환은 기준 RMS(예: 0.1) 기준으로 94 + 20 * log10(RMS / ref)를 사용합니다.

HNR 계산은 librosa.pyin을 기반으로 합니다.

✅ 오디오 파라미터 설명
파라미터 이름	설명	비명 소리와의 관계	폭발 소리와의 관계
rms (Root Mean Square)	오디오 에너지의 평균값 (음압의 크기)	비명이 일반 말보다 세기 때문에 높음	폭발은 순간적으로 매우 높음
peak	순간 최대 음압 (절대값)	비명도 클 수 있음	폭발은 보통 피크가 매우 높음 (임펄스적)
zcr (Zero Crossing Rate)	오디오 신호가 0을 넘는 횟수 → 신호의 진동성	비명은 높은 고주파 성분 때문에 진동이 많음	폭발은 노이즈지만 지속적인 진동은 적음
centroid (Spectral Centroid)	주파수 에너지의 "중심 위치" → 고주파일수록 높음	고음 비명은 centroid가 높게 나옴	폭발도 넓게 퍼지긴 하지만 centroid는 중간 정도
flatness (Spectral Flatness)	주파수 분포가 얼마나 평평한지 (0=하모닉, 1=노이즈)	비명은 하모닉 구조라 낮은 편	폭발은 노이즈성이라 높음
bandwidth (Spectral Bandwidth)	주파수 분포의 폭 (Hz)	고주파 비명은 대역폭이 넓음	폭발도 전체 주파수에 걸쳐있어 넓음
rolloff (Spectral Rolloff)	에너지의 85~95%가 누적되는 주파수 경계	고주파가 많으면 rolloff가 높음	폭발은 고주파까지 포함되어 rolloff가 큼
onset (Onset Strength)	순간적인 에너지 변화 (음의 발생 강도)	보통 중간~높음	폭발은 매우 빠른 에너지 증가로 onset 값이 큼
hnr (Harmonic-to-Noise Ratio)	소리에 포함된 조화 성분의 비율 (높을수록 "사람 목소리" 느낌)	비명은 사람의 목소리 → HNR이 높음	폭발은 noise → HNR이 매우 낮음 (~0)

✅ 파라미터 간 직관적 비교
파라미터	비명 소리	폭발 소리
에너지 (rms)	높음	매우 높음
피크 (peak)	중간~높음	매우 높음
진동성 (zcr)	높음 (고음 성분)	중간
중심주파수 (centroid)	높음 (~4kHz)	중간 (~2kHz)
주파수 분산 (bandwidth)	넓음	매우 넓음
하모닉/노이즈 (flatness)	낮음 (하모닉)	높음 (노이즈)
고주파 포함 정도 (rolloff)	높음	매우 높음
에너지 변화 (onset)	빠르지만 일정	매우 급격함
조화도 (hnr)	높음 (1020dB)	낮음 (05dB)

✅ 왜 이 파라미터들이 중요한가?
비명은 기본적으로 사람 음성의 특성을 가지며,
비정상적으로 크고 고주파 성분이 풍부한 음성이야.
→ 따라서 HNR, centroid, bandwidth, ZCR 등이 핵심 기준이 돼.

폭발은 광대역 노이즈 + 임펄스 같은 특성이 있어서,
매우 짧고 강하며 주파수가 넓게 분포해 있음.
→ 따라서 flatness, onset, rolloff, peak, rms가 중요해.

✅ 예시: 비명 vs 폭발 특징
예시	HNR	Flatness	Onset	Centroid
여성 비명	15 dB	0.1	0.2	4500 Hz
폭발 소리	3 dB	0.5	0.5	2000 Hz