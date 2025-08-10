"""
스크림 파일 분석 결과를 바탕으로 한 임계값 업데이트 결과

=== 분석 결과 요약 ===
총 12개의 높은 RMS 구간에서 분석한 결과:

📊 분석된 특성값들:
- RMS: 평균 0.2598 ± 0.0833
- Spectral Centroid: 평균 1789.9 ± 311.1 Hz  
- Spectral Flatness: 평균 0.0056 ± 0.0038
- Zero Crossing Rate: 평균 0.1407 ± 0.0195
- HNR: 평균 12.84 ± 7.50 dB

=== 기존 임계값 vs 실제 스크림 특성 ===

1. RMS 임계값: 
   - 기존: 0.02
   - 실제 스크림 평균: 0.2598
   → 기존 임계값이 너무 낮음

2. Spectral Centroid 임계값:
   - 기존: 3000Hz 이상
   - 실제 스크림 평균: 1789.9Hz
   → 기존 임계값이 너무 높음

3. Spectral Flatness 최대값:
   - 기존: 0.3 미만
   - 실제 스크림 평균: 0.0056
   → 스크림은 매우 낮은 flatness를 가짐 (톤이 뚜렷함)

4. Zero Crossing Rate:
   - 기존: 0.15 이상
   - 실제 스크림 평균: 0.1407
   → 기존 임계값이 적절함

5. HNR (Harmonic-to-Noise Ratio):
   - 기존: 10dB 이상
   - 실제 스크림 평균: 12.84dB
   → 기존 임계값이 적절함

=== 권장 업데이트 임계값 ===

보수적인 접근 (평균 - 1표준편차):
- RMS: 0.1764
- Centroid: 1478.8Hz (기존보다 훨씬 낮게)
- Flatness 최대: 0.0095 (기존보다 훨씬 낮게)
- ZCR: 0.1213
- HNR: 5.34dB

균형 잡힌 접근 (평균 - 0.5표준편차):
- RMS: 0.2182
- Centroid: 1634.4Hz  
- Flatness 최대: 0.0075
- ZCR: 0.1310
- HNR: 9.09dB
"""

# 업데이트된 임계값으로 기존 파일 수정
updated_scream_thresholds = {
    "rms": 0.18,           # 0.02에서 0.18로 대폭 상향 (실제 스크림은 훨씬 큰 RMS)
    "centroid": 1400,      # 3000에서 1400으로 대폭 하향 (실제 스크림은 더 낮은 주파수)
    "flatness_max": 0.01,  # 0.3에서 0.01로 대폭 하향 (스크림은 매우 톤적)
    "zcr": 0.12,           # 0.15에서 0.12로 약간 하향
    "hnr": 5               # 10에서 5로 하향 (다양한 스크림 포함)
}

print("🔧 권장 SCREAM_THRESHOLDS 업데이트:")
print("SCREAM_THRESHOLDS = {")
for key, value in updated_scream_thresholds.items():
    print(f'    "{key}": {value},')
print("}")
