import librosa
import soundfile as sf
import numpy as np

def amplify_audio(input_file, output_file, gain_factor=2.0):
    """
    오디오 파일의 볼륨을 증폭
    
    Parameters:
    - input_file: 입력 오디오 파일 경로
    - output_file: 출력 오디오 파일 경로  
    - gain_factor: 증폭 배수 (2.0이면 2배 증폭)
    """
    # 오디오 파일 로드
    audio, sr = librosa.load(input_file, sr=None)
    
    # 볼륨 증폭
    amplified_audio = audio * gain_factor
    
    # 클리핑 방지 (최대값이 1.0을 넘지 않도록)
    max_val = np.max(np.abs(amplified_audio))
    if max_val > 1.0:
        amplified_audio = amplified_audio / max_val
    
    # 증폭된 오디오 저장
    sf.write(output_file, amplified_audio, sr)
    
    print(f"원본 파일: {input_file}")
    print(f"증폭 배수: {gain_factor}배")
    print(f"저장된 파일: {output_file}")

# 사용 예시
if __name__ == "__main__":
    input_file = "envsound/gas/GAS_LEAK.wav"  # 증폭할 파일
    output_file = "gas_amplified_3.0.wav"  # 저장할 파일명
    gain = 3.0  # 3배 증폭
    
    amplify_audio(input_file, output_file, gain)
