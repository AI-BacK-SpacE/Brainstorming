# 1. Librosa
- **정의**
  -  **음악 및 오디오 신호 처리를 위한** 파이썬 라이브러리
  -  음악 분석, 오디오 신호 변환 및 기타 오디오 처리 작업을 수행하기 위한 다양한 기능을 제공
## 주요 기능
### 1) 오디오 파일 로드 및 저장
- 다양한 포맷의 오디오 파일(WAV, MP3 등) 로드 및 저장 가능
```py
# 가지고 있는 파일 업로드 경우
import librosa
y, sr = librosa.load('audio_file.wav', sr=22050)  # sr은 샘플링 레이트
librosa.output.write_wav('output.wav', y, sr)

# 내장 데이터 사용의 경우
filename = librosa.example('trumpet')  # trumpet 음원
y, sr = librosa.load(filename, sr=22050)
librosa.output.write_wav('output.wav', y, sr)
```
- **샘플링 레이트** : 아날로그 신호(예: 소리)를 디지털 신호로 변환할 때, 신호를 초당 몇 번 측정(샘플링)하는지를 나타내는 값
  - 일반적으로 **헤르츠(Hz)** 단위로 표현되며, **1초당 샘플링한 횟수를 의미**

### 2) 스펙트럼 분석
- 오디오 데이터를 스펙트럼(주파수 도메인)으로 변환
- 오디오 데이터를 주파수의 함수로 변환하는 과정
```py
D = librosa.stft(y)  # Short-Time Fourier Transform
S_db = librosa.amplitude_to_db(abs(D), ref=np.max)  # dB 변환
```
### 3) 특징 추출
```py
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
chroma = librosa.feature.chroma_stft(y=y, sr=sr)
```
