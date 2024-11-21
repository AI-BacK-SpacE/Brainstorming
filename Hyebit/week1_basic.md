# 1. Librosa
- **정의**
  -  **음악 및 오디오 신호 처리를 위한** 파이썬 라이브러리
  -  음악 분석, 오디오 신호 변환 및 기타 오디오 처리 작업을 수행하기 위한 다양한 기능을 제공
- [공식 GitHub](https://github.com/librosa/librosa)

## 1) 오디오 파일 로드
- 다양한 포맷의 오디오 파일(WAV, MP3 등) 로드 가능
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

## 2) 스펙트럼 분석
- 오디오 데이터를 스펙트럼(주파수 도메인)으로 변환해 시각화할 수 있음
```py
D = librosa.stft(y)  # Short-Time Fourier Transform
S_db = librosa.amplitude_to_db(abs(D), ref=np.max)  # dB 변환
```
## 3) 특징 추출
- Librosa는 MFCC, Chromagram 등 주요 음악적 특징을 쉽게 추출할 수 있음
```py
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
chroma = librosa.feature.chroma_stft(y=y, sr=sr)
```
### 그외 기타 등등
- 템포 및 비트 추출
- 오디오 변환
- 신호 분리 및 필터링

# 2. Aubio
- 오디오 및 음악 신호 분석을 위한 오픈 소스 라이브러리
- C 언어로 작성되었으며, 파이썬 모듈을 통해서도 사용가능
- [공식 메뉴얼](https://aubio.org/manual/latest/)
- [공식 GitHub](https://github.com/aubio/aubio)

## 1) 피치 추정(Pitch Detection)
- 입력 오디오 신호에서 주파수 기반의 피치 감지
```py
import aubio
import numpy as np

# 사인파 데이터 생성
samplerate = 44100
t = np.linspace(0, 1, samplerate, endpoint=False)
samples = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

# 피치 추정 (YIN 알고리즘)
win_s = 1024
hop_s = 512
pitch_o = aubio.pitch("yin", win_s, hop_s, samplerate) #-> 예시로 yin 알고리즘 사용
pitch_o.set_unit("Hz")
pitch_o.set_silence(-40)

# 피치 추정 결과
pitch = pitch_o(samples[:hop_s])[0]
print(f"추정된 피치: {pitch} Hz")
```
- 피치 추정을 위한 다양한 알고리즘 지원
  - yin
    - 신호의 주기성을 분석하여 피치를 감지하며, 낮은 왜곡과 높은 정확도 제공
    - 저주파 신호에 적합하며, 잡음에 강하지만 계산 비용이 다소 높음
    - 음악 신호의 안정적인 피치 추정 
  - yinfft
    - YIN 알고리즘의 주파수 도메인 변형 버전 (FFT를 사용하여 연산 속도 향상)
    - YIN보다 빠르지만, 정확도는 다소 낮을 수 있음
    - 성능과 정확도의 균형이 필요한 실시간 피치 추정에 적합 
  - schmitt
    - 신호를 특정 임계값으로 스레싱하여 주기성 감지
    - 다른 알고리즘보다 정확도가 낮지만 매우 빠르고 계산 비용이 낮음
    - 리소스가 제한된 시스템에 적합 
  - fcomb
    - 고조파(Harmonic)의 주파수 간격을 이용하여 기본 주파수를 추정
    - 하모닉 신호(음악)에 적합하지만, 잡음 환경에서는 성능이 저하될 수 있음
    - 하모닉이 뚜렷한 악기의 피치 추정 
  - mcomb
    - FComb와 유사하지만, 곱셈 기반 필터링을 사용하여 신호의 주기성을 강조
    - FComb보다 더 정밀하며, 하모닉 구조를 가진 신호에서 더 나은 결과 제공
    - 고정밀 음악 분석, 주기성이 명확한 신호 분석에 적합(특정 악기 등)  
      
| 알고리즘   | 정확도       | 속도       | 주요 특징                       | 사용 사례                         |
|------------|--------------|------------|----------------------------------|-----------------------------------|
| **YIN**    | 매우 높음    | 중간       | 저주파, 안정적                  | 음성, 음악의 기본 주파수 추정    |
| **YINFFT** | 높음         | 빠름       | FFT 기반, 실시간 처리 가능       | 실시간 피치 추정                  |
| **Schmitt**| 낮음         | 매우 빠름  | 간단한 시간 도메인 접근          | 단순한 실시간 피치 추정           |
| **FComb**  | 중간         | 빠름       | 주파수 도메인, 하모닉 분석        | 하모닉 악기의 피치 추정           |
| **MComb**  | 높음         | 중간       | 곱셈 기반 필터, 정밀한 하모닉 분석| 고정밀 음악 분석                  |


## 2) 비트 감지(Beat Detection)
- 음악 신호에서 비트(tempo)를 감지하여 박자 추출
```py
from aubio import tempo, source

filename = "example.wav"
samplerate = 44100
win_s = 512
hop_s = 256

audio_source = source(filename, samplerate, hop_s)
tempo_detector = tempo("default", win_s, hop_s, samplerate)

while True:
    samples, read = audio_source()
    if tempo_detector(samples):
        print("Beat detected")
    if read < hop_s:
        break
```

### 그외 기타
- 템포 추출(Tempo Extraction)
  - 음악의 BPM(Beat Per Minute)을 추출하여 곡의 속도 계산
- 오디오 분할(Audio Segmentation)
  - 오디오 신호를 에너지 변화 지점을 기준으로 구간별로 분할
- 특징 추출(Feature Extraction)
- 실시간 처리(Real-time Processing)
  - 오디오 신호의 실시간 분석 및 처리 기능 제공

# Librosa VS Aubio   

| **특징**         | **aubio**                                            | **librosa**                                        |
|-------------------|-----------------------------------------------------|---------------------------------------------------|
| **주요 목적**     | 실시간 오디오 분석 및 피치/템포 추출                | 음악 정보 및 오디오 분석, 데이터 과학 지원        |
| **특화 기능**     | 피치 감지, 템포 추출, 비트 감지, 오디오 분할        | 오디오 특징 추출 (MFCC, Chroma 등), 스펙트럼 분석 |
| **사용 편의성**   | 간결하고 경량화된 설계, 특정 작업에 최적화          | 다양한 오디오 처리 기능과 시각화 지원             |
| **데이터 입력**   | 스트리밍 및 파일 기반 데이터 처리 모두 지원          | 파일 기반 및 NumPy 배열 형태의 데이터 처리        |
| **주요 알고리즘** | YIN, YINFFT, Schmitt, FComb, MComb 등 제공          | 자체 알고리즘과 다양한 오디오 분석 툴 제공        |
| **MIDI 지원**     | 기본적으로 MIDI 지원 없음                           | 기본적으로 MIDI 지원 없음                         |
| **시각화 지원**   | 기본적으로 시각화 기능 없음 (matplotlib 연계 필요)   | 내장 시각화 기능 제공                             |
| **코드 기반**     | C 기반 경량화 Python 바인딩                         | Python 중심으로 구현                              |
| **적용 범위**     | 실시간 피치/템포 추출 및 음성 신호 처리에 최적화     | 음악 분석 및 연구, 데이터 분석에 적합             |

# 3. Pretty_midi
- Python에서 MIDI 데이터를 쉽게 처리하고 분석하기 위한 라이브러리

## 1) PrettyMIDI 클래스
- MIDI 파일을 로드하거나 새로운 MIDI 객체를 생성하는 클래스
```py
import pretty_midi

# MIDI 파일 로드
midi_data = pretty_midi.PrettyMIDI('example.mid')

# 템포 변화 확인
tempo_times, tempos = midi_data.get_tempo_changes()
print("Tempo changes at times:", tempo_times)
print("Tempos:", tempos)

# 평균 템포 추정
avg_tempo = midi_data.estimate_tempo()
print("Estimated Tempo:", avg_tempo)

# MIDI 파일 저장
midi_data.write('output_example.mid')
```

## 2) Instrument 클래스
- MIDI 파일에서 각 악기를 개별적으로 접근가능
```py
# 악기 정보 가져오기
for instrument in midi_data.instruments:
    print(f"Instrument Program: {instrument.program}") #-> .program : 악기 프로그램 번호 (0 = Acoustic Grand Piano)
    print(f"Is Drum: {instrument.is_drum}") #-> .is_drum : 드럼 채널 여부
    print(f"Number of Notes: {len(instrument.notes)}") #-> .notes : 악기에 포함된 노트의 리스트
```

## 3) Note 클래스
- MIDI 파일의 각 노트 정보를 출력하거나 새로운 노트를 추가가능
  - start: 노트 시작 시간 (초 단위)
  - end: 노트 종료 시간 (초 단위)
  - pitch: 노트의 음높이 (0–127)
  - velocity: 노트의 세기 (0–127)
```py
# MIDI 데이터에서 첫 번째 악기 가져오기
instrument = midi_data.instruments[0]

# 각 노트 정보 출력
for note in instrument.notes:
    print(f"Pitch: {note.pitch}, Start: {note.start}, End: {note.end}, Velocity: {note.velocity}")

# 새로운 노트 추가
note = pretty_midi.Note(velocity=100, pitch=60, start=0.5, end=1.0)  # C4
piano.notes.append(note)

# 저장
new_midi.write('new_note_example.mid')
```

## 4) get_chroma()
- 노트의 크로마(chroma) 특징을 반환
  - 크로마는 음높이를 12개의 반음(도, 도#, 레, ...)으로 매핑한 값
  - 주로 음악 분석에서 활용
```py
import librosa.display
import matplotlib.pyplot as plt

# MIDI 파일 로드
midi_data = pretty_midi.PrettyMIDI('example.mid')

# 크로마 특징 추출
chroma = midi_data.get_chroma()

# 시각화
plt.figure(figsize=(10, 4))
librosa.display.specshow(chroma, x_axis='time', y_axis='chroma', cmap='coolwarm')
plt.colorbar()
plt.title('Chroma Features')
plt.tight_layout()
plt.show()
```

## 5) Key Signature 및 템포 분석
- key_signature_changes: 키 변화 반환
- time_signature_changes: 박자 변화 반환
```py
# 키 시그니처 분석
for key_change in midi_data.key_signature_changes:
    print(f"Key: {key_change.key_number}, Time: {key_change.time}")

# 박자 분석
for time_signature in midi_data.time_signature_changes:
    print(f"Numerator: {time_signature.numerator}, Denominator: {time_signature.denominator}, Time: {time_signature.time}")
```
