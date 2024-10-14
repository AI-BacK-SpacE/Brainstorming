# 오픈소스
## 1. [OMNIZART](https://github.com/Music-and-Culture-Technology-Lab/omnizart)
-
- 코드 구현
  ```python
  !pip install omnizart
  
  import omnizart
  
  # OMNIZART에서 음악 모델을 다운로드하여 초기화
  omnizart_chord = omnizart.chord.app
  
  # MP3 파일 업로드
  from google.colab import files
  uploaded = files.upload()
  
  # 파일 이름 가져오기
  file_name = next(iter(uploaded))
  
  # MP3 파일에서 코드 추출
  result = omnizart_chord.transcribe(file_name)
  
  # 결과 출력
  for entry in result:
      print(f"Time: {entry['start_time']} - {entry['end_time']}, Chord: {entry['chord']}")
  ``` 


## 2. [Essentia](https://github.com/MTG/essentia)
-
- 코드 구현
  ```python
  !pip install essentia
  
  import essentia.standard as es
  
  # MP3 파일 로드 및 분석
  loader = es.MonoLoader(filename='your_file.mp3')
  audio = loader()
  
  # 코드 추출 및 피치 감지
  pitch_extractor = es.PredominantPitchMelodia()
  pitch_values, pitch_confidence = pitch_extractor(audio)
  ``` 

## 3. [madmom](https://github.com/CPJKU/madmom)
-
- 코드 구현
  ```python
  !pip install madmom
    
  import madmom
  
  # MP3 파일 로드 및 분석
  proc = madmom.features.chords.CNNChordFeatureProcessor()
  chords = proc('your_file.mp3')
  
  print(chords)
  ``` 

# 가장 기본적 방법 (입문자 추천)
- Librosa + Aubio + pretty_midi
  - 음성 파일 업로드 및 피치 추출
  - 추출된 피치를 midi를 활용해 코드로 변환

```python
!pip install librosa
!pip install aubio
!pip install pretty_midi

import librosa
import numpy as np
import pretty_midi
import matplotlib.pyplot as plt
from aubio import pitch
from google.colab import files

# 파일 업로드
uploaded = files.upload()

# 파일 이름 가져오기 (첫 번째 파일)
file_name = next(iter(uploaded))

# 오디오 파일 로드 (Librosa 사용)
y, sr = librosa.load(file_name, sr=None)

# 피치 추출 함수 (Aubio 사용)
def get_pitch(y, sr):
    tolerance = 0.8
    win_s = 4096  # FFT window size
    hop_s = 512  # hop size

    # Aubio pitch detection
    pitch_o = pitch("yin", win_s, hop_s, sr)
    pitch_o.set_unit("Hz")
    pitch_o.set_tolerance(tolerance)

    pitches = []
    for i in range(0, len(y), hop_s):
        sample = y[i:i + hop_s]
        pitch_val = pitch_o(sample)[0]
        pitches.append(pitch_val if pitch_val > 0 else np.nan)

    return np.array(pitches)

# 피치 추출
pitches = get_pitch(y, sr)

# 피치 값을 MIDI 값으로 변환하여 코드로 변환하는 함수
def pitch_to_chord(pitches):
    # MIDI 값으로 변환
    midi_values = [pretty_midi.note_number_to_name(int(librosa.hz_to_midi(p))) 
                   for p in pitches if not np.isnan(p)]
    
    # 코드 변환 (여기서는 단순히 피치 기반으로 코드 추론, 더 복잡한 코드 분석 필요)
    unique_notes = list(set(midi_values))
    return unique_notes

# 피치에서 코드 추출
chords = pitch_to_chord(pitches)

# 코드 출력
print("Detected chords:", chords)

# 피치 시각화
plt.figure(figsize=(10, 6))
plt.plot(pitches, label='Pitch')
plt.xlabel('Frames')
plt.ylabel('Frequency (Hz)')
plt.title('Pitch over time')
plt.legend()
plt.show()

```
