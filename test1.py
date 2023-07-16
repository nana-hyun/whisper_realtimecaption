# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 12:16:57 2023

@author: Nahyun
"""

import soundcard as sc
import time
import threading
import queue
import collections
import pyaudio
import easydict
import torch
import numpy as np
import whisper
from halo import Halo
import webrtcvad
import openai
import warnings

warnings.filterwarnings("ignore")


SAMPLE_RATE = 16000

openai.api_key = " "

class LoopbackAudio(threading.Thread):
    def __init__(self, callback, device, samplerate=SAMPLE_RATE):
        threading.Thread.__init__(self)
        self.callback = callback
        self.samplerate = samplerate
        self.mics = sc.all_microphones(include_loopback=True)
        self.mic_index = device
        self.stop_event = threading.Event()

    def run(self):
        if self.mic_index == None:
            mic = sc.default_microphone()
        else:
            mic = self.mics[self.mic_index]
        with mic.recorder(samplerate=self.samplerate) as recorder:
            while not self.stop_event.is_set():
                data = recorder.record(numframes=640)
                self.callback(data)

    def stop(self):
        self.stop_event.set()

class Audio(object):
    """Streams raw audio from microphone. Data is received in a separate thread, and stored in a buffer, to be read from."""

    RATE_PROCESS = 16000
    CHANNELS = 1
    BLOCKS_PER_SECOND = 50
    
    def __init__(self, callback=None, device=None, input_rate=RATE_PROCESS):
        def proxy_callback(in_data):
            callback(in_data)

        if callback is None:
            def callback(in_data): return self.buffer_queue.put(in_data)
        self.buffer_queue = queue.Queue()
        self.device = device
        self.input_rate = input_rate
        self.sample_rate = self.RATE_PROCESS
        self.block_size = int(self.RATE_PROCESS /
                              float(self.BLOCKS_PER_SECOND))

        self.soundcard_reader = LoopbackAudio(
            callback=proxy_callback, device=self.device, samplerate=self.sample_rate)
        self.soundcard_reader.daemon = True
        self.soundcard_reader.start()

    def read(self):
        # print("read")
        """Return a block of audio data, blocking if necessary."""
        return self.buffer_queue.get()

    def destroy(self):
        self.soundcard_reader.stop()
        self.soundcard_reader.join()

    frame_duration_ms = property(
        lambda self: 1000 * self.block_size // self.sample_rate)

class VADAudio(Audio):
    """Filter & segment audio with voice activity detection."""

    def __init__(self, aggressiveness=3, device=None, input_rate=None):
        super().__init__(device=device, input_rate=input_rate)
        self.vad = webrtcvad.Vad(aggressiveness)

    def frame_generator(self):
        """Generator that yields all audio frames from microphone."""
        if self.input_rate == self.RATE_PROCESS:
            while True:
                yield self.read()
        else:
            raise Exception("Resampling required")

    def vad_collector(self, padding_ms=300, ratio=0.75, frames=None):
        """Generator that yields series of consecutive audio frames comprising each utterence, separated by yielding a single None.
            Determines voice activity by ratio of frames in padding_ms. Uses a buffer to include padding_ms prior to being triggered.
            Example: (frame, ..., frame, None, frame, ..., frame, None, ...)
                      |---utterence---|        |---utterence---|
        """
        if frames is None:
            frames = self.frame_generator()
        num_padding_frames = padding_ms // self.frame_duration_ms
        ring_buffer = collections.deque(maxlen=num_padding_frames)
        triggered = False

        for frame in frames:
            if len(frame) < 640:
                return

            # 2 channel float data를 mono int16 data로 변환
            mono_frame = np.mean(frame, axis=1)
            frame = np.int16(mono_frame * 32768)

            is_speech = self.vad.is_speech(frame, self.sample_rate)

            if not triggered:
                ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])
                if num_voiced > ratio * ring_buffer.maxlen:
                    triggered = True
                    for f, s in ring_buffer:
                        yield f
                    ring_buffer.clear()

            else:
                yield frame
                ring_buffer.append((frame, is_speech))
                num_unvoiced = len(
                    [f for f, speech in ring_buffer if not speech])
                if num_unvoiced > ratio * ring_buffer.maxlen:
                    triggered = False
                    yield None
                    ring_buffer.clear()

def Int2Float(sound):
    _sound = np.copy(sound)  #
    abs_max = np.abs(_sound).max()
    _sound = _sound.astype('float32')
    if abs_max > 0:
        _sound *= 1/abs_max
    audio_float32 = torch.from_numpy(_sound.squeeze())
    return audio_float32


def get_chatgpt_response(content):
    prompt = f"""I will provide the text, so please summarize the contents in Korean in detail. Please summarize like markdown, with the topic and table of contents and with a detailed explanation.
\n\n```{content}```"""
    
    print(f"Sending prompt to GPT-3:\n{prompt}\n")
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a deep learning model for learning assistance and summarizing lecture."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=300,
        n=1,
        temperature=0,
    )

    answer = response.choices[0].message['content'].strip()
    return answer


def split_text(text, limit=1700):
    if len(text) <= limit:
        return [text]

    chunks = []
    while text:
        chunk = text[:limit]
        last_space_index = chunk.rfind(' ')

        # 추가: last_space_index가 -1이면 텍스트를 그대로 분할합니다.
        if last_space_index == -1:
            last_space_index = limit

        chunk = chunk[:last_space_index]
        chunks.append(chunk)
        text = text[last_space_index:].strip()

    return chunks


def process_text(text,ARGS):
    # 텍스트 분할
    content_chunks = split_text(text)

    answers = []
    for chunk in content_chunks:
        answer = get_chatgpt_response(chunk)
        answers.append(answer)
        print(f"Received response for chunk: {answer}")  # 응답을 출력합니다.

    # 결과를 하나의 문자열로 결합
    combined_answer = ' '.join(answers)
    file = open(ARGS.sum_path, "w")  # 파일을  열기
    file.write(combined_answer)  # 텍스트를 파일에 저장
    file.close()  # 파일 닫기

    return combined_answer       


      
def main():
    mics = sc.all_microphones(include_loopback=True)
    for i, mic in enumerate(mics):
        print(f"{i}: {mic.name}")

    mic_index = int(input("디바이스 번호를 선택하세요: "))


    DEFAULT_SAMPLE_RATE = 16000



    ARGS = easydict.EasyDict({
            "webRTC_aggressiveness" : 3,
            "nospinner" : 'store_true',
            "device" : mic_index,
            "silaro_model_name" : "silero_vad",
            "reload" : 'store_true',
            "trig_sum" :0.25,
            "neg_trig_sum" : 0.07,
            "num_steps": 8,
            "num_samples_per_window": 4000,
            "min_speech_samples": 10000,
            "min_silence_samples": 500,
            "nopython":False,
            "cuda":True,
            "model":"medium",
            "file_path": "./transcribe.txt",
            "sum_path" : "./summary.txt",
        })
    ARGS.rate = DEFAULT_SAMPLE_RATE 

    # Start audio with VAD
    vad_audio = VADAudio(aggressiveness=ARGS.webRTC_aggressiveness,
                         device=ARGS.device,
                         input_rate=ARGS.rate)
    
    frames = vad_audio.vad_collector()
    
    # load silero VAD
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model=ARGS.silaro_model_name,
                                  force_reload=ARGS.reload)
    (get_speech_ts, _, _, _, _) = utils
    
    whisper_model = whisper.load_model(ARGS.model)
    print("Whisper model loaded")
    
     # Stream from microphone to DeepSpeech using VAD
    spinner = None
    if not ARGS.nospinner:
        spinner = Halo(spinner='line')
    
    
    wav_data = bytearray()
    file = open(ARGS.file_path, "w")  # 파일을  열기
    try:
        for i, frame in enumerate(frames):
           
            if frame is not None:
                if spinner:
                    spinner.start()
                
                wav_data.extend(frame)
            else:
                if spinner:
                    spinner.stop()
                # print("webRTC has detected a possible speech")
                
                newsound = np.frombuffer(wav_data, np.int16)
                audio_float32 = Int2Float(newsound)
                time_stamps = get_speech_ts(
                    audio_float32, model, sampling_rate=ARGS.rate)
                # print("\n")
                if (len(time_stamps) > 0):
                    transcript = whisper_model.transcribe(audio=audio_float32, fp16=ARGS.cuda)
                    print(transcript['text'])
                    
                    text = transcript['text']
                    file.write(text + "\n")  # 텍스트를 파일에 저장
        
        
                else:
                    pass
                print()
        
                wav_data = bytearray()
    except KeyboardInterrupt:
        file.close()  # 파일 닫기
        with open("./transcribe.txt",'r', encoding='cp949') as file:
            text = file.read()
        print("요약을 진행합니다.\n")
        process_text(text,ARGS)
        return 0
    
    
    return 0

if __name__ == "__main__":
    main()
