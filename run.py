#!/usr/bin/env python3
# coding: utf-8

import time
import wave

from model_engine import OrpheusModel


if __name__ == '__main__':
    model = OrpheusModel('medium-3b')
    prompt = '''I used to be an adventurer like you. Then I took an arrow in the knee...'''


    start_time = time.monotonic()
    syn_tokens = model.generate_speech(
       prompt=prompt,
       voice='leo',
    )

    with wave.open('output.wav', 'wb') as wf:
       wf.setnchannels(1)
       wf.setsampwidth(2)
       wf.setframerate(24000)

       total_frames = 0
       chunk_counter = 0
       for audio_chunk in syn_tokens: # output streaming
          chunk_counter += 1
          frame_count = len(audio_chunk) // (wf.getsampwidth() * wf.getnchannels())
          total_frames += frame_count
          wf.writeframes(audio_chunk)
       duration = total_frames / wf.getframerate()

       end_time = time.monotonic()
       print(f'It took {end_time - start_time} seconds to generate {duration:.2f} seconds of audio')

