import globals as glo
import os
import random
import subprocess
import torch
import numpy as np
import soundfile as sf
import librosa
import pyrubberband as pyrb
from BeatNet.BeatNet import BeatNet
import madmom.audio.filters

# Hack madmom to work with recent python
madmom.audio.filters.np.float = float


class Generate:
    def __init__(self, bpm, seed, prompt, output_format):
        self.bpm = bpm
        self.prompt = prompt
        self.output_format = output_format
        self.save_path = glo.SAVE_PATH
        self.model = glo.MODEL
        self.sample_rate = glo.MODEL.sample_rate

        # Gets a random seed if none is specified
        if not seed or seed == -1:
            self.seed = torch.seed() % 2**32 - 1
        else:
            self.seed = seed

    def set_all_seeds(self):
        # Sets seed for the generation
        random.seed(self.seed)
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True

    def predict_from_text(self):
        # Generates audio from a text prompt
        self.set_all_seeds()

        print(f"Generating -> prompt: {self.prompt}, seed: {self.seed}")

        prediction = (
            self.model.generate([self.prompt], progress=True).cpu().numpy()[0, 0]
        )
        prediction = prediction / np.abs(prediction).max()
        self.seed += 1
        return prediction

    def estimate_beats(self, wav):
        # Maps out the beats
        beatnet = BeatNet(
            1,
            mode="offline",
            inference_model="DBN",
            plot=[],
            thread=False,
            device="cuda:0",
        )

        beatnet_input = librosa.resample(
            wav, orig_sr=self.sample_rate, target_sr=beatnet.sample_rate
        )
        return beatnet.process(beatnet_input)

    def get_loop_points(beats):
        # Trims + stretches the audio to match user BPM as a seamless loop
        downbeat_times = beats[:, 0][beats[:, 1] == 1]
        num_bars = len(downbeat_times) - 1
        if num_bars < 1:
            raise ValueError(
                "Less than one bar detected. Try increasing max_duration, or use a different seed."
            )
        even_num_bars = int(2 ** np.floor(np.log2(num_bars)))
        start_time = downbeat_times[0]
        end_time = downbeat_times[even_num_bars]
        num_beats = len(beats[(beats[:, 0] >= start_time) & (beats[:, 0] < end_time)])
        duration = end_time - start_time
        actual_bpm = num_beats / duration * 60

        # Handle possible octave errors
        if abs(actual_bpm / 2 - self.bpm) <= 10:
            actual_bpm = actual_bpm / 2
        elif abs(actual_bpm * 2 - self.bpm) <= 10:
            actual_bpm = actual_bpm * 2

        # Prepare the main audio loop
        start_sample = int(start_time * self.sample_rate)
        end_sample = int(end_time * self.sample_rate)
        loop = wav[start_sample:end_sample]

        # Process the audio loop for the main output
        stretched = pyrb.time_stretch(loop, self.sample_rate, self.bpm / actual_bpm)
        return stretched

    def write(self, audio, name):
        # Converts tensors into an audio files and saves it to the directory
        wav_path = self.save_path + name + ".wav"
        print(wav_path)
        sf.write(wav_path, audio, self.sample_rate)
        if self.output_format == "mp3":
            mp3_path = name + ".mp3"
            subprocess.call(
                ["ffmpeg", "-loglevel", "error", "-y", "-i", wav_path, mp3_path]
            )
            os.remove(wav_path)
            path = mp3_path
        else:
            path = wav_path
        return path

    def simple_predict(self, name):
        wav = self.predict_from_text()
        output_path = self.write(audio=wav, name=name)
        return output_path

    def loop_predict(self, name):
        wav = self.predict_from_text()
        beats = self.estimate_beats(wav=wav)
        loop = self.get_loop_points(beats=beats)
        output_path = self.write(audio=loop, name=name)
        return output_path
