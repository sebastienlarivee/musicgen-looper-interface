import globals as glo

import os
import datetime
import random
import uuid
import gc
import subprocess
import torch
import numpy as np
import soundfile as sf
import librosa
import pyrubberband as pyrb
from audiocraft.models import MusicGen
from BeatNet.BeatNet import BeatNet
import madmom.audio.filters

# Hack madmom to work with recent python
madmom.audio.filters.np.float = float


class Generate:
    def __init__(self, bpm, seed, prompt, output_format):
        self.bpm = bpm
        self.seed = seed
        self.prompt = prompt  # + f", {bpm} bpm"
        self.output_format = output_format
        self.save_path = glo.SAVE_PATH
        self.model = glo.MODEL
        self.sample_rate = glo.MODEL.sample_rate
        self.beatnet = BeatNet(
            1,
            mode="offline",
            inference_model="DBN",
            plot=[],
            thread=False,
            device="cuda:0",
        )

    def set_all_seeds(self):
        random.seed(self.seed)
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True

    def predict_from_text(self):
        if not self.seed or self.seed == -1:
            self.seed = torch.seed() % 2**32 - 1
            self.set_all_seeds()

        print(f"Generating -> prompt: {self.prompt}, seed: {self.seed}")

        prediction = (
            self.model.generate([self.prompt], progress=True).cpu().numpy()[0, 0]
        )
        prediction = prediction / np.abs(prediction).max()
        return prediction

    def write(self, audio, name):
        wav_path = self.save_path + name + ".wav"
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

    def simple_predict(self):
        wav = self.predict_from_text()
        output_path = self.write(audio=wav, name="simple_test")
        return output_path
