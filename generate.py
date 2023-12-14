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
    def __init__(self, bpm, seed, prompt):
        self.bpm = bpm
        self.seed = seed
        self.prompt = prompt + f", {bpm} bpm"
        self.model = glo.MODEL

    def set_all_seeds(self):
        random.seed(self.seed)
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True

    def main_predictor(params):
        beatnet = BeatNet(
            1,
            mode="offline",
            inference_model="DBN",
            plot=[],
            thread=False,
            device="cuda:0",
        )

        bpm = params["bpm"]
        seed = params["seed"]
        top_k = params["top_k"]
        top_p = params["top_p"]
        prompt = params["prompt"]
        variations = params["variations"]
        temperature = params["temperature"]
        max_duration = params["max_duration"]
        model_version = params["model_version"]
        output_format = params["output_format"]
        guidance = params["classifier_free_guidance"]
        base_save_path = params["save_path"]

        save_path = create_output_folders(base_save_path)

        model.set_generation_params(
            duration=max_duration,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            cfg_coef=guidance,
        )

        if not seed or seed == -1:
            seed = torch.seed() % 2**32 - 1
        set_all_seeds(seed)

        print(
            f"Generating: {model_version}, {variations} variation(s), prompt: {prompt}, seed: {seed}"
        )

        prompt = prompt + f", {bpm} bpm"
        print("Variation 01: generating...")
        wav = model.generate([prompt], progress=True).cpu().numpy()[0, 0]
        wav = wav / np.abs(wav).max()

        beats = estimate_beats(wav, model.sample_rate, beatnet)
        start_time, end_time = get_loop_points(beats)
        num_beats = len(beats[(beats[:, 0] >= start_time) & (beats[:, 0] < end_time)])
        duration = end_time - start_time
        actual_bpm = num_beats / duration * 60

        # Handle possible octave errors
        if abs(actual_bpm / 2 - bpm) <= 10:
            actual_bpm = actual_bpm / 2
        elif abs(actual_bpm * 2 - bpm) <= 10:
            actual_bpm = actual_bpm * 2

        # Prepare the main audio loop
        start_sample = int(start_time * model.sample_rate)
        end_sample = int(end_time * model.sample_rate)
        loop = wav[start_sample:end_sample]

        # Process the audio loop for the main output
        stretched = pyrb.time_stretch(loop, model.sample_rate, bpm / actual_bpm)

        # Generate a random string for this set of variations
        random_string = generate_random_string()

        # Initialize outputs list
        outputs = []

        # Save the main output
        main_output_path = write(
            stretched,
            model.sample_rate,
            output_format,
            f"{save_path}variation_01_{random_string}",
        )
        outputs.append(main_output_path)  # Append to list instead of dictionary
        # print(f"Outputs list: {outputs}")
