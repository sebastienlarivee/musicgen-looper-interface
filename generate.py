import globals as glo
import os
import random
import torch
import torchaudio
import numpy as np
import soundfile as sf
import librosa
import pyrubberband as pyrb
from BeatNet.BeatNet import BeatNet
import madmom.audio.filters

# Hack madmom to work with recent python (need to figure out that madmom is actually doing)
madmom.audio.filters.np.float = float


class Generate:
    def __init__(
        self,
        bpm: int,
        text_prompt: str,
        audio_prompt: str,
        duration: float,
        temperature: int,
        cfg_coef: float,
        seed: int,
    ):
        # Global variables:
        self.save_path = glo.SAVE_PATH
        self.model = glo.MODEL
        self.model_sample_rate = glo.MODEL.sample_rate

        # Variables passed from app.py
        self.bpm = bpm
        self.text_prompt = text_prompt

        if audio_prompt is not None:
            self.audio_prompt, self.prompt_sample_rate = torchaudio.load(audio_prompt)
            self.audio_prompt_wav = audio_prompt

        self.duration = duration
        self.temperature = temperature
        self.cfg_coef = cfg_coef

        # Gets a random seed if none is specified
        if not seed or seed == -1:
            self.seed = torch.seed() % 2**32 - 1
        else:
            self.seed = seed

    def set_generation_params(self):
        # Updates the MusicGen model parameters to user input
        self.model.set_generation_params(
            duration=self.duration,
            top_k=250,
            top_p=0,
            temperature=self.temperature,
            cfg_coef=self.cfg_coef,
        )

    def set_all_seeds(self):
        # Sets seed for the generation
        random.seed(self.seed)
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True

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
            wav, orig_sr=self.model_sample_rate, target_sr=beatnet.sample_rate
        )
        return beatnet.process(beatnet_input)

    def get_loop_points(self, beats, wav):
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
        start_sample = int(start_time * self.model_sample_rate)
        end_sample = int(end_time * self.model_sample_rate)
        loop = wav[start_sample:end_sample]

        # Process the audio loop for the main output
        stretched = pyrb.time_stretch(
            loop, self.model_sample_rate, self.bpm / actual_bpm
        )
        return stretched

    def write(self, audio, name):
        # Save's as file type given by user
        wav_path = self.save_path + name + ".wav"
        print(wav_path)
        sf.write(wav_path, audio, self.model_sample_rate)
        return wav_path

    def generate_from_text(self):
        # Generates audio from a text prompt
        self.set_all_seeds()

        generation = (
            self.model.generate([self.text_prompt], progress=True).cpu().numpy()[0, 0]
        )
        generation = generation / np.abs(generation).max()
        return generation

    def generate_from_audio(self, prompt=None, start_indice=None, end_indice=None):
        # Generates audio from an audio prompt
        if prompt == None:
            prompt = self.audio_prompt

        self.set_all_seeds()
        generation = (
            self.model.generate_continuation(
                prompt=prompt,
                prompt_sample_rate=self.prompt_sample_rate,
                descriptions=[self.text_prompt],
                progress=True,
            )
            .cpu()
            .numpy()[0, 0, start_indice:end_indice]
            # ^ Optional parameter to slice off the input audio
        )
        return generation

    ################################################
    # ENDPOINTS TO CALL THROUGH THE GRADIO INTERFACE
    ################################################

    def simple_generate_from_text(self, name):
        # ENDPOINT: Generate audio from a text prompt
        wav = self.predict_from_text()
        output_path = self.write(audio=wav, name=name)
        self.seed += 1  # For batch generation
        return output_path

    def simple_generate_from_audio(self, name):
        # ENDPOINT: Generate audio from an audio prompt
        prompt_duration = 3  # Placeholder value for testing

        self.audio_prompt = self.audio_prompt[
            ..., -int(prompt_duration * self.prompt_sample_rate) :
        ]

        wav = self.generate_from_audio(prompt=self.audio_prompt)
        output_path = self.write(audio=wav, name=name)
        self.seed += 1  # For batch generation
        return output_path

    def loop_generate_from_text(self, name):
        # ENDPOINT: Generate seamless loops from a text prompt
        print(f"Text to audio, seed: {self.seed}")
        wav = self.generate_from_text()
        beats = self.estimate_beats(wav=wav)
        loop = self.get_loop_points(beats=beats, wav=wav)
        output_path = self.write(audio=loop, name=name)
        self.seed += 1  # For batch generation
        return output_path

    def loop_generate_from_audio(self, name):
        # ENDPOINT: Generate seamless loops from an audio prompt (prompt must also be a loop to work well)
        print(f"Text to audio, seed: {self.seed}")

        # Select last 4 beats of input audio for use as audio prompt
        prompt_beats = 4  # Placeholder, make it a parameter?
        prompt_seconds = (60 / self.bpm) * prompt_beats
        prompt = self.audio_prompt[
            ..., -int(prompt_seconds * self.prompt_sample_rate) :
        ]

        # Calculate total duration (+ a bit of extra) and update generation params
        original_loop_seconds = self.audio_prompt.size(1) / self.prompt_sample_rate
        self.duration = prompt_seconds + original_loop_seconds + 0.1
        self.set_generation_params()

        # Slice generated audio from the end of the prompt to the length of the original loop
        start_indice = int(prompt_seconds * self.model_sample_rate)
        end_indice = int(start_indice + original_loop_seconds * self.model_sample_rate)

        # Generate
        wav = self.generate_from_audio(
            prompt=prompt,
            start_indice=start_indice,
            end_indice=end_indice,
        )

        print("GEN COMPLETE!")

        # Blending (I feel like this can be improved)
        num_lead = 500
        lead = sf.read(self.audio_prompt_wav, start=-num_lead)
        lead = lead[0]
        num_lead = len(lead)
        wav[-num_lead:] *= np.linspace(1, 0, num_lead)
        wav[-num_lead:] += np.linspace(0, 1, num_lead) * lead

        output_path = self.write(audio=wav, name=name)
        self.seed += 1  # For batch generation
        return output_path
