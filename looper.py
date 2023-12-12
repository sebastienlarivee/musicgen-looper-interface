import os
import random
import subprocess
import torch
import numpy as np
import soundfile as sf
import librosa
import pyrubberband as pyrb
from audiocraft.models import MusicGen
from audiocraft.models.loaders import (
    load_compression_model,
    load_lm_model,
    HF_MODEL_CHECKPOINTS_MAP,
)
from BeatNet.BeatNet import BeatNet
import madmom.audio.filters

# Hack madmom to work with recent python
madmom.audio.filters.np.float = float

# Global model path
MODEL_PATH = "/src/models/"
os.environ["TRANSFORMERS_CACHE"] = MODEL_PATH
os.environ["TORCH_HOME"] = MODEL_PATH


def set_all_seeds(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def estimate_beats(wav, sample_rate, beatnet):
    beatnet_input = librosa.resample(
        wav, orig_sr=sample_rate, target_sr=beatnet.sample_rate
    )
    return beatnet.process(beatnet_input)


def get_loop_points(beats):
    downbeat_times = beats[:, 0][beats[:, 1] == 1]
    num_bars = len(downbeat_times) - 1
    if num_bars < 1:
        raise ValueError(
            "Less than one bar detected. Try increasing max_duration, or use a different seed."
        )
    even_num_bars = int(2 ** np.floor(np.log2(num_bars)))
    start_time = downbeat_times[0]
    end_time = downbeat_times[even_num_bars]
    return start_time, end_time


def write(audio, sample_rate, output_format, name):
    wav_path = name + ".wav"
    sf.write(wav_path, audio, sample_rate)
    if output_format == "mp3":
        mp3_path = name + ".mp3"
        subprocess.call(
            ["ffmpeg", "-loglevel", "error", "-y", "-i", wav_path, mp3_path]
        )
        os.remove(wav_path)
        path = mp3_path
    else:
        path = wav_path
    return path


def load_model(model_path, cls, model_id, device):
    name = next(
        (key for key, val in HF_MODEL_CHECKPOINTS_MAP.items() if val == model_id), None
    )
    compression_model = load_compression_model(
        name, device=device, cache_dir=model_path
    )
    lm = load_lm_model(name, device=device, cache_dir=model_path)
    return MusicGen(name, compression_model, lm)


def add_output(outputs, path):
    for i in range(1, 21):
        field = f"variation_{i:02d}"
        if field not in outputs:
            outputs[field] = path
            return
    raise ValueError("Failed to add output")


def main_predictor(params):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    medium_model = load_model(MODEL_PATH, MusicGen, "facebook/musicgen-medium", device)
    large_model = load_model(MODEL_PATH, MusicGen, "facebook/musicgen-large", device)
    beatnet = BeatNet(
        1, mode="offline", inference_model="DBN", plot=[], thread=False, device="cuda:0"
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

    model = medium_model if model_version == "medium" else large_model

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

    prompt = prompt + f", {bpm} bpm"
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

    # Initialize outputs dictionary
    outputs = {}

    # Save the main output
    main_output_path = write(
        stretched, model.sample_rate, output_format, "variation_01"
    )
    add_output(outputs, main_output_path)

    # Generate additional variations if requested
    if variations > 1:
        for i in range(2, variations + 1):
            # Generate and process each variation loop
            # ... (variation generation code) ...

            # Save each variation
            variation_output_path = write(
                variation_stretched,
                model.sample_rate,
                output_format,
                f"variation_{i:02d}",
            )
            add_output(outputs, variation_output_path)

    return outputs
