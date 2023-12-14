# This code is adapted from:
# https://github.com/andreasjansson/cog-musicgen-looper by Andreas Jansson (Apache 2.0 license)
# With modifications

# You'll need to run this line for colab and stuff:
# !apt-get install -y ffmpeg portaudio19-dev rubberband-cli


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
from audiocraft.models.loaders import (
    load_compression_model,
    load_lm_model,
)
from BeatNet.BeatNet import BeatNet
import madmom.audio.filters

# Hack madmom to work with recent python
madmom.audio.filters.np.float = float

loaded_models = {}
output_folder_name = "Outputs"


def set_all_seeds(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def generate_random_string(length=5):
    random_str = str(uuid.uuid4()).replace("-", "")[:length]
    return random_str


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


def load_fb_model(model_id, device):
    # Check if the model is already loaded
    if model_id in loaded_models:
        return loaded_models[model_id]

    # Load the model since it's not loaded yet
    print(f"Loading model: {model_id}...")
    compression_model = load_compression_model(model_id, device=device)
    lm = load_lm_model(model_id, device=device)
    musicgen_model = MusicGen(model_id, compression_model, lm)

    # Store the loaded model in the dictionary
    loaded_models[model_id] = musicgen_model
    print(f"Model loaded!")
    return musicgen_model


def load_custom_model(custom_model_path):
    # Check if the model is already loaded
    if custom_model_path in loaded_models:
        return loaded_models[custom_model_path]

    # Load the model since it's not loaded yet
    print("Loading custom model...")
    musicgen_model = MusicGen.get_pretrained(custom_model_path)

    # Store the loaded model in the dictionary
    loaded_models[custom_model_path] = musicgen_model
    print("Model loaded!")
    return musicgen_model


def main_predictor(params):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Determine which model to load based on the 'model_version' parameter
    model_version = params["model_version"]

    if model_version == "custom model":
        custom_model_path = params["custom_model_path"]
        model_id = custom_model_path
        model = load_custom_model(custom_model_path)
    else:
        model_id = f"facebook/musicgen-{model_version}"  # Matches audiocraft naming scheme, needs to be more robust
        model = load_fb_model(model_id, device)

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

    # Generate additional variations if requested
    if variations > 1:
        # Use last 4 beats as audio prompt
        last_4beats = beats[beats[:, 0] <= end_time][-5:]
        audio_prompt_start_time = last_4beats[0][0]
        audio_prompt_end_time = last_4beats[-1][0]
        audio_prompt_start_sample = int(audio_prompt_start_time * model.sample_rate)
        audio_prompt_end_sample = int(audio_prompt_end_time * model.sample_rate)
        audio_prompt_seconds = audio_prompt_end_time - audio_prompt_start_time
        audio_prompt = torch.tensor(
            wav[audio_prompt_start_sample:audio_prompt_end_sample]
        )[None]
        audio_prompt_duration = audio_prompt_end_sample - audio_prompt_start_sample

        model.set_generation_params(
            duration=duration + audio_prompt_seconds + 0.1,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            cfg_coef=guidance,
        )

        for i in range(2, variations + 1):
            print(f"Variation {i:02d}: generating...")
            continuation = model.generate_continuation(
                prompt=audio_prompt,
                prompt_sample_rate=model.sample_rate,
                descriptions=[prompt],
                progress=True,
            )
            variation_loop = continuation.cpu().numpy()[
                0, 0, audio_prompt_duration : audio_prompt_duration + len(loop)
            ]

            # Process each variation loop
            num_lead = 100  # for blending to avoid clicks
            lead_start = start_sample - num_lead
            lead = wav[lead_start:start_sample]
            num_lead = len(lead)
            variation_loop[-num_lead:] *= np.linspace(1, 0, num_lead)
            variation_loop[-num_lead:] += np.linspace(0, 1, num_lead) * lead

            variation_stretched = pyrb.time_stretch(
                variation_loop, model.sample_rate, bpm / actual_bpm
            )

            # Save each variation
            variation_output_path = write(
                variation_stretched,
                model.sample_rate,
                output_format,
                f"{save_path}variation_{i:02d}_{random_string}",
            )
            outputs.append(variation_output_path)  # Append to list
            # print(f"Outputs list: {outputs}")

    torch.cuda.empty_cache()
    gc.collect()

    # print(f"Inference outputs: {outputs}")
    return outputs
