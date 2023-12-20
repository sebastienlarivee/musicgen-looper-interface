import torch
from audiocraft.models import MusicGen
import os
import datetime


# Store all global variable which are passed to most functions
MODEL = None
INTERRUPTED = False
SAVE_PATH = ""


def interrupt():
    global INTERRUPTED
    INTERRUPTED = True
    print("Interrupted!")


# Loading function from the fb gradio demo
def load_model(version):
    global MODEL
    print("Loading model", version)
    if MODEL is None or MODEL.name != version:
        if MODEL is not None:
            # Clear PyTorch CUDA cache and delete model
            del MODEL
            torch.cuda.empty_cache()
        MODEL = None  # in case loading would crash
        MODEL = MusicGen.get_pretrained(version)


# Create save path for generations
def create_output_folders(base_path, output_folder_name):
    global SAVE_PATH
    # Ensure base_path ends with a slash
    if not base_path.endswith(os.sep):
        base_path += os.sep

    # Create 'Outputs' folder
    outputs_path = base_path + output_folder_name
    if not os.path.exists(outputs_path):
        os.makedirs(outputs_path)

    # Create yyyy-mm-dd folder
    current_date = datetime.date.today().isoformat()
    date_folder_path = os.path.join(outputs_path, current_date)
    date_folder_path += os.sep
    if not os.path.exists(date_folder_path):
        os.makedirs(date_folder_path)

    # Return the path to the folder
    SAVE_PATH = date_folder_path
