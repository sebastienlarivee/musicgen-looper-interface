import gradio as gr
import globals as glo
import uuid

# from looper import main_predictor
from generate import Generate


max_audio_outputs = 10
output_folder_name = "Outputs"


def get_random_string(length=5):
    random_string = str(uuid.uuid4()).replace("-", "")[:length]
    return random_string


def model_loader(model, model_path):
    # Load custom model or base release
    if model == "custom model":
        # Need to test if download links work for this:
        glo.load_model(model_path)
    else:
        glo.load_model(f"facebook/musicgen-{model}")


def new_loops_from_text(
    model_version: str,
    custom_model_path: str,
    save_path: str,
    batch_size: int,
    bpm: int,
    text_prompt: str,
    duration: float,
    temperature: int,
    cfg_coef: float,
    seed: int,
):
    model_loader(model=model_version, model_path=custom_model_path)

    # Make the output folder(s) in the user specified location (make more efficient?)
    glo.create_output_folders(save_path, output_folder_name=output_folder_name)

    text_prompt = text_prompt + f", {bpm} bpm"
    output = []
    random_string = get_random_string()

    # Pass parameters from the gradio interface to the generation code
    predict = Generate(
        bpm=bpm,
        text_prompt=text_prompt,
        audio_prompt=None,
        duration=float(duration),
        temperature=temperature,
        cfg_coef=cfg_coef,
        seed=int(seed),
    )
    predict.set_generation_params()

    for i in range(batch_size):
        name = f"{random_string}_generation_{i+1:02d}"
        output.append(predict.loop_generate_from_text(name=name))

    # Pad with empty outputs so the returned number of outputs == max_audio_outputs
    padded_output = output + [None] * (max_audio_outputs - len(output))

    return padded_output


def new_loops_from_audio(
    model_version: str,
    custom_model_path: str,
    save_path: str,
    batch_size: int,
    bpm: int,
    text_prompt: str,
    audio_prompt: str,
    duration: float,
    temperature: int,
    cfg_coef: float,
    seed: int,
):
    model_loader(model=model_version, model_path=custom_model_path)

    # Make the output folder(s) in the user specified location (make more efficient?)
    glo.create_output_folders(save_path, output_folder_name=output_folder_name)

    text_prompt = text_prompt + f", {bpm} bpm"
    output = []
    random_string = get_random_string()

    # Pass parameters from the gradio interface to the generation code
    predict = Generate(
        bpm=bpm,
        text_prompt=text_prompt,
        audio_prompt=audio_prompt,
        duration=float(duration),
        temperature=temperature,
        cfg_coef=cfg_coef,
        seed=int(seed),
    )

    for i in range(batch_size):
        name = f"{random_string}_continuation_{i+1:02d}"
        output.append(predict.loop_generate_from_text(name=name))

    # Pad with empty outputs so the returned number of outputs == max_audio_outputs
    padded_output = output + [None] * (max_audio_outputs - len(output))

    return padded_output


##################
# GRADIO INTERFACE
##################


# Handles the rendering of variable audio outputs
def variable_outputs(k):
    k = int(k)
    return [gr.Audio(type="filepath", visible=True)] * k + [
        gr.Audio(type="filepath", visible=False)
    ] * (max_audio_outputs - k)


with gr.Blocks() as interface:
    # Generate from text tab
    with gr.Tab("Text to loop"):
        with gr.Row():
            with gr.Column():
                text_prompt_gen = gr.Textbox(
                    label="Prompt",
                    placeholder="chill lofi beat, hot summer day, relaxing",
                )

                with gr.Row():
                    bpm_slider_gen = gr.Slider(
                        minimum=50, maximum=250, value=100, label="BPM"
                    )
                    duration_slider_gen = gr.Slider(
                        minimum=5, maximum=30, value=10, step=0.5, label="Max Duration"
                    )
                    batch_slider_gen = gr.Slider(
                        minimum=1,
                        maximum=max_audio_outputs,
                        value=1,
                        step=1,
                        label="Variations",
                    )

                with gr.Row():
                    seed_input_gen = gr.Textbox(value=-1, label="Seed")
                    temperature_slider_gen = gr.Slider(
                        minimum=0, maximum=1, value=1, label="Temperature"
                    )
                    cfg_slider_gen = gr.Slider(
                        minimum=0, maximum=15, step=0.5, value=3, label="CFG Scale"
                    )
                submit_button_gen = gr.Button("Submit")

            with gr.Column():
                audio_outputs_gen = []
                for i in range(max_audio_outputs):
                    a = gr.Audio(type="filepath")
                    audio_outputs_gen.append(a)
        batch_slider_gen.change(variable_outputs, batch_slider_gen, audio_outputs_gen)

    # Generate continuations from audio tab
    with gr.Tab("Loop to loop"):
        with gr.Row():
            with gr.Column():
                audio_prompt_con = gr.Audio(type="filepath")
                text_prompt_con = gr.Textbox(
                    label="Prompt",
                    placeholder="chill lofi beat, hot summer day, relaxing",
                )

                with gr.Row():
                    bpm_slider_con = gr.Slider(
                        minimum=50, maximum=250, value=100, label="BPM"
                    )
                    duration_slider_con = gr.Slider(
                        minimum=5, maximum=30, value=10, step=0.5, label="Max Duration"
                    )
                    batch_slider_con = gr.Slider(
                        minimum=1,
                        maximum=max_audio_outputs,
                        value=1,
                        step=1,
                        label="Variations",
                    )

                with gr.Row():
                    seed_input_con = gr.Textbox(value=-1, label="Seed")
                    temperature_slider_con = gr.Slider(
                        minimum=0, maximum=1, value=1, label="Temperature"
                    )
                    cfg_slider_con = gr.Slider(
                        minimum=0, maximum=15, step=0.5, value=3, label="CFG Scale"
                    )

                submit_button_con = gr.Button("Submit")
            with gr.Column():
                audio_outputs_con = []
                for i in range(max_audio_outputs):
                    a = gr.Audio(type="filepath")
                    audio_outputs_con.append(a)
        batch_slider_con.change(variable_outputs, batch_slider_con, audio_outputs_con)

    # Settings tab
    with gr.Tab("Settings"):
        with gr.Column():
            with gr.Row():
                model_toggle_set = gr.Radio(
                    choices=[
                        "small",
                        "medium",
                        "large",
                        "stereo-small",
                        "stereo-medium",
                        "stereo-large",
                        "custom model",
                    ],
                    value="stereo-small",
                    label="Model Version",
                )
                model_path_set = gr.Textbox(
                    label="Custom Model Path",
                    placeholder="File path to your model",
                )
        with gr.Column():
            with gr.Row():
                save_path_set = gr.Textbox(
                    value="/content/musicgen-looper-interface", label="Save Path"
                )
    # Action handlers (NEED TO CLEAN THESE UP FOR NEXT RELEASE)
    submit_button_gen.click(
        fn=new_loops_from_text,
        inputs=[
            model_toggle_set,
            model_path_set,
            save_path_set,
            batch_slider_gen,
            bpm_slider_gen,
            text_prompt_gen,
            duration_slider_gen,
            temperature_slider_gen,
            cfg_slider_gen,
            seed_input_gen,
        ],
        outputs=audio_outputs_gen,
    )
    submit_button_con.click(
        fn=new_loops_from_audio,
        inputs=[
            model_toggle_set,
            model_path_set,
            save_path_set,
            batch_slider_con,
            bpm_slider_con,
            text_prompt_con,
            audio_prompt_con,
            duration_slider_con,
            temperature_slider_con,
            cfg_slider_con,
            seed_input_con,
        ],
        outputs=audio_outputs_con,
    )


if __name__ == "__main__":
    print("Launching MusicGen Looper interface...")
    interface.launch(share=True)
