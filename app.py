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


# Convert this to a class for better reusability (need to see if that's ok with gradio)
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
    output_format: str,
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
        duration=duration,
        temperature=temperature,
        cfg_coef=cfg_coef,
        output_format=output_format,
        seed=seed,
    )
    predict.set_generation_params()

    for i in range(batch_size):
        name = f"{random_string}_variation_{i+1:02d}"
        output.append(predict.loop_generate_from_audio(name=name))

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
    with gr.Tab("Generate"):
        with gr.Row():
            with gr.Column():
                prompt_input_gen = gr.Textbox(
                    label="Prompt",
                    placeholder="chill lofi beat, hot summer day, relaxing",
                )

                with gr.Row():
                    bpm_slider_gen = gr.Slider(
                        minimum=50, maximum=250, value=100, label="BPM"
                    )
                    duration_slider_gen = gr.Slider(
                        minimum=5, maximum=30, value=10, step=1, label="Max Duration"
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
                        minimum=0, maximum=15, value=3, label="CFG Scale"
                    )

                submit_button_gen = gr.Button("Submit")
            with gr.Column():
                audio_outputs_gen = []
                for i in range(max_audio_outputs):
                    a = gr.Audio(type="filepath")
                    audio_outputs_gen.append(a)
        batch_slider_gen.change(variable_outputs, batch_slider_gen, audio_outputs_gen)

    # Generate continuations from audio tab
    with gr.Tab("Continuations"):
        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(type="filepath")
                prompt_input2 = gr.Textbox(
                    label="Prompt",
                    placeholder="chill lofi beat, hot summer day, relaxing",
                )

                with gr.Row():
                    bpm_slider2 = gr.Slider(
                        minimum=50, maximum=250, value=100, label="BPM"
                    )
                    max_duration_slider2 = gr.Slider(
                        minimum=5, maximum=30, value=10, step=1, label="Max Duration"
                    )
                    variations_slider2 = gr.Slider(
                        minimum=1,
                        maximum=max_audio_outputs,
                        value=1,
                        step=1,
                        label="Variations",
                    )

                with gr.Row():
                    seed_input2 = gr.Textbox(value=-1, label="Seed")
                    temperature_slider2 = gr.Slider(
                        minimum=0, maximum=1, value=1, label="Temperature"
                    )
                    guidance_slider2 = gr.Slider(
                        minimum=0, maximum=15, value=3, label="CFG Scale"
                    )

                submit_button2 = gr.Button("Submit")
            with gr.Column():
                audio_outputs2 = []
                for i in range(max_audio_outputs):
                    a = gr.Audio(type="filepath")
                    audio_outputs2.append(a)
        variations_slider2.change(variable_outputs, variations_slider2, audio_outputs2)

    # Settings tab
    with gr.Tab("Settings"):
        with gr.Column():
            with gr.Row():
                model_version_toggle = gr.Radio(
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
                custom_model_path = gr.Textbox(
                    label="Custom Model Path",
                    placeholder="File path to your model",
                )
        with gr.Column():
            with gr.Row():
                save_path_input = gr.Textbox(
                    value="/content/musicgen-looper-interface", label="Save Path"
                )
                output_format_toggle = gr.Radio(
                    choices=["wav", "mp3"], value="wav", label="Output Format"
                )

    # Action handlers (NEED TO CLEAN THESE UP FOR NEXT RELEASE)
    submit_button_gen.click(
        fn=new_loops_from_text,
        inputs=[
            model_version_toggle,
            output_format_toggle,
            custom_model_path,
            save_path_input,
        ],
        outputs=audio_outputs_gen,
    )

    submit_button2.click(
        fn=inference_call,
        inputs=[
            bpm_slider2,
            seed_input2,
            prompt_input2,
            variations_slider2,
            temperature_slider2,
            max_duration_slider2,
            model_version_toggle,
            output_format_toggle,
            guidance_slider2,
            custom_model_path,
            save_path_input,
            audio_input,
        ],
        outputs=audio_outputs2,
    )


if __name__ == "__main__":
    print("Launching MusicGen Looper interface...")
    interface.launch(share=True)
