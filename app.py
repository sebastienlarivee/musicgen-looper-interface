import gradio as gr
import globals as glo
import uuid

# from looper import main_predictor
from generate import Generate


max_audio_outputs = 10
output_folder_name = "Outputs"


def get_random_string(length=5):
    random_str = str(uuid.uuid4()).replace("-", "")[:length]
    return random_str


# Convert this to a class for better reusability (need to see if that's ok with gradio)
def inference_call(
    bpm,
    seed,
    prompt,
    variations,
    temperature,
    max_duration,
    model_version,
    output_format,
    guidance,
    custom_model_path,
    save_path,
    audio_prompt,
):
    # Load custom model or base release
    if model_version == "custom model":
        # Need to test if download links work for this:
        glo.load_model(custom_model_path)
    else:
        glo.load_model(f"facebook/musicgen-{model_version}")

    # Make the output folder(s) in the user specified location (make more efficient?)
    glo.create_output_folders(save_path, output_folder_name=output_folder_name)

    prompt = prompt + f", {bpm} bpm"
    output = []
    random_string = get_random_string()

    # Pass parameters from the gradio interface to the generation code
    predict = Generate(
        bpm=bpm,
        text_prompt=prompt,
        audio_prompt=audio_prompt,
        duration=max_duration,
        temperature=temperature,
        cfg_coef=guidance,
        output_format=output_format,
        seed=int(seed),
    )
    predict.set_generation_params()

    for i in range(variations):
        name = f"{random_string}_variation_{i+1:02d}"
        output.append(predict.loop_generate_from_text(name=name))

    # Pad with empty outputs so the returned number of outputs == max_audio_outputs
    padded_output = output + [None] * (max_audio_outputs - len(output))

    # print(f"Padded: {padded_output}")

    return padded_output


# GRADIO INTERFACE


# Handles the rendering of variable audio outputs
def variable_outputs(k):
    k = int(k)
    return [gr.Audio(visible=True)] * k + [gr.Audio(visible=False)] * (
        max_audio_outputs - k
    )


with gr.Blocks() as interface:
    # Generate from text tab
    with gr.Tab("Generate"):
        with gr.Row():
            with gr.Column():
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="chill lofi beat, hot summer day, relaxing",
                )

                with gr.Row():
                    bpm_slider = gr.Slider(
                        minimum=50, maximum=250, value=100, label="BPM"
                    )
                    max_duration_slider = gr.Slider(
                        minimum=5, maximum=30, value=10, step=1, label="Max Duration"
                    )
                    variations_slider = gr.Slider(
                        minimum=1,
                        maximum=max_audio_outputs,
                        value=1,
                        step=1,
                        label="Variations",
                    )

                with gr.Row():
                    seed_input = gr.Textbox(value=-1, label="Seed")
                    temperature_slider = gr.Slider(
                        minimum=0, maximum=1, value=1, label="Temperature"
                    )
                    guidance_slider = gr.Slider(
                        minimum=0, maximum=15, value=3, label="CFG Scale"
                    )

                submit_button = gr.Button("Submit")
            with gr.Column():
                audio_outputs = []
                for i in range(max_audio_outputs):
                    a = gr.Audio()
                    audio_outputs.append(a)
        variations_slider.change(variable_outputs, variations_slider, audio_outputs)

    # Generate continuations tab
    with gr.Tab("Generate continuations"):
        with gr.Row():
            with gr.Column():
                prompt_input2 = gr.Textbox(
                    label="Prompt",
                    placeholder="chill lofi beat, hot summer day, relaxing",
                )
                audio_input = gr.Audio()

                with gr.Row():
                    bpm_slider2 = gr.Slider(
                        minimum=50, maximum=250, value=100, label="BPM"
                    )
                    max_duration_slider2 = gr.Slider(
                        minimum=5, maximum=30, value=10, step=1, label="Max Duration"
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
                audio_outputs2 = gr.Audio()

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
                    value="small",
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
    submit_button.click(
        fn=inference_call,
        inputs=[
            bpm_slider,
            seed_input,
            prompt_input,
            variations_slider,
            temperature_slider,
            max_duration_slider,
            model_version_toggle,
            output_format_toggle,
            guidance_slider,
            custom_model_path,
            save_path_input,
        ],
        outputs=audio_outputs,
    )

    submit_button2.click(
        fn=inference_call,
        inputs=[
            bpm_slider2,
            seed_input2,
            prompt_input2,
            variations_slider,
            temperature_slider2,
            max_duration_slider2,
            model_version_toggle,
            output_format_toggle,
            guidance_slider2,
            custom_model_path,
            save_path_input,
        ],
        outputs=audio_outputs2,
    )


if __name__ == "__main__":
    print("Launching MusicGen Looper interface...")
    interface.launch(share=True)
