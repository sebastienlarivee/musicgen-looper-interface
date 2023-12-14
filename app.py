import gradio as gr
import globals as glo

from looper import main_predictor
from generate import Generate


max_audio_outputs = 10
output_folder_name = "Outputs"


# Handles the rendering of variable audio outputs
def variable_outputs(k):
    k = int(k)
    return [gr.Audio(visible=True)] * k + [gr.Audio(visible=False)] * (
        max_audio_outputs - k
    )


# Can reuse this inference call component to send calls to multiple different types of calls!
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
):
    # Load custom model or base release
    if model_version == "custom model":
        # Need to test if download links work for this:
        glo.load_model(custom_model_path)
    else:
        glo.load_model(f"facebook/musicgen-{model_version}")

    # Make the output folder(s) in the user specified location (make more efficient?)
    glo.create_output_folders(save_path, output_folder_name=output_folder_name)

    glo.MODEL.set_generation_params(
        duration=max_duration,
        top_k=250,
        top_p=0,
        temperature=temperature,
        cfg_coef=guidance,
    )

    output = []
    predict = Generate(bpm=bpm, seed=seed, prompt=prompt, output_format=output_format)

    for _ in range(variations):
        output.append(predict.simple_predict())

    # Pad with empty outputs so the returned number of outputs == max_audio_outputs
    padded_output = output + [None] * (max_audio_outputs - len(output))

    # print(f"Padded: {padded_output}")

    return padded_output


# Gradio interface layout
with gr.Blocks() as demo:
    # Generate tab
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
        with gr.Column():
            gr.Markdown("Placeholder")
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


if __name__ == "__main__":
    print("Launching MusicGen Looper interface...")
    demo.launch(share=True)
