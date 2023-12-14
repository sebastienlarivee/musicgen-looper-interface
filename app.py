import gradio as gr

# from audiocraft.models import MusicGen
# from looper import main_predictor

MODEL = None
INTERRUPTED = False
UNLOAD_MODEL = False

max_audio_outputs = 10


def interrupt():
    global INTERRUPTED
    INTERRUPTED = True
    print("Interrupted!")


def load_model(version):
    global MODEL
    print("Loading model", version)
    if MODEL is None or MODEL.name != version:
        MODEL = MusicGen.get_pretrained(version)


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
    # Inference parameters
    params = {
        "bpm": bpm,
        "seed": int(seed),
        "top_k": 250,
        "top_p": 0,
        "prompt": prompt,
        "variations": variations,
        "temperature": temperature,
        "max_duration": max_duration,
        "model_version": model_version,
        "output_format": output_format,
        "classifier_free_guidance": guidance,
        "custom_model_path": custom_model_path,
        "save_path": save_path,
    }

    output = main_predictor(params)

    # To add later:
    # if len(output) < variations:

    # Pad with empty outputs so the returned number of outputs == max_audio_outputs
    padded_output = output + [None] * (max_audio_outputs - len(output))

    # print(f"Output: {output}")
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
