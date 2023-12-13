import gradio as gr

max_audio_outputs = 10


def variable_outputs(k):
    k = int(k)
    return [gr.Audio(visible=True)] * k + [gr.Audio(visible=False)] * (
        max_audio_outputs - k
    )


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
):
    # Prepare the API request

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
    }

    output = main_predictor(params)
    # Parse the output dictionary to extract file locations
    file_locations = [output[key] for key in output]

    # Return the list of file locations
    print(file_locations)
    return file_locations


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(label="Prompt")

            with gr.Row():
                bpm_slider = gr.Slider(minimum=50, maximum=250, value=100, label="BPM")
                max_duration_slider = gr.Slider(
                    minimum=5, maximum=30, value=10, label="Max Duration"
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
                    minimum=0, maximum=15, value=3, label="Classifier Free Guidance"
                )

            with gr.Row():
                output_format_toggle = gr.Radio(
                    choices=["wav", "mp3"], value="wav", label="Output Format"
                )
                model_version_toggle = gr.Radio(
                    choices=["medium", "large"], value="medium", label="Model Version"
                )

            submit_button = gr.Button("Submit")
        with gr.Column():
            audio_outputs = []
            for i in range(max_audio_outputs):
                a = gr.Audio()
                audio_outputs.append(a)

    variations_slider.change(variable_outputs, variations_slider, audio_outputs)

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
        ],
        outputs=audio_outputs,
    )

demo.launch(share=True)
