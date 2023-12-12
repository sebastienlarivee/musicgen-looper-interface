import gradio as gr
import replicate
import secret
import os


def api_call(
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
    os.environ["REPLICATE_API_TOKEN"]

    api_request_payload = {
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

    output = replicate.run(
        "andreasjansson/musicgen-looper:f8140d0457c2b39ad8728a80736fea9a67a0ec0cd37b35f40b68cce507db2366",
        input=api_request_payload,
    )
    return print(output)


with gr.Blocks() as demo:
    with gr.Row():
        prompt_input = gr.Textbox(label="Prompt")

    with gr.Row():
        bpm_slider = gr.Slider(minimum=50, maximum=250, value=100, label="BPM")
        max_duration_slider = gr.Slider(
            minimum=5, maximum=30, value=10, label="Max Duration"
        )
        variations_slider = gr.Slider(
            minimum=1, maximum=10, value=1, label="Variations"
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
            choices=["medium", "large"], value="large", label="Model Version"
        )

    submit_button = gr.Button("Submit")
    output_label = gr.Label()

    submit_button.click(
        fn=api_call,
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
        outputs=output_label,
    )


if __name__ == "__main__":
    demo.launch()
