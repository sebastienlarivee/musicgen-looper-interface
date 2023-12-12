import gradio as gr
import requests


def api_call(
    bpm, seed, prompt, variations, temperature, model_version, output_format, guidance
):
    # Prepare the API request
    api_request_payload = {
        "bpm": bpm,
        "seed": seed,
        "top_k": 250,
        "top_p": 0,
        "prompt": prompt,
        "variations": variations,
        "temperature": temperature,
        "max_duration": 10,
        "model_version": model_version,
        "output_format": output_format,
        "classifier_free_guidance": guidance,
    }

    # Make the API call
    response = requests.post(
        "API_ENDPOINT",  # Replace with the actual API endpoint
        json={"input": api_request_payload},
    )

    # Parse the response for audio file links
    # audio_links = parse_response(response)  # Define the parse_response function as needed

    # Download audio files
    # download_audio_files(audio_links)  # Define the download_audio_files function as needed

    return "Audio files downloaded successfully."


with gr.Blocks() as demo:
    with gr.Row():
        bpm_slider = gr.Slider(minimum=50, maximum=250, label="BPM")
        seed_input = gr.Textbox(label="Seed")
    with gr.Row():
        prompt_input = gr.Textbox(label="Prompt")
        variations_slider = gr.Slider(minimum=1, maximum=10, label="Variations")
    with gr.Row():
        temperature_slider = gr.Slider(minimum=0, maximum=1, label="Temperature")
        model_version_toggle = gr.Radio(
            choices=["medium", "large"], label="Model Version"
        )
    with gr.Row():
        output_format_toggle = gr.Radio(choices=["wav", "mp3"], label="Output Format")
        guidance_slider = gr.Slider(
            minimum=0, maximum=20, label="Classifier Free Guidance"
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
            model_version_toggle,
            output_format_toggle,
            guidance_slider,
        ],
        outputs=output_label,
    )


if __name__ == "__main__":
    demo.launch()
