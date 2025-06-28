import runpod
import os
import tempfile
from predict import FluxDevKontextPredictor, download_model_weights

# Initialize the predictor globally
# This will load models and compile the main model once when the worker starts.
print("Initializing FluxDevKontextPredictor...")
download_model_weights() # Ensure weights are downloaded before predictor init
predictor = FluxDevKontextPredictor()
predictor.setup() # Call the setup method to load models
print("FluxDevKontextPredictor initialized.")

def handler(event):
    """
    Processes incoming requests to the Serverless endpoint.
    Args:
        event (dict): Contains the input data and request metadata.
    Returns:
        dict or str: The result to be returned to the client.
                     If returning a file path, RunPod handles uploading.
    """
    print(f"Received event: {event}")
    job_input = event.get('input', {})

    # Extract parameters from the input
    prompt = job_input.get('prompt')
    input_image_url = job_input.get('input_image_url') # Expecting URL

    if not prompt or not input_image_url:
        return {
            "error": "Missing required parameters: 'prompt' and 'input_image_url'"
        }

    # Get other parameters with defaults, similar to predict.py
    aspect_ratio = job_input.get('aspect_ratio', "match_input_image")
    num_inference_steps = job_input.get('num_inference_steps', 28)
    guidance = job_input.get('guidance', 2.5)
    seed = job_input.get('seed', None)
    output_format = job_input.get('output_format', "webp")
    output_quality = job_input.get('output_quality', 80)
    disable_safety_checker = job_input.get('disable_safety_checker', False)
    go_fast = job_input.get('go_fast', True)

    try:
        print(f"Calling predictor with prompt: '{prompt}' and image URL: '{input_image_url}'")
        # The predict method will be modified to accept input_image_url
        output_path = predictor.predict(
            prompt=prompt,
            input_image_url=input_image_url, # Pass URL directly
            aspect_ratio=aspect_ratio,
            num_inference_steps=num_inference_steps,
            guidance=guidance,
            seed=seed,
            output_format=output_format,
            output_quality=output_quality,
            disable_safety_checker=disable_safety_checker,
            go_fast=go_fast
        )
        print(f"Prediction successful. Output path: {output_path}")
        # RunPod will handle uploading the file at output_path and return a URL
        # Or, if we return a dictionary, it will be JSON serialized.
        # For file output, just returning the path string is often sufficient.
        return str(output_path)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return {"error": str(e)}

# Start the RunPod serverless worker
if __name__ == '__main__':
    print("Starting RunPod serverless worker...")
    runpod.serverless.start({"handler": handler})
