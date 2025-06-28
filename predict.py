import os
import time
import torch
from PIL import Image
from cog import BasePredictor, Path, Input # Path will still be used for output, Input not directly by RunPod handler
import requests
import tempfile # For managing temporary image files

from flux.sampling import denoise, get_schedule, prepare_kontext, unpack
from flux.util import (
    configs,
    load_clip,
    load_t5
)
from flux.model import Flux
from flux.modules.autoencoder import AutoEncoder
from safetensors.torch import load_file as load_sft
from safety_checker import SafetyChecker
from util import print_timing, generate_compute_step_map
from weights import download_weights

from flux.util import ASPECT_RATIOS

# Kontext model configuration
KONTEXT_WEIGHTS_URL = "https://weights.replicate.delivery/default/black-forest-labs/kontext/release-candidate/kontext-dev.sft"
KONTEXT_WEIGHTS_PATH = "./models/kontext/kontext-dev.sft"
AE_WEIGHTS_URL = "https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-dev/safetensors/ae.safetensors"
AE_WEIGHTS_PATH = "./models/flux-dev/ae.safetensors"
T5_WEIGHTS_URL = "https://weights.replicate.delivery/default/official-models/flux/t5/t5-v1_1-xxl.tar"
T5_WEIGHTS_PATH = "./models/t5"
CLIP_URL = "https://weights.replicate.delivery/default/official-models/flux/clip/clip-vit-large-patch14.tar"
CLIP_PATH = "./models/clip"

TORCH_COMPILE_CACHE = "./torch-compile-cache-flux-dev-kontext.bin"

class FluxDevKontextPredictor(BasePredictor):
    """
    Flux.1 Kontext Predictor - Image-to-image transformation model using FLUX.1-dev architecture
    """

    def setup(self) -> None:
        """Load model weights and initialize the pipeline"""
        self.device = torch.device("cuda")

        # Download all weights if needed
        download_model_weights()

        # Initialize models
        st = time.time()
        print("Loading t5...")
        self.t5 = load_t5(self.device, max_length=512, t5_path=T5_WEIGHTS_PATH)
        print(f"Loaded t5 in {time.time() - st} seconds")
        st = time.time()
        self.clip = load_clip(self.device, clip_path=CLIP_PATH)
        print(f"Loaded clip in {time.time() - st} seconds")
        st = time.time()
        self.model = load_kontext_model(device=self.device)
        print(f"Loaded kontext model in {time.time() - st} seconds")
        st = time.time()
        self.ae = load_ae_local(device=self.device)
        print(f"Loaded ae in {time.time() - st} seconds")
        st = time.time()
        self.model = torch.compile(self.model, dynamic=True)

        # Initialize safety checker
        self.safety_checker = SafetyChecker()
        print("Compiling model with torch.compile...")
        start_time = time.time()
        self.predict(
            prompt="Make the hair blue",
            input_image=Path("lady.png"),
            aspect_ratio="1:1",
            num_inference_steps=30,
            guidance=2.5,
            seed=42,
            output_format="png",
            output_quality=100,
            disable_safety_checker=True,
            go_fast=True,
        )
        print(f"Compiled in {time.time() - start_time} seconds")
        print("FluxDevKontextPredictor setup complete")


    def predict(
        self,
        prompt: str,
        input_image_url: str, # Changed from input_image: Path
        aspect_ratio: str = "match_input_image",
        # megapixels: str = "1", # Commented out as in original
        num_inference_steps: int = 28,
        guidance: float = 2.5,
        seed: int = None,
        output_format: str = "webp",
        output_quality: int = 80,
        disable_safety_checker: bool = False,
        go_fast: bool = True,
    ) -> Path: # Still returns Path, RunPod handler will deal with it
        """
        Generate an image based on the text prompt and conditioning image URL using FLUX.1 Kontext
        """
        temp_image_file = None
        try:
            with torch.inference_mode(), print_timing("generate image"):
                seed = prepare_seed(seed)

                # Download image from URL
                print(f"Downloading image from URL: {input_image_url}")
                response = requests.get(input_image_url, stream=True)
                response.raise_for_status() # Raise an exception for bad status codes

                # Save to a temporary file
                # Create a named temporary file, ensuring it has an extension PIL can use.
                # The file will be deleted when closed.
                suffix = os.path.splitext(input_image_url)[1] or '.jpg' # Default to .jpg if no extension
                temp_image_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                for chunk in response.iter_content(chunk_size=8192):
                    temp_image_file.write(chunk)
                temp_image_file.close() # Close the file so PIL can open it

                print(f"Image downloaded and saved to temporary file: {temp_image_file.name}")

                if aspect_ratio == "match_input_image":
                    target_width, target_height = None, None
                else:
                    target_width, target_height = ASPECT_RATIOS[aspect_ratio]

                # Prepare input for kontext sampling using the downloaded image path
                inp, final_height, final_width = prepare_kontext(
                    t5=self.t5,
                    clip=self.clip,
                    prompt=prompt,
                    ae=self.ae,
                    img_cond_path=temp_image_file.name, # Use path of the temporary file
                    target_width=target_width,
                    target_height=target_height,
                    bs=1,
                    seed=seed,
                    device=self.device,
                )
            
                if go_fast:
                    compute_step_map = generate_compute_step_map("go really fast", num_inference_steps)
            else:
                compute_step_map = generate_compute_step_map("none", num_inference_steps)

            # Remove the original conditioning image from memory to save space
            inp.pop("img_cond_orig", None)

            # Get sampling schedule
            timesteps = get_schedule(
                num_inference_steps,
                inp["img"].shape[1],
                shift=True,  # flux-dev uses shift=True
            )

            # Generate image
            x = denoise(self.model, **inp, timesteps=timesteps, guidance=guidance, compute_step_map=compute_step_map)

            # Decode latents to pixel space
            x = unpack(x.float(), final_height, final_width)
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                x = self.ae.decode(x)

            # Convert to image
            x = x.clamp(-1, 1)
            x = (x + 1) / 2
            x = (x.permute(0, 2, 3, 1) * 255).to(torch.uint8).cpu().numpy()
            image = Image.fromarray(x[0])

            # Apply safety checking
            if not disable_safety_checker:
                images = self.safety_checker.filter_images([image])
                if not images:
                    raise Exception(
                        "Generated image contained NSFW content. Try running it again with a different prompt."
                    )
                image = images[0]

            # Save image
            output_path = f"output.{output_format}"
            if output_format == "png":
                image.save(output_path)
            elif output_format == "webp":
                image.save(
                    output_path, format="WEBP", quality=output_quality, optimize=True
                )
            else:  # jpg
                image.save(
                    output_path, format="JPEG", quality=output_quality, optimize=True
                )

            # Return the output path
            return Path(output_path)
        finally:
            # Clean up the temporary image file
            if temp_image_file and os.path.exists(temp_image_file.name):
                print(f"Cleaning up temporary file: {temp_image_file.name}")
                os.remove(temp_image_file.name)


def download_model_weights():
    """Download all required model weights if they don't exist"""
    # Download kontext weights
    if not os.path.exists(KONTEXT_WEIGHTS_PATH):
        print("Kontext weights not found, downloading...")
        download_weights(KONTEXT_WEIGHTS_URL, Path(KONTEXT_WEIGHTS_PATH))
        print("Kontext weights downloaded successfully")
    else:
        print("Kontext weights already exist")

    # Download autoencoder weights
    if not os.path.exists(AE_WEIGHTS_PATH):
        print("Autoencoder weights not found, downloading...")
        download_weights(AE_WEIGHTS_URL, Path(AE_WEIGHTS_PATH))
        print("Autoencoder weights downloaded successfully")
    else:
        print("Autoencoder weights already exist")

    if not os.path.exists(T5_WEIGHTS_PATH):
        print("T5 weights not found, downloading...")
        download_weights(T5_WEIGHTS_URL, Path(T5_WEIGHTS_PATH))
        print("T5 weights downloaded successfully")
    else:
        print("T5 weights already exist")
        
    if not os.path.exists(CLIP_PATH):
        print("CLIP weights not found, downloading...")
        download_weights(CLIP_URL, Path(CLIP_PATH))
        print("CLIP weights downloaded successfully")
    else:
        print("CLIP weights already exist")


def load_kontext_model(device: str | torch.device = "cuda"):
    """Load the kontext model with complete transformer weights"""
    # Use flux-dev config as base for kontext model
    config = configs["flux-dev"]

    print("Loading kontext model...")
    with torch.device("meta"):
        model = Flux(config.params).to(torch.bfloat16)

    # Load kontext weights (complete transformer)
    print(f"Loading kontext weights from {KONTEXT_WEIGHTS_PATH}")
    sd = load_sft(KONTEXT_WEIGHTS_PATH, device=str(device))
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)

    if missing:
        print(f"Missing keys: {missing}")
    if unexpected:
        print(f"Unexpected keys: {unexpected}")

    return model


def load_ae_local(device: str | torch.device = "cuda"):
    """Load autoencoder from local weights"""
    config = configs["flux-dev"]

    print("Loading autoencoder...")
    with torch.device("meta"):
        ae = AutoEncoder(config.ae_params)

    print(f"Loading autoencoder weights from {AE_WEIGHTS_PATH}")
    sd = load_sft(AE_WEIGHTS_PATH, device=str(device))
    missing, unexpected = ae.load_state_dict(sd, strict=False, assign=True)

    if missing:
        print(f"AE Missing keys: {missing}")
    if unexpected:
        print(f"AE Unexpected keys: {unexpected}")

    return ae


def prepare_seed(seed: int) -> int:
    if not seed:
        seed = int.from_bytes(os.urandom(2), "big")
    print(f"Using seed: {seed}")
    return seed
