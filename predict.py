# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import os
import time
import torch
import subprocess
import numpy as np
from PIL import Image
from transformers import pipeline
from diffusers.utils import load_image
from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel

MODEL_NAME = "DiffusionLight/DiffusionLight"
MODEL_CACHE = "checkpoints"
SDXL_URL = "https://weights.replicate.delivery/default/sdxl/sdxl-vae-upcast-fix.tar"

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

# create mask and depth map with mask for inpainting
def get_circle_mask(size=256):
    x = torch.linspace(-1, 1, size)
    y = torch.linspace(1, -1, size)
    y, x = torch.meshgrid(y, x)
    z = (1 - x**2 - y**2)
    mask = z >= 0
    return mask 

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        if not os.path.exists(MODEL_CACHE):
            download_weights(SDXL_URL, MODEL_CACHE)
        controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-depth-sdxl-1.0",
            torch_dtype=torch.float16,
            cache_dir=MODEL_CACHE,
        )
        self.pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnet,
            torch_dtype=torch.float16,
            cache_dir=MODEL_CACHE,
        ).to("cuda")
        self.pipe.load_lora_weights(MODEL_NAME)
        self.pipe.fuse_lora(lora_scale=0.75)
        self.depth_estimator = pipeline(task="depth-estimation", model="Intel/dpt-large")


    def predict(
        self,
        image: Path = Input(description="Input image"),
        prompt: str = Input(description="Prompt", default="a perfect mirrored reflective chrome ball sphere"),
        negative_prompt: str = Input(description="Negative Prompt", default="matte, diffuse, flat, dull"),
        num_inference_steps: int = Input(description="Number of inference steps", default=30),
        controlnet_conditioning_scale: float = Input(description="Controlnet conditioning scale", default=0.5),

    ) -> Path:
        """Run a single prediction on the model"""

        # prepare input image
        init_image = load_image(str(image))
        depth_image = self.depth_estimator(images=init_image)['depth']

        mask = get_circle_mask().numpy()
        depth = np.asarray(depth_image).copy()
        depth[384:640, 384:640] = depth[384:640, 384:640] * (1 - mask) + (mask * 255)
        depth_mask = Image.fromarray(depth)
        mask_image = np.zeros_like(depth)
        mask_image[384:640, 384:640] = mask * 255
        mask_image = Image.fromarray(mask_image)

        # run the pipeline
        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            image=init_image,
            mask_image=mask_image,
            control_image=depth_mask,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
        )

        output_path = "/tmp/output.png"
        output["images"][0].save(output_path)

        return Path(output_path)
