# abstract_image_postprocessing.py

import random
import torch
import numpy as np
from PIL import Image
from .abstract_image_utils import (
    pil_to_np, np_to_pil,
    add_grain_np, adjust_contrast_np, adjust_brightness_np,
    apply_grayscale_np, apply_final_blur_np
)

class AbstractImagePostprocessing:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",), # Input image
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "display": "number", "tooltip": "Seed for post-processing randomness."}),
                "randomize": ("BOOLEAN", {"default": False, "label": "Enable Randomization", "tooltip": "If true, randomize which effects are applied and their parameters using the seed."}),

                "apply_grain": ("BOOLEAN", {"default": False, "label": "Apply Grain", "tooltip": "Apply photographic grain (noise) to the image."}),
                "grain_amount": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.001, "display": "number", "tooltip": "Amount of grain to apply."}),
                "grain_amount_random_min": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.001, "display": "number", "tooltip": "Minimum grain amount if randomized."}),
                "grain_amount_random_max": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 1.0, "step": 0.001, "display": "number", "tooltip": "Maximum grain amount if randomized."}),


                "apply_contrast": ("BOOLEAN", {"default": False, "label": "Adjust Contrast", "tooltip": "Adjust the image contrast."}),
                "contrast_factor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01, "display": "number", "tooltip": "Contrast adjustment factor (1.0 is no change)."}),
                 "contrast_random_min": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 5.0, "step": 0.01, "display": "number", "tooltip": "Minimum contrast factor if randomized."}),
                 "contrast_random_max": ("FLOAT", {"default": 1.3, "min": 0.0, "max": 5.0, "step": 0.01, "display": "number", "tooltip": "Maximum contrast factor if randomized."}),


                "apply_brightness": ("BOOLEAN", {"default": False, "label": "Adjust Brightness", "tooltip": "Adjust the image brightness."}),
                "brightness_amount": ("INT", {"default": 0, "min": -255, "max": 255, "step": 1, "display": "number", "tooltip": "Brightness adjustment amount (-255 is black, 255 is white)."}),
                 "brightness_random_min": ("INT", {"default": -20, "min": -255, "max": 255, "step": 1, "display": "number", "tooltip": "Minimum brightness amount if randomized."}),
                 "brightness_random_max": ("INT", {"default": 20, "min": -255, "max": 255, "step": 1, "display": "number", "tooltip": "Maximum brightness amount if randomized."}),


                "apply_grayscale": ("BOOLEAN", {"default": False, "label": "Apply Grayscale", "tooltip": "Convert the image to grayscale."}),

                "apply_blur": ("BOOLEAN", {"default": False, "label": "Apply Blur", "tooltip": "Apply Gaussian blur to the image."}),
                "blur_sigma": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 10.0, "step": 0.01, "display": "number", "tooltip": "Sigma for Gaussian blur."}),
                 "blur_sigma_random_min": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 10.0, "step": 0.01, "display": "number", "tooltip": "Minimum blur sigma if randomized."}),
                 "blur_sigma_random_max": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 10.0, "step": 0.01, "display": "number", "tooltip": "Maximum blur sigma if randomized."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "AbstractImage/Postprocessing"

    def process(self, image, seed, randomize,
                apply_grain, grain_amount, grain_amount_random_min, grain_amount_random_max,
                apply_contrast, contrast_factor, contrast_random_min, contrast_random_max,
                apply_brightness, brightness_amount, brightness_random_min, brightness_random_max,
                apply_grayscale, apply_blur, blur_sigma, blur_sigma_random_min, blur_sigma_random_max):

        # Convert input tensor to PIL Image (RGBA for consistent processing)
        image = image.permute(0, 3, 1, 2) if image.shape[1] == 3 or image.shape[1] == 4 else image
        image = image.squeeze(0)
        image_np = image.cpu().numpy() * 255.0
        image_pil = Image.fromarray(np.clip(image_np, 0, 255).astype(np.uint8), 'RGB').convert('RGBA')
        current_np = pil_to_np(image_pil)

        main_rng = random.Random(seed)

        # --- Determine which effects to apply and their parameters based on randomization ---
        final_apply_grain = apply_grain
        final_grain_amount = grain_amount
        final_apply_contrast = apply_contrast
        final_contrast_factor = contrast_factor
        final_apply_brightness = apply_brightness
        final_brightness_amount = brightness_amount
        final_apply_grayscale = apply_grayscale
        final_apply_blur = apply_blur
        final_blur_sigma = blur_sigma

        if randomize:
             rand_rng = random.Random(seed + 100)

             final_apply_grain = apply_grain and rand_rng.random() < 0.5
             if final_apply_grain:
                  final_grain_amount = rand_rng.uniform(grain_amount_random_min, grain_amount_random_max)

             final_apply_contrast = apply_contrast and rand_rng.random() < 0.5
             if final_apply_contrast:
                  final_contrast_factor = rand_rng.uniform(contrast_random_min, contrast_random_max)

             final_apply_brightness = apply_brightness and rand_rng.random() < 0.5
             if final_apply_brightness:
                  final_brightness_amount = rand_rng.randint(brightness_random_min, brightness_random_max)

             final_apply_grayscale = apply_grayscale and rand_rng.random() < 0.5
             final_apply_blur = apply_blur and rand_rng.random() < 0.5
             if final_apply_blur:
                  final_blur_sigma = rand_rng.uniform(blur_sigma_random_min, blur_sigma_random_max)


        # --- Apply Post-processing Effects ---
        if final_apply_grain:
            grain_rng = random.Random(main_rng.randint(0, 2**32-1))
            current_np = add_grain_np(current_np, grain_rng, amount_range=(final_grain_amount, final_grain_amount))

        if final_apply_contrast:
            contrast_rng = random.Random(main_rng.randint(0, 2**32-1))
            current_np = adjust_contrast_np(current_np, contrast_rng, contrast_range=(final_contrast_factor, final_contrast_factor))

        if final_apply_brightness:
            brightness_rng = random.Random(main_rng.randint(0, 2**32-1))
            current_np = adjust_brightness_np(current_np, brightness_rng, brightness_range=(final_brightness_amount, final_brightness_amount))

        if final_apply_grayscale:
            current_np = apply_grayscale_np(current_np)

        if final_apply_blur:
            blur_rng = random.Random(main_rng.randint(0, 2**32-1))
            current_np = apply_final_blur_np(current_np, blur_rng, sigma_range=(final_blur_sigma, final_blur_sigma))


        # Convert final numpy array back to tensor
        final_pil = np_to_pil(current_np)
        final_rgb_pil = Image.new("RGB", final_pil.size, (0, 0, 0))
        final_rgb_pil.paste(final_pil, mask=final_pil.split()[3])


        img_np = np.array(final_rgb_pil).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np)[None,]

        return (img_tensor,)


# Node mapping for ComfyUI
NODE_CLASS_MAPPINGS = {
    "AbstractImagePostprocessing": AbstractImagePostprocessing
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AbstractImagePostprocessing": "Abstract Image Postprocessing"
}
