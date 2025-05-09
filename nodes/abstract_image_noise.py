# abstract_image_noise.py

import random
import torch
import numpy as np
from PIL import Image, ImageDraw
from .abstract_image_utils import (
    pil_to_np, np_to_pil, alpha_composite_with_blend,
    get_color_from_mode, hex_to_rgb, get_quadrant_coords,
    draw_perlin_noise_layer_fast, apply_feather_layer_np
)

# Define blend modes for the dropdown
BLEND_MODES = ["normal", "multiply", "screen", "overlay", "add", "subtract",
               "difference", "darken", "lighten", "color_dodge", "color_burn", "hard_light"]
# Map string names to function objects
from .abstract_image_utils import (
    blend_normal, blend_multiply, blend_screen, blend_overlay, blend_add,
    blend_subtract, blend_difference, blend_darken, blend_lighten,
    blend_color_dodge, blend_color_burn, blend_hard_light
)
BLEND_MODE_FUNCS = {
    "normal": blend_normal, "multiply": blend_multiply, "screen": blend_screen,
    "overlay": blend_overlay, "add": blend_add, "subtract": blend_subtract,
    "difference": blend_difference, "darken": blend_darken, "lighten": blend_lighten,
    "color_dodge": blend_color_dodge, "color_burn": blend_color_burn, "hard_light": blend_hard_light
}


class AbstractImageNoise:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",), # Input image
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "display": "number", "description": "Seed for noise pattern randomness."}),
                "randomize": ("BOOLEAN", {"default": False, "label": "Enable Randomization", "description": "If true, randomize noise scale, colors, position, and blend mode using the seed."}),
                "num_layers": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1, "display": "number", "description": "Number of noise layers to add."}),
                "noise_color_mode": (["random", "hex"] + [c.lower() for c in ["Red", "Green", "Blue", "Cyan", "Magenta", "Yellow", "Black", "White"]], {"default": "random", "description": "Color selection mode for noise. 'random' uses implicit RGB."}),
                "noise_color_hex": ("STRING", {"default": "", "placeholder": "#RRGGBB", "description": "Hex code for noise color (used if Noise Color Mode is 'hex')."}),
                "alpha": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1, "display": "number", "description": "Alpha (transparency) for noise (0 is fully transparent, 255 is fully opaque)."}),
                "shape_position": (["full", "upper_right", "upper_left", "lower_right", "lower_left", "center"], {"default": "full", "description": "Restrict noise pattern to a specific position/quadrant."},),
                "blend_mode": (BLEND_MODES, {"default": "normal", "description": "Blending mode to composite the noise layer onto the base image."}),
                 "feather_layer": ("BOOLEAN", {"default": False, "label": "Feather Layer", "description": "Apply Gaussian blur to the alpha channel of the noise layer."}),
                "feather_sigma_min": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1, "display": "number", "description": "Minimum sigma for feathering (if applied)."}),
                "feather_sigma_max": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 20.0, "step": 0.1, "display": "number", "description": "Maximum sigma for feathering (if applied)."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "AbstractImage/Layers"

    def generate(self, image, seed, randomize, num_layers,
                 noise_color_mode, noise_color_hex, alpha,
                 shape_position,
                 blend_mode, feather_layer, feather_sigma_min, feather_sigma_max):

# Convert input tensor to PIL Image (Assuming ComfyUI [B, H, W, C] format)
        image = image.squeeze(0) # Remove batch dimension [H, W, C]
        image_np = image.cpu().numpy() # [H, W, C] - values typically 0-1
        image_np = np.clip(image_np * 255.0, 0, 255).astype(np.uint8) # Scale to 0-255 and clip/cast

        # Check the number of channels and create PIL Image accordingly
        if image_np.shape[-1] == 3: # Input is RGB
             image_pil = Image.fromarray(image_np, 'RGB').convert('RGBA') # Convert to RGBA to add an alpha channel
        elif image_np.shape[-1] == 4: # Input is RGBA
             image_pil = Image.fromarray(image_np, 'RGBA') # Already has alpha, use directly
        else:
             print(f"Warning: Unexpected input image channel count: {image_np.shape[-1]}. Attempting conversion assuming RGB.")
             # Assume RGB and convert to RGBA, discarding extra channels if any
             image_pil = Image.fromarray(image_np[...,:3], 'RGB').convert('RGBA')


        width, height = image_pil.size
        # Use pil_to_np from utils to ensure consistent RGBA numpy array
        current_base_np = pil_to_np(image_pil)

        main_rng = random.Random(seed)

        for i in range(num_layers):
            layer_seed = main_rng.randint(0, 2**32-1)
            layer_rng = random.Random(layer_seed)

            # --- Randomization based on toggle ---
            final_noise_color_mode = noise_color_mode
            final_noise_color_hex = noise_color_hex
            final_alpha = alpha
            final_shape_position = shape_position
            final_blend_mode = blend_mode
            final_feather_layer = feather_layer
            final_feather_sigma_range = (feather_sigma_min, feather_sigma_max)

            if randomize:
                 rand_rng = random.Random(layer_seed + 100)

                 color_options = ["random", "hex"] + [c.lower() for c in ["Red", "Green", "Blue", "Cyan", "Magenta", "Yellow", "Black", "White"]]
                 final_noise_color_mode = rand_rng.choice(color_options)

                 if final_noise_color_mode == 'hex':
                      final_noise_color_hex = '#%06x' % rand_rng.randint(0, 0xFFFFFF)
                 else:
                      final_noise_color_hex = None

                 final_alpha = rand_rng.randint(100, 255)
                 final_shape_position = rand_rng.choice(["full", "upper_right", "upper_left", "lower_right", "lower_left", "center"])
                 final_blend_mode = rand_rng.choice(BLEND_MODES)
                 final_feather_layer = rand_rng.random() < 0.5
                 if final_feather_layer:
                      final_feather_sigma_range = (rand_rng.uniform(0.5, 5.0), rand_rng.uniform(5.0, 15.0))


            # --- Draw Noise Layer ---
            noise_drawing_rng = random.Random(layer_seed + 200)

            layer_pil = draw_perlin_noise_layer_fast(
                width, height,
                noise_drawing_rng,
                color_mode='rgb', # Noise's internal random color generation uses RGB
                hex_color=final_noise_color_hex,
                alpha=final_alpha,
                position=final_shape_position
            )

            layer_np = pil_to_np(layer_pil)

            # Apply Feathering if enabled
            if final_feather_layer:
                 feather_rng = random.Random(layer_seed + 300)
                 layer_np = apply_feather_layer_np(layer_np, feather_rng, sigma_range=final_feather_sigma_range)


            # Get the blending function
            blend_func = BLEND_MODE_FUNCS.get(final_blend_mode, blend_normal)

            # Composite the layer
            try:
                current_base_np = alpha_composite_with_blend(current_base_np, layer_np, blend_func=blend_func)
            except Exception as e:
                 print(f"!!! Compositing error on layer {i+1} (type: Noise, blend: {final_blend_mode}): {e}")


        # Convert final numpy array back to tensor
        final_pil = np_to_pil(current_base_np)
        final_rgb_pil = Image.new("RGB", final_pil.size, (0, 0, 0))
        final_rgb_pil.paste(final_pil, mask=final_pil.split()[3])

        img_np = np.array(final_rgb_pil).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np)[None,]

        return (img_tensor,)


# Node mapping for ComfyUI
NODE_CLASS_MAPPINGS = {
    "AbstractImageNoise": AbstractImageNoise
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AbstractImageNoise": "Abstract Image Noise Layer"
}
