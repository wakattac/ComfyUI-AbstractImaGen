# abstract_image_background.py

import random
import torch
import numpy as np
from PIL import Image
from .abstract_image_utils import draw_gradient_np, hex_to_rgb, get_color_from_mode, get_specific_color, np_to_pil

class AbstractImageBackground:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        color_modes = ["random"] + [c.lower() for c in ["Red", "Green", "Blue", "Cyan", "Magenta", "Yellow", "Black", "White"]]
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 8, "display": "number", "description": "Width of the generated image."}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 8, "display": "number", "description": "Height of the generated image."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "display": "number", "description": "Seed for background randomness."}),
                "randomize": ("BOOLEAN", {"default": False, "label": "Enable Randomization", "description": "If true, randomize background type, colors, and gradient direction using the seed. Hex inputs override random colors if valid."}),
                "background_type": (["solid", "gradient"], {"default": "solid", "description": "Choose between a solid color or gradient background."}),

                # Solid Color Options
                "solid_color_mode": (color_modes, {"default": "random", "description": "Color selection mode for solid background."}),
                "solid_color_hex": ("STRING", {"default": "", "placeholder": "#RRGGBB", "description": "Hex code for solid color background (overrides Solid Color Mode if valid). Example: #FF0000 for red."}),

                # Gradient Color Options
                "gradient_color_mode_start": (color_modes, {"default": "random", "description": "Color selection mode for the gradient start color."}), # New input
                "gradient_color_mode_end": (color_modes, {"default": "random", "description": "Color selection mode for the gradient end color."}),   # New input
                "gradient_start_hex": ("STRING", {"default": "", "placeholder": "#RRGGBB", "description": "Hex code for gradient start color (overrides Start Color Mode if valid). Example: #00FF00 for green."}),
                "gradient_end_hex": ("STRING", {"default": "", "placeholder": "#RRGGBB", "description": "Hex code for gradient end color (overrides End Color Mode if valid). Example: #0000FF for blue."}),
                "gradient_direction": (["vertical", "horizontal", "radial", "diagonal"], {"default": "vertical", "description": "Direction of the gradient."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "AbstractImage/Background"

    def generate(self, width, height, seed, randomize, background_type,
                 solid_color_mode, solid_color_hex,
                 gradient_color_mode_start, gradient_color_mode_end, gradient_start_hex, gradient_end_hex, gradient_direction): # Updated inputs

        rng = random.Random(seed)

        final_background_type = background_type
        final_solid_color = None
        final_gradient_direction = gradient_direction
        final_gradient_start_hex = gradient_start_hex
        final_gradient_end_hex = gradient_end_hex
        final_gradient_color_mode_start = gradient_color_mode_start # Use new input
        final_gradient_color_mode_end = gradient_color_mode_end   # Use new input

        alpha = 255

        if randomize:
            rand_rng = random.Random(seed) # Use the main seed for overall randomization control

            final_background_type = rand_rng.choice(["solid", "gradient"])

            # Randomize solid color (Hex overrides random mode if valid)
            solid_color_rng = random.Random(rand_rng.randint(0, 2**32-1))
            if solid_color_hex and self._is_valid_hex(solid_color_hex):
                 final_solid_color = hex_to_rgb(solid_color_hex) + (alpha,)
            else:
                 random_solid_mode_choice = solid_color_rng.choice(["random"] + [c.lower() for c in ["Red", "Green", "Blue", "Cyan", "Magenta", "Yellow", "Black", "White"]])
                 if random_solid_mode_choice == 'random':
                      final_solid_color = get_color_from_mode(solid_color_rng, 'rgb', alpha)
                 else:
                      final_solid_color = get_specific_color(random_solid_mode_choice, alpha)


            # Randomize gradient colors and direction (Hex overrides random mode if valid)
            grad_rng = random.Random(rand_rng.randint(0, 2**32-1))
            final_gradient_direction = grad_rng.choice(["vertical", "horizontal", "radial", "diagonal"])

            # Randomize gradient start color: prioritize provided hex, then random hex, then random specific color
            if gradient_start_hex and self._is_valid_hex(gradient_start_hex):
                 final_gradient_start_hex = gradient_start_hex
                 final_gradient_color_mode_start = 'hex' # Reflect that hex is being used
            else:
                 rand_start_choice_type = grad_rng.choice(['random_hex', 'specific_color', 'random_rgb_mode']) # Added 'random_rgb_mode' for completeness
                 if rand_start_choice_type == 'random_hex':
                      final_gradient_start_hex = '#%06x' % grad_rng.randint(0, 0xFFFFFF)
                      final_gradient_color_mode_start = 'random' # Set mode to random if generating random hex
                 elif rand_start_choice_type == 'specific_color':
                      rand_start_color_mode = grad_rng.choice([c.lower() for c in ["Red", "Green", "Blue", "Cyan", "Magenta", "Yellow", "Black", "White"]]) # Choose a specific color randomly
                      rgb_tuple = get_specific_color(rand_start_color_mode, alpha)[:3]
                      final_gradient_start_hex = '#%02x%02x%02x' % rgb_tuple # Corrected formatting
                      final_gradient_color_mode_start = rand_start_color_mode # Set the mode to the chosen specific color
                 else: # random_rgb_mode
                      rgb_tuple = get_color_from_mode(grad_rng, 'rgb', alpha)[:3]
                      final_gradient_start_hex = '#%02x%02x%02x' % rgb_tuple # Corrected formatting
                      final_gradient_color_mode_start = 'random'


            # Randomize gradient end color: prioritize provided hex, then random hex, then random specific color
            if gradient_end_hex and self._is_valid_hex(gradient_end_hex):
                 final_gradient_end_hex = gradient_end_hex
                 final_gradient_color_mode_end = 'hex' # Reflect that hex is being used
            else:
                 rand_end_choice_type = grad_rng.choice(['random_hex', 'specific_color', 'random_rgb_mode']) # Added 'random_rgb_mode' for completeness
                 if rand_end_choice_type == 'random_hex':
                      final_gradient_end_hex = '#%06x' % grad_rng.randint(0, 0xFFFFFF)
                      final_gradient_color_mode_end = 'random' # Set mode to random if generating random hex
                 elif rand_end_choice_type == 'specific_color':
                      rand_end_color_mode = grad_rng.choice([c.lower() for c in ["Red", "Green", "Blue", "Cyan", "Magenta", "Yellow", "Black", "White"]]) # Choose a specific color randomly
                      rgb_tuple = get_specific_color(rand_end_color_mode, alpha)[:3]
                      final_gradient_end_hex = '#%02x%02x%02x' % rgb_tuple # Corrected formatting
                      final_gradient_color_mode_end = rand_end_color_mode # Set the mode to the chosen specific color
                 else: # random_rgb_mode
                      rgb_tuple = get_color_from_mode(grad_rng, 'rgb', alpha)[:3]
                      final_gradient_end_hex = '#%02x%02x%02x' % rgb_tuple # Corrected formatting
                      final_gradient_color_mode_end = 'random'


        # --- Generate Background ---
        if final_background_type == 'solid':
            if final_solid_color is None: # If not randomized or hex was invalid/empty, determine based on fixed inputs
                 solid_color_rng = random.Random(seed + 1)
                 if solid_color_hex and self._is_valid_hex(solid_color_hex):
                      final_solid_color = hex_to_rgb(solid_color_hex) + (alpha,)
                 elif solid_color_mode == 'random':
                      final_solid_color = get_color_from_mode(solid_color_rng, 'rgb', alpha)
                 else:
                      final_solid_color = get_specific_color(solid_color_mode, alpha)

            base_np = np.full((height, width, 4), final_solid_color, dtype=np.uint8)

        elif final_background_type == 'gradient':
             grad_rng_gen = random.Random(seed + 2) # Use a separate RNG for the actual gradient generation if needed internally

             # Determine gradient start color based on hex or the new start dropdown
             start_color_rgb = None
             if final_gradient_start_hex and self._is_valid_hex(final_gradient_start_hex):
                  start_color_rgb = hex_to_rgb(final_gradient_start_hex)
             elif final_gradient_color_mode_start != 'random': # Use specific color from start dropdown
                  start_color_rgb = get_specific_color(final_gradient_color_mode_start, alpha)[:3]
             else: # Fallback for 'random' mode in start dropdown or other cases without hex/specific color
                 color_rng = random.Random(seed + 10) # Use a consistent fallback RNG seed
                 start_color_rgb = get_color_from_mode(color_rng, 'rgb', alpha)[:3]


             # Determine gradient end color based on hex or the new end dropdown
             end_color_rgb = None
             if final_gradient_end_hex and self._is_valid_hex(final_gradient_end_hex):
                  end_color_rgb = hex_to_rgb(final_gradient_end_hex)
             elif final_gradient_color_mode_end != 'random': # Use specific color from end dropdown
                  end_color_rgb = get_specific_color(final_gradient_color_mode_end, alpha)[:3]
             else: # Fallback for 'random' mode in end dropdown or other cases without hex/specific color
                 color_rng = random.Random(seed + 11) # Use a consistent fallback RNG seed with different salt
                 end_color_rgb = get_color_from_mode(color_rng, 'rgb', alpha)[:3]


             # Pass determined RGB colors to draw_gradient_np
             base_np = draw_gradient_np(
                 width, height,
                 grad_rng_gen, # Pass the gradient_rng for internal radial/diagonal randomness
                 start_color_rgb=start_color_rgb, # Pass explicit RGB tuple
                 end_color_rgb=end_color_rgb,     # Pass explicit RGB tuple
                 direction=final_gradient_direction,
             )


        # Convert to PIL and then to tensor
        base_pil = np_to_pil(base_np)
        # Ensure output is RGB for ComfyUI compatibility if alpha is not needed downstream
        final_rgb_pil = Image.new("RGB", base_pil.size, (0, 0, 0))
        final_rgb_pil.paste(base_pil, mask=base_pil.split()[3])


        img_np = np.array(final_rgb_pil).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np)[None,]

        return (img_tensor,)

    def _is_valid_hex(self, hex_string):
        """Helper to check if a string is a valid hex color."""
        if not isinstance(hex_string, str):
            return False
        hex_string = hex_string.lstrip('#')
        return len(hex_string) in [3, 6] and all(c in '0123456789abcdefABCDEF' for c in hex_string.lower())


# Node mapping for ComfyUI
NODE_CLASS_MAPPINGS = {
    "AbstractImageBackground": AbstractImageBackground
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AbstractImageBackground": "Abstract Image Background"
}
