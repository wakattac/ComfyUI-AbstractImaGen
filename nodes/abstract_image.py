from PIL import Image, ImageDraw, ImageOps, ImageFilter
import random
import os
import numpy as np
from scipy.ndimage import gaussian_filter
import torch 

# --- Configuration Class (Adapted for Node Inputs) ---
class AbstractImageConfig:
    def __init__(self, main_seed=None,
                 layer_probabilities=None,
                 feature_probabilities=None,
                 min_layers=2, max_layers=5,
                 min_shapes_per_layer=3, max_shapes_per_layer=12,
                 line_width_range=(5, 30),
                 grain_amount_range=(0.01, 0.08),
                 contrast_range=(0.8, 1.3),
                 brightness_range=(-20, 20),
                 final_blur_sigma_range=(0.4, 1.2),
                 feather_sigma_range=(1.0, 6.0),
                 component_salts=None,
                 toned_rgb_color=(128, 128, 128)
                 ):

        self.main_seed = main_seed if main_seed is not None else random.randint(0, 2**32 - 1)
        self.master_rng = random.Random(self.main_seed)

        self.layer_probabilities = layer_probabilities if layer_probabilities is not None else {
            'shapes': 0.5, 'pattern': 0.25, 'noise': 0.25
        }
        self._normalize_probabilities(self.layer_probabilities)

        self.feature_probabilities = feature_probabilities if feature_probabilities is not None else {
            'use_gradient_bg': 0.7, 'feather_layer': 0.2, 'add_grain': 0.15,
            'adjust_contrast': 0.2, 'adjust_brightness': 0.2, 'grayscale_final': 0.1, 'final_blur': 0.05
        }

        self.min_layers = min_layers
        self.max_layers = max_layers
        self.min_shapes_per_layer = min_shapes_per_layer
        self.max_shapes_per_layer = max_shapes_per_layer
        self.line_width_range = line_width_range

        self.grain_amount_range = grain_amount_range
        self.contrast_range = contrast_range
        self.brightness_range = brightness_range
        self.final_blur_sigma_range = final_blur_sigma_range
        self.feather_sigma_range = feather_sigma_range

        self._generate_component_seeds()
        self.component_salts = component_salts if component_salts is not None else {}

        self.toned_rgb_color = toned_rgb_color

    def _generate_component_seeds(self):
        """Generate seeds for different components based on main seed."""
        self.component_seeds = {
            'background': self.master_rng.randint(0, 2**32-1),
            'shapes': self.master_rng.randint(0, 2**32-1),
            'pattern': self.master_rng.randint(0, 2**32-1),
            'noise': self.master_rng.randint(0, 2**32-1),
            'post_processing': self.master_rng.randint(0, 2**32-1),
            'blending': self.master_rng.randint(0, 2**32-1),
            'color_generation': self.master_rng.randint(0, 2**32-1),
            'shape_type_bias': self.master_rng.randint(0, 2**32-1)
        }


    def _normalize_probabilities(self, probs_dict):
        """Normalizes probability dictionary values to sum to 1."""
        total = sum(probs_dict.values())
        if total > 0:
            for key in probs_dict:
                probs_dict[key] /= total
        else:
            num_items = len(probs_dict)
            if num_items > 0:
                for key in probs_dict:
                     probs_dict[key] = 1.0 / num_items


    def get_rng(self, component_name, salt=0):
        """Get a random number generator for a specific component."""
        base_seed = self.component_seeds.get(component_name, self.main_seed)
        user_salt = self.component_salts.get(component_name, 0)
        salted_seed = (base_seed + salt + user_salt) % (2**32)
        return random.Random(salted_seed)

    def should_apply(self, feature_name):
        """Determine if a feature should be applied based on its probability using post_processing RNG."""
        probability = self.feature_probabilities.get(feature_name, 0.0)
        feature_salt_offset = hash(feature_name) % (2**32)
        pp_rng = self.get_rng('post_processing', salt=feature_salt_offset)
        return pp_rng.random() < probability

    def choose_layer_type(self):
        """Choose a layer type based on configured probabilities using master RNG."""
        choices = list(self.layer_probabilities.keys())
        weights = list(self.layer_probabilities.values())
        if not choices or sum(weights) == 0:
             print("Warning: No valid layer types or zero probabilities. Choosing 'shapes' as fallback.")
             return 'shapes'
        return self.master_rng.choices(choices, weights=weights, k=1)[0]

    def choose_blend_mode(self, layer_index):
        """Choose a blend mode based on available options using blending RNG and layer index salt."""
        blend_modes = [
            blend_normal, blend_multiply, blend_screen, blend_overlay, blend_add,
            blend_subtract, blend_difference, blend_darken, blend_lighten,
            blend_color_dodge, blend_color_burn, blend_hard_light
        ]
        blend_rng = self.get_rng('blending', salt=layer_index)
        chosen_blend_func = blend_rng.choices(blend_modes, weights=[10] + [1] * (len(blend_modes) - 1), k=1)[0]
        return chosen_blend_func

    # Getter methods for parameters
    def get_grain_amount(self, salt=0):
        return self.get_rng('post_processing', salt).uniform(*self.grain_amount_range)

    def get_contrast(self, salt=0):
        return self.get_rng('post_processing', salt).uniform(*self.contrast_range)

    def get_brightness(self, salt=0):
        return self.get_rng('post_processing', salt).randint(*self.brightness_range)

    def get_final_blur_sigma(self, salt=0):
        return self.get_rng('post_processing', salt).uniform(*self.final_blur_sigma_range)

    def get_feather_sigma(self, salt=0):
        return self.get_rng('post_processing', salt).uniform(*self.feather_sigma_range)

    def get_num_layers(self):
        return self.master_rng.randint(self.min_layers, self.max_layers)

    def get_shapes_per_layer(self, salt=0):
        return self.get_rng('shapes', salt).randint(self.min_shapes_per_layer, self.max_shapes_per_layer)

    def get_line_width(self, salt=0):
        return self.get_rng('shapes', salt).randint(*self.line_width_range)


# --- NumPy <-> PIL Conversion ---
def pil_to_np(img):
    """Convert PIL Image to NumPy array (RGBA)."""
    img = img.convert('RGBA')
    return np.array(img)

def np_to_pil(arr):
    """Convert NumPy array (RGBA) to PIL Image."""
    if arr.ndim == 3 and arr.shape[2] == 3:
        arr = np.dstack((arr, np.full(arr.shape[:2], 255, dtype=np.uint8)))
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, 'RGBA')

# --- Blending Modes (NumPy) ---
def blend_normal(base, layer): return layer
def blend_multiply(base, layer): return base * layer
def blend_screen(base, layer): return 1.0 - (1.0 - base) * (1.0 - layer)
def blend_overlay(base, layer): return np.where(base < 0.5, 2.0 * base * layer, 1.0 - 2.0 * (1.0 - base) * (1.0 - layer))
def blend_add(base, layer): return np.minimum(base + layer, 1.0)
def blend_subtract(base, layer): return np.maximum(base - layer, 0.0)
def blend_difference(base, layer): return np.abs(base - layer)
def blend_darken(base, layer): return np.minimum(base, layer)
def blend_lighten(base, layer): return np.maximum(base, layer)
def blend_color_dodge(base, layer):
    mask = layer >= 1.0
    result = np.zeros_like(base)
    valid_pixels = ~mask
    result[valid_pixels] = np.minimum(base[valid_pixels] / (1.0 - layer[valid_pixels]), 1.0)
    result[mask] = 1.0
    return result
def blend_color_burn(base, layer):
    mask = layer <= 0.0
    result = np.zeros_like(base)
    valid_pixels = ~mask
    result[valid_pixels] = 1.0 - np.minimum((1.0 - base[valid_pixels]) / layer[valid_pixels], 1.0)
    return result
def blend_hard_light(base, layer): return np.where(layer < 0.5, 2.0 * base * layer, 1.0 - 2.0 * (1.0 - base) * (1.0 - layer))

# --- Alpha Compositing (NumPy) ---
def alpha_composite_with_blend(base_rgba, layer_rgba, blend_func=blend_normal):
    expected_shape_prefix = base_rgba.shape[:2]
    assert base_rgba.shape[:2] == layer_rgba.shape[:2], f"Base shape {base_rgba.shape[:2]} != Layer shape {layer_rgba.shape[:2]}"
    assert base_rgba.shape[2] == 4, f"Base must be RGBA, got shape {base_rgba.shape}"
    assert layer_rgba.shape[2] == 4, f"Layer must be RGBA, got shape {layer_rgba.shape}"

    base_rgb = base_rgba[..., :3].astype(np.float32) / 255.0
    base_a = base_rgba[..., 3:].astype(np.float32) / 255.0
    layer_rgb = layer_rgba[..., :3].astype(np.float32) / 255.0
    layer_a = layer_rgba[..., 3:].astype(np.float32) / 255.0

    expected_alpha_shape = expected_shape_prefix + (1,)
    assert base_a.shape == expected_alpha_shape, f"Base alpha shape {base_a.shape} != Expected {expected_alpha_shape}"
    assert layer_a.shape == expected_alpha_shape, f"Layer alpha shape {layer_a.shape} != Expected {expected_alpha_shape}"

    blended_rgb = blend_func(base_rgb, layer_rgb)

    OutA = layer_a + base_a * (1.0 - layer_a)
    OutRGB = np.zeros_like(base_rgb)
    mask = OutA > 1e-6

    numerator = blended_rgb * layer_a + base_rgb * base_a * (1.0 - layer_a)
    OutRGB[mask[..., 0]] = numerator[mask[..., 0]] / OutA[mask[..., 0]]

    inherit_base_mask = np.logical_and(~mask[..., 0], base_a[..., 0] > 1e-6)
    OutRGB[inherit_base_mask] = base_rgb[inherit_base_mask]

    final_rgba = np.dstack((np.clip(OutRGB, 0, 1) * 255, np.clip(OutA, 0, 1) * 255)).astype(np.uint8)
    return final_rgba

# --- Drawing & Feature Functions (Updated) ---

def get_color_from_mode(rng, color_mode, alpha_range, toned_rgb_color=None):
    """Generates a color based on the chosen color mode and provided RNG."""
    alpha = rng.randint(*alpha_range)
    if color_mode == 'grayscale':
        val = rng.randint(0, 255)
        return (val, val, val, alpha)
    elif color_mode == 'toned-random':
        r = rng.randint(100, 255)
        g = rng.randint(150, 255)
        b = rng.randint(0, 100)
        return (r, g, b, alpha)
    elif color_mode == 'toned-green-yellow':
         r = rng.randint(150, 255)
         g = rng.randint(180, 255)
         b = rng.randint(0, 80)
         return (r, g, b, alpha)
    elif color_mode == 'toned-red-magenta':
         r = rng.randint(180, 255)
         g = rng.randint(0, 80)
         b = rng.randint(150, 255)
         return (r, g, b, alpha)
    elif color_mode == 'toned-blue-cyan':
         r = rng.randint(0, 80)
         g = rng.randint(150, 255)
         b = rng.randint(180, 255)
         return (r, g, b, alpha)
    elif color_mode == 'toned-rgb' and toned_rgb_color is not None:
         return (*toned_rgb_color, alpha)
    # Default 'rgb' or fallback if toned-rgb is missing color
    r = rng.randint(0, 255)
    g = rng.randint(0, 255)
    b = rng.randint(0, 255)
    return (r, g, b, alpha)

def get_specific_color(color_name, alpha):
    """Returns a specific color tuple."""
    colors = {
        'red': (255, 0, 0, alpha),
        'green': (0, 255, 0, alpha),
        'blue': (0, 0, 255, alpha),
        'cyan': (0, 255, 255, alpha),
        'magenta': (255, 0, 255, alpha),
        'yellow': (255, 255, 0, alpha),
        'black': (0, 0, 0, alpha),
        'white': (255, 255, 255, alpha)
    }
    return colors.get(color_name.lower(), (128, 128, 128, alpha)) # Default to gray if not found


def draw_shapes_on_layer(width, height, num_shapes, color_mode, config, salt=0,
                         filled_shape_color_mode='random', line_zigzag_color_mode='random'):
    """Draws shapes onto a new transparent PIL Image layer using seed-based RNG."""
    shapes_rng = config.get_rng('shapes', salt)
    color_rng = config.get_rng('color_generation', salt) # Use color generation RNG

    layer_pil = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer_pil)

    shape_type_bias_rng = config.get_rng('shape_type_bias', salt)
    filled_shape_types = ['rectangle', 'ellipse', 'triangle', 'polygon']
    line_shape_types = ['line', 'zigzag']
    all_shape_types = filled_shape_types + line_shape_types

    filled_prob = 0.7
    line_prob = 0.3
    shape_type_weights = [filled_prob/len(filled_shape_types)] * len(filled_shape_types) + \
                         [line_prob/len(line_shape_types)] * len(line_shape_types)


    for i in range(num_shapes):
        shape_salt = salt + i
        shape_rng_instance = config.get_rng('shapes', shape_salt)
        color_rng_instance = config.get_rng('color_generation', shape_salt + 100)

        shape_type = shape_type_bias_rng.choices(all_shape_types, weights=shape_type_weights, k=1)[0]

        is_outline = shape_rng_instance.random() < 0.25
        line_width = config.get_line_width(salt=shape_salt) if shape_type in line_shape_types or is_outline else 1

        fill_color = None
        outline_color = None
        line_fill = None

        if shape_type in filled_shape_types and not is_outline:
             if filled_shape_color_mode == 'random':
                  fill_color = get_color_from_mode(color_rng_instance, color_mode, alpha_range=(200, 255), toned_rgb_color=config.toned_rgb_color)
             else:
                  fill_color = get_specific_color(filled_shape_color_mode, alpha=color_rng_instance.randint(200, 255))

        if shape_type in line_shape_types or is_outline:
             if line_zigzag_color_mode == 'random':
                  outline_color = get_color_from_mode(color_rng_instance, color_mode, alpha_range=(220, 255), toned_rgb_color=config.toned_rgb_color)
                  line_fill = get_color_from_mode(color_rng_instance, color_mode, alpha_range=(180, 255), toned_rgb_color=config.toned_rgb_color)
             else:
                  outline_color = get_specific_color(line_zigzag_color_mode, alpha=color_rng_instance.randint(220, 255))
                  line_fill = get_specific_color(line_zigzag_color_mode, alpha=color_rng_instance.randint(180, 255))


        x1 = shape_rng_instance.randint(-int(width*0.2), int(width*1.2))
        y1 = shape_rng_instance.randint(-int(height*0.2), int(height*1.2))
        x2 = shape_rng_instance.randint(x1, int(width*1.2))
        y2 = shape_rng_instance.randint(y1, int(height*1.2))
        min_size = 15
        if x2 - x1 < min_size: x2 = x1 + min_size
        if y2 - y1 < min_size: y2 = y1 + min_size

        try:
            if shape_type == 'rectangle':
                draw.rectangle([(x1, y1), (x2, y2)], fill=fill_color, outline=outline_color, width=line_width if is_outline else 1)
            elif shape_type == 'ellipse':
                draw.ellipse([(x1, y1), (x2, y2)], fill=fill_color, outline=outline_color, width=line_width if is_outline else 1)
            elif shape_type == 'triangle':
                 p1 = (shape_rng_instance.randint(x1, x2), shape_rng_instance.randint(y1, y2))
                 p2 = (shape_rng_instance.randint(x1, x2), shape_rng_instance.randint(y1, y2))
                 p3 = (shape_rng_instance.randint(x1, x2), shape_rng_instance.randint(y1, y2))
                 draw.polygon([p1, p2, p3], fill=fill_color, outline=outline_color, width=line_width if is_outline else 1)
            elif shape_type == 'polygon':
                 num_sides = shape_rng_instance.randint(5, 8)
                 cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                 radius = min(x2 - x1, y2 - y1) // 2
                 if radius < min_size // 2: radius = min_size // 2
                 angle_step = 2 * np.pi / num_sides
                 points = []
                 start_angle = shape_rng_instance.uniform(0, np.pi)
                 for j in range(num_sides):
                    angle = start_angle + j * angle_step
                    px = cx + int(radius * np.cos(angle))
                    py = cy + int(radius * np.sin(angle))
                    points.append((px, py))
                 if points:
                    draw.polygon(points, fill=fill_color, outline=outline_color, width=line_width if is_outline else 1)
            elif shape_type == 'line':
                draw.line([(x1, y1), (x2, y2)], fill=line_fill, width=line_width)
            elif shape_type == 'zigzag':
                 points = []
                 num_zigs = shape_rng_instance.randint(5, 20)
                 cur_x, cur_y = x1, y1
                 y_step = (y2 - y1) / num_zigs if num_zigs > 0 else 0
                 amplitude = (x2 - x1) / 2 if x2 > x1 else shape_rng_instance.randint(10, 50)
                 direction = 1
                 points.append((cur_x, cur_y))
                 for j in range(num_zigs):
                     next_y = y1 + (j + 1) * y_step
                     next_x = x1 + amplitude * direction
                     points.append((next_x, next_y))
                     cur_y = next_y
                     direction *= -1
                 if len(points) > 1:
                     draw.line(points, fill=line_fill, width=line_width, joint='miter')
        except (ValueError, TypeError) as e:
             continue
    return layer_pil

def draw_repeating_pattern_layer(width, height, color_mode, config, salt=0):
    """Creates a layer with a repeating tiled pattern using seed-based RNG."""
    pattern_rng = config.get_rng('pattern', salt)
    color_rng = config.get_rng('color_generation', salt + 1000)

    tile_sizes = [16, 32, 64, 128, 256, 512]
    tile_size = pattern_rng.choice(tile_sizes)

    tile_img_pil = Image.new('RGBA', (tile_size, tile_size), (0,0,0,0))
    tile_draw = ImageDraw.Draw(tile_img_pil)

    num_tile_shapes = pattern_rng.randint(1, 3)
    for i in range(num_tile_shapes):
        shape_salt = salt + 2000 + i
        shape_rng = config.get_rng('pattern', shape_salt)
        color_rng_instance = config.get_rng('color_generation', shape_salt + 100)

        shape = shape_rng.choice(['ellipse', 'rectangle', 'line'])
        fill = get_color_from_mode(color_rng_instance, color_mode, alpha_range=(150, 255), toned_rgb_color=config.toned_rgb_color)

        x1, y1 = shape_rng.randint(0, tile_size//2), shape_rng.randint(0, tile_size//2)
        x2, y2 = shape_rng.randint(x1, tile_size), shape_rng.randint(y1, tile_size)
        if shape == 'ellipse':
            tile_draw.ellipse([(x1,y1), (x2,y2)], fill=fill)
        elif shape == 'rectangle':
            tile_draw.rectangle([(x1,y1), (x2,y2)], fill=fill)
        elif shape == 'line':
            tile_draw.line([(x1,y1), (x2,y2)], fill=fill, width=shape_rng.randint(1,4))

    tile_np = pil_to_np(tile_img_pil)

    nx = int(np.ceil(width / tile_size))
    ny = int(np.ceil(height / tile_size))

    tiled_layer_np = np.tile(tile_np, (ny, nx, 1))

    final_layer_np = tiled_layer_np[:height, :width, :]

    alpha_factor = pattern_rng.uniform(0.5, 1.0)
    final_layer_np[..., 3] = (final_layer_np[..., 3] * alpha_factor).astype(np.uint8)

    return np_to_pil(final_layer_np)


def draw_perlin_noise_layer_fast(width, height, color_mode, config, salt=0):
    """Creates a layer using a faster sine-based noise approach with seed control."""
    noise_rng = config.get_rng('noise', salt)
    color_rng = config.get_rng('color_generation', salt + 2000)

    scale = noise_rng.uniform(0.005, 0.05)

    x = np.linspace(0, width * scale, width)
    y = np.linspace(0, height * scale, height)
    X, Y = np.meshgrid(x, y)

    noise = np.zeros((height, width))
    num_waves = noise_rng.randint(3, 8)

    for i in range(num_waves):
        wave_salt = salt + 3000 + i
        wave_rng = config.get_rng('noise', wave_salt)

        freq_x = wave_rng.uniform(0.1, 2.0)
        freq_y = wave_rng.uniform(0.1, 2.0)
        phase_x = wave_rng.uniform(0, 2 * np.pi)
        phase_y = wave_rng.uniform(0, 2 * np.pi)
        amplitude = 1.0 / (i + 1)

        noise += amplitude * np.sin(freq_x * X + phase_x) * np.sin(freq_y * Y + phase_y)

    min_noise, max_noise = np.min(noise), np.max(noise)
    if max_noise > min_noise:
        noise = (noise - min_noise) / (max_noise - min_noise)
    else:
        noise = np.zeros_like(noise)

    color1 = np.array(get_color_from_mode(color_rng, color_mode, alpha_range=(50, 150), toned_rgb_color=config.toned_rgb_color)[:3])
    color2 = np.array(get_color_from_mode(color_rng, color_mode, alpha_range=(100, 200), toned_rgb_color=config.toned_rgb_color)[:3])
    alpha_val = noise_rng.randint(80, 180)

    noise_rgb = color1[np.newaxis, np.newaxis, :] * (1 - noise[..., np.newaxis]) + \
                color2[np.newaxis, np.newaxis, :] * noise[..., np.newaxis]

    noise_rgba = np.dstack((noise_rgb.astype(np.uint8),
                           np.full((height, width), alpha_val, dtype=np.uint8)))

    return np_to_pil(noise_rgba)

def draw_gradient_np(width, height, color_mode, config, salt=0):
    """Creates a gradient as a NumPy array (RGBA) using seed-based RNG."""
    gradient_rng = config.get_rng('background', salt)
    color_rng = config.get_rng('color_generation', salt + 3000)

    gradient = np.zeros((height, width, 4), dtype=np.uint8)
    directions = ['vertical', 'horizontal', 'radial', 'diagonal']
    weights = [1, 1, 0.8, 0.8]
    direction = gradient_rng.choices(directions, weights=weights, k=1)[0]

    color1 = np.array(get_color_from_mode(color_rng, color_mode, alpha_range=(150, 255), toned_rgb_color=config.toned_rgb_color))[np.newaxis, np.newaxis, :]
    color2 = np.array(get_color_from_mode(color_rng, color_mode, alpha_range=(100, 200), toned_rgb_color=config.toned_rgb_color))[np.newaxis, np.newaxis, :]

    h_indices = np.linspace(0, 1, height)[:, np.newaxis, np.newaxis]
    w_indices = np.linspace(0, 1, width)[np.newaxis, :, np.newaxis]

    if direction == 'vertical':
        weights = h_indices
        gradient = color1 * (1 - weights) + color2 * weights
    elif direction == 'horizontal':
        weights = w_indices
        gradient = color1 * (1 - weights) + color2 * weights
    elif direction == 'radial':
        center_x, center_y = gradient_rng.uniform(0, 1), gradient_rng.uniform(0, 1)
        y, x = np.indices((height, width))
        y_norm, x_norm = y / (height - 1), x / (width - 1)
        dist = np.sqrt((x_norm - center_x)**2 + (y_norm - center_y)**2)
        max_dist = np.sqrt(max(center_x, 1-center_x)**2 + max(center_y, 1-center_y)**2)
        if max_dist < 1e-6: max_dist = 1.0
        dist = dist / max_dist
        weights = np.clip(dist, 0, 1)[:, :, np.newaxis]
        gradient = color1 * (1 - weights) + color2 * weights
    elif direction == 'diagonal':
        diag_indices = ((np.linspace(0, 1, height)[:, np.newaxis] + np.linspace(0, 1, width)[np.newaxis, :])) / 2
        weights = diag_indices[..., np.newaxis]
        gradient = color1 * (1 - weights) + color2 * weights

    target_shape = (height, width, 4)
    gradient = np.broadcast_to(gradient, target_shape)
    return gradient.astype(np.uint8)


# --- Post Processing Functions ---
def add_grain_np(arr, config, salt=0):
    """Adds Gaussian noise (grain) using seed-based RNG for parameters."""
    pp_rng_params = config.get_rng('post_processing', salt + 1)
    amount = config.get_grain_amount(salt=salt + 2)

    np_rng = np.random.RandomState(config.get_rng('post_processing', salt).randint(0, 2**32-1))

    noise = np_rng.normal(loc=0, scale=int(255 * amount), size=arr.shape[:2] + (3,))
    noisy_arr = arr[..., :3].astype(np.float32) + noise
    noisy_arr = np.clip(noisy_arr, 0, 255)
    return np.dstack((noisy_arr.astype(np.uint8), arr[..., 3]))

def adjust_contrast_np(arr, config, salt=0):
    """Adjusts contrast using seed-based RNG for parameters."""
    pp_rng = config.get_rng('post_processing', salt)
    contrast = config.get_contrast(salt=salt + 1)

    rgb = arr[..., :3].astype(np.float32)
    adjusted_rgb = rgb * contrast
    adjusted_rgb = np.clip(adjusted_rgb, 0, 255)
    return np.dstack((adjusted_rgb.astype(np.uint8), arr[..., 3]))

def adjust_brightness_np(arr, config, salt=0):
    """Adjusts brightness using seed-based RNG for parameters."""
    pp_rng = config.get_rng('post_processing', salt)
    brightness = config.get_brightness(salt=salt + 1)

    rgb = arr[..., :3].astype(np.float32)
    adjusted_rgb = rgb + brightness
    adjusted_rgb = np.clip(adjusted_rgb, 0, 255)
    return np.dstack((adjusted_rgb.astype(np.uint8), arr[..., 3]))


def apply_grayscale_np(arr, config, salt=0):
     pp_rng = config.get_rng('post_processing', salt)
     rgb = arr[...,:3].astype(np.float32)
     gray = np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
     gray_rgb = np.dstack((gray, gray, gray))
     return np.dstack((np.clip(gray_rgb, 0, 255).astype(np.uint8), arr[..., 3]))

def apply_final_blur_np(arr, config, salt=0):
    """Applies Gaussian blur using seed-based RNG for parameters."""
    pp_rng = config.get_rng('post_processing', salt)
    sigma = config.get_final_blur_sigma(salt=salt + 1)

    try:
       blurred_rgb = gaussian_filter(arr[..., :3].astype(float), sigma=(sigma, sigma, 0))
       blurred_rgb = np.clip(blurred_rgb, 0, 255).astype(np.uint8)
       return np.dstack((blurred_rgb, arr[..., 3]))
    except Exception as e:
       print(f"Post-processing blur failed with sigma {sigma}: {e}")
       return arr

def apply_feather_layer_np(arr, config, salt=0):
    """Applies Gaussian blur (feathering) to the alpha channel."""
    pp_rng = config.get_rng('post_processing', salt)
    sigma = config.get_feather_sigma(salt=salt + 1)

    if np.any(arr[..., 3]):
        try:
            feathered_alpha = gaussian_filter(arr[..., 3].astype(float), sigma=sigma)
            feathered_alpha = np.clip(feathered_alpha, 0, 255).astype(np.uint8)
            return np.dstack((arr[..., :3], feathered_alpha))
        except Exception as e:
            print(f"Feathering failed with sigma {sigma}: {e}")
            return arr
    return arr


# --- Main Generation Function ---
def generate_abstract_image_seeded(width, height, config, color_mode,
                                   filled_shape_color_mode, line_zigzag_color_mode):
    """
    Generates a randomized abstract image using controlled seeding and config.
    """
    bg_salt = 0
    use_gradient_bg = config.should_apply('use_gradient_bg')

    if use_gradient_bg:
        base_np = draw_gradient_np(width, height, color_mode, config, salt=bg_salt)
    else:
        bg_rng = config.get_rng('background', salt=bg_salt + 1)
        solid_color = get_color_from_mode(bg_rng, 'rgb', alpha_range=(255, 255), toned_rgb_color=config.toned_rgb_color)
        base_np = np.full((height, width, 4), solid_color, dtype=np.uint8)


    current_base = base_np.copy()
    num_layers = config.get_num_layers()

    for i in range(num_layers):
        layer_salt = 10000 + i * 1000
        layer_type = config.choose_layer_type()
        layer_pil = None

        if layer_type == 'shapes':
            shapes_per_layer = config.get_shapes_per_layer(salt=layer_salt)
            layer_pil = draw_shapes_on_layer(
                width, height, shapes_per_layer, color_mode, config, salt=layer_salt,
                filled_shape_color_mode=filled_shape_color_mode,
                line_zigzag_color_mode=line_zigzag_color_mode
                )
        elif layer_type == 'pattern':
            layer_pil = draw_repeating_pattern_layer(width, height, color_mode, config, salt=layer_salt)
        elif layer_type == 'noise':
            layer_pil = draw_perlin_noise_layer_fast(width, height, color_mode, config, salt=layer_salt)

        if layer_pil is not None:
            layer_np = pil_to_np(layer_pil)

            feather_salt = layer_salt + 500
            if config.should_apply('feather_layer'):
                layer_np = apply_feather_layer_np(layer_np, config, salt=feather_salt)

            chosen_blend_func = config.choose_blend_mode(layer_index=i)

            try:
                current_base = alpha_composite_with_blend(current_base, layer_np, blend_func=chosen_blend_func)
            except Exception as e:
                 print(f"!!! Compositing error on layer {i+1} (type: {layer_type}, blend: {chosen_blend_func.__name__}): {e}")


    final_np = current_base

    # --- Post-processing (on final NumPy array) ---
    grain_salt = 20000
    if config.should_apply('add_grain'):
        final_np = add_grain_np(final_np, config, salt=grain_salt)

    contrast_salt = 21000
    apply_contrast = config.should_apply('adjust_contrast')
    apply_brightness = config.should_apply('adjust_brightness')

    if apply_contrast:
         final_np = adjust_contrast_np(final_np, config, salt=contrast_salt)

    brightness_salt = 21500
    if apply_brightness:
         final_np = adjust_brightness_np(final_np, config, salt=brightness_salt)


    grayscale_salt = 22000
    if config.should_apply('grayscale_final') and color_mode != 'grayscale':
        final_np = apply_grayscale_np(final_np, config, salt=grayscale_salt)

    blur_salt = 23000
    if config.should_apply('final_blur'):
         final_np = apply_final_blur_np(final_np, config, salt=blur_salt)


    final_pil = np_to_pil(final_np)
    final_rgb_pil = Image.new("RGB", final_pil.size, (0, 0, 0))
    final_rgb_pil.paste(final_pil, mask=final_pil.split()[3])

    return final_rgb_pil


# --- ComfyUI Node Definition ---
class AbstractImageGenerator:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 8, "display": "number", "tooltip": "Width of the generated image."}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 8, "display": "number", "tooltip": "Height of the generated image."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "display": "number", "tooltip": "Main seed controlling overall randomness."}),
                "color_mode": (["rgb", "grayscale", "toned-random", "toned-green-yellow", "toned-red-magenta", "toned-blue-cyan", "toned-rgb"], {"default": "rgb", "tooltip": "Overall color scheme for the image."}),

                 "toned_rgb_r": ("INT", {"default": 128, "min": 0, "max": 255, "step": 1, "display": "number", "tooltip": "Red component for Toned RGB color mode."}),
                 "toned_rgb_g": ("INT", {"default": 128, "min": 0, "max": 255, "step": 1, "display": "number", "tooltip": "Green component for Toned RGB color mode."}),
                 "toned_rgb_b": ("INT", {"default": 128, "min": 0, "max": 255, "step": 1, "display": "number", "tooltip": "Blue component for Toned RGB color mode."}),

                "layer_shapes_prob": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "display": "number", "tooltip": "Probability of adding a layer of shapes."}),
                "layer_pattern_prob": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01, "display": "number", "tooltip": "Probability of adding a tiled pattern layer."}),
                "layer_noise_prob": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01, "display": "number", "tooltip": "Probability of adding a noise pattern layer."}),

                "num_layers_min": ("INT", {"default": 2, "min": 0, "max": 10, "step": 1, "display": "number", "tooltip": "Minimum number of feature layers."}),
                "num_layers_max": ("INT", {"default": 5, "min": 0, "max": 10, "step": 1, "display": "number", "tooltip": "Maximum number of feature layers."}),
                "shapes_per_layer_min": ("INT", {"default": 3, "min": 0, "max": 50, "step": 1, "display": "number", "tooltip": "Minimum number of shapes to draw per shapes layer."}),
                "shapes_per_layer_max": ("INT", {"default": 12, "min": 0, "max": 50, "step": 1, "display": "number", "tooltip": "Maximum number of shapes to draw per shapes layer."}),
                "line_width_min": ("INT", {"default": 5, "min": 1, "max": 100, "step": 1, "display": "number", "tooltip": "Minimum width for lines and zig-zags in shapes layers."}),
                "line_width_max": ("INT", {"default": 30, "min": 1, "max": 100, "step": 1, "display": "number", "tooltip": "Maximum width for lines and zig-zags in shapes layers."}),

                 "filled_shape_color_mode": (["random", "red", "green", "blue", "cyan", "magenta", "yellow", "black", "white"], {"default": "random", "tooltip": "Color selection mode for filled shapes."}),
                 "line_zigzag_color_mode": (["random", "red", "green", "blue", "cyan", "magenta", "yellow", "black", "white"], {"default": "random", "tooltip": "Color selection mode for lines and zig-zags."}),

                "bg_type_gradient_prob": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01, "display": "number", "tooltip": "Probability of using a gradient background instead of a solid color."}),
                "feather_layer_prob": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01, "display": "number", "tooltip": "Probability of feathering (blurring the alpha of) each feature layer."}),
                "add_grain_prob": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01, "display": "number", "tooltip": "Probability of adding photographic grain (noise) to the final image."}),
                "adjust_contrast_prob": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01, "display": "number", "tooltip": "Probability of adjusting the contrast of the final image."}),
                "adjust_brightness_prob": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01, "display": "number", "tooltip": "Probability of adjusting the brightness of the final image."}),
                "grayscale_final_prob": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01, "display": "number", "tooltip": "Probability of converting the final image to grayscale (if not already)."}),
                "final_blur_prob": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01, "display": "number", "tooltip": "Probability of applying a final overall blur to the image."}),

                "grain_amount_min": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.001, "display": "number", "tooltip": "Minimum amount of grain to add (if applied)."}),
                "grain_amount_max": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 1.0, "step": 0.001, "display": "number", "tooltip": "Maximum amount of grain to add (if applied)."}),
                "contrast_min": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 5.0, "step": 0.01, "display": "number", "tooltip": "Minimum contrast adjustment factor (if applied)."}),
                "contrast_max": ("FLOAT", {"default": 1.3, "min": 0.0, "max": 5.0, "step": 0.01, "display": "number", "tooltip": "Maximum contrast adjustment factor (if applied)."}),
                "brightness_min": ("INT", {"default": -20, "min": -255, "max": 255, "step": 1, "display": "number", "tooltip": "Minimum brightness adjustment amount (if applied)."}),
                "brightness_max": ("INT", {"default": 20, "min": -255, "max": 255, "step": 1, "display": "number", "tooltip": "Maximum brightness adjustment amount (if applied)."}),
                "final_blur_sigma_min": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 10.0, "step": 0.01, "display": "number", "tooltip": "Minimum sigma for the final blur (if applied)."}),
                "final_blur_sigma_max": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 10.0, "step": 0.01, "display": "number", "tooltip": "Maximum sigma for the final blur (if applied)."}),
                "feather_sigma_min": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1, "display": "number", "tooltip": "Minimum sigma for feathering layers (if applied)."}),
                "feather_sigma_max": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 20.0, "step": 0.1, "display": "number", "tooltip": "Maximum sigma for feathering layers (if applied)."}),

                "bg_salt": ("INT", {"default": 0, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff, "display": "number", "tooltip": "Salt for background generation randomness."}),
                "shapes_salt": ("INT", {"default": 0, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff, "display": "number", "tooltip": "Salt for shapes generation randomness."}),
                "pattern_salt": ("INT", {"default": 0, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff, "display": "number", "tooltip": "Salt for pattern generation randomness."}),
                "noise_salt": ("INT", {"default": 0, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff, "display": "number", "tooltip": "Salt for noise layer generation randomness."}),
                "post_processing_salt": ("INT", {"default": 0, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff, "display": "number", "tooltip": "Salt for post-processing randomness (e.g., applying features, parameters)."}),
                "blending_salt": ("INT", {"default": 0, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff, "display": "number", "tooltip": "Salt for blending mode selection randomness."}),
                "color_generation_salt": ("INT", {"default": 0, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff, "display": "number", "tooltip": "Salt for random color generation randomness."}),
                "shape_type_bias_salt": ("INT", {"default": 0, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff, "display": "number", "tooltip": "Salt for random shape type selection randomness (filled vs line/zigzag)."}),

            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "AbstractImage/Old-V1"

    def generate(self, width, height, seed, color_mode,
                 toned_rgb_r, toned_rgb_g, toned_rgb_b,
                 layer_shapes_prob, layer_pattern_prob, layer_noise_prob,
                 num_layers_min, num_layers_max, shapes_per_layer_min, shapes_per_layer_max,
                 line_width_min, line_width_max,
                 filled_shape_color_mode, line_zigzag_color_mode,
                 bg_type_gradient_prob, feather_layer_prob, add_grain_prob,
                 adjust_contrast_prob, adjust_brightness_prob, grayscale_final_prob, final_blur_prob,
                 grain_amount_min, grain_amount_max, contrast_min, contrast_max,
                 brightness_min, brightness_max, final_blur_sigma_min, final_blur_sigma_max,
                 feather_sigma_min, feather_sigma_max,
                 bg_salt, shapes_salt, pattern_salt, noise_salt, post_processing_salt, blending_salt,
                 color_generation_salt, shape_type_bias_salt
                 ):

        layer_probs = {
            'shapes': layer_shapes_prob,
            'pattern': layer_pattern_prob,
            'noise': layer_noise_prob,
        }

        feature_probs = {
            'use_gradient_bg': bg_type_gradient_prob,
            'feather_layer': feather_layer_prob,
            'add_grain': add_grain_prob,
            'adjust_contrast': adjust_contrast_prob,
            'adjust_brightness': adjust_brightness_prob,
            'grayscale_final': grayscale_final_prob,
            'final_blur': final_blur_prob,
        }

        component_salts = {
            'background': bg_salt,
            'shapes': shapes_salt,
            'pattern': pattern_salt,
            'noise': noise_salt,
            'post_processing': post_processing_salt,
            'blending': blending_salt,
            'color_generation': color_generation_salt,
            'shape_type_bias': shape_type_bias_salt
        }

        toned_rgb_color = (toned_rgb_r, toned_rgb_g, toned_rgb_b)

        config = AbstractImageConfig(
            main_seed=seed,
            layer_probabilities=layer_probs,
            feature_probabilities=feature_probs,
            min_layers=num_layers_min,
            max_layers=num_layers_max,
            min_shapes_per_layer=shapes_per_layer_min,
            max_shapes_per_layer=shapes_per_layer_max,
            line_width_range=(line_width_min, line_width_max),
            grain_amount_range=(grain_amount_min, grain_amount_max),
            contrast_range=(contrast_min, contrast_max),
            brightness_range=(brightness_min, brightness_max),
            final_blur_sigma_range=(final_blur_sigma_min, final_blur_sigma_max),
            feather_sigma_range=(feather_sigma_min, feather_sigma_max),
            component_salts=component_salts,
            toned_rgb_color=toned_rgb_color
        )

        generated_image_pil = generate_abstract_image_seeded(
            width=width,
            height=height,
            config=config,
            color_mode=color_mode,
            filled_shape_color_mode=filled_shape_color_mode,
            line_zigzag_color_mode=line_zigzag_color_mode
        )

        img_np = np.array(generated_image_pil).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np)[None,]

        return (img_tensor,)


NODE_CLASS_MAPPINGS = {
    "AbstractImageGenerator": AbstractImageGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AbstractImageGenerator": "Abstract Image Generator V1 (Random-Gacha-Mode)"
}
