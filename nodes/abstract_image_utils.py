# abstract_image_utils.py

from PIL import Image, ImageDraw
import random
import numpy as np
from scipy.ndimage import gaussian_filter
import torch

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
    divisor = 1.0 - layer[valid_pixels]
    divisor[divisor < 1e-6] = 1e-6
    result[valid_pixels] = np.minimum(base[valid_pixels] / divisor, 1.0)
    result[mask] = 1.0
    return result
def blend_color_burn(base, layer):
    mask = layer <= 0.0
    result = np.zeros_like(base)
    valid_pixels = ~mask
    divisor = layer[valid_pixels]
    divisor[divisor < 1e-6] = 1e-6
    result[valid_pixels] = 1.0 - np.minimum((1.0 - base[valid_pixels]) / divisor, 1.0)
    return result
def blend_hard_light(base, layer): return np.where(layer < 0.5, 2.0 * base * layer, 1.0 - 2.0 * (1.0 - base) * (1.0 - layer))

# --- Alpha Compositing (NumPy) ---
def alpha_composite_with_blend(base_rgba, layer_rgba, blend_func=blend_normal):
    """Composites with an optional blending mode for the RGB channels."""
    assert base_rgba.shape[:2] == layer_rgba.shape[:2], \
        f"Base shape {base_rgba.shape[:2]} != Layer shape {layer_rgba.shape[:2]}"
    assert base_rgba.shape[2] == 4, f"Base must be RGBA, got shape {base_rgba.shape}"
    assert layer_rgba.shape[2] == 4, f"Layer must be RGBA, got shape {layer_rgba.shape}"

    base_rgb = base_rgba[..., :3].astype(np.float32) / 255.0
    base_a = base_rgba[..., 3:].astype(np.float32) / 255.0
    layer_rgb = layer_rgba[..., :3].astype(np.float32) / 255.0
    layer_a = layer_rgba[..., 3:].astype(np.float32) / 255.0

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

# --- Color Generation ---

def hex_to_rgb(hex_string):
    """Converts a hex color string to an RGB tuple (0-255)."""
    hex_string = hex_string.lstrip('#')
    if len(hex_string) == 3:
        hex_string = ''.join([c*2 for c in hex_string])
    if len(hex_string) != 6:
        return (128, 128, 128) # Default to gray
    try:
        return tuple(int(hex_string[i:i+2], 16) for i in (0, 2, 4))
    except ValueError:
        return (128, 128, 128) # Default to gray


def get_color_from_mode(rng, color_mode, alpha, toned_rgb_color=None, hex_color=None):
    """Generates a color based on the chosen color mode and provided RNG."""
    if hex_color:
        rgb = hex_to_rgb(hex_color)
        return (*rgb, alpha)

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
    # Default 'rgb' mode or fallback
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
    default_color = (128, 128, 128, alpha)
    return colors.get(color_name.lower(), default_color)

# --- Quadrant/Position Handling ---
def get_quadrant_coords(width, height, position):
    """Returns the bounding box coordinates for a given position."""
    if position == 'full':
        return (0, 0, width, height)
    elif position == 'upper_right':
        return (width // 2, 0, width, height // 2)
    elif position == 'upper_left':
        return (0, 0, width // 2, height // 2)
    elif position == 'lower_right':
        return (width // 2, height // 2, width, height)
    elif position == 'lower_left':
        return (0, height // 2, width // 2, height)
    elif position == 'center':
        cx = width // 2
        cy = height // 2
        half_width = width // 4
        half_height = height // 4
        return (cx - half_width, cy - half_height, cx + half_width, cy + half_height)
    return (0, 0, width, height)

# --- Drawing Functions ---

def draw_filled_shapes_on_layer(width, height, num_shapes,
                                rng,
                                filled_shape_color_mode='random', filled_hex_color=None,
                                alpha=255,
                                max_shape_size=None, position='full'):
    """Draws ONLY filled shapes onto a new transparent PIL Image layer using provided RNG."""

    layer_pil = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer_pil)

    filled_shape_types = ['rectangle', 'ellipse', 'triangle', 'polygon']
    shape_weights = [1] * len(filled_shape_types)

    position_x1, position_y1, position_x2, position_y2 = get_quadrant_coords(width, height, position)

    for i in range(num_shapes):
        shape_rng = random.Random(rng.randint(0, 2**32-1))
        color_rng = random.Random(rng.randint(0, 2**32-1))

        shape_type = shape_rng.choices(filled_shape_types, weights=shape_weights, k=1)[0]

        is_outline = shape_rng.random() < 0.25

        fill_color = None
        outline_color = None

        current_alpha = alpha

        if not is_outline:
             if filled_shape_color_mode == 'random':
                  fill_color = get_color_from_mode(color_rng, 'rgb', current_alpha)
             elif filled_shape_color_mode == 'hex' and filled_hex_color:
                  fill_color = get_color_from_mode(color_rng, 'rgb', current_alpha, hex_color=filled_hex_color)
             else:
                  fill_color = get_specific_color(filled_shape_color_mode, current_alpha)

        if is_outline:
             if filled_shape_color_mode == 'random':
                  outline_color = get_color_from_mode(color_rng, 'rgb', current_alpha)
             elif filled_shape_color_mode == 'hex' and filled_hex_color:
                  outline_color = get_color_from_mode(color_rng, 'rgb', current_alpha, hex_color=filled_hex_color)
             else:
                  outline_color = get_specific_color(filled_shape_color_mode, current_alpha)

        line_width = shape_rng.randint(1, 8)


        x1 = shape_rng.randint(position_x1 - int(width*0.1), position_x2 + int(width*0.1))
        y1 = shape_rng.randint(position_y1 - int(height*0.1), position_y2 + int(height*0.1))
        x2 = shape_rng.randint(x1, position_x2 + int(width*0.1))
        y2 = shape_rng.randint(y1, position_y2 + int(height*0.1))

        if max_shape_size is not None and max_shape_size > 0:
             current_width = abs(x2 - x1)
             current_height = abs(y2 - y1)
             if current_width > max_shape_size:
                 if x1 < x2: x2 = x1 + max_shape_size
                 else: x1 = x2 + max_shape_size
             if current_height > max_shape_size:
                 if y1 < y2: y2 = y1 + max_shape_size
                 else: y1 = y2 + max_shape_size

        min_size = 15
        if abs(x2 - x1) < min_size: x2 = x1 + min_size * (1 if x2 > x1 else -1)
        if abs(y2 - y1) < min_size: y2 = y1 + min_size * (1 if y2 > y1 else -1)


        try:
            if shape_type == 'rectangle':
                draw.rectangle([(x1, y1), (x2, y2)], fill=fill_color, outline=outline_color, width=line_width if is_outline else 1)
            elif shape_type == 'ellipse':
                draw.ellipse([(x1, y1), (x2, y2)], fill=fill_color, outline=outline_color, width=line_width if is_outline else 1)
            elif shape_type == 'triangle':
                 p1 = (shape_rng.randint(x1, x2), shape_rng.randint(y1, y2))
                 p2 = (shape_rng.randint(x1, x2), shape_rng.randint(y1, y2))
                 p3 = (shape_rng.randint(x1, x2), shape_rng.randint(y1, y2))
                 draw.polygon([p1, p2, p3], fill=fill_color, outline=outline_color, width=line_width if is_outline else 1)
            elif shape_type == 'polygon':
                 num_sides = shape_rng.randint(5, 8)
                 cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                 radius = min(abs(x2 - x1), abs(y2 - y1)) // 2
                 if radius < min_size // 2: radius = min_size // 2
                 angle_step = 2 * np.pi / num_sides
                 points = []
                 start_angle = shape_rng.uniform(0, np.pi)
                 for j in range(num_sides):
                    angle = start_angle + j * angle_step
                    px = cx + int(radius * np.cos(angle))
                    py = cy + int(radius * np.sin(angle))
                    points.append((px, py))
                 if points:
                    draw.polygon(points, fill=fill_color, outline=outline_color, width=line_width if is_outline else 1)
        except (ValueError, TypeError) as e:
             continue
        except Exception as e:
             print(f"Error drawing shape {shape_type}: {e}")
             continue

    return layer_pil

def draw_lines_on_layer(width, height, num_shapes,
                        rng,
                        line_zigzag_color_mode='random', line_zigzag_hex=None,
                        alpha=255,
                        line_width_range=(5, 30), position='full'):
    """Draws ONLY lines and zigzags onto a new transparent PIL Image layer using provided RNG."""
    layer_pil = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer_pil)

    line_shape_types = ['line', 'zigzag']
    shape_weights = [1] * len(line_shape_types)

    position_x1, position_y1, position_x2, position_y2 = get_quadrant_coords(width, height, position)


    for i in range(num_shapes):
        shape_rng = random.Random(rng.randint(0, 2**32-1))
        color_rng = random.Random(rng.randint(0, 2**32-1))

        shape_type = shape_rng.choices(line_shape_types, weights=shape_weights, k=1)[0]

        line_width = shape_rng.randint(*line_width_range)

        line_fill = None
        current_alpha = alpha

        if line_zigzag_color_mode == 'random':
             line_fill = get_color_from_mode(color_rng, 'rgb', current_alpha)
        elif line_zigzag_color_mode == 'hex' and line_zigzag_hex:
             line_fill = get_color_from_mode(color_rng, 'rgb', current_alpha, hex_color=line_zigzag_hex)
        else:
             line_fill = get_specific_color(line_zigzag_color_mode, current_alpha)


        x1 = shape_rng.randint(position_x1 - int(width*0.1), position_x2 + int(width*0.1))
        y1 = shape_rng.randint(position_y1 - int(height*0.1), position_y2 + int(height*0.1))
        x2 = shape_rng.randint(x1, position_x2 + int(width*0.1))
        y2 = shape_rng.randint(y1, position_y2 + int(height*0.1))

        min_line_length = 20
        if abs(x2 - x1) < min_line_length and abs(y2 - y1) < min_line_length:
             if shape_rng.random() < 0.5:
                  x2 = x1 + min_line_length * (1 if x2 > x1 else -1)
             else:
                  y2 = y1 + min_line_length * (1 if y2 > y1 else -1)


        try:
            if shape_type == 'line':
                draw.line([(x1, y1), (x2, y2)], fill=line_fill, width=line_width)
            elif shape_type == 'zigzag':
                 points = []
                 num_zigs = shape_rng.randint(5, 20)
                 cur_x, cur_y = x1, y1
                 y_step = (y2 - y1) / num_zigs if num_zigs > 0 else 0
                 amplitude = (x2 - x1) / 2 if abs(x2 - x1) > 0 else shape_rng.randint(10, 50)
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
        except Exception as e:
             print(f"Error drawing shape {shape_type}: {e}")
             continue
    return layer_pil


# abstract_image_utils.py - Updated draw_repeating_pattern_layer function

def draw_repeating_pattern_layer(width, height,
                                 rng, # This is the RNG from the calling node
                                 pattern_color_mode='random', # Use the node's input parameter name
                                 pattern_color_hex=None, # Use the node's input parameter name
                                 alpha=255,
                                 tile_size=None, position='full'):
    """Creates a layer with a repeating tiled pattern using provided RNG."""
    pattern_rng = random.Random(rng.randint(0, 2**32-1)) # RNG for pattern structure
    # We need one color RNG for the tile if the mode is random.
    # Use the main RNG or a derived one for consistency if the mode is random.
    color_rng_for_tile = random.Random(rng.randint(0, 2**32-1)) # RNG for color determination *for the tile*


    if tile_size is None or tile_size <= 0:
        tile_sizes = [16, 32, 64, 128, 256, 512]
        tile_size = pattern_rng.choice(tile_sizes)

    tile_img_pil = Image.new('RGBA', (tile_size, tile_size), (0,0,0,0))
    tile_draw = ImageDraw.Draw(tile_img_pil)

    num_tile_shapes = pattern_rng.randint(1, 3)

    # --- Determine the color for all shapes in this tile based on mode/hex ---
    tile_shape_color = None
    current_alpha = alpha

    # Prioritize hex color if provided and valid
    if pattern_color_hex and len(pattern_color_hex.lstrip('#')) in [3, 6]: # Check for valid hex length (#RGB or #RRGGBB)
         try:
             rgb = hex_to_rgb(pattern_color_hex)
             tile_shape_color = (*rgb, current_alpha)
         except ValueError:
             print(f"Warning: Invalid hex color provided for pattern: {pattern_color_hex}. Falling back to color mode.")
             pass # Fallback to color mode logic

    # If hex was not used or invalid, use the color mode dropdown
    if tile_shape_color is None:
         if pattern_color_mode == 'random':
              # If mode is random, generate a single random color for this tile using color_rng_for_tile
              tile_shape_color = get_color_from_mode(color_rng_for_tile, 'rgb', current_alpha) # Generate random RGB for the tile
         else:
              # If a specific color name is chosen, get that color for the tile
              tile_shape_color = get_specific_color(pattern_color_mode, current_alpha)


    for i in range(num_tile_shapes):
        shape_rng = random.Random(pattern_rng.randint(0, 2**32-1)) # RNG for shape position/size *within the tile*
        # No need for a new color_rng_instance here, we use the determined tile_shape_color

        shape = shape_rng.choice(['ellipse', 'rectangle', 'line'])
        fill = tile_shape_color # Use the determined color for all shapes in this tile

        # ... shape drawing using 'fill' color ...
        x1, y1 = shape_rng.randint(0, tile_size//2), shape_rng.randint(0, tile_size//2)
        x2, y2 = shape_rng.randint(x1, tile_size), shape_rng.randint(y1, tile_size)
        if shape == 'ellipse':
            tile_draw.ellipse([(x1,y1), (x2,y2)], fill=fill)
        elif shape == 'rectangle':
            tile_draw.rectangle([(x1,y1), (x2,y2)], fill=fill)
        elif shape == 'line':
            tile_draw.line([(x1,y1), (x2,y2)], fill=fill, width=shape_rng.randint(1,4))


    tile_np = pil_to_np(tile_img_pil)

    # ... tiling and masking (same as before) ...

    nx = int(np.ceil(width / tile_size))
    ny = int(np.ceil(height / tile_size))

    tiled_layer_np = np.tile(tile_np, (ny, nx, 1))

    full_layer_np = tiled_layer_np[:height, :width, :]

    mask = np.zeros((height, width, 1), dtype=np.uint8)
    px1, py1, px2, py2 = get_quadrant_coords(width, height, position)
    mask[py1:py2, px1:px2, :] = 255

    final_layer_np = full_layer_np.copy()
    # Apply alpha from the tile drawing and the position mask
    final_layer_np[..., 3] = (final_layer_np[..., 3] * (mask[..., 0] / 255.0)).astype(np.uint8)


    return np_to_pil(final_layer_np)


def draw_perlin_noise_layer_fast(width, height,
                                 rng,
                                 color_mode='rgb',
                                 toned_rgb_color=None, hex_color=None,
                                 alpha=255,
                                 position='full'):
    """Creates a layer using a faster sine-based noise approach with seed control."""
    noise_rng = random.Random(rng.randint(0, 2**32-1))
    color_rng = random.Random(rng.randint(0, 2**32-1))

    scale = noise_rng.uniform(0.005, 0.05)

    x = np.linspace(0, width * scale, width)
    y = np.linspace(0, height * scale, height)
    X, Y = np.meshgrid(x, y)

    noise = np.zeros((height, width))
    num_waves = noise_rng.randint(3, 8)

    for i in range(num_waves):
        wave_rng = random.Random(noise_rng.randint(0, 2**32-1))

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

    current_alpha = alpha
    if hex_color:
         color1 = np.array(get_color_from_mode(color_rng, 'rgb', current_alpha, hex_color=hex_color)[:3])
         color2 = np.array(get_color_from_mode(color_rng, 'rgb', current_alpha, hex_color=hex_color)[:3])
    else:
         color1 = np.array(get_color_from_mode(color_rng, color_mode, current_alpha, toned_rgb_color=toned_rgb_color)[:3])
         color2 = np.array(get_color_from_mode(color_rng, color_mode, current_alpha, toned_rgb_color=toned_rgb_color)[:3])

    noise_rgb = color1[np.newaxis, np.newaxis, :] * (1 - noise[..., np.newaxis]) + \
                color2[np.newaxis, np.newaxis, :] * noise[..., np.newaxis]

    noise_rgba = np.dstack((noise_rgb.astype(np.uint8),
                           np.full((height, width), current_alpha, dtype=np.uint8)))

    mask = np.zeros((height, width, 1), dtype=np.uint8)
    px1, py1, px2, py2 = get_quadrant_coords(width, height, position)
    mask[py1:py2, px1:px2, :] = 255

    final_layer_np = noise_rgba.copy()
    final_layer_np[..., 3] = (final_layer_np[..., 3] * (mask[..., 0] / 255.0)).astype(np.uint8)

    return np_to_pil(final_layer_np)


# abstract_image_utils.py - Updated draw_gradient_np function

def draw_gradient_np(width, height,
                     rng,
                     start_hex=None, end_hex=None,
                     start_color_rgb=None, end_color_rgb=None,
                     direction='vertical'):
    """Creates a gradient as a NumPy array (RGBA) using provided RNG."""
    gradient_rng = random.Random(rng.randint(0, 2**32-1))

    # Determine start and end colors (This part remains the same as the last correction)
    alpha = 255
    if start_hex and end_hex:
        color1_rgb = hex_to_rgb(start_hex)
        color2_rgb = hex_to_rgb(end_hex)
    elif start_color_rgb is not None and end_color_rgb is not None: # Use explicit RGB tuples if provided
        color1_rgb = start_color_rgb
        color2_rgb = end_color_rgb
    else:
        # Fallback to internal random RGB generation if no explicit colors or hex are provided
        print("Warning: No explicit gradient colors provided (hex or RGB). Falling back to random RGB gradient.")
        color_rng = random.Random(rng.randint(0, 2**32-1)) # Use a new RNG for fallback random colors
        color1_rgb = get_color_from_mode(color_rng, 'rgb', alpha)[:3]
        color2_rgb = get_color_from_mode(color_rng, 'rgb', alpha)[:3]

    color1 = np.array((*color1_rgb, alpha))[np.newaxis, np.newaxis, :] # Shape (1, 1, 4)
    color2 = np.array((*color2_rgb, alpha))[np.newaxis, np.newaxis, :] # Shape (1, 1, 4)

    gradient_rgb = np.zeros((height, width, 3), dtype=np.float32) # Use float for calculation


    if direction == 'vertical':
        # Randomize start and end rows for the gradient interpolation
        start_y = gradient_rng.uniform(0, height)
        end_y = gradient_rng.uniform(0, height)
        y_indices = np.arange(height)[:, np.newaxis] # Shape (height, 1)
        weights_1d = np.clip((y_indices - start_y) / (end_y - start_y + 1e-6), 0, 1) # Shape (height, 1)
        # Repeat weights across the width and add a channel dimension
        weights = np.repeat(weights_1d, width, axis=1)[:, :, np.newaxis] # Shape (height, width, 1)

    elif direction == 'horizontal':
        # Randomize start and end columns for the gradient interpolation
        start_x = gradient_rng.uniform(0, width)
        end_x = gradient_rng.uniform(0, width)
        x_indices = np.arange(width)[np.newaxis, :] # Shape (1, width)
        weights_1d = np.clip((x_indices - start_x) / (end_x - start_x + 1e-6), 0, 1) # Shape (1, width)
        # Repeat weights across the height and add a channel dimension
        weights = np.repeat(weights_1d, height, axis=0)[:, :, np.newaxis] # Shape (height, width, 1)

    elif direction == 'radial':
        # Randomize the center of the radial gradient using the seeded RNG
        center_x, center_y = gradient_rng.uniform(0, width), gradient_rng.uniform(0, height) # Use pixel coordinates directly
        y, x = np.indices((height, width))
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        # Normalize distance based on max distance from center to corners
        max_dist = np.sqrt(max(center_x, width - 1 - center_x)**2 + max(center_y, height - 1 - center_y)**2)
        if max_dist < 1e-6: max_dist = 1.0
        dist_norm = dist / max_dist
        weights = np.clip(dist_norm, 0, 1)[:, :, np.newaxis] # Shape (height, width, 1)

    elif direction == 'diagonal':
        # For diagonal, we can randomize the starting and ending values of the sum of normalized indices
        y_indices_norm = np.arange(height)[:, np.newaxis] / (height - 1.0 + 1e-6) # Normalize 0-1, Shape (height, 1)
        x_indices_norm = np.arange(width)[np.newaxis, :] / (width - 1.0 + 1e-6) # Normalize 0-1, Shape (1, width)

        # Summing these broadcasts them to (height, width)
        diag_sum_normalized = (y_indices_norm + x_indices_norm) # Shape (height, width)

        # The range of diag_sum_normalized is approximately 0 to 2.
        # Randomize the values within this range that correspond to weight 0 and weight 1.
        min_sum_norm = 0.0
        max_sum_norm = 2.0 # Max possible sum of normalized indices is 1+1=2

        start_val_norm = gradient_rng.uniform(min_sum_norm, max_sum_norm)
        end_val_norm = gradient_rng.uniform(min_sum_norm, max_sum_norm)

        # Calculate weights based on the randomized start and end values, clipped to 0-1
        weights = np.clip((diag_sum_normalized - start_val_norm) / (end_val_norm - start_val_norm + 1e-6), 0, 1)[:, :, np.newaxis] # Shape (height, width, 1)

    # Now perform the color interpolation using the correctly shaped weights
    # Shape (1, 1, 3) * (height, width, 1) broadcasts to (height, width, 3) * (height, width, 3) -> (height, width, 3)
    gradient_rgb = color1[:,:,:3] * (1 - weights) + color2[:,:,:3] * weights


    # Stack with alpha channel (assuming alpha is constant 255 for background)
    final_rgba = np.dstack((np.clip(gradient_rgb, 0, 255).astype(np.uint8),
                           np.full((height, width), alpha, dtype=np.uint8)))


    return final_rgba.astype(np.uint8) # Ensure uint8 return type

# --- Post Processing Functions ---
def add_grain_np(arr, rng, amount_range=(0.01, 0.08)):
    """Adds Gaussian noise (grain) using provided RNG for parameters."""
    amount = rng.uniform(*amount_range)
    np_rng = np.random.RandomState(rng.randint(0, 2**32-1))
    noise = np_rng.normal(loc=0, scale=int(255 * amount), size=arr.shape[:2] + (3,))
    noisy_arr = arr[..., :3].astype(np.float32) + noise
    noisy_arr = np.clip(noisy_arr, 0, 255)
    return np.dstack((noisy_arr.astype(np.uint8), arr[..., 3]))

def adjust_contrast_np(arr, rng, contrast_range=(0.8, 1.3)):
    """Adjusts contrast using provided RNG for parameters."""
    contrast = rng.uniform(*contrast_range)
    rgb = arr[..., :3].astype(np.float32)
    adjusted_rgb = rgb * contrast
    adjusted_rgb = np.clip(adjusted_rgb, 0, 255)
    return np.dstack((adjusted_rgb.astype(np.uint8), arr[..., 3]))

def adjust_brightness_np(arr, rng, brightness_range=(-20, 20)):
    """Adjusts brightness using provided RNG for parameters."""
    brightness = rng.randint(*brightness_range)
    rgb = arr[..., :3].astype(np.float32)
    adjusted_rgb = rgb + brightness
    adjusted_rgb = np.clip(adjusted_rgb, 0, 255)
    return np.dstack((adjusted_rgb.astype(np.uint8), arr[..., 3]))

def apply_grayscale_np(arr):
     """Converts to grayscale."""
     rgb = arr[...,:3].astype(np.float32)
     gray = np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
     gray_rgb = np.dstack((gray, gray, gray))
     return np.dstack((np.clip(gray_rgb, 0, 255).astype(np.uint8), arr[..., 3]))

def apply_final_blur_np(arr, rng, sigma_range=(0.4, 1.2)):
    """Applies Gaussian blur using provided RNG for parameters."""
    sigma = rng.uniform(*sigma_range)
    try:
       blurred_rgb = gaussian_filter(arr[..., :3].astype(float), sigma=(sigma, sigma, 0))
       blurred_rgb = np.clip(blurred_rgb, 0, 255).astype(np.uint8)
       return np.dstack((blurred_rgb, arr[..., 3]))
    except Exception as e:
       print(f"Post-processing blur failed with sigma {sigma}: {e}")
       return arr

def apply_feather_layer_np(arr, rng, sigma_range=(1.0, 6.0)):
    """Applies Gaussian blur (feathering) to the alpha channel."""
    sigma = rng.uniform(*sigma_range)
    if np.any(arr[..., 3]):
        try:
            feathered_alpha = gaussian_filter(arr[..., 3].astype(float), sigma=sigma)
            feathered_alpha = np.clip(feathered_alpha, 0, 255).astype(np.uint8)
            return np.dstack((arr[..., :3], feathered_alpha))
        except Exception as e:
            print(f"Feathering failed with sigma {sigma}: {e}")
            return arr
    return arr

