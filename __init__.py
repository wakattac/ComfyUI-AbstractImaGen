# __init__.py

# Import all your node classes
from .nodes.abstract_image import NODE_CLASS_MAPPINGS as generator_mappings, NODE_DISPLAY_NAME_MAPPINGS as generator_names
from .nodes.abstract_image_background import NODE_CLASS_MAPPINGS as background_mappings, NODE_DISPLAY_NAME_MAPPINGS as background_names
from .nodes.abstract_image_filled_shapes import NODE_CLASS_MAPPINGS as filled_shapes_mappings, NODE_DISPLAY_NAME_MAPPINGS as filled_shapes_names
from .nodes.abstract_image_lines import NODE_CLASS_MAPPINGS as lines_mappings, NODE_DISPLAY_NAME_MAPPINGS as lines_names
from .nodes.abstract_image_pattern import NODE_CLASS_MAPPINGS as pattern_mappings, NODE_DISPLAY_NAME_MAPPINGS as pattern_names
from .nodes.abstract_image_postprocessing import NODE_CLASS_MAPPINGS as postprocessing_mappings, NODE_DISPLAY_NAME_MAPPINGS as postprocessing_names
from .nodes.abstract_image_noise import NODE_CLASS_MAPPINGS as noise_mappings, NODE_DISPLAY_NAME_MAPPINGS as noise_names

# Combine the mappings and names from all node files
NODE_CLASS_MAPPINGS = {
    **generator_mappings,
    **background_mappings,
    **filled_shapes_mappings,
    **lines_mappings,
    **pattern_mappings,
    **postprocessing_mappings,
    **noise_mappings,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **generator_names,
    **background_names,
    **filled_shapes_names,
    **lines_names,
    **pattern_names,
    **postprocessing_names,
    **noise_names,
}

# List all node class names in __all__
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

