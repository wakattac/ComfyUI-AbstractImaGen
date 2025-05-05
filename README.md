# ComfyUI Abstract Image Generator

A custom node for ComfyUI that generates abstract images with comprehensive seed-based control over layers, shapes, patterns, noise, colors, and post-processing effects.

This node is designed to create unique abstract base images on the fly within your ComfyUI workflows, which can then be used as input for VAE encoding, image-to-image generation, or other creative processes.

## Features

* **Seed-Based Generation:** A main seed controls the overall randomness, ensuring reproducibility.
* **Component Salts:** Individual salts for different generation components (background, shapes, patterns, noise, post-processing, blending, color generation, shape type bias) allow for targeted variations while keeping the main seed consistent.
* **Layer Control:** Define the minimum and maximum number of layers and their probabilities (shapes, patterns, noise).
* **Shape Customization:** Control the number of shapes per layer, line widths for lines/zig-zags, and color modes for filled shapes and lines/zig-zags (random, specific colors).
* **Color Modes:** Choose from various color modes including RGB, Grayscale, and several Toned options (Random, Green-Yellow, Red-Magenta, Blue-Cyan), plus a custom Toned RGB mode with sliders.
* **Post-Processing Effects:** Probabilistically apply and control parameters for gradient background, layer feathering, grain (noise), contrast, brightness, grayscale conversion, and final blur.
* **ComfyUI Integration:** Seamlessly integrates into your ComfyUI workflow, outputting a standard `IMAGE` tensor.

## Installation

### ComfyUI-manager
TBD

### Manual Installation
1.  Navigate to your ComfyUI installation directory.
2.  Go into the `custom_nodes` folder.
3.  Open a terminal and run
    ```bash
    git clone https://github.com/wakattac/ComfyUI-AbstractImaGen
    ```
5.  If you have a Python environment set up for ComfyUI, you can install the dependencies listed in `requirements.txt` (though most are likely already installed with ComfyUI).
    Linux:
    ```bash
    cd ../..
    source venv/bin/activate
    cd path/to/comfyui/custom_nodes/ComfyUI-AbstractImaGen
    pip install -r requirements.txt
    ```
    Windows:
    ```powershell
    cd ..\..
    venv\Scripts\activate
    cd path\to\comfyui\custom_nodes\ComfyUI-AbstractImaGen
    pip install -r requirements.txt
    ```
7.  Restart ComfyUI.

You should now find the node under the "AbstractImage" category in the add node menu.

## Usage

Add the "Abstract Image Generator (Seeded)" node to your workflow. Connect its `IMAGE` output to any node that accepts an image input (e.g., VAE Encode, Save Image).

Adjust the various input parameters to control the generated image:

* **width, height:** Dimensions of the output image.
* **seed:** The main seed for overall randomness. Changing this will produce a completely different image structure and appearance.
* **color_mode:** Select the general color style. Use the `toned_rgb_r/g/b` sliders when `color_mode` is set to `toned-rgb`.
* **layer\_*_prob:** Probabilities for including different types of layers (shapes, pattern, noise). These are automatically normalized.
* **num\_layers\_min/max:** Range for the total number of feature layers.
* **shapes\_per\_layer\_min/max:** Range for the number of shapes drawn on a shapes layer.
* **line\_width\_min/max:** Range for the width of line and zigzag shapes.
* **filled\_shape\_color\_mode, line\_zigzag\_color\_mode:** Choose between random colors (based on the main `color_mode`) or specific predefined colors for different shape types.
* **bg\_type\_gradient\_prob:** Probability of the background being a gradient instead of a solid color.
* **feather\_layer\_prob:** Probability of applying a feathering effect (alpha blur) to individual layers.
* **add\_grain\_prob:** Probability of adding photographic grain to the final image.
* **adjust\_contrast\_prob, adjust\_brightness\_prob:** Probabilities for applying contrast and brightness adjustments.
* **grayscale\_final\_prob:** Probability of converting the final image to grayscale.
* **final\_blur\_prob:** Probability of applying a final overall blur.
* **\*\_amount/sigma/contrast/brightness\_min/max:** Ranges for the parameters of the post-processing effects when they are applied.
* **\*\_salt:** Individual salts for each component. Modifying a component's salt will change the randomness *only* within that component's operations, allowing for fine-tuning variations without altering other parts of the image structure controlled by different components or the main seed.

Experiment with the parameters and salts to discover a wide range of abstract visuals!

## Contributing

If you find issues or have suggestions, please open an issue or submit a pull request on the GitHub repository.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
