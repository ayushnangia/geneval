# Indianized Prompt Generation Script

This script generates text prompts with enhanced Indian cultural context, designed for evaluating text-to-image generation models. It builds upon a base set of standard prompt types (single object, multiple objects, counting, color, position, color attribution) and adds specific categories relevant to the Indian context.

## Purpose

The goal is to create a diverse evaluation suite that tests a model's ability to generate images based on prompts involving:
-   Common objects (including many specific to India).
-   Object counts and colors.
-   Relative spatial positions.
-   Objects within typical Indian settings (e.g., markets, festivals, specific locations).
-   People performing common Indian activities or wearing traditional Indian clothing.
-   Specific Indian food items and cultural objects.

## Dependencies

The script requires Python 3 and the following libraries:
-   `PyYAML`: For parsing the configuration file (`config/config.yaml`).
-   `NumPy`: For random sampling and numerical operations.

You can install the dependencies using pip:
```bash
pip3 install pyyaml numpy
```

## Configuration

Most customization is now handled through the `config/config.yaml` file. This file defines:
-   **File Paths:** Location of the `object_names.txt` file (relative to the script directory) and the default output path.
-   **Generation Parameters:** Default random seed, default number of prompts per category, probability of adding a setting, maximum count for counting prompts.
-   **Core Lists:** The lists used for generating prompts, including `colors`, `positions`, `indian_settings`, `indian_food`, `indian_clothing`, `indian_activities`, `indian_cultural_objects`, and `non_colorable_objects`.

**To customize the generation, modify the values within `config/config.yaml`.**

## Input File

The script still requires an `object_names.txt` file. The path to this file is now specified in `config/config.yaml` under `files.object_names_file`. By default, it expects the file to be in the same directory as the script. This file should contain a list of object class names, one per line. An example file is included.

## Usage

Run the script from your terminal using `python3` from the `indian_prompts` directory:

```bash
python3 create_prompts.py [OPTIONS]
```

**Available Options:**

Command-line options override the defaults specified in `config/config.yaml`:

-   `--seed SEED`: An integer seed for the random number generator. (Default: loaded from `config.yaml`)
-   `--num-prompts N`, `-n N`: The approximate number of prompts to generate *per category*. (Default: loaded from `config.yaml`)
-   `--output-path PATH`, `-o PATH`: The directory where the output files will be saved. (Default: loaded from `config.yaml`)

**Example:**

```bash
# Generate prompts using config defaults, output to current dir (default)
python3 create_prompts.py

# Override seed and number of prompts, output to a specific folder
python3 create_prompts.py --seed 123 -n 50 -o ./generated_prompts
```

## Output Files

The script generates two files in the specified output path:

1.  **`generation_prompts.txt`**: A plain text file containing the generated prompts, with one prompt per line.
2.  **`evaluation_metadata.jsonl`**: A JSON Lines file (`.jsonl`) where each line is a JSON object containing structured metadata for the corresponding prompt in `generation_prompts.txt`. This metadata includes:
    -   `tag`: The category of the prompt (e.g., `single_object`, `two_object`, `colors`, `indian_food`, `indian_activity`).
    -   `prompt`: The exact text prompt.
    -   `include`: A list of objects/attributes that should be present.
    -   `exclude`: (For `counting` tag) Objects/counts that should *not* be present.
    -   `setting`: (Optional) The Indian setting specified in the prompt.
    -   `food_item`: (For `indian_food` tag) The specific food item.
    -   `attire`: (For `indian_clothing` tag) The specific clothing item.
    -   `activity`: (For `indian_activity` tag) The specific activity.
    -   `cultural_item`: (For `indian_cultural` tag) The specific cultural object.
    -   Other keys like `color`, `position` depending on the prompt type.

## Further Customization

Beyond editing `config/config.yaml` and `object_names.txt`, you can further customize by:
-   Modifying the Python logic within the generation functions in `create_prompts.py` (e.g., changing prompt templates).
-   Adding new generation functions for other types of prompts and integrating them into the `generate_suite` function. 