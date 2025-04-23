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
-   `PyYAML`: For parsing configuration/data (though primarily used for deduplication structure in the original script, the dependency remains).
-   `NumPy`: For random sampling and numerical operations.

You can install the dependencies using pip:
```bash
pip3 install pyyaml numpy
```

## Input File

The script requires an `object_names.txt` file to be present in the **same directory** where the script is run. This file should contain a list of object class names, one per line. The script uses this list as the vocabulary for generating prompts. An example file with a mix of general and Indian-specific objects is included in this directory.

## Usage

Run the script from your terminal using `python3`:

```bash
python3 create_prompts.py [OPTIONS]
```

**Available Options:**

-   `--seed SEED`: (Optional) An integer seed for the random number generator to ensure reproducible results. Defaults to `43`.
-   `--num-prompts N`, `-n N`: (Optional) The approximate number of prompts to generate *per category* (both standard and Indian-specific). Defaults to `100`. Note that the total number of unique prompts generated may vary due to deduplication and the inclusion of single-object prompts for every item in `object_names.txt`.
-   `--output-path PATH`, `-o PATH`: (Optional) The directory where the output files will be saved. Defaults to the current directory (`.`).

**Example:**

```bash
# Generate prompts with seed 123, aiming for ~50 per category, output to current dir
python3 create_prompts.py --seed 123 -n 50
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

## Customization

You can customize the generated prompts by:
-   Editing the `indian_prompts/object_names.txt` file to add, remove, or modify object names.
-   Modifying the Python lists within `create_prompts.py` (e.g., `indian_settings`, `indian_food`, `indian_clothing`, `indian_activities`, `indian_cultural_objects`, `colors`, `positions`) to change the available options.
-   Adjusting the logic within the generation functions (e.g., changing the probability of adding a setting, modifying prompt templates).
-   Adding new generation functions for other types of prompts. 