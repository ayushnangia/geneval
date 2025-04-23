"""
Generate Indianized prompts for evaluation
"""

import argparse
import json
import os
import random
import yaml

import numpy as np

# --- Configuration Loading ---
CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'config', 'config.yaml')

try:
    with open(CONFIG_FILE, 'r') as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print(f"Error: Configuration file not found at {CONFIG_FILE}")
    exit(1)
except yaml.YAMLError as e:
    print(f"Error parsing configuration file {CONFIG_FILE}: {e}")
    exit(1)

# Extract config values
lists_config = config.get('lists', {})
params_config = config.get('parameters', {})
files_config = config.get('files', {})

# --- Load classnames ---
# Use path relative to the script's directory specified in config
object_names_file = os.path.join(os.path.dirname(__file__), files_config.get('object_names_file', 'object_names.txt'))
try:
    with open(object_names_file) as cls_file:
        classnames = [line.strip() for line in cls_file if line.strip()]
except FileNotFoundError:
    print(f"Error: Object names file not found at {object_names_file}")
    exit(1)

# --- Get lists from config ---
indian_settings = lists_config.get('indian_settings', [])
indian_food = lists_config.get('indian_food', [])
indian_clothing = lists_config.get('indian_clothing', [])
indian_activities = lists_config.get('indian_activities', [])
indian_cultural_objects = lists_config.get('indian_cultural_objects', [])
non_colorable_objects = lists_config.get('non_colorable_objects', [])
colors = lists_config.get('colors', [])
positions = lists_config.get('positions', [])


# --- Get parameters from config ---
setting_probability = params_config.get('setting_probability', 0.2)
counting_max_count = params_config.get('counting_max_count', 4)
default_seed = params_config.get('default_seed', 43)
default_num_prompts = params_config.get('default_num_prompts_per_category', 100)
default_output_path = files_config.get('output_path_default', '.')


# --- Derived Lists (Calculated after loading config and classnames) ---
colorable_idxs = [i for i, name in enumerate(classnames) if name not in non_colorable_objects]
colorable_classnames = [classnames[i] for i in colorable_idxs]


# Proper a vs an
def with_article(name: str):
    name = name.strip() # Handle potential leading spaces from lists
    if not name: return "" # Handle empty strings if any somehow occur
    if name.split()[0].lower() in "aeiou":
        return f"an {name}"
    return f"a {name}"

# Proper plural
def make_plural(name: str):
    name = name.strip()
    if not name: return ""
    # Simple pluralization, may need refinement for irregular nouns
    if name[-1] in "sxy" or name[-2:] in ["sh", "ch"]:
        return f"{name}es"
    elif name[-1] == 'y' and name[-2] not in 'aeiou':
         return f"{name[:-1]}ies"
    else:
        return f"{name}s"

# --- Modified Generation Functions ---

def generate_single_object_sample(rng: np.random.Generator, size: int = None):
    TAG = "single_object"
    if size > len(classnames):
        size = len(classnames)
        print(f"Warning: Not enough distinct classes for single_object, generating only {size} samples")
    return_scalar = size is None
    size = size or 1
    idxs = rng.choice(len(classnames), size=size, replace=False)
    samples = []
    for idx in idxs:
        # Use setting_probability from config
        setting = rng.choice(indian_settings) if rng.random() < setting_probability else None
        prompt = f"a photo of {with_article(classnames[idx])}"
        if setting:
            prompt += f" {setting}"
        sample = dict(
            tag=TAG,
            include=[{"class": classnames[idx], "count": 1}],
            prompt=prompt
        )
        if setting:
            sample["setting"] = setting
        samples.append(sample)

    if return_scalar:
        return samples[0]
    return samples

def generate_two_object_sample(rng: np.random.Generator):
    TAG = "two_object"
    idx_a, idx_b = rng.choice(len(classnames), size=2, replace=False)
    # Use setting_probability from config
    setting = rng.choice(indian_settings) if rng.random() < setting_probability else None
    prompt = f"a photo of {with_article(classnames[idx_a])} and {with_article(classnames[idx_b])}"
    if setting:
        prompt += f" {setting}"

    sample = dict(
        tag=TAG,
        include=[
            {"class": classnames[idx_a], "count": 1},
            {"class": classnames[idx_b], "count": 1}
        ],
        prompt=prompt
    )
    if setting:
        sample["setting"] = setting
    return sample

numbers = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"] # Can also move to config if needed
def generate_counting_sample(rng: np.random.Generator, max_count=None): # Use config default
    TAG = "counting"
    # Use counting_max_count from config, allow override
    current_max_count = max_count if max_count is not None else counting_max_count
    if current_max_count >= len(numbers):
        print(f"Warning: counting_max_count ({current_max_count}) exceeds available number words ({len(numbers)}). Adjusting.")
        current_max_count = len(numbers) - 1

    idx = rng.choice(len(classnames))
    num = int(rng.integers(2, current_max_count + 1)) # Inclusive endpoint

    # Ensure number word exists
    if num >= len(numbers):
         print(f"Error: Number {num} is too large for the 'numbers' list. Skipping sample.")
         return None # Or handle appropriately

    return dict(
        tag=TAG,
        include=[
            {"class": classnames[idx], "count": num}
        ],
        exclude=[
            {"class": classnames[idx], "count": num + 1} # Exclude n+1
        ],
        prompt=f"a photo of {numbers[num]} {make_plural(classnames[idx])}"
    )

# Colors list is loaded from config
def generate_color_sample(rng: np.random.Generator):
    TAG = "colors"
    if not colorable_classnames:
        print("Warning: No colorable classnames found based on config and object list. Skipping color sample.")
        return None # Or handle appropriately
    idx = rng.choice(colorable_idxs) # Choose from filtered indices
    # Ensure person is not the only colorable item or handle if needed
    if classnames[idx] == "person" and len(colorable_classnames) > 1:
         # Avoid "a [color] person" if other options exist, reroll once
         new_idx = rng.choice(colorable_idxs)
         if classnames[new_idx] != "person":
             idx = new_idx

    color = rng.choice(colors) # Use colors list from config
    return dict(
        tag=TAG,
        include=[
            {"class": classnames[idx], "count": 1, "color": color}
        ],
        prompt=f"a photo of {with_article(color)} {classnames[idx]}"
    )


# Positions list is loaded from config
def generate_position_sample(rng: np.random.Generator):
    TAG = "position"
    idx_a, idx_b = rng.choice(len(classnames), size=2, replace=False)
    position = rng.choice(positions) # Use positions list from config
    return dict(
        tag=TAG,
        include=[
            {"class": classnames[idx_b], "count": 1}, # Object B (reference)
            {"class": classnames[idx_a], "count": 1, "position": (position, 0)} # Object A relative to B (index 0)
        ],
        prompt=f"a photo of {with_article(classnames[idx_a])} {position} {with_article(classnames[idx_b])}"
    )


def generate_color_attribution_sample(rng: np.random.Generator):
    TAG = "color_attr"
    if len(colorable_idxs) < 2:
         print("Warning: Not enough colorable classnames for color attribution based on config/objects. Skipping.")
         return None
    idxs = rng.choice(colorable_idxs, size=2, replace=False)
    idx_a, idx_b = idxs[0], idxs[1]

    cidx_a, cidx_b = rng.choice(len(colors), size=2, replace=False) # Use colors from config
    return dict(
        tag=TAG,
        include=[
            {"class": classnames[idx_a], "count": 1, "color": colors[cidx_a]},
            {"class": classnames[idx_b], "count": 1, "color": colors[cidx_b]}
        ],
        prompt=f"a photo of {with_article(colors[cidx_a])} {classnames[idx_a]} and {with_article(colors[cidx_b])} {classnames[idx_b]}"
    )


# --- New Indian-Specific Generation Functions ---
# Use corresponding lists from config

def generate_indian_food_prompt(rng: np.random.Generator):
    TAG = "indian_food"
    if not indian_food: return None # Handle empty list
    food_item = rng.choice(indian_food)
    return dict(
        tag=TAG,
        include=[{"class": food_item, "count": 1}], # Assume food item is also in classnames or treat as attribute
        food_item=food_item,
        prompt=f"a photo of {with_article(food_item)}"
    )

def generate_indian_clothing_prompt(rng: np.random.Generator):
    TAG = "indian_clothing"
    if not indian_clothing: return None
    clothing_item = rng.choice(indian_clothing)
    return dict(
        tag=TAG,
        include=[{"class": "person", "count": 1}], # Assumes a person is wearing it
        attire=clothing_item.strip(),
        prompt=f"a photo of a person wearing {with_article(clothing_item)}"
    )

def generate_indian_activity_prompt(rng: np.random.Generator):
    TAG = "indian_activity"
    if not indian_activities: return None
    activity = rng.choice(indian_activities)
    # Determine if article is needed (simple check for starting verb)
    prompt_activity = activity.strip()
    if prompt_activity.split()[0].endswith('ing'): # heuristic
         prompt = f"a photo of a person {prompt_activity}"
    else:
         prompt = f"a photo of a person performing {with_article(prompt_activity)}" # Less common

    return dict(
        tag=TAG,
        include=[{"class": "person", "count": 1}], # Assumes a person is doing it
        activity=activity.strip(),
        prompt=prompt
    )

def generate_indian_cultural_prompt(rng: np.random.Generator):
    TAG = "indian_cultural"
    if not indian_cultural_objects: return None
    cultural_item = rng.choice(indian_cultural_objects)
    return dict(
        tag=TAG,
        include=[{"class": cultural_item, "count": 1}], # Assume item is also in classnames or treat as attribute
        cultural_item=cultural_item,
        prompt=f"a photo of {with_article(cultural_item)}"
    )


# --- Modified Generation Suite ---

def generate_suite(rng: np.random.Generator, n: int, output_path: str): # n and output_path passed from main
    samples = []
    num_base_classes = len(classnames) # Total number including Indian items
    num_each_std_type = n // 2 # Generate fewer standard types to make room for Indian ones
    num_each_indian_type = n // 2 # Generate specific Indian types

    print(f"Generating prompts (approx {num_each_std_type} per standard task, {num_each_indian_type} per Indian task)...")

    # Generate single object samples for ALL classes
    samples.extend(generate_single_object_sample(rng, size=num_base_classes))
    print(f"  Added {len(samples)} single object samples.")

    # Generate two object samples
    temp_samples = []
    for _ in range(num_each_std_type * 2): # Generate more initially due to potential duplicates/settings variation
        temp_samples.append(generate_two_object_sample(rng))
    samples.extend(temp_samples)
    print(f"  Added {len(temp_samples)} two object samples.")

    # Generate counting samples
    temp_samples = []
    for _ in range(num_each_std_type):
        sample = generate_counting_sample(rng) # max_count comes from config by default
        if sample: temp_samples.append(sample) # Check if sample was generated
    samples.extend(temp_samples)
    print(f"  Added {len(temp_samples)} counting samples (max count: {counting_max_count}).")

    # Generate color samples
    temp_samples = []
    for _ in range(num_each_std_type):
        sample = generate_color_sample(rng)
        if sample: temp_samples.append(sample)
    samples.extend(temp_samples)
    print(f"  Added {len(temp_samples)} color samples.")

    # Generate position samples
    temp_samples = []
    for _ in range(num_each_std_type):
        temp_samples.append(generate_position_sample(rng))
    samples.extend(temp_samples)
    print(f"  Added {len(temp_samples)} position samples.")

    # Generate color attribution samples
    temp_samples = []
    for _ in range(num_each_std_type):
         sample = generate_color_attribution_sample(rng)
         if sample: temp_samples.append(sample)
    samples.extend(temp_samples)
    print(f"  Added {len(temp_samples)} color attribution samples.")

    # Generate Indian food samples
    temp_samples = []
    for _ in range(num_each_indian_type):
        sample = generate_indian_food_prompt(rng)
        if sample: temp_samples.append(sample)
    samples.extend(temp_samples)
    print(f"  Added {len(temp_samples)} Indian food samples.")

    # Generate Indian clothing samples
    temp_samples = []
    for _ in range(num_each_indian_type):
         sample = generate_indian_clothing_prompt(rng)
         if sample: temp_samples.append(sample)
    samples.extend(temp_samples)
    print(f"  Added {len(temp_samples)} Indian clothing samples.")

    # Generate Indian activity samples
    temp_samples = []
    for _ in range(num_each_indian_type):
        sample = generate_indian_activity_prompt(rng)
        if sample: temp_samples.append(sample)
    samples.extend(temp_samples)
    print(f"  Added {len(temp_samples)} Indian activity samples.")

     # Generate Indian cultural object samples
    temp_samples = []
    for _ in range(num_each_indian_type):
        sample = generate_indian_cultural_prompt(rng)
        if sample: temp_samples.append(sample)
    samples.extend(temp_samples)
    print(f"  Added {len(temp_samples)} Indian cultural object samples.")


    # De-duplicate
    unique_samples, used_prompts = [], set()
    for sample in samples:
        # Deduplicate based on the prompt text itself
        prompt_text = sample['prompt']
        if prompt_text not in used_prompts:
            unique_samples.append(sample)
            used_prompts.add(prompt_text)

    print(f"Total samples generated: {len(samples)}, Unique prompts: {len(unique_samples)}")

    # Write to files
    os.makedirs(output_path, exist_ok=True)
    prompts_file = os.path.join(output_path, "generation_prompts.txt")
    metadata_file = os.path.join(output_path, "evaluation_metadata.jsonl")

    print(f"Writing prompts to {prompts_file}")
    with open(prompts_file, "w") as fp:
        for sample in unique_samples:
            print(sample['prompt'], file=fp)

    print(f"Writing metadata to {metadata_file}")
    with open(metadata_file, "w") as fp:
        for sample in unique_samples:
             # Ensure metadata is serializable
             serializable_sample = {}
             for key, value in sample.items():
                 if isinstance(value, tuple): # Handle tuple like position
                     serializable_sample[key] = list(value)
                 else:
                     serializable_sample[key] = value
             print(json.dumps(serializable_sample), file=fp)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Indianized prompts for evaluation using config/config.yaml.")
    # Use defaults from loaded config
    parser.add_argument("--seed", type=int, default=default_seed,
                        help=f"Generation seed (default: {default_seed} from config)")
    parser.add_argument("--num-prompts", "-n", type=int, default=default_num_prompts,
                        help=f"Approx number of prompts per task category (default: {default_num_prompts} from config)")
    parser.add_argument("--output-path", "-o", type=str, default=default_output_path,
                        help=f"Output folder for prompts and metadata (default: '{default_output_path}' from config)")
    args = parser.parse_args()

    # Use random seed for Python's random module as well
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # Pass n and output_path explicitly
    generate_suite(rng, args.num_prompts, args.output_path) 