"""
Generate Indianized prompts for evaluation
"""

import argparse
import json
import os
import random
import yaml

import numpy as np

# Load classnames
# Assumes object_names.txt is in the same directory as this script
try:
    with open("object_names.txt") as cls_file:
        classnames = [line.strip() for line in cls_file if line.strip()]
except FileNotFoundError:
    print("Error: object_names.txt not found in the current directory.")
    exit(1)

# Indian context lists
indian_settings = [
    "on a busy street in Delhi", "in a quiet village in Kerala",
    "at a crowded Mumbai railway station", "during a monsoon downpour",
    "in a Himalayan valley", "on a Ganges river ghat", "inside a Haveli",
    "at a bustling market", "during a wedding procession", "at a Diwali celebration",
    "during Holi festival", "near a chai stall", "in a paddy field",
    "outside a colourful Rajasthan fort"
]

indian_food = [
    "samosa", "jalebi", "biryani", "dosa", "idli", "chai", "lassi", "pani puri",
    "chapati", "naan", "paratha", "vada pav", "pav bhaji", "gulab jamun"
]

indian_clothing = [
    "sari", "kurta", "dhoti", "lehenga", "sherwani", "salwar kameez", "turban",
    " Nehru jacket" # Added space for article use
]

indian_activities = [
    "selling flowers", "drinking chai", "cooking roti", "playing cricket",
    "riding an autorickshaw", "carrying water pots", "offering prayers at a temple",
    "weaving a carpet", " haggling at a market", " celebrating Holi" # Added space for article use
]

indian_cultural_objects = [
    "diya", "rangoli", "puja thali", "kalash", "incense stick",
    "steel tiffin box", "earthen pot", "charpai", "mandir", "gurdwara",
    "mosque", "dhaba", "paan shop", "roadside tea stall", "hand pump",
    "weaving loom", "neem tree", "banyan tree", "marigold flower garland",
    "lotus flower", "langur monkey", "peacock", "cricket bat", "harmonium",
    "tabla", "sitar"
]

# Objects typically not described by simple colors (e.g., "a red biryani")
non_colorable_objects = [
    "autorickshaw", "cycle rickshaw", "bajaj scooter", "ambassador car", "tata truck",
    "mandir", "gurdwara", "mosque", "dhaba", "paan shop", "roadside tea stall",
    "hand pump", "weaving loom", "neem tree", "banyan tree", "langur monkey", "peacock",
    "harmonium", "tabla", "sitar", "biryani", "dosa", "idli", "chai", "lassi",
    "pani puri", "chapati", "naan", "paratha", "vada pav", "pav bhaji", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "computer keyboard", "tv remote",
    "microwave", "oven", "toaster", "refrigerator"
]
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
        setting = rng.choice(indian_settings) if rng.random() < 0.2 else None
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
    setting = rng.choice(indian_settings) if rng.random() < 0.2 else None
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

numbers = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
def generate_counting_sample(rng: np.random.Generator, max_count=4):
    TAG = "counting"
    idx = rng.choice(len(classnames))
    num = int(rng.integers(2, max_count + 1)) # Inclusive endpoint
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

colors = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white", "gold", "silver", "bright"]
def generate_color_sample(rng: np.random.Generator):
    TAG = "colors"
    if not colorable_classnames:
        print("Warning: No colorable classnames found. Skipping color sample.")
        return None # Or handle appropriately
    idx = rng.choice(colorable_idxs) # Choose from filtered indices
    # Ensure person is not the only colorable item or handle if needed
    if classnames[idx] == "person" and len(colorable_classnames) > 1:
         # Avoid "a [color] person" if other options exist, reroll once
         new_idx = rng.choice(colorable_idxs)
         if classnames[new_idx] != "person":
             idx = new_idx

    color = rng.choice(colors)
    return dict(
        tag=TAG,
        include=[
            {"class": classnames[idx], "count": 1, "color": color}
        ],
        prompt=f"a photo of {with_article(color)} {classnames[idx]}"
    )


positions = ["left of", "right of", "above", "below", "next to", "behind", "in front of", "near"]
def generate_position_sample(rng: np.random.Generator):
    TAG = "position"
    idx_a, idx_b = rng.choice(len(classnames), size=2, replace=False)
    position = rng.choice(positions)
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
         print("Warning: Not enough colorable classnames for color attribution. Skipping.")
         return None
    idxs = rng.choice(colorable_idxs, size=2, replace=False)
    idx_a, idx_b = idxs[0], idxs[1]

    cidx_a, cidx_b = rng.choice(len(colors), size=2, replace=False)
    return dict(
        tag=TAG,
        include=[
            {"class": classnames[idx_a], "count": 1, "color": colors[cidx_a]},
            {"class": classnames[idx_b], "count": 1, "color": colors[cidx_b]}
        ],
        prompt=f"a photo of {with_article(colors[cidx_a])} {classnames[idx_a]} and {with_article(colors[cidx_b])} {classnames[idx_b]}"
    )


# --- New Indian-Specific Generation Functions ---

def generate_indian_food_prompt(rng: np.random.Generator):
    TAG = "indian_food"
    food_item = rng.choice(indian_food)
    return dict(
        tag=TAG,
        include=[{"class": food_item, "count": 1}], # Assume food item is also in classnames or treat as attribute
        food_item=food_item,
        prompt=f"a photo of {with_article(food_item)}"
    )

def generate_indian_clothing_prompt(rng: np.random.Generator):
    TAG = "indian_clothing"
    clothing_item = rng.choice(indian_clothing)
    return dict(
        tag=TAG,
        include=[{"class": "person", "count": 1}], # Assumes a person is wearing it
        attire=clothing_item.strip(),
        prompt=f"a photo of a person wearing {with_article(clothing_item)}"
    )

def generate_indian_activity_prompt(rng: np.random.Generator):
    TAG = "indian_activity"
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
    cultural_item = rng.choice(indian_cultural_objects)
    return dict(
        tag=TAG,
        include=[{"class": cultural_item, "count": 1}], # Assume item is also in classnames or treat as attribute
        cultural_item=cultural_item,
        prompt=f"a photo of {with_article(cultural_item)}"
    )


# --- Modified Generation Suite ---

def generate_suite(rng: np.random.Generator, n: int = 100, output_path: str = "."):
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
        temp_samples.append(generate_counting_sample(rng, max_count=4))
    samples.extend(temp_samples)
    print(f"  Added {len(temp_samples)} counting samples.")

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
        temp_samples.append(generate_indian_food_prompt(rng))
    samples.extend(temp_samples)
    print(f"  Added {len(temp_samples)} Indian food samples.")

    # Generate Indian clothing samples
    temp_samples = []
    for _ in range(num_each_indian_type):
        temp_samples.append(generate_indian_clothing_prompt(rng))
    samples.extend(temp_samples)
    print(f"  Added {len(temp_samples)} Indian clothing samples.")

    # Generate Indian activity samples
    temp_samples = []
    for _ in range(num_each_indian_type):
        temp_samples.append(generate_indian_activity_prompt(rng))
    samples.extend(temp_samples)
    print(f"  Added {len(temp_samples)} Indian activity samples.")

     # Generate Indian cultural object samples
    temp_samples = []
    for _ in range(num_each_indian_type):
        temp_samples.append(generate_indian_cultural_prompt(rng))
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
    parser = argparse.ArgumentParser(description="Generate Indianized prompts for evaluation.")
    parser.add_argument("--seed", type=int, default=43, help="Generation seed (default: 43)")
    parser.add_argument("--num-prompts", "-n", type=int, default=100, help="Approx number of prompts per task category (default: 100)")
    parser.add_argument("--output-path", "-o", type=str, default=".", help="Output folder for prompts and metadata (default: '.')")
    args = parser.parse_args()

    # Use random seed for Python's random module as well
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    generate_suite(rng, args.num_prompts, args.output_path) 