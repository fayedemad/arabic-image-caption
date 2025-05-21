"""
Script to split captions into train and validation sets for CNN-based model.
"""

import os
import random
from collections import defaultdict

def split_captions(caption_file, train_ratio=0.8, val_ratio=0.2, seed=42):
    """
    Split captions into train and validation sets.
    
    Args:
        caption_file: Path to the caption file
        train_ratio: Ratio of training data (default: 0.8)
        val_ratio: Ratio of validation data (default: 0.2)
        seed: Random seed for reproducibility
        
    Returns:
        train_captions: List of training captions
        val_captions: List of validation captions
    """
    # Set random seed
    random.seed(seed)
    
    # Read all captions
    with open(caption_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Group captions by image
    image_captions = defaultdict(list)
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Split by tab
        parts = line.split('\t')
        if len(parts) != 2:
            print(f"Warning: Invalid line format (no tab separator): {line}")
            continue
            
        image_id_with_number, caption = parts
        
        # Split image_id and number
        image_id_parts = image_id_with_number.split('#')
        if len(image_id_parts) != 2:
            print(f"Warning: Invalid image ID format (no #): {image_id_with_number}")
            continue
            
        image_id, caption_number = image_id_parts
        
        # Clean up
        image_id = image_id.strip()
        caption = caption.strip()
        
        if image_id and caption:
            image_captions[image_id].append((int(caption_number), caption))
    
    # Sort captions by number for each image
    for image_id in image_captions:
        image_captions[image_id].sort(key=lambda x: x[0])
    
    # Split images into train and val sets
    image_ids = list(image_captions.keys())
    random.shuffle(image_ids)
    
    split_idx = int(len(image_ids) * train_ratio)
    train_images = image_ids[:split_idx]
    val_images = image_ids[split_idx:]
    
    # Create train and val caption lists
    train_captions = []
    val_captions = []
    
    for image_id in train_images:
        for caption_number, caption in image_captions[image_id]:
            train_captions.append(f"{image_id}#{caption_number}\t{caption}")
            
    for image_id in val_images:
        for caption_number, caption in image_captions[image_id]:
            val_captions.append(f"{image_id}#{caption_number}\t{caption}")
    
    return train_captions, val_captions

def main():
    # Input and output paths
    caption_file = "data/arabic_captions.txt"
    train_output = "data/train_captions.txt"
    val_output = "data/val_captions.txt"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(train_output), exist_ok=True)
    
    # Split captions
    train_captions, val_captions = split_captions(
        caption_file,
        train_ratio=0.8,
        val_ratio=0.2,
        seed=42
    )
    
    # Save train captions
    with open(train_output, 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_captions))
    print(f"Saved {len(train_captions)} training captions to {train_output}")
    
    # Save validation captions
    with open(val_output, 'w', encoding='utf-8') as f:
        f.write('\n'.join(val_captions))
    print(f"Saved {len(val_captions)} validation captions to {val_output}")

if __name__ == "__main__":
    main() 