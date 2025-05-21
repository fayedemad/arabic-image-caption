"""
Script to create a small test dataset for quick validation of the training process.
"""

import os
import shutil
import random

def create_test_dataset(image_dir, caption_file, val_caption_file, output_dir, num_samples=10):
    """
    Create a small test dataset by randomly selecting images and their captions.
    
    Args:
        image_dir: Directory containing all images
        caption_file: Path to training caption file
        val_caption_file: Path to validation caption file
        output_dir: Directory to save test dataset
        num_samples: Number of samples to include in test dataset
    """
    # Create output directories
    test_image_dir = os.path.join(output_dir, 'images')
    os.makedirs(test_image_dir, exist_ok=True)
    
    # Read caption files
    with open(caption_file, 'r', encoding='utf-8') as f:
        train_captions = f.readlines()
    
    with open(val_caption_file, 'r', encoding='utf-8') as f:
        val_captions = f.readlines()
    
    # Combine and shuffle captions
    all_captions = train_captions + val_captions
    random.shuffle(all_captions)
    
    # Select random samples
    test_captions = all_captions[:num_samples]
    
    # Split into train and val (80% train, 20% val)
    split_idx = int(len(test_captions) * 0.8)
    train_test_captions = test_captions[:split_idx]
    val_test_captions = test_captions[split_idx:]
    
    # Create test caption files
    train_test_file = os.path.join(output_dir, 'train_test_captions.txt')
    val_test_file = os.path.join(output_dir, 'val_test_captions.txt')
    
    with open(train_test_file, 'w', encoding='utf-8') as f:
        f.writelines(train_test_captions)
    
    with open(val_test_file, 'w', encoding='utf-8') as f:
        f.writelines(val_test_captions)
    
    # Copy selected images
    processed_images = set()  # Keep track of processed images
    for caption in test_captions:
        # Extract base image name without suffix
        image_name = caption.split('\t')[0].split('#')[0].strip()
        
        # Skip if we've already processed this image
        if image_name in processed_images:
            continue
            
        src_path = os.path.join(image_dir, image_name)
        dst_path = os.path.join(test_image_dir, image_name)
        
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"Copied {image_name}")
            processed_images.add(image_name)
        else:
            print(f"Warning: Image {image_name} not found")
    
    print(f"\nCreated test dataset with {len(processed_images)} unique images")
    print(f"Training samples: {len(train_test_captions)}")
    print(f"Validation samples: {len(val_test_captions)}")
    print(f"\nTest dataset saved to: {output_dir}")

if __name__ == '__main__':
    # Create test dataset with 10 samples
    create_test_dataset(
        image_dir='data/images',
        caption_file='data/train_captions.txt',
        val_caption_file='data/val_captions.txt',
        output_dir='data/test_dataset',
        num_samples=10
    ) 