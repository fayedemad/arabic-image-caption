import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from transformers import AutoTokenizer
import re
from functools import lru_cache

class CNNArabicCaptionDataset(Dataset):
    """
    Dataset class for CNN-based Arabic image captioning.
    Handles the format:
    image_id#number    caption
    Where:
    - image_id is the image filename
    - number is the caption number for that image
    - caption is the Arabic caption text
    """
    def __init__(self, image_dir, caption_file, tokenizer=None, max_length=50):
        self.image_dir = os.path.join(image_dir, 'images')
        self.caption_file = caption_file
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv2")
        self.max_length = max_length
        
        special_tokens = {
            "pad_token": "[PAD]",
            "bos_token": "[BOS]",
            "eos_token": "[EOS]",
            "unk_token": "[UNK]"
        }
        self.tokenizer.add_special_tokens(special_tokens)
        
        self.transform = transforms.Compose([
            transforms.Resize((380, 380)), 
            transforms.CenterCrop(380),
            transforms.ToTensor(),
        ])
        
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        self.samples = []
        
        if not os.path.exists(self.image_dir):
            raise ValueError(f"Image directory not found: {self.image_dir}")
            
        self.available_images = set()
        for root, _, files in os.walk(self.image_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.available_images.add(file)
        
        print(f"Found {len(self.available_images)} images in {self.image_dir}")
        
        with open(caption_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        print(f"Reading {len(lines)} lines from {caption_file}")
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
                
            try:
                if '\t' in line:
                    image_id_with_number, caption = line.split('\t', 1)
                else:
                    parts = re.split(r'[\s,]+', line, 1)
                    if len(parts) != 2:
                        print(f"Warning: Line {line_num}: Invalid format (no separator): {line}")
                        continue
                    image_id_with_number, caption = parts
                
                if '#' in image_id_with_number:
                    image_id, caption_number = image_id_with_number.split('#', 1)
                else:
                    image_id = image_id_with_number
                    caption_number = "1"
                
                image_id = image_id.strip()
                caption = caption.strip()
                
                if not caption:
                    print(f"Warning: Line {line_num}: Empty caption for image {image_id}")
                    continue
                    
                image_path = os.path.join(self.image_dir, image_id)
                if not os.path.exists(image_path):
                    print(f"Warning: Line {line_num}: Image not found: {image_id}")
                    continue
                
                encoding = self.tokenizer(
                    caption,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                self.samples.append({
                    'image_id': image_id,
                    'caption': caption,
                    'caption_number': int(caption_number),
                    'input_ids': encoding['input_ids'].squeeze(),
                    'attention_mask': encoding['attention_mask'].squeeze()
                })
                
            except Exception as e:
                print(f"Warning: Line {line_num}: Error processing line: {line}")
                print(f"Error details: {str(e)}")
                continue
        
        print(f"Loaded {len(self.samples)} valid samples")
        if len(self.samples) == 0:
            raise ValueError("No valid samples found in the dataset")
    
    @staticmethod
    def preprocess_image(image):
        """Preprocess a PIL Image for the model."""
        transform = transforms.Compose([
            transforms.Resize((380, 380)),
            transforms.CenterCrop(380),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        return transform(image)
    
    @lru_cache(maxsize=1000)
    def _load_image(self, image_id):
        """Cache image loading to avoid repeated disk reads."""
        image_path = os.path.join(self.image_dir, image_id)
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
            return image
        except Exception as e:
            print(f"Error loading image {image_id}: {str(e)}")
            return torch.zeros((3, 380, 380))
            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        image = self._load_image(sample['image_id'])
        
        return {
            'image_id': sample['image_id'],
            'pixel_values': image,
            'caption_ids': sample['input_ids'],
            'attention_mask': sample['attention_mask'],
            'caption': sample['caption'],
            'caption_number': sample['caption_number']
        }

def collate_fn(batch):
    """
    Collate function for the dataloader.
    """
    caption_ids = torch.stack([item['caption_ids'] for item in batch])
    attention_masks = torch.stack([item['attention_mask'] for item in batch])
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    
    image_ids = [item['image_id'] for item in batch]
    captions = [item['caption'] for item in batch]
    caption_numbers = [item['caption_number'] for item in batch]
    
    return {
        'image_id': image_ids,
        'pixel_values': pixel_values,
        'caption_ids': caption_ids,
        'attention_mask': attention_masks,
        'caption': captions,
        'caption_number': caption_numbers
    } 