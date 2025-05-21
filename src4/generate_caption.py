"""
Script to generate captions for images using a trained CNN-based Arabic image captioning model.
"""

import os
import argparse
import torch
from PIL import Image
from transformers import AutoTokenizer
from models.cnn_captioner import CNNArabicCaptioner
import torchvision.transforms as transforms

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
    
    tensor = transform(image)
    return tensor

def generate_caption(model, image, tokenizer, device, max_length=50, temperature=1.0, debug=False):
    """Generate caption for a single image."""
    model.eval()
    with torch.no_grad():
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        image = image.to(device)
        
        if debug:
            print(f"Image shape: {image.shape}")
        
        hidden = model.get_initial_hidden(image)
        current_token = torch.tensor([[tokenizer.bos_token_id]], device=device)
        generated_tokens = [tokenizer.bos_token_id]
        
        for _ in range(max_length):
            outputs, hidden = model.decode_step(current_token, hidden)
            next_token_logits = outputs[0, -1, :] / temperature
            
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.argmax(probs, dim=-1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
                
            current_token = next_token.unsqueeze(0)
            generated_tokens.append(next_token.item())
        
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return generated_text

def main():
    parser = argparse.ArgumentParser(description='Generate captions for images')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--image_path', type=str, required=True, help='Path to image or directory of images')
    parser.add_argument('--tokenizer_dir', type=str, default='tokenizer_new', help='Directory containing tokenizer')
    parser.add_argument('--output_file', type=str, help='Path to save generated captions')
    parser.add_argument('--max_length', type=int, default=50, help='Maximum caption length')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    parser.add_argument('--force_cpu', action='store_true', help='Force using CPU even if CUDA is available')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with verbose output')
    args = parser.parse_args()
    
    device = torch.device('cpu') if args.force_cpu else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    print(f"Tokenizer vocabulary size: {len(tokenizer)}")
    
    print("Loading model...")
    checkpoint = torch.load(args.model_path, map_location=device)
    model = CNNArabicCaptioner(
        vocab_size=checkpoint['vocab_size'],
        hidden_size=512,
        num_layers=2,
        dropout=0.1
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    if os.path.isfile(args.image_path):
        image_paths = [args.image_path]
    else:
        image_paths = [os.path.join(args.image_path, f) for f in os.listdir(args.image_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Found {len(image_paths)} images")
    
    generated_captions = []
    for image_path in image_paths:
        try:
            image = Image.open(image_path).convert('RGB')
            tensor = preprocess_image(image)
            
            caption = generate_caption(
                model=model,
                image=tensor,
                tokenizer=tokenizer,
                device=device,
                max_length=args.max_length,
                temperature=args.temperature,
                debug=args.debug
            )
            
            generated_captions.append((image_path, caption))
            print(f"\nImage: {image_path}")
            print(f"Generated caption: {caption}")
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            continue
    
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for image_path, caption in generated_captions:
                f.write(f"{image_path}\t{caption}\n")
        print(f"\nSaved generated captions to {args.output_file}")

if __name__ == '__main__':
    main() 