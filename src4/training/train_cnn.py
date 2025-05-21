import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import wandb
import time
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from src4.data.dataset import CNNArabicCaptionDataset, collate_fn
from src4.models.cnn_captioner import CNNArabicCaptioner

def train(args):
    if args.use_wandb:
        try:
            wandb.init(project="arabic-image-captioning", config=vars(args))
        except Exception as e:
            print(f"Warning: Failed to initialize wandb: {e}")
            print("Continuing without wandb logging...")
            args.use_wandb = False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    print(f"Tokenizer vocabulary size: {len(tokenizer)}")
    
    print("Creating datasets...")
    train_dataset = CNNArabicCaptionDataset(
        image_dir=args.image_dir,
        caption_file=args.caption_file,
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    
    val_dataset = CNNArabicCaptionDataset(
        image_dir=args.image_dir,
        caption_file=args.val_caption_file,
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        collate_fn=collate_fn
    )
    
    print("Creating model...")
    model = CNNArabicCaptioner(
        vocab_size=len(tokenizer),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout
    )
    model = model.to(device)
    
    total_steps = len(train_loader) * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    
    print("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        total_batches = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for batch_idx, batch in enumerate(train_pbar):
            images = batch['pixel_values'].to(device)
            captions = batch['caption_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_time = time.time()
            outputs = model(images, captions, attention_mask)
            forward_time = time.time() - start_time
            loss = criterion(outputs.view(-1, outputs.size(-1)), captions.view(-1))
            start_time = time.time()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            scheduler.step()  # Update learning rate
            backward_time = time.time() - start_time
            
            total_loss += loss.item()
            total_batches += 1
            
            train_pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss/total_batches:.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}",
                'forward_time': f"{forward_time:.2f}s",
                'backward_time': f"{backward_time:.2f}s"
            })
            
            if args.use_wandb:
                wandb.log({
                    'train_loss': loss.item(),
                    'train_avg_loss': total_loss/total_batches,
                    'learning_rate': scheduler.get_last_lr()[0],
                    'forward_time': forward_time,
                    'backward_time': backward_time
                })
            
            if batch_idx % 100 == 0:
                print(f"\nBatch {batch_idx} timing:")
                print(f"Forward pass: {forward_time:.2f}s")
                print(f"Backward pass: {backward_time:.2f}s")
                print(f"Learning rate: {scheduler.get_last_lr()[0]:.2e}")
                if torch.cuda.is_available():
                    print(f"GPU Memory: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="Validation")
            for batch in val_pbar:
                images = batch['pixel_values'].to(device)
                captions = batch['caption_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                outputs = model(images, captions, attention_mask)
                loss = criterion(outputs.view(-1, outputs.size(-1)), captions.view(-1))
                
                val_loss += loss.item()
                val_batches += 1
                
                val_pbar.set_postfix({
                    'val_loss': f"{loss.item():.4f}",
                    'val_avg_loss': f"{val_loss/val_batches:.4f}"
                })
        
        avg_val_loss = val_loss / val_batches
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(args.output_dir, f"model_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': avg_val_loss,
                'vocab_size': len(tokenizer)  # Save vocab size with checkpoint
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
        
        if args.use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'val_loss': avg_val_loss,
                'best_val_loss': best_val_loss
            })
    
    if args.use_wandb:
        wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CNN-based Arabic image captioning model')
    
    # Data arguments
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing images')
    parser.add_argument('--caption_file', type=str, required=True, help='Path to training caption file')
    parser.add_argument('--val_caption_file', type=str, required=True, help='Path to validation caption file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save checkpoints')
    parser.add_argument('--tokenizer_dir', type=str, default='tokenizer_new', help='Directory containing tokenizer')
    
    # Model arguments
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--max_length', type=int, default=50, help='Maximum caption length')
    parser.add_argument('--hidden_size', type=int, default=512, help='Hidden size')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of transformer layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--gradient_clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Ratio of total steps for warmup')
    
    # Logging arguments
    parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging')
    
    args = parser.parse_args()
    train(args) 