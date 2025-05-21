"""
CNN-based Arabic Image Captioning model using EfficientNet for feature extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]

class CNNArabicCaptioner(nn.Module):
    """
    CNN-based model using EfficientNet for feature extraction with transformer decoder.
    """
    def __init__(self, vocab_size, hidden_size=512, num_layers=2, dropout=0.3, max_length=50, nhead=8):
        super().__init__()
        
        self.image_encoder = EfficientNet.from_pretrained('efficientnet-b0')
        self.image_encoder._fc = nn.Identity()
        
        self.feature_projection = nn.Sequential(
            nn.Linear(1280, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(1280, 512, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(512, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size, max_length)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.output = nn.Linear(hidden_size, vocab_size)
        
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.vocab_size = vocab_size
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for better training."""
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        
        for p in self.transformer_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        nn.init.normal_(self.output.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.output.bias, 0.0)
        
        for layer in self.feature_projection:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.0)
        
        for layer in self.spatial_attention:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.0)
    
    def generate_square_subsequent_mask(self, sz):
        """Generate a square mask for the sequence."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, images, captions=None, attention_mask=None):
        """
        Forward pass of the model.
        
        Args:
            images: Batch of images [batch_size, channels, height, width]
            captions: Optional batch of captions [batch_size, seq_len]
            attention_mask: Optional attention mask [batch_size, seq_len]
            
        Returns:
            If captions are provided (training or validation):
                - logits: [batch_size, seq_len, vocab_size]
            If no captions (inference):
                - generated_captions: List of generated captions
        """
        batch_size = images.size(0)
        
        features = self.image_encoder.extract_features(images)
        
        attention_weights = self.spatial_attention(features)
        attended_features = features * attention_weights
        
        features = F.adaptive_avg_pool2d(attended_features, (1, 1))
        features = features.squeeze(-1).squeeze(-1)
        
        features = self.feature_projection(features)
        
        if captions is not None:
            embedded = self.embedding(captions)
            embedded = self.pos_encoder(embedded)
            
            tgt_mask = self.generate_square_subsequent_mask(captions.size(1)).to(captions.device)
            
            memory = features.unsqueeze(1).repeat(1, captions.size(1), 1)
            
            output = self.transformer_decoder(
                embedded,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=None
            )
            
            logits = self.output(output)
            
            if attention_mask is not None:
                attention_mask = attention_mask.unsqueeze(-1).expand_as(logits)
                logits = logits * attention_mask
            
            return logits
        else:
            generated_captions = []
            
            for i in range(batch_size):
                current_tokens = torch.tensor([[0]], device=images.device)
                generated = []
                
                memory = features[i].unsqueeze(0).unsqueeze(0)
                
                for _ in range(self.max_length):
                    tgt_mask = self.generate_square_subsequent_mask(current_tokens.size(1)).to(images.device)
                    
                    embedded = self.embedding(current_tokens)
                    embedded = self.pos_encoder(embedded)
                    
                    output = self.transformer_decoder(
                        embedded,
                        memory,
                        tgt_mask=tgt_mask,
                        memory_mask=None
                    )
                    
                    logits = self.output(output[:, -1:])
                    
                    next_token = torch.argmax(logits, dim=-1)
                    
                    if next_token.item() == 1:
                        break
                    
                    generated.append(next_token.item())
                    
                    current_tokens = torch.cat([current_tokens, next_token], dim=1)
                
                generated_captions.append(generated)
            
            return generated_captions
            
    def generate_caption(self, image, max_length=50):
        """Generate a caption for a single image."""
        self.eval()
        with torch.no_grad():
            image = image.unsqueeze(0)
            caption = self.forward(image, max_length=max_length)
            return caption[0] 