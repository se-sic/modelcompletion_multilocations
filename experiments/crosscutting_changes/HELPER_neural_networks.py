import math
import torch.nn as nn
import torch

# Linear Model
class LinearModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input from (batch_size, 2, 1536) -> (batch_size, 3072)
        return self.linear(x)

class NonLinearModel(nn.Module):
    def __init__(self, input_dim, hidden_sizes):
        super(NonLinearModel, self).__init__()

        layers = []
        dims = [input_dim] + hidden_sizes

        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(dims[-1], 1))  # Final output layer

        self.layers = nn.Sequential(*layers)

    def forward(self, x, more_embeddings=False):
        x = x.view(x.size(0), -1)
        return self.layers(x)
    

class AttentionClassifier(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        """
        Args:
            embed_dim (int): Dimensionality of each embedding vector.
            num_heads (int): Number of parallel attention heads.
            dropout (float): Dropout probability applied inside multi-head attention.
        """
        super(AttentionClassifier, self).__init__()
        
        # MultiheadAttention with batch_first=True expects input shape (batch, seq_len, embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Final linear layer mapping from 'embed_dim' to a single logit for binary classification
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, x, more_embeddings ):
        """
        Args:
            x (Tensor): Input of shape (batch_size, seq_len, embed_dim).
        
        Returns:
            logits (Tensor): Raw output of shape (batch_size, 1). 
                             Apply sigmoid outside or use a loss function that expects logits.
            attn_weights (Tensor): The attention weights of shape 
                                   (batch_size, num_heads, seq_len, seq_len).
        """
        if more_embeddings=="ALL_EMBEDDINGS":
            batch_size, num_items, num_embeddings, embed_dim = x.shape
            x = x.view(batch_size, num_items, num_embeddings * embed_dim)

        # self-attention over x
        attn_output, attn_weights = self.attention(x, x, x) 
        # Simple pooling over sequence dimension (mean)
        pooled = attn_output.mean(dim=1)  # => (batch_size, embed_dim)
        
        # Linear layer -> logits for binary classification
        logits = self.fc(pooled)          # => (batch_size, 1)
        return logits#, attn_weights




####################

