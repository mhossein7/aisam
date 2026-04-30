import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Basic building blocks ----------

class FeedForward(nn.Module):
    """Feedforward sublayer in a Transformer block."""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """One Transformer block with self-attention + cross-attention."""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()

        # self-attention for feature sequence
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)

        # cross-attention for conditioning on input sequence
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)

        # Feedforward network
        self.ffn = FeedForward(d_model, d_ff, dropout)

        # Normalizations
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, context):
        # x: [n, batch, d_model]
        # context: [n+1, batch, d_model] (input embeddings)

        # --- Self-attention ---
        x2, _ = self.self_attn(x, x, x)
        x = self.norm1(x + x2)

        # --- Cross-attention ---
        x2, _ = self.cross_attn(x, context, context)
        x = self.norm2(x + x2)

        # --- Feedforward ---
        x2 = self.ffn(x)
        x = self.norm3(x + x2)

        return x


# ---------- Full Transformer forecaster ----------

class TransformerForecaster(nn.Module):
    def __init__(self, feature_dim, input_dim, d_model=128, n_heads=4, 
                 num_layers=3, d_ff=256, dropout=0.1):
        super().__init__()

        # Embedding layers
        self.feature_embed = nn.Linear(feature_dim, d_model)
        self.input_embed = nn.Linear(input_dim, d_model)

        # Positional encoding (simple learnable)
        self.pos_embed = nn.Parameter(torch.zeros(1, 512, d_model))  # supports up to 512 timesteps

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)
        ])

        # Output projection
        self.output_layer = nn.Linear(d_model, feature_dim)

    def forward(self, X, U):
        """
        X: (batch_size, seq_len, feature_dim)
        U: (batch_size, seq_len+1, input_dim)
        """

        batch_size, seq_len, _ = X.size()

        # --- Embedding ---
        x_embed = self.feature_embed(X) + self.pos_embed[:, :seq_len, :]
        u_embed = self.input_embed(U) + self.pos_embed[:, :seq_len + 1, :]

        # Transpose to shape [seq_len, batch, d_model] for attention API
        x_embed = x_embed.transpose(0, 1)
        u_embed = u_embed.transpose(0, 1)

        # --- Pass through transformer layers ---
        for layer in self.layers:
            x_embed = layer(x_embed, u_embed)

        # Take the last token representation (for next-step prediction)
        last_state = x_embed[-1]  # shape: (batch, d_model)

        # --- Predict next feature vector ---
        out = self.output_layer(last_state)  # shape: (batch, feature_dim)

        return out


# ---------- Example usage ----------
if __name__ == "__main__":
    batch_size, seq_len, feature_dim, input_dim = 32, 20, 16, 1
    X = torch.randn(batch_size, seq_len, feature_dim)
    U = torch.randn(batch_size, seq_len + 1, input_dim)  # includes next input

    model = TransformerForecaster(feature_dim, input_dim)
    y_pred = model(X, U)
    print("Predicted next feature vector shape:", y_pred.shape)
