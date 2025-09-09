import torch
import torch.nn as nn
from torch.nn.functional import softmax, sigmoid
from einops import rearrange

class HybridTemporalNet(nn.Module):
    """
    A hybrid network combining TPGN's temporal modeling and BiImprovedLSTM's weak signal gate.
    Args:
        input_dim (int): Input feature dimension (D).
        hidden_dim (int): Hidden feature dimension (D).
        seq_len (int): Sequence length (T).
        period (int): Period for TPGN.
    """
    def __init__(self, input_dim, hidden_dim, seq_len, period):
        super(HybridTemporalNet, self).__init__()
        
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        
        # TPGN-like short-term and long-term feature extractor
        self.short_term_extractor = nn.Conv1d(
            in_channels=input_dim, out_channels=hidden_dim, kernel_size=period, stride=1, padding=period//2, groups=input_dim
        )
        self.long_term_extractor = nn.Conv1d(
            in_channels=input_dim, out_channels=hidden_dim, kernel_size=seq_len, stride=1, padding=seq_len//2, groups=input_dim
        )
        
        # Weak signal gate (inspired by BiImprovedLSTM)
        self.weak_signal_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            #nn.ReLU(),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # Attention mechanism for temporal feature fusion
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True)
        
        # Bidirectional modeling (double-layer LSTM)
        self.bi_lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        
        # Output linear layer
        self.output_layer = nn.Linear(hidden_dim * 2, hidden_dim * 2)

    def forward(self, x):
        """
        Forward pass for the hybrid network.
        Args:
            x: [B, T, D] - Input time series data.
        Returns:
            [B, T, 2D] - Combined output.
        """
        B, T, D = x.shape
        
        # Step 1: Short-term and long-term feature extraction
        x_short = self.short_term_extractor(x.permute(0, 2, 1))  # [B, D, T] -> [B, hidden_dim, T]
        x_long = self.long_term_extractor(x.permute(0, 2, 1))    # [B, D, T] -> [B, hidden_dim, T]
        
        # Combine features and permute back
        x_combined = x_short + x_long  # [B, hidden_dim, T]
        x_combined = x_combined.permute(0, 2, 1)  # [B, T, hidden_dim]
        
        # Step 2: Weak signal gating
        weak_signal = self.weak_signal_gate(x_combined)  # [B, T, hidden_dim]
        x_gate = weak_signal * x_combined  # Apply gating
        
        # Step 3: Attention mechanism
        attn_output, _ = self.attention(x_gate, x_gate, x_gate)  # [B, T, hidden_dim]
        
        # Step 4: Bidirectional LSTM
        lstm_output, _ = self.bi_lstm(attn_output)  # [B, T, hidden_dim * 2]
        
        # Step 5: Output layer
        output = self.output_layer(lstm_output)  # [B, T, hidden_dim * 2]
        
        return output
    
if __name__ == "__main__":
    # Define input parameters
    batch_size = 8
    seq_len = 7
    input_dim = 128
    hidden_dim = 128
    period = 3

    # Create dummy inputs
    x = torch.randn(batch_size, seq_len, input_dim)  # Input: [B, T, D]

    # Initialize the model
    model = HybridTemporalNet(input_dim=input_dim, hidden_dim=hidden_dim, seq_len=seq_len, period=period)

    # Forward pass
    output = model(x)

    # Print input and output shapes
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)