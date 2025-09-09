import torch
import torch.nn as nn
from torch.nn.functional import softmax, sigmoid
from einops import rearrange


class DUGNet(nn.Module):
    """
    A hybrid network combining temporal modeling with uncertainty-aware mechanisms.
    Args:
        input_dim (int): Input feature dimension (D).
        hidden_dim (int): Hidden feature dimension (D).
        seq_len (int): Sequence length (T).
        period (int): Period for TPGN.
    """
    def __init__(self, input_dim, hidden_dim, seq_len, period):
        super(DUGNet, self).__init__()
        
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        
        # Short-term and long-term feature extractor
        self.short_term_extractor = nn.Conv1d(
            in_channels=input_dim, out_channels=hidden_dim, kernel_size=period, stride=1, padding=period // 2, groups=input_dim
        )
        self.long_term_extractor = nn.Conv1d(
            in_channels=input_dim, out_channels=hidden_dim, kernel_size=seq_len, stride=1, padding=seq_len // 2, groups=input_dim
        )

        # Uncertainty branches with Monte Carlo Dropout
        self.uncertainty_branch_short = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=1),
            nn.GELU(),
            nn.Dropout(p=0.1),  # Monte Carlo Dropout
            nn.Conv1d(hidden_dim // 2, hidden_dim // 2, kernel_size=1),
            nn.Softmax(dim=1),
            nn.Dropout(p=0.1),  # Monte Carlo Dropout
            nn.Conv1d(hidden_dim // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )
        



       
        self.uncertainty_branch_long = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=1),
            nn.GELU(),
            nn.Dropout(p=0.1),  # Monte Carlo Dropout
            nn.Conv1d(hidden_dim // 2, hidden_dim // 2, kernel_size=1),
            nn.Softmax(dim=1),
            nn.Dropout(p=0.1),  # Monte Carlo Dropout
            nn.Conv1d(hidden_dim // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )
        

        # Weak signal gate
        self.weak_signal_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            #nn.ReLU(),
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
        
        # Step 2: Uncertainty modeling
        x_short_uncertainty = self.uncertainty_branch_short(x_short)  # [B, 1, T]
        x_long_uncertainty = self.uncertainty_branch_long(x_long)    # [B, 1, T]
        
        # Step 3: Apply uncertainty-aware weighting
        x_short = x_short * (1 + x_short_uncertainty * 0.001)
        x_long = x_long * (1 + x_long_uncertainty * 0.001)
        
        # Combine features
        x_combined = x_short + x_long  # [B, hidden_dim, T]
        x_combined = x_combined.permute(0, 2, 1)  # [B, T, hidden_dim]
        
        # Step 4: Weak signal gating
        weak_signal = self.weak_signal_gate(x_combined)  # [B, T, hidden_dim]
        x_gate = weak_signal * x_combined  # Apply gating
        
        # Step 5: Attention mechanism
        #attn_output = self.battn_layer(x_gate)
        attn_output, _ = self.attention(x_gate, x_gate, x_gate)  # [B, T, hidden_dim]
        
        # Step 6: Bidirectional LSTM
        lstm_output, _ = self.bi_lstm(attn_output)  # [B, T, hidden_dim * 2]
        
        # Step 7: Output layer
        output = self.output_layer(lstm_output)  # [B, T, hidden_dim * 2]
        
        return output#, x_short_uncertainty, x_long_uncertainty

