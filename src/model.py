"""
CNN-LSTM Hybrid Model for Solar Irradiance Prediction

Architecture:
- 1D CNN layers for local pattern extraction
- LSTM layers for temporal dependency modeling
- Fully connected output layer

Research Project: IIT Bombay (December 2025 - February 2026)
"""

import torch
import torch.nn as nn


class CNNLSTMModel(nn.Module):
    """CNN-LSTM hybrid model for GHI time-series forecasting."""

    def __init__(self, input_channels=1, lstm_hidden=64, output_size=1,
                 num_features=8, dropout=0.2):
        """
        Args:
            input_channels: Number of CNN input channels (1 for univariate baseline)
            lstm_hidden: Number of LSTM hidden units
            output_size: Number of output predictions
            num_features: Number of engineered features fed to LSTM
            dropout: Dropout rate for regularization
        """
        super(CNNLSTMModel, self).__init__()

        self.input_channels = input_channels
        self.lstm_hidden = lstm_hidden
        self.output_size = output_size

        # CNN feature extraction
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=num_features, out_channels=32,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # LSTM temporal modeling
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )

        # Output layer
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_size),
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, seq_len, num_features)
        Returns:
            Predictions of shape (batch, output_size)
        """
        # CNN expects (batch, channels, seq_len)
        x_cnn = x.permute(0, 2, 1)
        x_cnn = self.cnn(x_cnn)

        # LSTM expects (batch, seq_len, features)
        x_lstm = x_cnn.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x_lstm)

        # Take last time step
        out = self.fc(lstm_out[:, -1, :])
        return out

    def predict(self, x, device='cpu'):
        """Run inference on a single sample or batch."""
        self.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x).to(device)
            if x_tensor.dim() == 2:
                x_tensor = x_tensor.unsqueeze(0)
            output = self.forward(x_tensor)
        return output.cpu().numpy()
