import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class ChanneledCausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.utils.weight_norm(nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation))
        self.chomp = nn.ConstantPad1d((0, -padding), 0)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.relu(self.chomp(self.conv(x)))

class TCNAutoencoder(nn.Module):
    def __init__(self, num_features, seq_len):
        super().__init__()
        self.encoder = nn.Sequential(
            ChanneledCausalConv1d(num_features, 32, kernel_size=3, dilation=1),
            ChanneledCausalConv1d(32, 16, kernel_size=3, dilation=2)
        )
        self.decoder = nn.Sequential(
            ChanneledCausalConv1d(16, 32, kernel_size=3, dilation=2),
            nn.utils.weight_norm(nn.Conv1d(32, num_features, kernel_size=3, padding=2)),
            nn.ConstantPad1d((0, -2), 0)
        )

    def forward(self, x):
        x = x.transpose(1, 2) # Switch to (Batch, Features, Seq_len)
        latent = self.encoder(x)
        out = self.decoder(latent)
        return out.transpose(1, 2)

if __name__ == "__main__":
    print("⏳ Starting model training process...")
    X_train = np.load("data/processed/train.npy")
    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    model = TCNAutoencoder(num_features=25, seq_len=100)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(5): # Fast training execution optimized for setup verification
        for batch in loader:
            inputs = batch[0]
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/tcn_autoencoder.pth")
    print("✅ Model trained and preserved successfully at models/tcn_autoencoder.pth")