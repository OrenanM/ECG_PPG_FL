import torch.nn as nn
import torch

class BioCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=53):  # Ajuste num_classes conforme necessário
        super(BioCNN, self).__init__()
        
        # Stream 1: ECG Spectrogram or PPG Spectrogram
        self.ecg_conv = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),  # (Entrada Escala de cinza)
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # Fully Connected Layers
        self.fc = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, signal):
        signal_features = self.ecg_conv(signal)
        signal_features = torch.flatten(signal_features, start_dim=1)
        
        output = self.fc(signal_features)
        return output
    
##################################################################3
class MultiStreamBioCNN(nn.Module):
    def __init__(self, num_classes=53):  # Ajuste num_classes conforme necessário
        super(MultiStreamBioCNN, self).__init__()
        
        # Stream 1: ECG Spectrogram
        self.ecg_conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),  # (Entrada Escala de cinza)
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Stream 2: PPG Spectrogram
        self.emg_conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),  # (Entrada RGB)
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Fully Connected Layers
        self.fc = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, ecg_emg):
        ecg = ecg_emg[:,0,:,:].unsqueeze(1)
        emg = ecg_emg[:,1,:,:].unsqueeze(1)

        ecg_features = self.ecg_conv(ecg)
        emg_features = self.emg_conv(emg)
        
        ecg_features = torch.flatten(ecg_features, start_dim=1)
        emg_features = torch.flatten(emg_features, start_dim=1)
        
        combined_features = torch.cat((ecg_features, emg_features), dim=1)
        output = self.fc(combined_features)
        return output