import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from utils.dataset_utils import check, separate_data, split_data, save_file
from Generate_ECG_PPG.generate_data_csv import generate_dataset_csv
from torch.utils.data import Dataset
from scipy.signal import stft

random.seed(1)
np.random.seed(1)
num_clients = 40
dir_path = "PPG_ECG/"

class CustomDatasetMultiStream(Dataset):
    def __init__(self, df_dataset_ecg, df_dataset_ppg, T = 0.008, nperseg=65, type_signal=None):
        fs = 1/T

        data_ecg = df_dataset_ecg[df_dataset_ecg.columns[:-1]].values
        # aplica short-time fourier transform
        _, _, Zxx_ecg = stft(data_ecg, fs, nperseg=nperseg)
        
        self.data_ecg = torch.abs(torch.tensor(Zxx_ecg))  # Magnitude (escala de cinza)
        self.labels_ecg = torch.tensor(df_dataset_ecg['label'].values, dtype=torch.long)

        data_ppg = df_dataset_ppg[df_dataset_ppg.columns[:-1]].values
        # aplica short-time fourier transform
        _, _, Zxx_ppg = stft(data_ppg, fs, nperseg=nperseg)
        
        self.data_ppg = torch.abs(torch.tensor(Zxx_ppg))  # Magnitude (escala de cinza)
        self.labels_ppg = torch.tensor(df_dataset_ppg['label'].values, dtype=torch.long)
        self.data = torch.stack([self.data_ecg, self.data_ppg], dim=1)
        self.labels = self.labels_ecg

    def __len__(self):
        return len(self.data_ecg)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
class CustomDataset(Dataset):
    def __init__(self, df_dataset, T = 0.008, nperseg=65):
        data = df_dataset[df_dataset.columns[:-1]].values

        fs = 1/T
        # aplica short-time fourier transform
        frequencies, times, Zxx = stft(data, fs, nperseg=nperseg)
        
        self.data = torch.abs(torch.tensor(Zxx))  # Magnitude (escala de cinza)
        self.data = self.data.unsqueeze(1)
        self.labels = torch.tensor(df_dataset['label'].values, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Allocate data to users
def generate_dataset(dir_path, num_clients, niid, balance, partition, size_win, hold_size, type_signal):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"
    
    if check(config_path, train_path, test_path, num_clients, niid, balance, partition,
             size_win, hold_size, type_signal):
        return
    
    if type_signal == "ECG_PPG":
        trainset_csv_ecg, testset_csv_ecg = generate_dataset_csv(
            size_win, hold_size, type_signal=' PLETH', root = 'Generate_ECG_PPG')
        trainset_csv_ppg, testset_csv_ppg = generate_dataset_csv(
            size_win, hold_size, type_signal=' II', root = 'Generate_ECG_PPG')
        
        trainset = CustomDatasetMultiStream(trainset_csv_ecg, trainset_csv_ppg)
        testset = CustomDatasetMultiStream(testset_csv_ecg, testset_csv_ppg)

    else:
        trainset_csv, testset_csv = generate_dataset_csv(
            size_win, hold_size, type_signal, root = 'Generate_ECG_PPG')
        
        trainset = CustomDataset(trainset_csv)
        testset = CustomDataset(testset_csv)
    
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    dataset_image = []
    dataset_label = []

    dataset_image.extend(trainset.data.cpu().detach().numpy())
    dataset_image.extend(testset.data.cpu().detach().numpy())
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    num_classes = len(set(dataset_label))
    print(f'Number of classes: {num_classes}')

    # dataset = []
    # for i in range(num_classes):
    #     idx = dataset_label == i
    #     dataset.append(dataset_image[idx])

    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes,  
                                    niid, balance, partition, class_per_client=2)
    train_data, test_data = split_data(X, y)
 
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        statistic, niid, balance, partition, size_win, hold_size, type_signal)


if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    size_win = int(sys.argv[4]) if sys.argv[4] != None else 80
    hold_size = int(sys.argv[5]) if sys.argv[5] != None else 0
    type_signal = sys.argv[6] if sys.argv[6] != None else ' PLETH'

    dir_path = type_signal + '/'

    if type_signal == "ECG":
        type_signal = ' II'
    elif type_signal == "PPG":
        type_signal = ' PLETH'
    elif type_signal == 'Fusion':
        type_signal = 'ECG_PPG'
    
    generate_dataset(dir_path, num_clients, niid, balance, partition, 
                     size_win, hold_size, type_signal)