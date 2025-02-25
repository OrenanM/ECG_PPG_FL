import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from utils.dataset_utils import check, separate_data, split_data, save_file
from PPG_ECG.generate_data_csv import generate_dataset_csv
from torch.utils.data import Dataset

random.seed(1)
np.random.seed(1)
num_clients = 20
dir_path = "PPG_ECG/"

class CustomDataset(Dataset):
    def __init__(self, df_dataset):
        labels = torch.tensor(df_dataset['label'].values)
        datas  = torch.tensor(df_dataset[df_dataset.columns[:-1]].values)
        self.data = [(datas[i], labels[i]) for i in range(df_dataset.shape[0])]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Allocate data to users
def generate_dataset(dir_path, num_clients, niid, balance, partition, size_win, hold_size, type_signal):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"
    
    if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
        return
        
    trainset_csv, testset_csv = generate_dataset_csv(size_win, hold_size, type_signal, root = 'PPG_ECG')
    
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
        statistic, niid, balance, partition)


if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    size_win = int(sys.argv[4]) if sys.argv[4] != None else 80
    hold_size = int(sys.argv[5]) if sys.argv[5] != None else 0
    type_signal = sys.argv[6] if sys.argv[6] != None else ' PLETH'

    if type_signal == "ECG":
        type_signal = ' PLETH'
    elif type_signal == "PPG":
        type_signal = ' II'
    
    generate_dataset(dir_path, num_clients, niid, balance, partition, 
                     size_win, hold_size, type_signal)