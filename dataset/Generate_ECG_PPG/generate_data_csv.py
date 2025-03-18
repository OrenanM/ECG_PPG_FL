import numpy as np
import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split

def create_windows(arr, size_win, hold=0):
    windows = list()  # Lista para armazenar as janelas geradas

    # Percorre o array com um passo de (tamanho da janela - hold)
    for w in range(0, arr.shape[0], size_win-hold):
        win = arr[w:w+size_win]  # cria uma janela de tamanho size_win

        # Verifica se a janela tem o tamanho exato
        if len(win) == size_win:  
            # Adiciona a janela à lista
            windows.append(win)  
    
    return np.array(windows)

def create_dataset_signal(path_signals, index, size_win=0, hold_size = 0, type_signal=' PLETH'):
    # carrega o csv
    df_signal = pd.read_csv(path_signals[index])
    
    # filtra a caracteristica utilizada
    if size_win > 0:
        values_arr = df_signal[type_signal].values #array dos valores

        # cria um dataframe em que as linhas são uma janela temporal
        windows = create_windows(values_arr, size_win, hold_size)
        df_signal = pd.DataFrame(windows)

    else:
        # sem janelamento
        df_signal = df_signal[[' PLETH']].T

    df_signal['label'] = index + 1 # label
        
    return df_signal

def create_dataset(path_signals, size_win=80, hold_size=0, type_signal=' PLETH'):
    n_register = len(path_signals) # numero de dados
    df = pd.DataFrame() # armazena todos os dasets

    for i in range(n_register):
        # cria o dataset de cada usuario
        df1 = create_dataset_signal(path_signals, index=i, size_win=size_win, 
                                    hold_size=hold_size, type_signal=type_signal)
        # concatena todos os datasets
        df = pd.concat((df, df1), axis=0)

    return df

def generate_dataset_csv(size_win=80, hold_size=0, type_signal=' PLETH', save=False, root=None):
    # diretorio dos dados
    dir = 'bidmc-ppg-and-respiration-dataset-1.0.0/bidmc_csv'
    if root:
        dir = os.path.join(root, dir)

    # todos os arquivos no diretório
    paths = os.listdir(dir)
    paths.sort() # ordena as pastas

    # filtra os sinais
    path_signals = [os.path.join(dir, path) for path in paths if 'Signals' in path]

    # cria os datasets particionados
    df_ecg = create_dataset(path_signals, size_win=size_win, type_signal=type_signal, hold_size=hold_size)

    # Definir a proporção de dados para teste
    test_per = 0.8

    # Dividir os dados em treino e teste
    signal_train, signal_test = train_test_split(df_ecg.values, test_size=test_per, random_state=42)

    # Criar os DataFrames de treino e teste
    df_sgn_train = pd.DataFrame(signal_train)
    df_sgn_test = pd.DataFrame(signal_test)

    df_sgn_train.rename(columns={df_sgn_train.columns[-1]: 'label'}, inplace=True)
    df_sgn_test.rename(columns={df_sgn_test.columns[-1]: 'label'}, inplace=True)

    # salva o dataset
    if save:
        if type_signal == " PLETH":
            type_signal = 'ECG'
        elif type_signal == " II":
            type_signal = 'PPG'
        df_sgn_train.to_csv(f'data/{type_signal}_train.csv', index=False)
        df_sgn_test.to_csv(f'data/{type_signal}_test.csv', index=False)

    return df_sgn_train, df_sgn_test

if __name__ == '__main__':
    size_win = int(sys.argv[1]) if sys.argv[1] != None else 80
    hold_size = int(sys.argv[2]) if sys.argv[2] != None else 0
    type_signal = sys.argv[3] if sys.argv[3] != None else ' PLETH'

    if type_signal == "ECG":
        type_signal = ' PLETH'
    elif type_signal == "PPG":
        type_signal = ' II'

    generate_dataset_csv(size_win, hold_size, type_signal, save=True)