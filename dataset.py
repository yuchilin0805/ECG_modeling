import os

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import wfdb
from utils import split_data

def scaling(X, sigma=0.1):
    scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1, X.shape[1]))
    myNoise = np.matmul(np.ones((X.shape[0], 1)), scalingFactor)
    return X * myNoise


def shift(sig, interval=20):
    for col in range(sig.shape[1]):
        offset = np.random.choice(range(-interval, interval))
        sig[:, col] += offset / 1000 
    return sig


def transform(sig, train=False):
    if train:
        if np.random.randn() > 0.5: sig = scaling(sig)
        if np.random.randn() > 0.5: sig = shift(sig)
    return sig


class ECGDataset(Dataset): ###cpsc_2018
    def __init__(self, phase, data_dir, label_csv, folds, leads):
        super(ECGDataset, self).__init__()
        self.phase = phase
        df = pd.read_csv(label_csv)
        df = df[df['fold'].isin(folds)]
        self.data_dir = data_dir
        self.labels = df
        self.leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        if leads == 'all':
            self.use_leads = np.where(np.in1d(self.leads, self.leads))[0]
        else:
            self.use_leads = np.where(np.in1d(self.leads, leads))[0]
        self.nleads = len(self.use_leads)
        self.classes = ['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'PVC', 'STD', 'STE']
        self.n_classes = len(self.classes)
        self.data_dict = {}
        self.label_dict = {}

    def __getitem__(self, index: int):
        row = self.labels.iloc[index]
        patient_id = row['patient_id']
        ecg_data, _ = wfdb.rdsamp(os.path.join(self.data_dir, patient_id))
        ecg_data = transform(ecg_data, self.phase == 'train')
        nsteps, _ = ecg_data.shape

        
        

        ecg_data = ecg_data[-15000:, self.use_leads]
        result = np.zeros((15000, self.nleads))  # 30 s, 500 Hz
        result[-nsteps:, :] = ecg_data

        # Correct way to handle label retrieval
        if patient_id in self.label_dict:
            labels = self.label_dict[patient_id]
        else:
            labels = row[self.classes].to_numpy(dtype=np.float32)
            self.label_dict[patient_id] = labels

        return torch.from_numpy(result.transpose()).float(), torch.from_numpy(labels).float()
    
    def __len__(self):
        return len(self.labels)
    

class PTBDataset(Dataset):
    def __init__(self, phase, data_dir, label_csv, folds, leads):
        super(PTBDataset, self).__init__()
        self.phase = phase
        df = pd.read_csv(label_csv)
        df = df[df['fold'].isin(folds)]
        self.data_dir = data_dir
        self.labels = df
        self.leads = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        if leads == 'all':
            self.use_leads = np.where(np.in1d(self.leads, self.leads))[0]
        else:
            self.use_leads = np.where(np.in1d(self.leads, leads))[0]
        self.nleads = len(self.use_leads)
        self.classes = ['NORM','MI','STTC','CD','HYP']
        self.n_classes = len(self.classes)
        self.data_dict = {}
        self.label_dict = {}

    def __getitem__(self, index: int):
        row = self.labels.iloc[index]
        patient_id = row['patient_id']
        
        file_name=patient_id[-8:]
        ecg_data, _ = wfdb.rdsamp(os.path.join(self.data_dir, file_name))      
        ecg_data = transform(ecg_data, self.phase == 'train')
        nsteps, _ = ecg_data.shape


        ecg_data = ecg_data[-15000:, self.use_leads]
        result = np.zeros((15000, self.nleads))  # 30 s, 500 Hz
        result[-nsteps:, :] = ecg_data

        # Correct way to handle label retrieval
        if patient_id in self.label_dict:
            labels = self.label_dict[patient_id]
        else:
            labels = row[self.classes].to_numpy(dtype=np.float32)
            self.label_dict[patient_id] = labels

        return torch.from_numpy(result.transpose()).float(), torch.from_numpy(labels).float()
    
    def __len__(self):
        return len(self.labels)
    
class shaoxingDataset(Dataset):
    def __init__(self, phase, data_dir, label_csv, folds, leads):
        super(shaoxingDataset, self).__init__()
        self.phase = phase
        df = pd.read_csv(label_csv)
        df = df[df['fold'].isin(folds)]
        self.data_dir = data_dir
        self.labels = df
        self.leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        if leads == 'all':
            self.use_leads = np.where(np.in1d(self.leads, self.leads))[0]
        else:
            self.use_leads = np.where(np.in1d(self.leads, leads))[0]
        self.nleads = len(self.use_leads)
        self.classes = ['1AVB', '2AVB', '2AVB1', '2AVB2', '3AVB', 'ABI', 'ALS', 'APB', 'AQW', 'ARS', 'AVB', 
                        'CCR', 'CR', 'ERV', 'FQRS', 'IDC', 'IVB', 'JEB', 'JPT', 'LBBB', 'LBBBB', 
                        'LFBBB', 'LVH', 'LVQRSAL', 'LVQRSCL', 'LVQRSLL', 'MI', 'MIBW', 'MIFW', 'MILW', 
                        'MISW', 'PRIE', 'PWC', 'QTIE', 'RAH', 'RBBB', 'RVH', 'STDD', 'STE', 'STTC', 'STTU', 
                        'TWC', 'TWO', 'UW', 'VB', 'VEB', 'VFW', 'VPB', 'VPE', 'VET', 'WAVN', 'WPW', 'SB', 'SR', 
                        'AFIB', 'ST', 'AF', 'SA', 'SVT', 'AT', 'AVNRT', 'AVRT', 'SAAWR'] #63 classes
        self.n_classes = len(self.classes)
        self.data_dict = {}
        self.label_dict = {}

    def __getitem__(self, index: int):
        row = self.labels.iloc[index]
        patient_id = row['patient_id']


        ecg_data, _ = wfdb.rdsamp(os.path.join(self.data_dir, patient_id))

        
        ecg_data = transform(ecg_data, self.phase == 'train')
        nsteps, _ = ecg_data.shape

        #result=ecg_data

        ecg_data = ecg_data[-15000:, self.use_leads]        
        result = np.zeros((15000, self.nleads))  # 30 s, 500 Hz
        result[-nsteps:, :] = ecg_data

       


        # Correct way to handle label retrieval
        if patient_id in self.label_dict:
            labels = self.label_dict[patient_id]
        else:
            labels = row[self.classes].to_numpy(dtype=np.float32)
            self.label_dict[patient_id] = labels

        return torch.from_numpy(result.transpose()).float(), torch.from_numpy(labels).float()
    
    def __len__(self):
        return len(self.labels)

