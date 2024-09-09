import tensorflow as tf
import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
import wfdb
import os
import pandas as pd
def list_physical_gpus():
    """
    List available physical GPUs for TensorFlow.
    """
    gpus = tf.config.list_physical_devices('GPU')
    print(f"TensorFlow GPUs: {gpus}")

def check_cuda_availability():
    """
    Check CUDA availability and print GPU details for PyTorch.
    """
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    if cuda_available:
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPU(s) available: {num_gpus}")

        for i in range(num_gpus):
            gpu_info = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {gpu_info.name}, Memory: {gpu_info.total_memory / (1024 ** 3)} GB")
    else:
        print("No GPU available.")

def split_data(seed=42):
    """
    Split data into training, validation, and test sets.
    """
    folds = range(1, 11)
    folds = np.random.RandomState(seed).permutation(folds)
    #return folds[:8], folds[8:9], folds[9:]
    return folds

def prepare_input(ecg_file: str):
    """
    Prepare input data from ECG file.
    """
    if ecg_file.endswith('.mat'):
        ecg_file = ecg_file[:-4]
    ecg_data, _ = wfdb.rdsamp(ecg_file)
    nsteps, nleads = ecg_data.shape
    ecg_data = ecg_data[-15000:, :]
    result = np.zeros((15000, nleads))  # 30 s, 500 Hz
    result[-nsteps:, :] = ecg_data
    return result.transpose()

def cal_scores(y_true, y_pred):
    """
    Calculate evaluation scores.
    """
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    #auc = roc_auc_score(y_true, y_score)
    acc = accuracy_score(y_true, y_pred)
    #return precision, recall, f1, auc, acc
    return precision, recall, f1, acc
def find_optimal_threshold(y_true, y_score):
    """
    Find the optimal threshold for classification.
    """
    thresholds = np.linspace(0, 1, 100)
    f1s = [f1_score(y_true, y_score > threshold) for threshold in thresholds]
    return thresholds[np.argmax(f1s)]

def cal_f1(y_true, y_score, find_optimal):
    """
    Calculate F1 score.
    """
    if find_optimal:
        thresholds = np.linspace(0, 1, 100)
    else:
        thresholds = [0.5]
    f1s = [f1_score(y_true, y_score > threshold) for threshold in thresholds]
    return np.max(f1s)

def cal_f1s(y_trues, y_scores, find_optimal=True):
    """
    Calculate F1 scores for multiple classes.
    """
    f1s = []
    for i in range(y_trues.shape[1]):
        f1 = cal_f1(y_trues[:, i], y_scores[:, i], find_optimal)
        f1s.append(f1)
    return np.array(f1s)

def cal_aucs(y_trues, y_scores):
    """
    Calculate AUC scores.
    """
    return roc_auc_score(y_trues, y_scores, average=None)

def write_csv(csv_name, dataset, accuracy_per_label, precision_per_label, recall_per_label, f1_per_label, 
              avg_accuracy, avg_precision, avg_recall, avg_f1):
    classes=dataset.classes
    dicts={'labels':['avg'],
        'accuracy':[avg_accuracy],
        'precision':[avg_precision],
        'recall':[avg_recall],
        'f1':[avg_f1]}
    df = pd.DataFrame(dicts)    
    for i,label in enumerate(classes):
        df.loc[len(df.index)]=[label,accuracy_per_label[i],precision_per_label[i],recall_per_label[i],f1_per_label[i]]
    df.to_csv(csv_name, index=None)

def write_summary_csv(summary_csv, args, avg_accuracy, avg_precision, avg_recall, avg_f1,
                      model_path,fold_idx,use_gpu,inference_speed, model_size, balanced_accuracy):
    dicts={'model_file':[model_path],
           'folds':[fold_idx],
           'use_gpu':[use_gpu],
           'seed':[args.seed],
           'lr' :[args.lr],
           'batch_size':[args.batch_size],
           'epochs':[args.epochs],
           'inference_speed':[inference_speed],
           'model_size':[model_size],
           'accuracy':[avg_accuracy],
           'precision':[avg_precision],
           'recall':[avg_recall],
           'f1':[avg_f1],
           'balanced_accuracy':[balanced_accuracy]}
    df = pd.DataFrame(dicts)
    if(os.path.isfile(summary_csv)):
        df.to_csv(summary_csv, mode='a', index=None, header=None)
    else:
        df.to_csv(summary_csv,index=None)

def print_prediction(args, prediction, true_label,y_output,dataset,path):
    
    pred_list=[]
    label_list=[]
    
    for i,j in zip(prediction,true_label):
        nonzeroi=np.nonzero(i)[0]        
        nonzeroj=np.nonzero(j)[0]
        tmp=[]
        tmp2=[]
        for idx in nonzeroi:
            tmp.append(dataset.classes[idx])
        for idx in nonzeroj:
            tmp2.append(dataset.classes[idx])

        pred_list.append(str(tmp))
        label_list.append(str(tmp2))


    dicts={'predictions':pred_list,
           'true_labels':label_list           
           }
    df = pd.DataFrame(dicts)
    for i,label in enumerate(dataset.classes):
        df.insert(len(df.columns),label,y_output[:,i])

    df.to_csv(path, index=None)
