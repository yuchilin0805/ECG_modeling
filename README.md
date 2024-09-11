# Design and Development of Energy-efficient Edge AI Models for 12-lead ECG Diagnosis System in Rural Areas

## Requirement
### Dataset
#### cpsc_2018
CPSC2018 is the dataset of 2018 PhysioNet/CinC Challenge. The challenge aims to encourage the
development of algorithms to identify the rhythm/morphology abnormalities from 12-lead Electrocardiograms (ECGs).
#### Shaoxing-Ninbo 
This dataset for 12-lead electrocardiogram signals was created under the auspices of
Chapman University, Shaoxing People’s Hospital (Shaoxing Hospital Zhejiang University School of Medicine), and
Ningbo First Hospital. 
#### PTB-XL
The PTB-XL ECG dataset is a large dataset of 21799 clinical 12-lead ECGs from 18869 patients of 10
second length. Where 52% are male and 48% are female with ages covering the whole range from 0 to 95 years. 

## Run
The files collectively define the infrastructure for training, validating, and performing inference on ECG classification models using various deep learning architectures. \
The models.py file defines different neural network models (LSTM, GRU, ResNet, and Transformer variants). These models are tailored for tasks like ECG classification, handling temporal dependencies and feature extraction in sequence data.\
The K_fold_cross.py script manages the training and evaluation processes, leveraging cross-validation to assess model performance across multiple folds, and tracks metrics. \
The dataset.py file defines PyTorch Dataset classes to handle different ECG datasets (e.g., CPSC-2018, PTB-XL, Shaoxing-Ninbo), supporting data augmentation and transformation.\
The inference.py is designed for performing inference on new ECG data, using trained models to predict outcomes and track inference performance.\

### Preprocessing
To get labels.csv for training
```sh
$ python utils/data_preprocessor/datapreprocess.py 
```
### K_fold Cross validation
#### training
```sh
$ python K_fold_cross.py -config_file configuration/config.txt
```
#### quantization
Change the model_path in config_quantize.txt to the trained model name
```sh
$ python K_fold_cross.py -config_file configuration/config_quantize.txt
```
