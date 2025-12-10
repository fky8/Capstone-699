import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import hmmlearn.hmm as hmm
from math import floor
from sklearn.metrics import accuracy_score,roc_auc_score,f1_score,roc_curve,auc
from sklearn.preprocessing import MinMaxScaler,StandardScaler,label_binarize
import os
import glob
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from itertools import product
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings


from model.model_module import get_all_raw_data,create_train_test_data,create_lag_train_test_data\
,model_initialization,forecasting_intialization,blocked_cv,compare_scores,create_results_frame\
,sequence_data_creator,MultivariateLSTMDataset,LSTMClassifier,train_classifier,validate_classifier\
,create_results,compare_lstm_results

warnings.filterwarnings("ignore")

#1. Prepare data
features=pd.read_csv("pipeline/data_files/features_combined.csv").drop_duplicates()
features['Datetime']=pd.to_datetime(features['Datetime'])
ohclv=get_all_raw_data()
ohclv['Datetime']=pd.to_datetime(ohclv['Datetime'])
ohclv['Datetime']=ohclv.Datetime.apply(lambda x:x.replace(tzinfo=None))-pd.Timedelta(hours=5)
df=ohclv.merge(features,on='Datetime',how='inner')
train_x,test_x,train_y,test_y=create_train_test_data(alternative_data=False)
train_x_alt,test_x_alt,train_y,test_y=create_train_test_data(alternative_data=True)

#2. Modelling
#2.1 HMM
hmm_model =model_initialization(train_x,train_y)
log_prob, predicted_regimes = hmm_model.decode(test_x, algorithm="viterbi")
probs=hmm_model.predict_proba(test_x)
print("ROC AUC for HMM with Normal Data:",roc_auc_score(test_y,probs,multi_class='ovr'))
hmm_model =model_initialization(train_x_alt,train_y)
log_prob, predicted_regimes = hmm_model.decode(test_x_alt, algorithm="viterbi")
probs=hmm_model.predict_proba(test_x_alt)
print("ROC AUC for HMM with Alternative Data:",roc_auc_score(test_y,probs,multi_class='ovr'))
      
#2.2 Random Forest
train_x_lag,test_x_lag,train_y_lag,test_y_lag=create_lag_train_test_data(alternative_data=False)
train_x_lag_alt,test_x_lag_alt,train_y_lag_alt,test_y_lag_alt=create_lag_train_test_data(alternative_data=True)

train_x_lstm,val_x_lstm,test_x_lstm,train_y_lstm,val_y_lstm,test_y_lstm=create_lag_train_test_data(alternative_data=False,validation=True)

train_x_lstm_alt,val_x_lstm_alt,test_x_lstm_alt,train_y_lstm_alt,val_y_lstm,test_y_lstm_alt=create_lag_train_test_data(alternative_data=True,validation=True)
score_alt,score_norm,probs_alt,probs=\
compare_scores(train_x_lag,train_y_lag,test_x_lag,test_y_lag,train_x_lag_alt,test_x_lag_alt)
print("ROC AUC for Random Forest with Normal Data:",score_norm)
print("ROC AUC for Random Forest with Alternate Data:",score_alt)
create_results_frame(features,probs_alt,probs)

#2.3 LSTM
SEQUENCE_LENGTH = 10
BATCH_SIZE = 32
# Create Dataset instances
train_dataset = MultivariateLSTMDataset(train_x_lstm, SEQUENCE_LENGTH, train_y_lstm)
val_dataset = MultivariateLSTMDataset(val_x_lstm, SEQUENCE_LENGTH, val_y_lstm)
test_dataset = MultivariateLSTMDataset(test_x_lstm, SEQUENCE_LENGTH, test_y_lstm)

# Create DataLoader instances
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False) # Shuffle can be True for train data in classification if temporal order across batches isn't critical
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Create Dataset instances
train_dataset_alt = MultivariateLSTMDataset(train_x_lstm_alt, SEQUENCE_LENGTH, train_y_lstm)
val_dataset_alt = MultivariateLSTMDataset(val_x_lstm_alt, SEQUENCE_LENGTH, val_y_lstm)
test_dataset_alt = MultivariateLSTMDataset(test_x_lstm_alt, SEQUENCE_LENGTH, test_y_lstm)

# Create DataLoader instances
train_loader_alt = DataLoader(train_dataset_alt, batch_size=BATCH_SIZE, shuffle=False) # Shuffle can be True for train data in classification if temporal order across batches isn't critical
val_loader_alt = DataLoader(val_dataset_alt, batch_size=BATCH_SIZE, shuffle=False)
test_loader_alt = DataLoader(test_dataset_alt, batch_size=BATCH_SIZE, shuffle=False)

#Train model with no alternative data
input_size=train_loader.dataset[0][0].shape[1]
batch_size=32
hidden_size=64
num_layers=2
num_classes=3
num_epochs=50

model =LSTMClassifier(input_size, hidden_size, num_layers, num_classes)
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters())

for epoch in range(num_epochs):
    train_classifier(model, train_loader, 1,criterion) # Train for one epoch
    validate_classifier(model, val_loader, criterion)

#Train model with alternative data
input_size=train_loader_alt.dataset[0][0].shape[1]
batch_size=32
hidden_size=128
num_layers=4
num_classes=3
num_epochs=20

model_alt =LSTMClassifier(input_size, hidden_size, num_layers, num_classes)
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model_alt.parameters())

for epoch in range(num_epochs):
    train_classifier(model_alt, train_loader_alt, 1,criterion) # Train for one epoch
    validate_classifier(model_alt, val_loader_alt, criterion)

compare_lstm_results(test_loader,model,test_loader_alt,model_alt)