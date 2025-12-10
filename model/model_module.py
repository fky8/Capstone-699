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

#Data Collection
def get_all_raw_data(file_prefix='pipeline/data_files/raw_mags_1m', file_extension="csv"):
    """"
    Takes in the daily financial data and outputs a dataframe of the concatenated dataset of OHCLV data for the entire available period.
    
    file_prefix: filepath and prefix for all raw OHCLV data files
    file_extension: File type for all raw OHCLV data files 
    """
    
    # Define the file prefix and extension 
    file_extension = 'csv'

    # Create the search pattern using a wildcard '*'
    # e.g., 'your_prefix_*.csv'
    pattern = f"{file_prefix}*.{file_extension}" 

    # Get a list of all files matching the pattern
    # You can specify a full directory path if needed, e.g., 'path/to/files/{pattern}'
    file_list = glob.glob(pattern)

    # Initialize an empty list to store individual DataFrames
    dfs = []

    # Loop through the files, read each into a DataFrame, and append to the list
    for filename in file_list:
        df = pd.read_csv(filename)
        dfs.append(df)

    # Concatenate all DataFrames in the list
    # ignore_index=True ensures a continuous index in the final DataFrame, discarding original indices
    combined_df = pd.concat(dfs, ignore_index=True)

    # Display the first few rows of the combined DataFrame
    #print(combined_df.head())
    return combined_df.drop_duplicates()

def create_train_test_data(feature_set_name='pipeline/data_files/features_combined.csv'
                           ,ohclv_name='pipeline/data_files/raw_mags_1m'
                           ,alternative_data=False):
    """
    Creates training and test data from a given feature table and OHCLV data.
    
    feature_set_name: relative filepath and name of the alternative data feature table
    ohclv_name: prefix of all OHCLV data files
    alternative_data: Boolean value indicating whether to include the alternative data in final training and test sets
    
    """
    df=pd.read_csv(feature_set_name)
    df['Datetime']=pd.to_datetime(df['Datetime'])
    ohclv=get_all_raw_data(ohclv_name)
    ohclv['Datetime']=pd.to_datetime(ohclv['Datetime'])
    ohclv['Datetime']=ohclv.Datetime.apply(lambda x:x.replace(tzinfo=None))-pd.Timedelta(hours=5)
    features=ohclv.merge(df,on='Datetime',how='inner')
    ids=features['Datetime']
    if alternative_data:
        X=features.drop(['Datetime','regime_label','pred_vol','hist_vol','vol_ratio'],axis=1)[features.regime_label!='Unknown']
    else:
        X=features[['open','high','low','close','close','volume']][features.regime_label!='Unknown']
    y=features['regime_label'].replace({'Unknown':3,
                                       'Low':0,
                                       'Neutral':1,
                                       'High':2})[features.regime_label!='Unknown']
    train_x,test_x=X[:floor(len(features)*0.7)],X[floor(len(features)*0.7):]
    train_y,test_y=y[:floor(len(features)*0.7)],y[floor(len(features)*0.7):]
    
    return train_x,test_x,train_y,test_y

def create_lag_train_test_data(feature_set_name='pipeline/data_files/features_combined.csv'
                           ,ohclv_name='pipeline/data_files/raw_mags_1m.csv'
                           ,alternative_data=False,validation=False):
    df=pd.read_csv(feature_set_name)
    df['Datetime']=pd.to_datetime(df['Datetime'])
    ohclv=get_all_raw_data()
    ohclv['Datetime']=pd.to_datetime(ohclv['Datetime'])
    ohclv['Datetime']=ohclv.Datetime.apply(lambda x:x.replace(tzinfo=None))-pd.Timedelta(hours=5)
    features=ohclv.merge(df,on='Datetime',how='inner')
    ids=features['Datetime']
    if alternative_data:
        X=features.drop(['Datetime','regime_label','pred_vol','hist_vol','vol_ratio'],axis=1)[features.regime_label!='Unknown']
    else:
        X=features[['open','high','low','close','close','volume']][features.regime_label!='Unknown']
    X=X.shift(10)
    
    scaler=StandardScaler()
    y=features['regime_label'].replace({'Unknown':3,
                                       'Low':0,
                                       'Neutral':1,
                                       'High':2})[features.regime_label!='Unknown']
    X['target_lag']=y.shift(1)
    if validation:
        train_x,val_x,test_x=X[:floor(len(features)*0.7)],X[floor(len(features)*0.7):floor(len(features)*0.85)],X[floor(len(features)*0.85):]
        train_x_scaled=pd.DataFrame(scaler.fit_transform(train_x),columns=train_x.columns)
        val_x_scaled=pd.DataFrame(scaler.transform(val_x),columns=val_x.columns)
        test_x_scaled=pd.DataFrame(scaler.transform(test_x),columns=test_x.columns)
        train_y,val_y,test_y=y[:floor(len(features)*0.7)],y[floor(len(features)*0.7):floor(len(features)*0.85)],y[floor(len(features)*0.85):]
        #train_x_scaled=pd.DataFrame(scaler.fit_transform(train_x),columns=X.columns)
        #test_x_scaled=pd.DataFrame(scaler.transform(test_x),columns=X.columns)
        
        results=train_x_scaled[10:],val_x_scaled[10:],test_x_scaled[10:],train_y[10:],val_y[10:],test_y[10:]
    else:
        train_x,test_x=X[:floor(len(features)*0.7)],X[floor(len(features)*0.7):]
        train_x_scaled=pd.DataFrame(scaler.fit_transform(train_x),columns=train_x.columns)
        test_x_scaled=pd.DataFrame(scaler.transform(test_x),columns=test_x.columns)
        train_y,test_y=y[:floor(len(features)*0.7)],y[floor(len(features)*0.7):]
        #train_x_scaled=pd.DataFrame(scaler.fit_transform(train_x),columns=X.columns)
        #test_x_scaled=pd.DataFrame(scaler.transform(test_x),columns=X.columns)
        results=train_x[10:],test_x[10:],train_y[10:],test_y[10:]
    
    return results
#Modelling

def model_initialization(train_x,train_y):
    """
    Initiliaze parameters for a Hidden Markov Model using a given training set and return a trained model
    
    train_x:dataset/numpy array of predictive features
    train_y: pandas series of numpy array of state labels
    """
    # Number of hidden states (regimes)
    n_components = len(np.unique(train_y))
    n_features = train_x.shape[1]

    # Manually compute the parameters
    startprob = np.zeros(n_components)
    transmat = np.zeros((n_components, n_components))
    means = np.zeros((n_components, n_features))
    covars = np.zeros((n_components, n_features, n_features))
    
    # Loop through each regime to calculate emission parameters (mean and covariance)
    for i in range(n_components):
        regime_data = train_x[train_y == i]
        means[i] = regime_data.mean(axis=0)
        covars[i] = np.cov(regime_data, rowvar=False) + 1e-6 * np.eye(n_features) # Add epsilon for stability
    
    # Calculate transition probabilities from the labeled sequence
    for i in range(len(train_y) - 1):
        transmat[train_y[i], train_y[i+1]] += 1
        
    # Normalize transition matrix
    transmat = transmat / transmat.sum(axis=1, keepdims=True)
    
    # Calculate initial state probabilities (assuming first data point is labeled correctly)
    startprob[train_y[0]] = 1.0

    # Ensure no zero probabilities
    transmat[transmat == 0] = 1e-9
    transmat = transmat / transmat.sum(axis=1, keepdims=True)
    
    # Create a Gaussian HMM model and initialize with computed parameters
    model = hmm.GaussianHMM(n_components=n_components, covariance_type="full", init_params="")
    model.startprob_ = startprob
    model.transmat_ = transmat
    model.means_ = means
    model.covars_ = covars


    return model

def forecasting_intialization(X,y):
    """
    Function to initialize the 
    """
    X=X.values
    lengths = [len(X)] # For a single time series

    # Initialize a GaussianHMM with the correct number of regimes (hidden states)
    n_regimes = y.nunique()
    model = hmm.GaussianHMM(n_components=n_regimes, n_iter=100)
    regimes=y

    # Manually set the known state sequence to guide training
    # hmmlearn does not have a built-in supervised training method, so we use
    # an iterative approach to find the most likely parameters for the given state sequence.
    # This is an alternative to standard EM for when labels are known.

    # Function to re-estimate parameters based on labeled data
    def fit_supervised_hmm(model, X, regimes):
        for i in range(model.n_iter):
            # E-step with known states
            # The E-step is skipped in favor of a known state sequence

            # M-step: Re-estimate parameters (startprob, transmat, means, covars)
            # We manually re-calculate the necessary statistics based on the known labels

            # Re-estimate start probabilities
            model.startprob_ = np.bincount(regimes[:1], minlength=model.n_components) / len(regimes[:1])

            # Re-estimate transition matrix
            for i in range(model.n_components):
                for j in range(model.n_components):
                    # Count transitions from state i to j
                    transitions = np.sum( (regimes[:-1] == i) & (regimes[1:] == j) )
                    total_transitions = np.sum(regimes[:-1] == i)
                    model.transmat_[i, j] = transitions / total_transitions if total_transitions > 0 else 0

            # Re-estimate means and covariances
            for i in range(model.n_components):
                # Isolate data for the current regime
                regime_data = X[regimes == i]
                if len(regime_data) > 0:
                    model.means_[i, :] = np.mean(regime_data, axis=0)
                    model.covars_[i, :, :] = np.cov(regime_data, rowvar=False) + 1e-6 * np.eye(X.shape[1]) # Add small value for numerical stability

        return model

    # Fit the supervised HMM model
    # Note: For real applications, a custom implementation or a different library might be more robust
    # For demonstration, the manual approach is shown.
    supervised_model = fit_supervised_hmm(model, X, y.values)


    return model

def blocked_cv(X,y,classifier,param_grid):

    # Initialize TimeSeriesSplit
    # n_splits determines the number of splits (folds)
    # gap allows for a gap between the training and testing sets
    # max_train_size can limit the maximum size of the training set
    results=[]
    best_score = -np.inf
    result_grid={}
    tscv = TimeSeriesSplit(n_splits=5)
    param_combinations=list(product(*param_grid.values()))
    param_keys = list(param_grid.keys())
    for comb in param_combinations:
        current_params = dict(zip(param_keys, comb))
        model=RandomForestClassifier(random_state=24,**current_params)
        
        fold_score=[]
        # Iterate through the splits
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            model.fit(X_train,y_train)
            probs=model.predict_proba(X_test)
            fold_score.append(roc_auc_score(y_test,probs,multi_class='ovr'))
        avg_score=np.mean(fold_score)
        std_score=np.std(fold_score)
        results.append({'params': current_params, 'avg_score': avg_score,'std_score':std_score})
        
        if avg_score > best_score:
            best_score = avg_score
            best_params = current_params
    #for res in results:
    #    key={res['params']}
    #    score={res['avg_score']}
    #    result_grid[key]=score
        
    return results

#Evaluation

def plot_ovr_roc(test_y,probs,n_classes=3,name='Normal Data',model_type='Random Forest'):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    test_y_bin=label_binarize(test_y, classes=np.arange(n_classes))
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(test_y_bin[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 7. Plot the ROC curves for each class
    plt.figure(figsize=(8, 6))
    colors = ['blue', 'red', 'green', 'purple'] # Define colors for each class
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of class {i} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier') # Diagonal line for random classifier
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve - One-vs-Rest:{name}')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(f"model/model_graphs/ROC_Curve_{model_type}_{name}.png")

def compare_scores(train_x,train_y,test_x,test_y,
                  train_x_alt,test_x_alt):

    rf_alt=RandomForestClassifier(random_state=42,n_estimators=172,max_depth=2,max_features=10)
    rf=RandomForestClassifier(random_state=42,n_estimators=64,max_depth=9,max_features=10)
        
    rf.fit(train_x,train_y)
    rf_alt.fit(train_x_alt,train_y)
    probs_train=rf.predict_proba(train_x)
    probs=rf.predict_proba(test_x)
    probs_alt_train=rf_alt.predict_proba(train_x_alt)
    probs_alt=rf_alt.predict_proba(test_x_alt)
    score=roc_auc_score(test_y,probs,multi_class='ovr')
    score_alt=roc_auc_score(test_y,probs_alt,multi_class='ovr')
    
    n_classes=train_y.nunique()
    plot_ovr_roc(test_y,probs,name='Normal Data')
    plot_ovr_roc(test_y,probs_alt,name='Alternative Data')

    
    
    
    probs_frame_1=pd.DataFrame(probs_train,columns=['probability_low','probability_medium','probability_high'])
    probs_frame_2=pd.DataFrame(probs,columns=['probability_low','probability_medium','probability_high'])
    probs_frame_1['set']='train'
    probs_frame_2['set']='test'
    
    alt_frame_1=pd.DataFrame(probs_alt_train,columns=['probability_low_with_alt','probability_medium_with_alt','probability_high_with_alt'])
    alt_frame_2=pd.DataFrame(probs_alt,columns=['probability_low_with_alt','probability_medium_with_alt','probability_high_with_alt']) 
    alt_frame_1['set']='train'
    alt_frame_2['set']='test'
    
    
    
    return score_alt,score,pd.concat([alt_frame_1,alt_frame_2]),pd.concat([probs_frame_1,probs_frame_2])
    
def create_results_frame(features,probs_alt,probs,output_path='model/data_files/results.csv'):
    labels=pd.concat([features[10:floor(len(features)*0.7)],features[floor(len(features)*0.7):][10:]])[['Datetime','regime_label']]
    labels=labels[labels.regime_label!='Unknown']
    
    results=pd.concat([labels.reset_index(drop=True),probs.reset_index(drop=True),probs_alt.reset_index(drop=True)],axis=1)\
    .dropna(subset='Datetime')
    
    results.to_csv(output_path)
    
    return results

def sequence_data_creator(sequence_length=10,feature_set_name='pipeline/data_files/features_combined.csv'
                           ,ohclv_name='pipeline/data_files/raw_mags_1m'
                           ,alternative_data=False):
    """
    Converts a dataframe of structured multivariate time series data to a dataloader containing sequences
    
    sequence_length:The number of consecutive timestamps to compute a sequence
    feature_set_name: the path of the file containing the feature set
    ohclv_name: the path of the 
    """
    df=pd.read_csv(feature_set_name)
    df['Datetime']=pd.to_datetime(df['Datetime'])
    ohclv=get_all_raw_data(ohclv_name)
    ohclv['Datetime']=pd.to_datetime(ohclv['Datetime'])
    ohclv['Datetime']=ohclv.Datetime.apply(lambda x:x.replace(tzinfo=None))-pd.Timedelta(hours=5)
    features=ohclv.merge(df,on='Datetime',how='inner')
    ids=features['Datetime']
    if alternative_data:
        X=features.drop(['Datetime','regime_label','pred_vol','hist_vol','vol_ratio'],axis=1)[features.regime_label!='Unknown']
    else:
        X=features[['open','high','low','close','close','volume']][features.regime_label!='Unknown']
    y=features['regime_label'].replace({'Unknown':3,
                                       'Low':0,
                                       'Neutral':1,
                                       'High':2})[features.regime_label!='Unknown']
    scaler=StandardScaler()
    X=scaler.fit_transform(X)
    num_features = X.shape[1]
    
    # Create sequences
    # Using a simple sliding window to create sequences
    sequences = []
    labels = [] # Assuming you have a label for each sequence
    for i in range(len(X) - sequence_length):
        sequences.append(X[i : i + sequence_length])
        # Assuming the label corresponds to the last element in the sequence
        labels.append(y[i + sequence_length - 1])
        
    sequences = np.array(sequences)
    labels = np.array(labels)

    # Convert to PyTorch tensors
    X = torch.tensor(sequences, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long) # Use long for classification labels


    class StructuredDataset(Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    dataset = StructuredDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    return dataloader
    
class MultivariateLSTMDataset(Dataset):
    def __init__(self, data, sequence_length, target_column):
        self.sequence_length = sequence_length
        self.X, self.y = self.create_sequences(data, target_column)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long) # Use torch.long for classification targets

    def create_sequences(self, data, target_column):
        X, y = [], []
        # Exclude the target column from features
        features = data.values
        targets = target_column.values

        for i in range(len(data) - self.sequence_length + 1):
            # Get the input sequence of features
            seq_x = features[i:(i + self.sequence_length)]
            X.append(seq_x)
            # Get the target label (e.g., at the end of the sequence) for classification
            # For classification, the label is usually a single value for the entire sequence
            seq_y = targets[i + self.sequence_length - 1]
            y.append(seq_y)
        return np.array(X), np.array(y)

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Take the output from the last time step for classification
        out = self.fc(out[:, -1, :])
        return out
    
def train_classifier(model,train_loader,num_epochs=1,criterion=nn.CrossEntropyLoss()): 
    """
    Function to train a pytorch RNN
    model: Pytorch classifier
    train_loader:Pytorch dataloader
    num_epochs: Number of training cycles to compute
    criterion: metric with with to compute loss function
    
    """
    optimizer=optim.Adam(model.parameters())
    for epoch in range(num_epochs):
        epochLoss = []
        for batch_idx, (data, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            predictions = model(data)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            epochLoss.append(loss.item())

def validate_classifier(model, val_loader, criterion):
    """
    Function to validate the performance of a pytorch RNN for hyperparameter tuning
    model: Pytorch classifier
    val_loader:Pytorch dataloader
    criterion: metric with with to compute loss function
    
    """
    model.eval() # Set the model to evaluation mode
    with torch.no_grad(): # Disable gradient calculation for validation
        total_loss = 0
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
        avg_loss = total_loss / len(val_loader)
        #print(f'Validation Loss: {avg_loss:.4f}')
    return avg_loss

def create_results(dataloader,model):
    """Takes in a dataloader and trained LSTM model and outputs a numpy array of class probabilities
        
        dataloader: Pytorch dataloader of prediction data and results
        model: trained pytorch RNN
    """
    all_data_tensors = []
    labels=[]
    for data,label in dataloader:
        all_data_tensors.append(data)
        labels.append(label)
        data_tensor = torch.cat(all_data_tensors, dim=0)
        label_tensor=torch.cat(labels,dim=0)
    probabilities=torch.softmax(model(data_tensor),dim=1)
    
    return probabilities.detach().numpy(),label_tensor.numpy()

def compare_lstm_results(test_loader,model,test_loader_alt,model_alt):
    "Function to compare test results with and without the alternative data"
    probabilities,labels=create_results(test_loader,model)
    probabilities_alt,_=create_results(test_loader_alt,model_alt)
    

    plot_ovr_roc(labels,probabilities,name='Normal Data',model_type='LSTM')
    plot_ovr_roc(labels,probabilities_alt,name='Alternative Data',model_type='LSTM')
    print("ROC AUC with normal data (LSTM):",roc_auc_score(labels,probabilities,multi_class='ovr'))
    print("ROC AUC with alternative data (LSTM):",roc_auc_score(labels,probabilities_alt,multi_class='ovr'))