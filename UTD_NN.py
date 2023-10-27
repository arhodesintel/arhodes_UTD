import csv
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import normalize
from sklearn.impute import SimpleImputer
from lightgbm import LGBMClassifier
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch import nn
from torch import optim
import torch
from torch import Tensor
from sklearn.metrics import confusion_matrix,accuracy_score, roc_curve, auc
from torch.autograd import Function
import seaborn as sns
import matplotlib.pyplot as plt
import os
from PIL import Image
import PyPDF2
import glob
import pickle
import dask.dataframe as dd
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def viz_pair(data_0, data_1,feature,idx, viz_file_path):
    '''

    data_0: reference data

    data_1: new data

    '''
    fig, axes = plt.subplots(1, 4, figsize=(32, 8), )
    if pd.api.types.is_numeric_dtype(data_0):
        data_0 = data_0[~np.isnan(data_0)]
        data_1 = data_1[~np.isnan(data_1)]
        sns.kdeplot(data_0, fill=True,
                    color="r", ax=axes[0], legend=True, label='reference')
        sns.kdeplot(data_1, fill=True,
                    color="b", ax=axes[0], legend=True, label='new')
        sns.ecdfplot(data_0, ax=axes[1],
                     color="r", legend=True, label='reference')
        sns.ecdfplot(data_1, ax=axes[1],
                     color="b", legend=True, label='new')

    else:
        sns.countplot(data_0, ax=axes[0], color="r", label='reference')
        sns.countplot(data_1, ax=axes[0], color="b", label='new')
        sns.ecdfplot(data_0, ax=axes[1], color="r", label='reference')
        sns.ecdfplot(data_1, ax=axes[1], color="b", label='new')

    percs = np.linspace(1, 99, 50)
    qn_a = np.percentile(data_0, percs)
    qn_b = np.percentile(data_1, percs)

    axes[2].plot(percs, qn_a, ls="", marker="o", color='r', label='reference')

    axes[2].plot(percs, qn_b, ls="", marker="x", color='b', label='new')

    # axes[2].set_yscale('log')

    percs = np.linspace(0, 100, 51)
    qn_a = np.percentile(data_0, percs)
    qn_b = np.percentile(data_1, percs)
    axes[3].plot(percs, qn_a, ls="", marker="o", color='r', label='reference')
    axes[3].plot(percs, qn_b, ls="", marker="x", color='b', label='new')
    axes[3].set_yscale('log')

    # axes[2].plot(x, x, color="k", ls="--")
    axes[0].set(xlabel=None)
    axes[1].set(xlabel=None)
    axes[0].title.set_text('Density')
    axes[1].title.set_text('CDF')
    axes[2].title.set_text('1% ~ 99% Pencentiles')
    axes[3].title.set_text('0% ~ 100% Pencentiles')
    plt.suptitle(str(idx+1) + '. ' + feature)
    plt.savefig(viz_file_path+str(idx+1)+'.png')
    

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.drop1 = nn.Dropout(0.2)
        self.drop2 = nn.Dropout(0.2)
        self.drop3 = nn.Dropout(0.2)
        self.BN1 = nn.BatchNorm1d(hidden_dim)
        self.BN2 = nn.BatchNorm1d(hidden_dim)
        self.BN3 = nn.BatchNorm1d(hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.drop1(x)
        x = self.BN1(x)
        x = torch.relu(x)
        x = self.layer_2(x)
        x = self.drop2(x)
        x = self.BN2(x)
        x = torch.relu(x)        
        x = self.layer_3(x)
        x = self.drop3(x)
        x = self.BN3(x)
        x = torch.relu(x)
        x = torch.sigmoid((self.layer_4(x)))
        return x


def gen_data():
    source_rows = np.where(dataset['meta'][source_target_feature_name] == 0)
    target_rows = np.where(dataset['meta'][source_target_feature_name] == 1)
    source_list = source_rows[0].tolist()
    Y = dataset['meta'][GT_feature_name].to_numpy()
    Y_Nan_list = np.where(np.isnan(Y))[0].tolist()
    intersect_source_NanY = [x for x in source_list if x in Y_Nan_list]
    dataset['data'].drop(intersect_source_NanY, inplace=True, errors='ignore', axis=0)
    dataset['meta'].drop(intersect_source_NanY, inplace=True, errors='ignore', axis=0)    
    X = dataset['data'].to_numpy()
    imp = SimpleImputer(missing_values=np.nan, fill_value=-100)
    imp.fit(X)
    X = imp.transform(X)
    X = normalize(X, axis=0, norm='max')  # normalize data by column max
    X_source = X[source_rows[0]]
    X_target = X[target_rows[0]]
    Y = dataset['meta'][GT_feature_name].to_numpy()
    Y_source = Y[source_rows]
    Y_target = Y[target_rows]

    if save_processed_data_flag:
        np.save(save_processed_features_dir + 'X_target_aug.npy', X_target)
        np.save(save_processed_features_dir + 'Y_target_aug.npy', Y_target)
        np.save(save_processed_features_dir + 'X_source_aug.npy', X_source)
        np.save(save_processed_features_dir + 'Y_source_aug.npy', Y_source)
        np.save(save_processed_features_dir + 'common_feats_aug.npy', dataset['data'].keys().to_list())
    del X, Y, dataset

def get_lgbm_feats():
    model = LGBMClassifier()
    if LGBM_feature_importance_criterion == 'domain_GT':
        X_train = np.concatenate((POR_train_X, HDR_train_X), 0)
        Y_train = np.concatenate((POR_train_Y, HDR_train_Y), 0)
        X_test = np.concatenate((POR_test_X, HDR_test_X), 0)
        Y_test = np.concatenate((POR_test_Y, HDR_test_Y), 0)
        model.fit(X_train, Y_train[:, 1], eval_set=[(X_test, Y_test[:, 1])],
                  eval_metric=['auc'])  # , verbosity=2) #0 index for GT prediction
    elif LGBM_feature_importance_criterion == 'yield_GT':
        X_train = POR_train_X
        Y_train = POR_train_Y
        X_test = POR_test_X
        Y_test = POR_test_Y
        model.fit(X_train, Y_train[:, 0], eval_set=[(X_test, Y_test[:, 0])],
                  eval_metric=['auc'])  # , verbosity=2) #0 index for GT prediction
    top_k_feats = (-model.feature_importances_).argsort()[:num_lgbm_top_features]
    np.save(save_processed_features_dir + 'top_k_lgbm_feats_y', top_k_feats)
    return top_k_feats


#####
# set of basic parameters, file paths, etc.
#####

test_pct = .20 #percentage of the data that you wish to use for testing vs. training
gen_data_flag = False #set to true the first time you load the dataset, this will create a pandas data frame, etc.
dataset = pd.read_pickle('/home/anthonyrhodes/Projects/Labs/UTD/data/RPL81_Class_212702_beac1_212880.csv.pd.dat') #point this to the location of your dataset
drop_features_array = [] #include any features as an array of string that you wish to drop from the training/testing of the predictive model
if len(drop_features_array) != 0:
    a.drop(drop_features_array, inplace=True, errors='ignore', axis=1)
num_lgbm_top_features = 500 #the number of "top k" features -- as determined by first pass with lgbm model -- you'd like to retain for training the NN, e.g., 500
total_epochs = 100 #total epochs to train model, perform feature ranking; typically, 100 is sufficient
num_warmup_epochs = 10 #number of epochs to train prior to running feature ranking, recommended: 10
decorrelation_hp = .01 #HP for decorrelation/regularization weight, recommendation ~.01
gt_prediction_hp = 10 #HP weight for gt prediction vs. domain prediction
source_target_feature_name = 'GT_pilot' #the feature name from your dataset corresponding with the source/target dataset for distribution shift study
GT_feature_name = 'GT_Hot' #the featture name from your dataset corresponding the yield prediction
save_processed_features_dir = '/home/anthonyrhodes/Projects/Labs/UTD/test/' #the local directory where you want results, processesed features saved
viz_file_path = '/home/anthonyrhodes/Desktop/UTD_images/' #the local directory where you want data visualization pdf saved

save_processed_data_flag = False  #flag to save processed data after running gen_data() function; recommended, so you don't have to load the entire dataframe again
run_LGBM_feature_reduction_flag = False #flag to run LGBM top-k feature analysis; run this once for a new dataset
LGBM_feature_importance_criterion = 'yield_GT' #options are: 'domain_GT', 'yield_GT' -- when generating LGBM top-k features; if set to 'domain_GT' the features are chosen according to the ones
#that are most significant for domain prediction; otherwise, if set to 'yield_GT', features are chosen according to the ones that are most significant fcr yield prediction


if gen_data_flag:
    gen_data() #generate/process dataframe from file
else: #otherwise, if you already ran gen_data(), simply bypass the processing, and load the data tensors
    common_features_set = np.load(save_processed_features_dir + 'common_feats_aug.npy', allow_pickle=True)
    X_target = np.load(save_processed_features_dir + 'X_target_aug.npy')
    Y_target = np.load(save_processed_features_dir + 'Y_target_aug.npy')
    X_source = np.load(save_processed_features_dir + 'X_source_aug.npy')
    Y_source = np.load(save_processed_features_dir + 'Y_source_aug.npy')

########
# data processing for training yield/domain prediction model
#######
POR_X=X_source
POR_Y=Y_source
POR_Y_domain = np.zeros(len(POR_Y))
POR_train_X=POR_X[:int(len(POR_X)*(1-test_pct)),:]
POR_test_X=POR_X[int(len(POR_X)*(1-test_pct)):,:]
POR_Y_2d = np.concatenate((np.expand_dims(POR_Y,1),np.expand_dims(POR_Y_domain,1)),1)
POR_train_Y=POR_Y_2d[:int(len(POR_Y)*(1-test_pct))]
POR_test_Y=POR_Y_2d[int(len(POR_Y)*(1-test_pct)):]
HDR_X=X_target
HDR_Y=Y_target
HDR_train_X=HDR_X[:int(len(HDR_X)*(1-test_pct)),:]
HDR_test_X=HDR_X[int(len(HDR_X)*(1-test_pct)):,:]
HDR_Y_domain = np.ones(len(HDR_X))
HDR_Y_2d = np.concatenate((np.expand_dims(HDR_Y,1),np.expand_dims(HDR_Y_domain,1)),1)
HDR_train_Y=HDR_Y_2d[:int(len(HDR_Y)*(1-test_pct))]
HDR_test_Y=HDR_Y_2d[int(len(HDR_Y)*(1-test_pct)):]


if run_LGBM_feature_reduction_flag: ##run this to generate top-k, lgbm-based features
    top_k_feats = get_lgbm_feats(POR_train_X,HDR_train_X,POR_test_X,POR_test_Y, HDR_train_Y, num_lgbm_top_features, LGBM_feature_importance_criterion)
else: #if you've already run the lgbm feature selection, run this instead to simply load the top-k features, bypassing the lgbm training execution
    top_k_feats = np.load(save_processed_features_dir + 'top_k_lgbm_feats_y.npy')[:num_lgbm_top_features]


input_dim = num_lgbm_top_features
hidden_dim = 100 #number hidden neurons for NN
output_dim = 2
model = NeuralNetwork(input_dim, hidden_dim, output_dim).to(device)
LR = 0.001
loss_fn = nn.BCELoss()
loss_fn_no_reduce = nn.BCELoss(reduce=False)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
batch_size = 128
train_dataset = TensorDataset( Tensor(np.concatenate((POR_train_X[:,top_k_feats],HDR_train_X[:,top_k_feats]),0)), Tensor(np.concatenate((POR_train_Y,HDR_train_Y,),0)))
test_dataset = TensorDataset( Tensor(np.concatenate((POR_test_X[:,top_k_feats],HDR_test_X[:,top_k_feats]),0)), Tensor(np.concatenate((POR_test_Y,HDR_test_Y,),0)))
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
best_test_performance = 0


for epoch in range(total_epochs):
    model.train()
    idx = 1
    epoch_loss = 0
    i = 0
    b_ix = 0
    len_dataloader = len(train_dataloader.dataset.tensors[0])
    for X, y in train_dataloader:
        X=X.to(device)
        y=y.to(device)
        optimizer.zero_grad()
        pred = model(X.to(device))
        loss_domain = loss_fn(pred[:,1],y[:,1])
        y = torch.nan_to_num(y, nan=0.0)
        loss_y_pred = torch.mean((1-y[:,1])*loss_fn_no_reduce(pred[:,0],y[:,0]))
        loss = loss_domain + gt_prediction_hp*loss_y_pred
        decorrelate_flag = True
        if decorrelate_flag:
            cov_feat = torch.corrcoef(model.layer_1.weight.swapaxes(0, 1))
            corr_loss = 1 / (cov_feat.shape[0]) * (torch.norm(cov_feat, 'fro') ** 2 - torch.norm(torch.diagonal(cov_feat)) ** 2)
            loss = loss + decorrelation_hp * corr_loss
        #loss_values.append(loss.item())
        epoch_loss = epoch_loss + loss
        loss.backward()
        optimizer.step()
        idx = idx+1
        i = i + 1
        b_ix = b_ix + 1
    print('Epoch #'+str(epoch))
    print('Train epoch loss: ' + str(epoch_loss / idx))

    with torch.no_grad():
        model.eval()
        loss_domain = 0
        loss_y_pred = 0
        domain_total_cnt = 0
        source_total_cnt = 0
        total_domain_correct = 0
        total_source_correct = 0
        test_epoch_loss = 0
        idx = 1
        gt_list = []
        pred_list = []
        por_list = []
        pred_por_list=[]
        target_list=[]
        pred_target_list=[]
        for X, y in test_dataloader:
            X=X.to(device)
            y=y.to(device)
            #pred = model(X[:, top_300_feats])
            pred = model(X)
            y = torch.nan_to_num(y, nan=0.0)
            domain_correct = len(torch.where(torch.round(pred[:, 1]) == y[:, 1])[0])           
            source_indices = torch.where(y[:, 0] == 1)[0]
            source_correct = len(torch.where(y[source_indices, 0] == torch.round(pred[source_indices, 0]))[0])
            source_total_cnt = source_total_cnt + len(source_indices)
            domain_total_cnt = domain_total_cnt + len(y)
            total_domain_correct = total_domain_correct + domain_correct
            total_source_correct = total_source_correct + source_correct
            loss_domain = loss_fn(pred[:, 1], y[:, 1])
            loss_y_pred = torch.mean((1 - y[:, 1]) * loss_fn_no_reduce(pred[:, 0], y[:, 0]))
            loss = loss_y_pred #loss_domain + loss_y_pred
            test_epoch_loss = test_epoch_loss + loss
            pred_list.append(pred)
            gt_list.append(y)
            if y[0,1]==0: #POR class (source)                
                por_list.append(y[0,0])
                pred_por_list.append(pred[0,0])
            if y[0,1]==1: #target class
                target_list.append(y[0,0])
                pred_target_list.append(pred[0,0])
            idx = idx + 1

        pred_domain=torch.stack(pred_list)[:,0,1]
        test_domain =torch.stack(gt_list)[:,0,1]      
        false_positive_rate, recall, thresholds = roc_curve(test_domain.detach().cpu().numpy(), pred_domain.detach().cpu().numpy())
        roc_auc_domain = auc(false_positive_rate, recall)

        test_y_source=torch.stack(por_list)
        pred_y_source=torch.stack(pred_por_list)
        false_positive_rate, recall, thresholds = roc_curve(test_y_source.detach().cpu().numpy(), pred_y_source.detach().cpu().numpy())
        roc_auc_y_source = auc(false_positive_rate, recall) 
        print("Epoch test loss: " +str(test_epoch_loss/idx))
        print("Epoch test source domain Y AUC: " + str(roc_auc_y_source))     
        print("Epoch test domain AUC: " + str(roc_auc_domain))

    run_post = True #flag to run data ranking / visualization analysis
    if run_post and epoch >= num_warmup_epochs and (epoch % 1 == 0):
        if (roc_auc_domain)  > best_test_performance: #if performance is best to date, run full ranking/visualization processes
            print('Best Result!************************')
            best_test_performance = roc_auc_domain#(roc_auc_y_source+roc_auc_y_target+roc_auc_domain)/3.
            idx = 1
            grad_array = torch.zeros(1,num_lgbm_top_features).to(device)
            for X, y in test_dataloader:
                X = X.to(device)
                y = y.to(device)
                # pred = model(X[:, top_300_feats])
                X.requires_grad = True
                pred = model(X)          
                output_signal = pred.mean() #uses both domain prediction and yield prediction
                output_signal.backward()
                grad_array = grad_array + torch.abs(X.grad)
                idx = idx + 1

            torch.save(model.state_dict(), save_processed_features_dir +'best_utd_model' + str(best_test_performance)) #saves best model
            torch.save(grad_array/idx, save_processed_features_dir + 'grad_array')
            vals, indices = torch.topk(grad_array, 500) #500
            vals = vals.detach().cpu().numpy()
            indices = indices.detach().cpu().numpy()

            with open(save_processed_features_dir +'UTD_result500_.txt', 'w') as f: #saved ranked features to txt file
                for indx in range( num_lgbm_top_features):
                    #f.write(str(df_columnlist[indices[0]][indx]+'     *****    '+str(vals[0][indx].item()))+'\n')
                    ##f.write(str(common_features_set[indices[0]][indx] + '     *****    ' + str(vals[0][indx].item()) + '\n'))
                    f.write(str(common_features_set[top_k_feats[indices[0][indx]]] + '     *****    ' + str(vals[0][indx].item()) + '\n'))

            
            for idx in range(50): #visualization for top-50 features
                c_feature = top_k_feats[indices[0][idx]]
                feature_name = common_features_set[top_k_feats[indices[0][idx]]]
                ##c_feature = indices[0][idx]
                ##feature_name = common_features_set[[indices[0][idx]]][0]
                viz_pair(X_source[:, c_feature], X_target[:,c_feature],feature_name,idx,viz_file_path)

            folder_path = save_processed_features_dir
            # ^change the string "C:\username\document",into the path you're using
            list_of_files = sorted(filter(os.path.isfile, glob.glob(folder_path + '*')))
                # for filename in list_of_files: 
            for filename in os.listdir(folder_path):
                if filename.endswith('.png'):
                    img_path = os.path.join(folder_path, filename)
                    img = Image.open(img_path)
                    background = Image.new("RGB", img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[3])  # 3 is the alpha channel
                    img = background
                    pdf_path = os.path.join(folder_path, filename[:-4] + '.pdf')
                    img.save(pdf_path, 'PDF', resolution=100.0)
    
            pdf_merger = PyPDF2.PdfMerger()
            flist = os.listdir(viz_file_path )
            flist.sort(key=lambda fname:int(fname.split('.')[0]))
            for filename in flist:
                # for filename in list_of_files:
                if filename.endswith('.pdf'):
                    pdf_path = os.path.join(viz_file_path , filename)
                    pdf_merger.append(pdf_path)
            pdf_merger.write(os.path.join(save_processed_features_dir, 'merge_viz.pdf')) #saves single pdf for feature visualization
                    # may change the name "merged.pdf" into anyname, and add on the extension: ".pdf"
pdf_merger.close()
