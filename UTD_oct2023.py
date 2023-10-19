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


def viz_pair(data_0, data_1,feature,idx):
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
    plt.savefig('/home/anthonyrhodes/Desktop/UTD_images/'+str(idx+1)+'.png')
    
class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.drop1 = nn.Dropout(0.2)
        self.drop2 = nn.Dropout(0.2)
        #self.BN1 = nn.BatchNorm1d(hidden_dim)
        #self.dropout1 = nn.Dropout1d(p=0.2)

        #self.relu = torch.relu()
        #nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="relu")
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, hidden_dim)
        #self.BN2 = nn.BatchNorm1d(hidden_dim)
        #self.dropout2 = nn.Dropout1d(p=0.2)
        self.layer_4 = nn.Linear(hidden_dim, output_dim)
        #self.sigmoid = torch.sigmoid()

    def forward(self, x):
        #x = torch.nn.functional.relu(self.layer_1(x))
        x = self.layer_1(x)
        x = self.drop1(x)
        #x = self.BN1(x)
        x = torch.relu(x)
        #x = self.dropout1(x)
        x = self.layer_2(x)
        
        x = self.drop2(x)
        
        #x = self.BN2(x)
        x = torch.relu(x)        
        x = self.layer_3(x)
        x = torch.relu(x)
        #x = self.dropout2(x)
        x = self.drop2(x)
        #x = torch.nn.functional.sigmoid(self.layer_2(x))
        x = torch.sigmoid((self.layer_4(x)))

        return x

class NN_reversal(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NN_reversal, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, hidden_dim)

        self.layer_4a = nn.Linear(hidden_dim, 50)
        self.layer_5a = nn.Linear(50, 1)

        self.layer_4b = nn.Linear(hidden_dim, 50)
        self.layer_5b = nn.Linear(50, 1)


    def forward(self, x):
        x = self.layer_1(x)
        x = torch.relu(x)
        x = self.layer_2(x)
        x = torch.relu(x)
        feats = torch.relu((self.layer_3(x)))
        #domain_x = self.layer_4a(ReverseLayerF.apply(feats,alpha))
        domain_x = self.layer_4a(feats)
        domain_x = torch.relu(domain_x)
        #domain_x = torch.sigmoid(domain_x)
        domain_x = torch.sigmoid(self.layer_5a(domain_x))
        #domain_x = self.layer_5a(domain_x)
        #domain_x = torch.softmax(self.layer_5a(domain_x),axis=1)
        x = self.layer_4b(feats)
        #x = torch.sigmoid(x)
        x = torch.relu(x)
        x = torch.sigmoid(self.layer_5b(x))
        #x = self.layer_5b(x)
        #x = torch.softmax(self.layer_5b(x),axis=1)

        return torch.cat((x,domain_x),1)


# df = pd.read_csv('/home/anthonyrhodes/Projects/Labs/UTD/RPL81_SDT_pilot_short.csv')
# pd.read_csv('/home/anthonyrhodes/Projects/Labs/UTD/data/UTD/RPL81_Class_212702_beac1_212880.csv.pd.dat')
# df=pd.read_pickle('/home/anthonyrhodes/Projects/Labs/UTD/data/UTD/RPL81_Class_212702_beac1_212880.csv.pd.dat')
# df=pd.read_pickle('/home/anthonyrhodes/Projects/Labs/UTD/data/UTD/new_report_3b719_250566.csv.pd.dat')
# df=pd.read_pickle('/home/anthonyrhodes/Documents/new_report_3b719_250566.csv.pd.dat')

#'VISUAL_ID', 'Group', 'GT_SDS', 'GT_SDT', 'DN', 'GT_SDT_o', 'GT_SDT_comb', 'SUBSTRUCTURE_ID', 'SORT_LOT', 'Lot7', 'SORT_WAFER', 'SORT_X', 'SORT_Y',
#list(df.columns)
#cols to exclude: visual_id, group, gt_sds, gt_sdt, dn, gt_sdt_o, gt_sdt_comb, substrsucture_df, sort_lot, sort_wafer, sort_x, sort_y, program_name_119325, DevRevStep_119325, FUNCTIONAL_BIN_119325, DATA_BIN_119325, INTERFACE_IBN_119325, FUNCTIONAL_TOTAL_BIN_119325
#list(df.columns.values)
# Drop_list=['VISUAL_ID', 'Group', 'GT_SDS', 'GT_SDT', 'DN', 'GT_SDT_o', 'GT_SDT_comb', 'SUBSTRUCTURE_ID', 'SORT_LOT', 'Lot7', 'SORT_WAFER', 'SORT_X', 'SORT_Y', 'Program_Name_119325', 'DevRevStep_119325', 'FUNCTIONAL_BIN_119325', 'DATA_BIN_119325', 'INTERFACE_BIN_119325', 'FUNCTIONAL_TOTAL_BIN_119325']
#
# POR_row_indices = np.where(df['Group']=='POR')[0]
# HDR_fan_indices = np.where(df['Group']=='HDR@FAN swap')[0]
# HDR_fan_no_anneal_indices = np.where(df['Group']=='HDR@FAN no-anneal')[0]
# HDR_fan_indices=HDR_fan_no_anneal_indices
#
# GT_success_indices = np.where(df['GT_SDS']==1)[0]
# GT_fail_indices = np.where(df['GT_SDS']==0)[0]

#df.keys()
#dict_keys(['data', 'meta', 'feature_list', 'meta_list', 'feature_support'])

# with open('/home/anthonyrhodes/Desktop/top50_feats_UTD.txt') as file:
#     lines = [line.rstrip() for line in file]
test_pct = .20
gen_data = False
folder = '/home/anthonyrhodes/Downloads/'
file = '/home/anthonyrhodes/Downloads/subset_New_Report_RunId_338336_adc1d_338423.csv.pd.dat'
file = '/home/anthonyrhodes/Downloads/RPL81_SDT_c43c8_381746_short1.csv.80.pd.dat'
file = '/home/anthonyrhodes/Downloads/RPL81_SDT_c43c8_381746_short2.csv.pd.dat'
#file = 'Subset of New Report - RunId 338336_adc1d_338423.csv'
out_path = '/home/anthonyrhodes/Projects/Labs/UTD/shift_detection_results_aug'

feature_file = None

if not os.path.exists(out_path):
    os.makedirs(out_path)

file = os.path.join(folder, file)
#with open(file, 'rb') as f:
#    #dataset = pickle.load(f)
#dataset = pd.read_csv('/home/anthonyrhodes/Downloads/338336_adc1d_338423.csv')
#dataset = dd.read_csv('/home/anthonyrhodes/Downloads/338336_adc1d_338423.csv', engine="pyarrow")

filename = '/home/anthonyrhodes/Downloads/subset_New_Report_RunId_338336_adc1d_338423.csv.pd.dat'
filename = '/home/anthonyrhodes/Downloads/RPL81_SDT_c43c8_381746_short1.csv.80.pd.dat'
filename = '/home/anthonyrhodes/Downloads/RPL81_SDT_c43c8_381746_short2.csv.pd.dat'
n = 1000000 #n = sum(1 for line in open(filename))-1  # Calculate number of rows in file
s = n//100#n//4 # sample size of 10%
skip = sorted(random.sample(range(1, n+1), n-s))  # n+1 to compensate for header
#df = pd.read_csv('/home/anthonyrhodes/Downloads/subset_New_Report_RunId_338336_adc1d_338423.csv.pd.dat', skiprows=skip)
dataset = pd.read_pickle('/home/anthonyrhodes/Downloads/RPL81_SDT_c43c8_381746_short2.csv.pd.dat')

# 
# dataset['meta'].drop(['VISUAL_ID','VFC_HDR_GRP','GT_pilot','GT_Hot','PPV_IM','SUBSTRUCTURE_ID','SORT_LOT','Lot7','SORT_WAFER','SORT_X','SORT_Y','Program_Name_119325','DevRevStep_119325','LATO Start WW_119325','FUNCTIONAL_BIN_119325','DATA_BIN_119325',
# 'INTERFACE_BIN_119325','FUNCTIONAL_TOTAL_BIN_119325','SORT_LOT_U1','SORT_WAFER_U1','SORT_X_U1','SORT_Y_U1','LOTS End WW_119325_U1','Program Name_119325_U1','DevRevStep_119325_U1','LATO Start WW_119325_U1','FUNCTIONAL_BIN_119325_U1',
# 'DATA_BIN_119325_U1','INTERFACE_BIN_119325_U1','LOTS End WW_7820_CLASSHOT','Program Name_7820_CLASSHOT','DevRevStep_7820_CLASSHOT','LATO Start WW_7820_CLASSHOT','FUNCTIONAL_BIN_7820_CLASSHOT','DATA_BIN_7820_CLASSHOT','INTERFACE_BIN_7820_CLASSHOT',],
#                       inplace=True,
#                       errors='ignore',
#                       axis=1)




# %% prepare data

num_top_feat = 100
# gt = 'FUNCTIONAL_BIN_119325'
# group = 'SM_DEP_APC'
# site = 'SITE'
# 
# # names = dataset['feature_list']
# # all_data = dataset['data']
# meta = dataset['meta']
# meta['GT'] = (meta[gt] < 600)

'''

if feature_file:
    feature_selected = data_utils.parse_token_list(feature_file)
    feature_selected = [feat for feat in feature_selected if feat in names]
    all_data, names = data_utils.get_cols_by_name(
        all_data, names, feature_selected, num_top_feat)

'''

#
# all_data = all_data.iloc[:, :num_top_feat]
# names = names[:num_top_feat]
# group = dataset['meta'][group]
# site = dataset['meta'][site]
# 
# fabs = ['D1', 'F28']
# populations = ['OLD', 'WiW']
# 
# data = {}
# meta = {}
# group_label = {}

# for fab in fabs:
#     group_label[fab] = np.zeros(((site == fab).sum(),))
#     data[fab] = dataset['data'][site == fab]
#     meta[fab] = dataset['meta'][site == fab]
#     group_label[fab][meta[fab][group] == populations[1]] = 1
#     print(data[fab].shape)


if gen_data:

    
    ##X = dataset['data'].to_numpy()
    ##source_rows = np.where(dataset['meta']['VFC_HDR_GRP']=='POR')
    ##target_rows = np.where(dataset['meta']['VFC_HDR_GRP'] == 'VFC_HDR_SWAP')
    source_rows = np.where(dataset['meta']['GT_pilot']==0)
    target_rows = np.where(dataset['meta']['GT_pilot'] == 1)
    source_list = source_rows[0].tolist()
    Y = dataset['meta']['GT_Hot'].to_numpy()
    Y_Nan_list = np.where(np.isnan(Y))[0].tolist()
    intersect_source_NanY = [x for x in source_list if x in Y_Nan_list]
    dataset['data'].drop(intersect_source_NanY, inplace=True, errors='ignore', axis=0)
    dataset['meta'].drop(intersect_source_NanY, inplace=True, errors='ignore', axis=0)

    source_rows = np.where(dataset['meta']['GT_pilot'] == 0)
    target_rows = np.where(dataset['meta']['GT_pilot'] == 1)
    X = dataset['data'].to_numpy()
    
    # nan_col_indices = np.where((data['D1'].isna().sum() == data['D1'].shape[0]) == True)
    # nan_col_names = data['D1'].columns[nan_col_indices]

    #Y_source=df['meta']['GT_Hot'].to_numpy()
    #X = data['D1'].to_numpy()[source_rows[0]]
    # X = data['D1'].drop(nan_col_names, axis=1)
    imp = SimpleImputer(missing_values=np.nan, fill_value=-100)
    imp.fit(X)
    X = imp.transform(X)
    X = normalize(X, axis=0, norm='max') # normalize data by column max
    X_source = X[source_rows[0]]
    X_target = X[target_rows[0]]
    #Y = dataset['meta']['GT_pilot'].to_numpy()
    Y = dataset['meta']['GT_Hot'].to_numpy()
    Y_source = Y[source_rows]
    Y_target = Y[target_rows]
    #Y_source = meta['D1']['GT'].to_numpy()[source_rows[0]].astype(int)
    #Y_target = meta['D1']['GT'].to_numpy()[target_rows[0]].astype(int)
    
                
                
    # #del X
    # 
    # 
    # X = X.to_numpy()[source_rows[0]]
    # source_features = df['feature_list'].drop(df['feature_list'][nan_col_indices])
    # X = df['data'].drop(df['data'].columns[nan_col_indices],axis=1)
    # del df
    # X = X.to_numpy()
    # 
    # # do pre-processing
    # #X[np.isnan(X)] = np.nanmean(X,axis=0) #replace nans with column means
    # #imp = SimpleIm(puter(missing_values=np.nan, strategy='mean')
    # imp = SimpleImputer(missing_values=np.nan, fill_value=-100)
    # imp.fit(X)
    # X=imp.transform(X)
    # X_source=normalize(X, axis=0, norm='max') #normalize data by column max
    # del X
    # 
    ###############
    ###Source


    #######################################################
    ### Target
    # df=pd.read_pickle('/home/anthonyrhodes/Projects/Labs/UTD/data/UTD/RPL81_Class_212702_beac1_212880.csv.pd.dat')
    # nan_col_indices = np.where((df['data'].isna().sum() == 64818) == True)
    # Y_target = df['meta']['GT_Hot'].to_numpy()
    # target_features = df['feature_list'].drop(df['feature_list'][nan_col_indices])
    # X = df['data'].drop(df['data'].columns[nan_col_indices], axis=1)
    # del df
    # X = X.to_numpy()
    # imp.fit(X)
    # X = imp.transform(X)
    # X_target = normalize(X, axis=0, norm='max')  # normalize data by column max
    # del X
    #
    # common_feature_list=source_features.intersection(target_features)
    # source_feats_common_indices = [i for i, item in enumerate(source_features) if item in set(common_feature_list)]
    # target_feats_common_indices = [i for i, item in enumerate(target_features) if item in set(common_feature_list)]
    # X_source = X_source[:,source_feats_common_indices]
    # X_target = X_target[:,target_feats_common_indices]

    save_flag = False
    if save_flag:
        np.save('/home/anthonyrhodes/Projects/Labs/UTD/X_target_aug.npy', X_target)
        np.save('/home/anthonyrhodes/Projects/Labs/UTD/Y_target_aug.npy', Y_target)
        np.save('/home/anthonyrhodes/Projects/Labs/UTD/X_source_aug.npy', X_source)
        np.save('/home/anthonyrhodes/Projects/Labs/UTD/Y_source_aug.npy', Y_source)
        np.save('/home/anthonyrhodes/Projects/Labs/UTD/common_feats_aug.npy', dataset['data'].keys().to_list())
    #np.save('/home/anthonyrhodes/Projects/Labs/UTD/common_feats_D1.npy', data['D1'].drop(nan_col_names, axis=1).keys().to_list())
    del X, Y, dataset
            

else:
    common_features_set = np.load('/home/anthonyrhodes/Projects/Labs/UTD/common_feats_aug.npy', allow_pickle=True)
    X_target = np.load('/home/anthonyrhodes/Projects/Labs/UTD/X_target_aug.npy')
    Y_target = np.load('/home/anthonyrhodes/Projects/Labs/UTD/Y_target_aug.npy')
    X_source = np.load('/home/anthonyrhodes/Projects/Labs/UTD/X_source_aug.npy')
    Y_source = np.load('/home/anthonyrhodes/Projects/Labs/UTD/Y_source_aug.npy')

# source_target_top50 = np.zeros((50,2))
# for idx in range(len(lines)):
#     print(idx)
#     print('source ' + str(np.where(source_features==lines[idx])))
#     print('target ' + str(np.where(target_features == lines[idx])))
#     feature = source_features[np.where(source_features == lines[idx])[0].item()]
#     source_indx = np.where(source_features == lines[idx])[0].item()
#     target_indx = np.where(target_features == lines[idx])[0].item()
#     source_target_top50[idx,:]=[np.where(source_features==lines[idx])[0].item(),np.where(target_features==lines[idx])[0].item()]
#     viz_pair(X[:,source_indx], X_target[:, target_indx], feature, idx)

POR_X=X_source#[0:150000]#[0:150000,:]#[POR_row_indices,:]
POR_Y=Y_source#[0:150000]#[0:150000]#[POR_row_indices]
POR_Y_domain = np.zeros(len(POR_Y))
POR_train_X=POR_X[:int(len(POR_X)*(1-test_pct)),:]
POR_test_X=POR_X[int(len(POR_X)*(1-test_pct)):,:]
POR_Y_2d = np.concatenate((np.expand_dims(POR_Y,1),np.expand_dims(POR_Y_domain,1)),1)
POR_train_Y=POR_Y_2d[:int(len(POR_Y)*(1-test_pct))]
POR_test_Y=POR_Y_2d[int(len(POR_Y)*(1-test_pct)):]

HDR_X=X_target#[0:20000,:]
HDR_Y=Y_target#[0:20000]
#nan_rows=ws=np.where(np.isnan(HDR_Y))
#HDR_X=np.delete(HDR_X,nan_rows,axis=0)
#HDR_Y=np.delete(HDR_Y,nan_rows,axis=0)

HDR_train_X=HDR_X[:int(len(HDR_X)*(1-test_pct)),:]
HDR_test_X=HDR_X[int(len(HDR_X)*(1-test_pct)):,:]
#HDR_Y = np.zeros(len(HDR_X))
HDR_Y_domain = np.ones(len(HDR_X))
HDR_Y_2d = np.concatenate((np.expand_dims(HDR_Y,1),np.expand_dims(HDR_Y_domain,1)),1)
HDR_train_Y=HDR_Y_2d[:int(len(HDR_Y)*(1-test_pct))]
HDR_test_Y=HDR_Y_2d[int(len(HDR_Y)*(1-test_pct)):]




#model = GradientBoostingClassifier()

recal_LGBM = False
if recal_LGBM:

    model = LGBMClassifier()
    ##model.fit(POR_train_X, POR_train_Y,eval_set =[(POR_test_X,POR_test_Y)] , eval_metric=['auc'],verbose = 2)
    ##X_train = np.concatenate((POR_train_X, HDR_train_X), 0)
    X_train = POR_train_X
    del POR_train_X, HDR_train_X
    ##Y_train = np.concatenate((POR_train_Y, HDR_train_Y), 0)
    Y_train = POR_train_Y
    del POR_train_Y, HDR_train_Y
    ##X_test = np.concatenate((POR_test_X, HDR_test_X), 0)
    X_test = POR_test_X
    del POR_test_X, HDR_test_X
    Y_test = POR_test_Y
    ###Y_test = np.concatenate((POR_test_Y, HDR_test_Y), 0)
    del POR_test_Y, HDR_test_Y
    #model.fit(X_train, Y_train, eval_set=[(X_test, Y_test)], eval_metric=['auc'], verbose=2)


    model.fit(X_train, Y_train[:, 0], eval_set=[(X_test, Y_test[:, 0])], eval_metric=['auc'])#, verbosity=2) #0 index for GT prediction
    ###model.fit(X_train, Y_train[:, 0], eval_set=[(X_test, Y_test[:, 0])], eval_metric=['auc'])  # , verbosity=2) #0 index for GT prediction
    #model.fit(X_train, Y_train[:, 1], eval_set=[(X_test, Y_test[:, 1])], eval_metric=['auc'], verbose=2)
    top_300_feats = (-model.feature_importances_).argsort()[:500]
    np.save('top_300_utd_both_domains_aug2',top_300_feats)
else:
    #top_300_feats2 = np.load('top_300_utd.npy')
    top_300_feats = np.load('top_300_utd_both_domains_aug.npy')[:500]
    #top_300_feats = np.concatenate((top_300_feats, top_300_feats2))

input_dim = 500#16782#300#16621#300#16621 #2997 #300
hidden_dim = 100 #100
output_dim = 2
model = NeuralNetwork(input_dim, hidden_dim, output_dim).to(device)
#model = NN_reversal(input_dim, hidden_dim, output_dim).to(device)
LR = 0.001 #.001
loss_fn = nn.BCELoss()
#loss_fn = nn.CrossEntropyLoss()

weights = torch.tensor([1.0, 10.0])
loss_fn_no_reduce = nn.BCELoss(reduce=False)#, weight=weights.repeat(512,1))
#loss_fn_no_reduce = torch.nn.CrossEntropyLoss(reduce=False, weight=weights)

#loss_fn_no_reduce = nn.CrossEntropyLoss(reduce=False)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

batch_size = 128#512 #1024
#POR_train_dataset = TensorDataset( Tensor(POR_train_X), Tensor(POR_train_Y))
#POR_train_dataset = TensorDataset( Tensor(np.concatenate((POR_train_X,HDR_train_X),0)), Tensor(POR_train_Y))



##train_dataset = TensorDataset(Tensor(POR_train_X[:,top_300_feats]),Tensor(POR_train_Y))  ##( Tensor(np.concatenate((POR_train_X,HDR_train_X),0)), Tensor(np.concatenate((POR_train_Y,HDR_train_Y,),0)))
##test_dataset = TensorDataset(Tensor(POR_test_X[:,top_300_feats]),Tensor(POR_test_Y)) #Tensor(np.concatenate((POR_test_X,HDR_test_X),0)), Tensor(np.concatenate((POR_test_Y,HDR_test_Y,),0)))

##train_dataset = TensorDataset(Tensor(np.concatenate((POR_train_X,HDR_train_X),0)), Tensor(np.concatenate((POR_train_Y,HDR_train_Y,),0)))
##test_dataset = TensorDataset(Tensor(np.concatenate((POR_test_X,HDR_test_X),0)), Tensor(np.concatenate((POR_test_Y,HDR_test_Y,),0)))
train_dataset = TensorDataset( Tensor(np.concatenate((POR_train_X[:,top_300_feats],HDR_train_X[:,top_300_feats]),0)), Tensor(np.concatenate((POR_train_Y,HDR_train_Y,),0)))
test_dataset = TensorDataset( Tensor(np.concatenate((POR_test_X[:,top_300_feats],HDR_test_X[:,top_300_feats]),0)), Tensor(np.concatenate((POR_test_Y,HDR_test_Y,),0)))

#POR_test_dataset = TensorDataset( Tensor(POR_test_X), Tensor(POR_test_Y))
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
#POR_train_dataloader = DataLoader(dataset=(torch.from_numpy(POR_train_X),torch.from_numpy(POR_train_Y)), batch_size=batch_size, shuffle=True)
#POR_test_dataloader = DataLoader(dataset=(torch.from_numpy(POR_test_X),torch.from_numpy(POR_test_Y)), batch_size=batch_size, shuffle=True)

best_test_performance = 0
for epoch in range(1000):

    model.train()
    idx = 1
    epoch_loss = 0
    i = 0
    b_ix = 0
    len_dataloader = len(train_dataloader.dataset.tensors[0])
    for X, y in train_dataloader:
    # zero the parameter gradients
        X=X.to(device)
        y=y.to(device)
        p = float(i + (epoch+1) * len_dataloader) / (epoch+1) / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        #alpha = 1-1/(epoch+1)
        alpha = .01
        optimizer.zero_grad()
        #print(b_ix)

    # forward + backward + optimize

        #loss = loss_fn(pred, y.unsqueeze(-1))
        #pred = model(X[:, top_300_feats])
        pred = model(X.to(device))
        loss_domain = loss_fn(pred[:,1],y[:,1])
        #loss_domain = loss_fn(pred[:, 0], y[:, 0])
        y = torch.nan_to_num(y, nan=0.0)
        loss_y_pred = torch.mean((1-y[:,1])*loss_fn_no_reduce(pred[:,0],y[:,0]))
        domain_y_hp = 10
        loss = loss_domain + domain_y_hp*loss_y_pred

        decorrelate_flag = True
        if decorrelate_flag:
            cov_feat = torch.corrcoef(model.layer_1.weight.swapaxes(0, 1))
            corr_loss = 1 / (cov_feat.shape[0]) * (torch.norm(cov_feat, 'fro') ** 2 - torch.norm(torch.diagonal(cov_feat)) ** 2)
            loss = loss + .01 * corr_loss #.1
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
            ##domain_correct = len(torch.where(torch.round(pred[:, 0]) == y[:, 0])[0])
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
        ##pred_domain=torch.stack(pred_list)[:,0,0]
        ##test_domain =torch.stack(gt_list)[:,0,0]
        false_positive_rate, recall, thresholds = roc_curve(test_domain.detach().cpu().numpy(), pred_domain.detach().cpu().numpy())
        roc_auc_domain = auc(false_positive_rate, recall)

        test_y_source=torch.stack(por_list)
        pred_y_source=torch.stack(pred_por_list)
        false_positive_rate, recall, thresholds = roc_curve(test_y_source.detach().cpu().numpy(), pred_y_source.detach().cpu().numpy())
        roc_auc_y_source = auc(false_positive_rate, recall)
        #
        # test_y_target = torch.stack(target_list)
        # pred_y_target = torch.stack(pred_target_list)
        # false_positive_rate, recall, thresholds = roc_curve(test_y_target.detach().cpu().numpy(), pred_y_target.detach().cpu().numpy())
        # roc_auc_y_target = auc(false_positive_rate, recall)

        print("Epoch test loss: " +str(test_epoch_loss/idx))
        print("Epoch test domain accuracy: " +str(total_domain_correct/domain_total_cnt))
        print("Epoch test source domain Y AUC: " + str(roc_auc_y_source))
        #print("Epoch test target domain Y AUC: " + str(roc_auc_y_target))
        #print("Epoch test Y/GT accuracy: " +str(total_source_correct/source_total_cnt))
        print("Epoch test domain AUC: " + str(roc_auc_domain))

    run_post = True
    if run_post and epoch >= 50 and (epoch % 10 == 0):
        #if (roc_auc_y_source+roc_auc_y_target+roc_auc_domain)/3. > best_test_performance:
        if (roc_auc_domain)  > best_test_performance:
            print('Best Result!************************')
            best_test_performance = roc_auc_domain#(roc_auc_y_source+roc_auc_y_target+roc_auc_domain)/3.
            idx = 1
            grad_array = torch.zeros(1,500).to(device)  #2997 #16621
            for X, y in test_dataloader:
                X = X.to(device)
                y = y.to(device)
                # pred = model(X[:, top_300_feats])
                X.requires_grad = True
                pred = model(X)
                d_loss = loss_fn(pred[:, 1], y[:, 1])
                #d_loss = loss_fn(pred[:, 0], y[:, 0])
                d_loss.backward() #algo decision: backpropagation based on domain loss signal!
                grad_array = grad_array + torch.abs(X.grad) #X.grad #algo decision here: use abs or raw gradient values!
                idx = idx + 1

            torch.save(model.state_dict(), '/home/anthonyrhodes/Projects/Labs/UTD/best_utd_model_aug' + str(best_test_performance))
            torch.save(grad_array/idx, '/home/anthonyrhodes/Projects/Labs/UTD/grad_array_aug')
            vals, indices = torch.topk(grad_array, 500) #500
            vals = vals.detach().cpu().numpy()
            indices = indices.detach().cpu().numpy()

            with open('/home/anthonyrhodes/Projects/Labs/UTD/UTD_result300_aug.txt', 'w') as f:
                for indx in range(300): #500
                    #f.write(str(df_columnlist[indices[0]][indx]+'     *****    '+str(vals[0][indx].item()))+'\n')
                    ##f.write(str(common_features_set[indices[0]][indx] + '     *****    ' + str(vals[0][indx].item()) + '\n'))
                    f.write(str(common_features_set[top_300_feats[indices[0][indx]]] + '     *****    ' + str(vals[0][indx].item()) + '\n'))

            
            for idx in range(50):
                c_feature = top_300_feats[indices[0][idx]]
                feature_name = common_features_set[top_300_feats[indices[0][idx]]]
                ##c_feature = indices[0][idx]
                ##feature_name = common_features_set[[indices[0][idx]]][0]
                viz_pair(X_source[:, c_feature], X_target[:,c_feature],feature_name,idx)

            folder_path = r'/home/anthonyrhodes/Desktop/UTD_images'
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
            flist = os.listdir(folder_path)
            flist.sort(key=lambda fname: int(fname.split('.')[0]))
            for filename in flist:
                # for filename in list_of_files:
                if filename.endswith('.pdf'):
                    pdf_path = os.path.join(folder_path, filename)
                    pdf_merger.append(pdf_path)
            pdf_merger.write(os.path.join('/home/anthonyrhodes/Desktop/', 'merge_july.pdf'))
                    # may change the name "merged.pdf" into anyname, and add on the extension: ".pdf"
pdf_merger.close()
            
            
            #     #print(idx)
            #     #print('source ' + str(np.where(source_features == lines[idx])))
            #     #print('target ' + str(np.where(target_features == lines[idx])))
            #     c_feature = common_features_set[top_300_feats[indices[0][indx]]]
            #     feature = source_features[np.where(source_features == lines[idx])[0].item()]
            #     source_indx = np.where(source_features == lines[idx])[0].item()
            #     target_indx = np.where(target_features == lines[idx])[0].item()
            #     source_target_top50[idx, :] = [np.where(source_features == lines[idx])[0].item(),
            #                                    np.where(target_features == lines[idx])[0].item()]
            #     viz_pair(X[:, source_indx], X_target[:, target_indx], feature, idx)
                    



        #model.load_state_dict(torch.load('/home/anthonyrhodes/Projects/Labs/UTD/utd_model'))
#
# X.requires_grad=True
# loss = loss_fn(pred[:, 1], y[:, 1])
# loss.backward()
# X.grad
#train_predictions=model.predict(POR_train_X)
#test_predictions=model.predict(POR_test_X)

(-model.feature_importances_).argsort()[:100] #top 100 feats
print()

#gbm.fit(train_X, train_Y,eval_set =[(test_X,test_Y)] , eval_metric=['auc'],
        #early_stopping_rounds=10,verbose = 2)