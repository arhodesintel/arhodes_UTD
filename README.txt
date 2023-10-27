Algorithm workflow for feature ranking with distribution shift data: please see UTD_NN.pdf for schematic details. 

Summary
   (1) Data are loaded and processed from pandas dataframe (see gen_data() function)
   (2) Run LGBM to determine 'top-k' (e.g., k=500) features 
   (3) Using these top-k features, we train a NN to concurrently predict: (i) domain (e.g., source/target) and (2) yield GT 
   (4) From best-performing model, the algorithm ranks the data features according to importance (producing a txt file), in addition to providing visualization of the top ranked features (in pdf format)
   
CPU Run-time: ~27 seconds per epoch (with Batch size = 128)

Import requirements
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


Algorithm parameters, file paths, etc.
test_pct = .20 #percentage of the data that you wish to use for testing vs. training
gen_data_flag = False #set to true the first time you load the dataset, this will create a pandas data frame, etc.
dataset = pd.read_pickle('/home/anthonyrhodes/Projects/Labs/UTD/data/RPL81_Class_212702_beac1_212880.csv.pd.dat') #point this to the location of your dataset
drop_features_array = [] #include any features as an array of string that you wish to drop from the training/testing of the predictive model
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


