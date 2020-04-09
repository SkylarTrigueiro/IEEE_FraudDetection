import config
import joblib
import numpy as np
import sys
import pandas as pd
import features as ft
import feat_selection as fs
from data_management import load_data
from sklearn.model_selection import train_test_split
import preprocessor as pp
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from feature_engine import discretisers as dis
from sklearn.preprocessing import LabelEncoder

#%%
    
train, submission = load_data()
print('data has been loaded.')
   
train, y = ft.get_features(train, submission )
print("train and target have been partitioned.")
    
both = pd.concat([train, submission])
    
train, submission, both = ft.feature_generator( train, submission, both )
print('New features have been added.')
    
NUMERIC, NUMERIC_NA, CATEG, CATEG_NA, DISCRETE, DISCRETE_NA = ft.partition_features( train, submission, both )
print('features have been partitioned.')

#%%

train = both.iloc[:len(train), :]
submission = both.iloc[len(train):, :]

#%%

Xy_train = train.copy()
Xy_train['isFraud'] = y.values

#%%
    
def resumetable(df):
    print(f"Dataset Shape: {df.shape}")
    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])
    summary = summary.reset_index()
    summary['Name'] = summary['index']
    summary = summary[['Name','dtypes']]
    summary['% Missing'] = 100*round(df.isnull().sum()/len(df),3).values    
    summary['Uniques'] = df.nunique().values

    for name in summary['Name'].value_counts().index:
        summary.loc[summary['Name'] == name, 'Entropy'] = round(stats.entropy(df[name].value_counts(normalize=True), base=2),2) 

    return summary

#%%
    
summary = resumetable(Xy_train)

#%%

def cat_summary(df, cat, target):
    
    summ = df.groupby(cat)[target].mean()*100
    summ = pd.DataFrame( summ.values, columns = ['% Target'], index = summ.index )
    summ = summ.join( df[cat].value_counts() )
    summ.columns = ['% Target', 'Value Counts']
    
    return summ

#%% 
    
columns = list(Xy_train.columns)

#%%

missing = train.isnull().mean()*100

#%%

submission.isnull().mean()*100

#%%

dic = {}
for col in train.columns:
    dic[col] = cat_summary(Xy_train, col, 'isFraud')
    
#%%
    
locals().update(dic)
del dic

#%%

feature_importance = pd.read_csv('feature_importances.csv')


#%%

plt.figure(figsize=(16, 150))
sns.barplot(data=feature_importance.sort_values(by='average', ascending=False), x='average', y='feature');
plt.title('50 TOP feature importance over 5 folds average');

#%%

most_important_features = feature_importance.sort_values(by='average', ascending=False).head(100)
most_important_features = list(most_important_features['feature'])

#%%

def corr1(col):
    N = None #10000

    feature_importance = pd.read_csv('feature_importances.csv')
    most_important_features = feature_importance.sort_values(by='average', ascending=False).head(50)
    most_important_features = list(most_important_features['feature'])
    
    num_vars = [f for f in most_important_features if both[f].dtype != 'object']
    trx = both.head(N) if N is not None else both.copy()
    corrs = abs(trx[num_vars].corrwith(trx[col])).reset_index().sort_values(0, ascending=False).reset_index(drop=True).rename({'index':'Feature',0:'Correlation with ' + col}, axis=1)
    trx = corrs.dropna().tail(5)
    
    return trx

#%%
    
sub_1 = pd.read_csv('lgb_uid1_934124.csv', index_col='TransactionID')
sub_2 = pd.read_csv('lgb_uid3_934457.csv', index_col='TransactionID')
sub_3 = pd.read_csv('lgb_uid5_93975.csv', index_col='TransactionID')
sub_4 = pd.read_csv('lgb_uid7_939616.csv', index_col='TransactionID')
sub_5 = pd.read_csv('lgb_uid9_937851.csv', index_col = 'TransactionID')
sub_6 = pd.read_csv('lgb_uid10_93788.csv', index_col='TransactionID')

#%%

sub_1 += sub_2
sub_1 += sub_3
sub_1 += sub_4
sub_1 += sub_5
sub_1 += sub_6

#%%

sub_1.to_csv('6ensemble_time_series_cv.csv')

#%%
    
phone_specs_links = joblib.load('data/phone_specs_links.joblib')

#%%

model_and_specs_info = joblib.load('data/model_and_specs_info.joblib')

#%%

def get_manufacturer(url):
    if url is not None and 'handsetdetection' in url:
        return url.split('/devices/')[1].split('/', 1)[0]
    else:
        return "missing"
    
#%%
        
Xy_train['DeviceInfo'] = Xy_train['DeviceInfo'].astype('str')
Xy_train['ClearDeviceInfo'] = np.vectorize(lambda x : x.lower().split('build')[0].strip())(Xy_train['DeviceInfo'])
Xy_train['ClearDeviceInfo'] = Xy_train['ClearDeviceInfo'].astype('str')
Xy_train['manufacturer'] = Xy_train['ClearDeviceInfo'].apply(lambda model: get_manufacturer(phone_specs_links.get(model, "")))

