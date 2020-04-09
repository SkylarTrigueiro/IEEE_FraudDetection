# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 17:56:07 2019

@author: striguei
"""
import config
import joblib
import numpy as np
import pandas as pd
from feature_engine import missing_data_imputers as mdi
from feature_engine import categorical_encoders as ce
from tqdm import tqdm

def get_features(train, submission ):

    TARGET = 'isFraud'
    y = train[TARGET]
    train.drop(TARGET, axis = 1, inplace = True)
    
    return train, y

def partition_features(train, submission, both):
    
    if( not config.KAGGLE_CHEATS ):

        FEATURES = list(train.columns)
        if( config.DISC_IS_CATEG ):
            CATEG = categ_list(train, FEATURES)
        else:
            CATEG = [ var for var in FEATURES if train[var].dtype == 'O' ]
        
        NUMERIC = [var for var in FEATURES if ( (train[var].dtype != 'O') and (var not in CATEG)) ]
        DISCRETE = [ var for var in NUMERIC if ( (train[var].nunique() < config.MAX_CAT_CARDINALITY) and (submission[var].nunique() < config.MAX_CAT_CARDINALITY) )]
        DISCRETE = [ var for var in DISCRETE if var not in CATEG]
        NUMERIC = [ var for var in NUMERIC if var not in DISCRETE]
        CATEG_NA = [ var for var in CATEG if ( (train[var].isnull().sum() > 0) or (submission[var].isnull().sum() > 0) )]
        NUMERIC_NA = [ var for var in NUMERIC if ( (train[var].isnull().sum() > 0) or (submission[var].isnull().sum() > 0) )]
        DISCRETE_NA = [ var for var in DISCRETE if ( (train[var].isnull().sum() > 0) or (submission[var].isnull().sum() > 0) )]
    
    else:
        
        FEATURES = list(both.columns)
        if( config.DISC_IS_CATEG ):
            CATEG = categ_list(both, FEATURES)
        else:
            CATEG = [ var for var in FEATURES if both[var].dtype == 'O' ]
        
        NUMERIC = [var for var in FEATURES if ( (both[var].dtype != 'O') and (var not in CATEG)) ]
        DISCRETE = [ var for var in NUMERIC if ( (both[var].nunique() < config.MAX_CAT_CARDINALITY) ) ]
        DISCRETE = [ var for var in DISCRETE if var not in CATEG]
        NUMERIC = [ var for var in NUMERIC if var not in DISCRETE]
        CATEG_NA = [ var for var in CATEG if ( (both[var].isnull().sum() > 0) )]
        NUMERIC_NA = [ var for var in NUMERIC if ( (both[var].isnull().sum() > 0) )]
        DISCRETE_NA = [ var for var in DISCRETE if ( (both[var].isnull().sum() > 0) )]
        
    return NUMERIC, NUMERIC_NA, CATEG, CATEG_NA, DISCRETE, DISCRETE_NA

def categ_list(train, FEATURES):
    
    CATEG = [var for var in FEATURES if (train[var].dtype == 'O' ) ]
    for i in range(1,7):
        new = 'card' + str(i)
        if( new not in CATEG):
            CATEG.append(new)
    for i in range(12, 39):
        new = 'id_' + str(i)
        if( new not in CATEG):
            CATEG.append(new)
    CATEG.sort()

    return CATEG

def feature_generator( train, submission, both ):
    
    train, submission, both = correlated_features( train, submission, both)
    print('correlated features engineered')
    train, submission, both = D9(train, submission, both)
    print('D9 values filled.')
    train, submission, both = fill_unique_value_pairs(train, submission, both)
    print('Unique values filled.')
    #train, submission, both = ft.TransactionAmt_agg(train, submission, both)
    #train, submission, both = ft.id_02_agg(train, submission, both)
    #train, submission, both = ft.D15_agg(train, submission, both)
    train, submission, both = P_emaildomain(train, submission, both)
    print('P_emaildomain features engineered.')
    train, submission, both = R_emaildomain(train, submission, both)
    print('R_emaildomain features engineered.')
    train, submission, both = D16(train, submission, both)
    print('D16 features engineered.')
    train, submission, both = e2e_1(train, submission, both)
    print('e2e1 features engineered.')
    train, submission, both = e2e_2(train, submission, both)
    print('e2e2 features engineered.')
    train, submission, both = e2e_3(train, submission, both)
    print('e2e3 features engineered.')
    train, submission, both = browser( train,submission, both )
    print('browser features engineered.')
    train, submission, both = OS( train,submission, both )
    print('OS features engineered.')
    train, submission, both = PhoneSpecs( train,submission, both )
    print('PhoneSpecs features engineered.')
    train, submission, both = ScreenSize( train,submission, both )
    print('ScreenSize features engineered.')
    train, submission, both = TransactionAmtDecimal( train,submission, both )
    print('TransactionAmtDecimal features engineered.')
    train, submission, both = TransactionAmt_Last_Decimal( train, submission, both )
    print('TransactionAmt_Last_tDecimal features engineered.')
    #train, submission, both = TransactionAmtLog( train, submission, both )
    #print('TransactionAmtLog features engineered.')
    train, submission, both = M(train, submission, both)
    print('M features engineered.')
    train, submission, both = time( train, submission, both )
    print('time features engineered.')
    train, submission, both = UserID(train, submission, both)
    print('User features engineered.')
    train, submission, both = interaction(train, submission, both)
    print('Interaction features engineered.')
    #train, submission, both = TransactionAmt_time( train, submission, both )
    #train, submission, both = TransactionAmt_UserID_Avg( train, submission, both )
    train, submission, both = TransactionDT_UserID_timediff(train, submission, both)
    
    
    if( not config.KAGGLE_CHEATS ):
        train.drop('P_emaildomain', axis=1, inplace = True )
        train.drop('R_emaildomain', axis=1, inplace = True )
        submission.drop('P_emaildomain', axis=1, inplace = True )
        submission.drop('R_emaildomain', axis=1, inplace = True )
    else:
        submission.drop('P_emaildomain', axis=1, inplace = True )
        submission.drop('R_emaildomain', axis=1, inplace = True )
    
    return train, submission, both

def feature_generator2( train, submission, both, CATEG):
    
    train, submission, both = Frequent_Categorical_Encoder( train, submission, both, CATEG )
    train, submission, both = D_normalized(train, submission, both)
    
    return train, submission, both

def top_x( train, submission, both, CATEG):
    
    encoder = ce.OneHotCategoricalEncoder(top_categories = config.MAX_CAT_CARDINALITY, variables=CATEG)
    encoder.fit(train)
    
    train = encoder.transform(train)
    submission = encoder.transform(submission)
    
    return train, submission, both
        
def get_OHE(df, var_list):
    df_OHE = pd.concat([df[var_list], 
                         pd.get_dummies(df[var_list], drop_first=True)],
                        axis=1
                       )
    return df_OHE

def D9(train, submission, both):
    
    if( not config.KAGGLE_CHEATS ):
        
        train['D9'] = train['TransactionDT']//(3600)%24
        train['D9'] = train['D9'].astype(int)
        
        submission['D9'] = submission['TransactionDT']//(3600)%24
        submission['D9'] = submission['D9'].astype(int)
        
    else:
        
        both['D9'] = both['TransactionDT']//(3600)%24
        both['D9'] = both['D9'].astype(int)
    
    return train, submission, both

def D16( train, submission, both ):
    
    if( not config.KAGGLE_CHEATS ):
        
        train['D16'] = train['TransactionDT']//(3600*24)%7
        submission['D16'] = submission['TransactionDT']//(3600*24)%7
        
    else:
        
        both['D16'] = both['TransactionDT']//(3600*24)%7
    
    return train, submission, both

def e2e_1( train, submission, both):
    
    if( not config.KAGGLE_CHEATS ):
        
        imputer = mdi.CategoricalVariableImputer(variables = ['P_emaildomain_1', 'R_emaildomain_1'])
        imputer.fit(train)
    
        train = imputer.transform(train)
        submission = imputer.transform(submission)
    
        train['e2e_1'] = train['P_emaildomain_1'] + '_to_' + train['R_emaildomain_1']
        submission['e2e_1'] = submission['P_emaildomain_1'] + '_to_' + submission['R_emaildomain_1']    
        
    else:
    
        imputer = mdi.CategoricalVariableImputer(variables = ['P_emaildomain_1', 'R_emaildomain_1'])
        imputer.fit(both)
    
        both = imputer.transform(both)
    
        both['e2e_1'] = both['P_emaildomain_1'] + '_to_' + both['R_emaildomain_1']
        
        
    return train, submission, both

def e2e_2( train, submission, both):
    
    if( not config.KAGGLE_CHEATS ):
        
        imputer = mdi.CategoricalVariableImputer(variables = ['P_emaildomain_2', 'R_emaildomain_2'])
        imputer.fit(train)
    
        train = imputer.transform(train)
        submission = imputer.transform(submission)
    
        train['e2e_2'] = train['P_emaildomain_2'] + '_to_' + train['R_emaildomain_2']
        submission['e2e_2'] = submission['P_emaildomain_2'] + '_to_' + submission['R_emaildomain_2']    
        
    else:
        
        imputer = mdi.CategoricalVariableImputer(variables = ['P_emaildomain_2', 'R_emaildomain_2'])
        imputer.fit(both)
    
        both = imputer.transform(both)
    
        both['e2e_2'] = both['P_emaildomain_2'] + '_to_' + both['R_emaildomain_2']
    
    return train, submission, both

def e2e_3( train, submission, both):
    
    if( not config.KAGGLE_CHEATS ):

        imputer = mdi.CategoricalVariableImputer(variables = ['P_emaildomain_3', 'R_emaildomain_3'])
        imputer.fit(train)
    
        train = imputer.transform(train)
        submission = imputer.transform(submission)
    
        train['e2e_3'] = train['P_emaildomain_3'] + '_to_' + train['R_emaildomain_3']
        submission['e2e_3'] = submission['P_emaildomain_3'] + '_to_' + submission['R_emaildomain_3']    
        
    else:
        
        imputer = mdi.CategoricalVariableImputer(variables = ['P_emaildomain_3', 'R_emaildomain_3'])
        imputer.fit(both)
    
        both = imputer.transform(both)
    
        both['e2e_3'] = both['P_emaildomain_3'] + '_to_' + both['R_emaildomain_3']
    
    return train, submission, both

def TransactionAmt_agg( train, submission, both):

    if( not config.KAGGLE_CHEATS ): 
        
        train['TransactionAmt_to_mean_card1'] = train['TransactionAmt'] / train.groupby(['card1'])['TransactionAmt'].transform('mean')
        train['TransactionAmt_to_mean_card4'] = train['TransactionAmt'] / train.groupby(['card4'])['TransactionAmt'].transform('mean')
        train['TransactionAmt_to_std_card1'] = train['TransactionAmt'] / train.groupby(['card1'])['TransactionAmt'].transform('std')
        train['TransactionAmt_to_std_card4'] = train['TransactionAmt'] / train.groupby(['card4'])['TransactionAmt'].transform('std')
 
        submission['TransactionAmt_to_mean_card1'] = submission['TransactionAmt'] /submission.groupby(['card1'])['TransactionAmt'].transform('mean')
        submission['TransactionAmt_to_mean_card4'] = submission['TransactionAmt'] /submission.groupby(['card4'])['TransactionAmt'].transform('mean')
        submission['TransactionAmt_to_std_card1'] = submission['TransactionAmt'] /submission.groupby(['card1'])['TransactionAmt'].transform('std')
        submission['TransactionAmt_to_std_card4'] = submission['TransactionAmt'] /submission.groupby(['card4'])['TransactionAmt'].transform('std')
        
    else:
        
        both['TransactionAmt_to_mean_card1'] = both['TransactionAmt'] /both.groupby(['card1'])['TransactionAmt'].transform('mean')
        both['TransactionAmt_to_mean_card4'] = both['TransactionAmt'] /both.groupby(['card4'])['TransactionAmt'].transform('mean')
        both['TransactionAmt_to_std_card1'] = both['TransactionAmt'] /both.groupby(['card1'])['TransactionAmt'].transform('std')
        both['TransactionAmt_to_std_card4'] = both['TransactionAmt'] /both.groupby(['card4'])['TransactionAmt'].transform('std')
   
    return train, submission, both

def id_02_agg(train, submission, both):

    if( not config.KAGGLE_CHEATS ):
        
        train['id_02_to_mean_card1'] = train['id_02'] / train.groupby(['card1'])['id_02'].transform('mean')
        train['id_02_to_mean_card4'] = train['id_02'] / train.groupby(['card4'])['id_02'].transform('mean')
        train['id_02_to_std_card1'] = train['id_02'] / train.groupby(['card1'])['id_02'].transform('std')
        train['id_02_to_std_card4'] = train['id_02'] / train.groupby(['card4'])['id_02'].transform('std')
   
        submission['id_02_to_mean_card1'] = submission['id_02'] / submission.groupby(['card1'])['id_02'].transform('mean')
        submission['id_02_to_mean_card4'] = submission['id_02'] / submission.groupby(['card4'])['id_02'].transform('mean')
        submission['id_02_to_std_card1'] = submission['id_02'] / submission.groupby(['card1'])['id_02'].transform('std')
        submission['id_02_to_std_card4'] = submission['id_02'] / submission.groupby(['card4'])['id_02'].transform('std')
        
    else:
        
        both['id_02_to_mean_card1'] = both['id_02'] / both.groupby(['card1'])['id_02'].transform('mean')
        both['id_02_to_mean_card4'] = both['id_02'] / both.groupby(['card4'])['id_02'].transform('mean')
        both['id_02_to_std_card1'] = both['id_02'] / both.groupby(['card1'])['id_02'].transform('std')
        both['id_02_to_std_card4'] = both['id_02'] / both.groupby(['card4'])['id_02'].transform('std')
   
    return train, submission, both

def D15_agg( train, submission, both ):
    
    if( not config.KAGGLE_CHEATS ):
    
        train['D15_to_mean_card1'] = train['D15'] / train.groupby(['card1'])['D15'].transform('mean')
        train['D15_to_mean_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('mean')
        train['D15_to_std_card1'] = train['D15'] / train.groupby(['card1'])['D15'].transform('std')
        train['D15_to_std_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('std')
    
        submission['D15_to_mean_card1'] = submission['D15'] / submission.groupby(['card1'])['D15'].transform('mean')
        submission['D15_to_mean_card4'] = submission['D15'] / submission.groupby(['card4'])['D15'].transform('mean')
        submission['D15_to_std_card1'] = submission['D15'] / submission.groupby(['card1'])['D15'].transform('std')
        submission['D15_to_std_card4'] = submission['D15'] / submission.groupby(['card4'])['D15'].transform('std')

        train['D15_to_mean_addr1'] = train['D15'] / train.groupby(['addr1'])['D15'].transform('mean')
        train['D15_to_mean_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('mean')
        train['D15_to_std_addr1'] = train['D15'] / train.groupby(['addr1'])['D15'].transform('std')
        train['D15_to_std_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('std')
    
        submission['D15_to_mean_addr1'] = submission['D15'] / submission.groupby(['addr1'])['D15'].transform('mean')
        submission['D15_to_mean_card4'] = submission['D15'] / submission.groupby(['card4'])['D15'].transform('mean')
        submission['D15_to_std_addr1'] = submission['D15'] / submission.groupby(['addr1'])['D15'].transform('std')
        submission['D15_to_std_card4'] = submission['D15'] / submission.groupby(['card4'])['D15'].transform('std')
        
    else:
        
        both['D15_to_mean_card1'] = both['D15'] / both.groupby(['card1'])['D15'].transform('mean')
        both['D15_to_mean_card4'] = both['D15'] / both.groupby(['card4'])['D15'].transform('mean')
        both['D15_to_std_card1'] = both['D15'] / both.groupby(['card1'])['D15'].transform('std')
        both['D15_to_std_card4'] = both['D15'] / both.groupby(['card4'])['D15'].transform('std')

        both['D15_to_mean_addr1'] = both['D15'] / both.groupby(['addr1'])['D15'].transform('mean')
        both['D15_to_mean_card4'] = both['D15'] / both.groupby(['card4'])['D15'].transform('mean')
        both['D15_to_std_addr1'] = both['D15'] / both.groupby(['addr1'])['D15'].transform('std')
        both['D15_to_std_card4'] = both['D15'] / both.groupby(['card4'])['D15'].transform('std')
    
    return train, submission, both

def P_emaildomain(train, submission, both):
    
    if( not config.KAGGLE_CHEATS ):
        
        train[['P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3']] = train['P_emaildomain'].str.split('.', expand=True)
        submission[['P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3']] = submission['P_emaildomain'].str.split('.', expand=True)
    
    else:
        
        both[['P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3']] = both['P_emaildomain'].str.split('.', expand=True)
    
    return train, submission, both
    
def R_emaildomain(train, submission, both):
    
    if( not config.KAGGLE_CHEATS ):
        
        train[['R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3']] = train['R_emaildomain'].str.split('.', expand=True)    
        submission[['R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3']] = submission['R_emaildomain'].str.split('.', expand=True)
    
    else:
    
        both[['R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3']] = both['R_emaildomain'].str.split('.', expand=True)
    
    return train, submission, both

def browser( train, submission, both ):
    
    if( not config.KAGGLE_CHEATS ):
        
        train['browser'] = train['id_31'].str.split(' ', expand = True)[0]
        submission['browser'] = submission['id_31'].str.split(' ', expand = True)[0]
        
        train.drop(['id_31'], axis=1, inplace = True )
        submission.drop(['id_31'], axis=1, inplace = True )
        
    else:
        
        both['browser'] = both['id_31'].str.split(' ', expand=True)[0]
        both.drop(['id_31'], axis=1, inplace = True )
    
    return train, submission, both

def OS( train, submission, both ):
    
    if( not config.KAGGLE_CHEATS ):
        
        train['OS'] = train['id_30'].str.split(' ', expand = True)[0]
        submission['OS'] = submission['id_30'].str.split(' ', expand = True)[0]
        
        train.drop(['id_30'], axis=1, inplace = True )
        submission.drop(['id_30'], axis=1, inplace = True )
        
    else:
        
        both['OS'] = both['id_30'].str.split(' ', expand = True)[0]
        both.drop(['id_30'], axis=1, inplace = True )
        
    return train, submission, both

def get_manufacturer(url):
    if url is not None and 'handsetdetection' in url:
        return url.split('/devices/')[1].split('/', 1)[0]
    else:
        return "missing"
    
def try_get_ram(text):
    parts = text.split(',')
    if len(parts) != 2:
        return "missing"
    else:
        return parts[1].strip()
    
def try_get_storage(text):
    parts = text.split(',')
    if len(parts) != 2:
        return "missing"
    else:
        return parts[0].strip()

def PhoneSpecs( train, submission, both ):
    
    phone_specs_links = joblib.load('data/phone_specs_links.joblib')
    model_and_specs = joblib.load('data/model_and_specs_info.joblib')
    
    if( not config.KAGGLE_CHEATS ):
        
        train['DeviceInfo'] = train['DeviceInfo'].astype('str')
        train['ClearDeviceInfo'] = np.vectorize(lambda x : x.lower().split('build')[0].strip())(train['DeviceInfo'])
        train['ClearDeviceInfo'] = train['ClearDeviceInfo'].astype('str')
        train['manufacturer'] = train['ClearDeviceInfo'].apply(lambda model: get_manufacturer(phone_specs_links.get(model, "")))
        train['RAM'] = train['ClearDeviceInfo'].apply(lambda x :\
                 try_get_ram(model_and_specs.get(x, {'memory internal':","}).get('memory internal', ",")))
        
        submission['DeviceInfo'] = submission['DeviceInfo'].astype('str')
        submission['ClearDeviceInfo'] = np.vectorize(lambda x : x.lower().split('build')[0].strip())(submission['DeviceInfo'])
        submission['ClearDeviceInfo'] = submission['ClearDeviceInfo'].astype('str')
        submission['manufacturer'] = submission['ClearDeviceInfo'].apply(lambda model: get_manufacturer(phone_specs_links.get(model, "")))
        submission['RAM'] = submission['ClearDeviceInfo'].apply(lambda x :\
                 try_get_ram(model_and_specs.get(x, {'memory internal':","}).get('memory internal', ",")))

        train.drop(['ClearDeviceInfo'], axis=1, inplace = True)
        submission.drop(['ClearDeviceInfo'], axis=1, inplace = True)
        
    else:
        
        both['DeviceInfo'] = both['DeviceInfo'].astype('str')
        both['ClearDeviceInfo'] = np.vectorize(lambda x : x.lower().split('build')[0].strip())(both['DeviceInfo'])
        both['ClearDeviceInfo'] = both['ClearDeviceInfo'].astype('str')
        both['manufacturer'] = both['ClearDeviceInfo'].apply(lambda model: get_manufacturer(phone_specs_links.get(model, "")))
        both['RAM'] = both['ClearDeviceInfo'].apply(lambda x :\
                 try_get_ram(model_and_specs.get(x, {'memory internal':","}).get('memory internal', ",")))

        both.drop(['ClearDeviceInfo'], axis=1, inplace = True)    
    
    return train, submission, both

def ScreenSize( train, submission, both ):
    
    if( not config.KAGGLE_CHEATS ):
        
        train['ScreenSizeX'] = train['id_33'].str.split('x', expand = True)[0]
        train['ScreenSizeY'] = train['id_33'].str.split('x', expand = True)[1]
        
        submission['ScreenSizeX'] = submission['id_33'].str.split('x', expand = True )[0]
        submission['ScreenSizeY'] = submission['id_33'].str.split('x', expand = True )[1]
    
        
    else:
        
        both['ScreenSizeX'] = both['id_33'].str.split('x', expand = True )[0]
        both['ScreenSizeY'] = both['id_33'].str.split('x', expand = True )[1]
        
    return train, submission, both

def TransactionAmtLog( train, submission, both ):
    
    if( not config.KAGGLE_CHEATS ):
        
        train['TransactionAmt_Log'] = np.log(train['TransactionAmt'])
        submission['TransactionAmt_Log'] = np.log(submission['TransactionAmt'])
        
    else:
    
        both['TransactionAmt_Log'] = np.log1p(both['TransactionAmt'])
        #both['TransactionAmt'] = np.log1p(both['TransactionAmt'])
        
    return train, submission, both

def TransactionAmtDecimal( train, submission, both ):
    
    if( not config.KAGGLE_CHEATS ):
        
        train['TransactionAmtDecimal'] = train['TransactionAmt'] - np.floor(train['TransactionAmt'])
        train['TransactionAmtDecimal'] = np.floor(1000*train['TransactionAmtDecimal'])
        submission['TransactionAmtDecimal'] = submission['TransactionAmt'] - np.floor(submission['TransactionAmt'])
        submission['TransactionAmtDecimal'] = np.floor(1000*submission['TransactionAmtDecimal'])
        
    else:
    
        both['TransactionAmtDecimal'] = both['TransactionAmt'] - np.floor(both['TransactionAmt'])
        both['TransactionAmtDecimal'] = np.floor(1000*both['TransactionAmtDecimal'])
        
    return train, submission, both

def TransactionAmt_Last_Decimal( train, submission, both ):
    
    if( not config.KAGGLE_CHEATS ):
        
        train['TransactionAmtDecimal'] = train['TransactionAmt'] - np.floor(train['TransactionAmt'])
        train['TransactionAmtDecimal'] = np.floor(1000*train['TransactionAmtDecimal'])%10
        submission['TransactionAmtDecimal'] = submission['TransactionAmt'] - np.floor(submission['TransactionAmt'])
        submission['TransactionAmtDecimal'] = np.floor(1000*submission['TransactionAmtDecimal'])%10
        
    else:
    
        both['TransactionAmtDecimal'] = both['TransactionAmt'] - np.floor(both['TransactionAmt'])
        both['TransactionAmtDecimal'] = np.floor(1000*both['TransactionAmtDecimal'])%10
        
    return train, submission, both

def TF_2_binary( value ):
    
    if( value == 'T' ):        
        return 1
    elif( value == 'F' ):
        return 0
    else:        
        return -1

def M(train, submission, both):
    
    if( not config.KAGGLE_CHEATS ):
    
        train['M1'] = train['M1'].apply(TF_2_binary)
        train['M2'] = train['M2'].apply(TF_2_binary)
        train['M3'] = train['M3'].apply(TF_2_binary)
        train['M5'] = train['M5'].apply(TF_2_binary)
        train['M6'] = train['M6'].apply(TF_2_binary)
        train['M7'] = train['M7'].apply(TF_2_binary)
        train['M8'] = train['M8'].apply(TF_2_binary)
        train['M9'] = train['M9'].apply(TF_2_binary)
        train['M10'] = train['M1'] + train['M2'] + train['M3'] + train['M5'] + train['M6'] + train['M7'] + train['M8'] + train['M9']
    
        submission['M1'] = submission['M1'].apply(TF_2_binary)
        submission['M2'] = submission['M2'].apply(TF_2_binary)
        submission['M3'] = submission['M3'].apply(TF_2_binary)
        submission['M5'] = submission['M5'].apply(TF_2_binary)
        submission['M6'] = submission['M6'].apply(TF_2_binary)
        submission['M7'] = submission['M7'].apply(TF_2_binary)
        submission['M8'] = submission['M8'].apply(TF_2_binary)
        submission['M9'] = submission['M9'].apply(TF_2_binary)
        submission['M10'] = submission['M1'] + submission['M2'] + submission['M3'] + submission['M5'] + submission['M6'] + submission['M7'] + submission['M8'] + submission['M9']
        
    else:
        
        both['M1'] = both['M1'].apply(TF_2_binary)
        both['M2'] = both['M2'].apply(TF_2_binary)
        both['M3'] = both['M3'].apply(TF_2_binary)
        both['M5'] = both['M5'].apply(TF_2_binary)
        both['M6'] = both['M6'].apply(TF_2_binary)
        both['M7'] = both['M7'].apply(TF_2_binary)
        both['M8'] = both['M8'].apply(TF_2_binary)
        both['M9'] = both['M9'].apply(TF_2_binary)
        both['M10'] = both['M1'] + both['M2'] + both['M3'] + both['M5'] + both['M6'] + both['M7'] + both['M8'] + both['M9']
    
    return train, submission, both

def interaction(train, submission, both):
    
    if( not config.KAGGLE_CHEATS ):
        
        #train['card1/TransactionAmt'] = train['card1']/np.round(train['TransactionAmt'])
        #train['card2/TransactionAmt'] = train['card2']/np.round(train['TransactionAmt'])
        #train['card1/card2'] = train['card1']/train['card2']
        
        #submission['card1/TransactionAmt'] = submission['card1']/np.round(submission['TransactionAmt'])
        #submission['card2/TransactionAmt'] = submission['card2']/np.round(submission['TransactionAmt'])
        #submission['card1/card2'] = submission['card1']/submission['card2']
        
        i_cols = ['card1',
                  'card2',
                  'card3',
                  'card5',
                  'dist1',
                  'dist2',
                  'userArea',
                  'userID1',
                  'userID2',
                  'userID3',
                  'userID4',
                  #'userID4_ProductCD', 
                  #'userID4_browser',
                  #'userID4_manufacturer', 
                  #'userID4_R_email',
                  #'userID4_e2e', 
                  #'userID4_D9',
                  'daysPassed', 
                  'weeksPassed', 
                  '2weeksPassed', 
                  'monthsPassed',
                  #'userID4_daysPassed', 
                  #'userID4_weeksPassed', 
                  #'userID4_2weeksPassed',
                  #'userID4_monthsPassed',
                  #'userID4_C13',
                  'userID5',
                  'userID5_ProductCD',
                  'userID5_browser',
                  'userID5_manufacturer',
                  'userID5_D9',
                  'userID5_10minPassed',
                  'userID5_20minPassed',
                  'userID5_30minPassed',
                  'userID5_hoursPassed',
                  'userID5_8hoursPassed',
                  'userID5_daysPassed',
                  'userID5_weeksPassed',
                  'userID5_2weeksPassed',
                  'userID5_monthsPassed',
                  'userID5_C13'
                  ]

        for col in tqdm(i_cols):
            for agg_type in ['mean','std']:
                new_col_name = col+'_TransactionAmt_'+agg_type
                temp_df = pd.concat([train[[col, 'TransactionAmt']], submission[[col,'TransactionAmt']]])
                temp_df = temp_df.groupby([col])['TransactionAmt'].agg([agg_type]).reset_index().rename(
                                                    columns={agg_type: new_col_name})
                temp_df.index = list(temp_df[col])
                temp_df = temp_df[new_col_name].to_dict()   
    
                train[new_col_name] = train[col].map(temp_df)
                submission[new_col_name] = submission[col].map(temp_df)
 
        for col in tqdm(i_cols):
            new_col_name = col + '_TransactionAmt_value_counts'
            temp_df = pd.concat([train[[col, 'TransactionAmt']], submission[[col,'TransactionAmt']]])
            temp_dict = dict(temp_df.groupby([col])['TransactionAmt'].value_counts())
            
            train[new_col_name] = train[[col,'TransactionAmt']].apply(tuple,axis=1)
            submission[new_col_name] = submission[[col,'TransactionAmt']].apply(tuple,axis=1)
            
            train[new_col_name] = train[new_col_name].map(temp_dict)
            submission[new_col_name] = submission[new_col_name].map(temp_dict)
     
    else:
        
        i_cols = ['card1','card2','card3','card4','card5','dist1','dist2','userArea','userID1']
        time_cols = ['10minPassed','20minPassed','30minPassed','hoursPassed', '8hoursPassed', 'daysPassed', 'weeksPassed', '2weeksPassed', 'monthsPassed']
        cat_cols = ['R_email', 'ProductCD', 'D9', 'D16', 'browser', 'manufacturer']
        
        if( config.USER_ID_LEVEL == 1):
            for tc in time_cols:
                i_cols.append('userID1_'+tc)
            for cc in cat_cols:
                i_cols.append('userID1_'+cc)
        if( config.USER_ID_LEVEL == 2):
            for tc in time_cols:
                i_cols.append('userID2_'+tc)
            for cc in cat_cols:
                i_cols.append('userID2_'+cc)
        if( config.USER_ID_LEVEL == 3):
            for tc in time_cols:
                i_cols.append('userID3_'+tc)
            for cc in cat_cols:
                i_cols.append('userID3_'+cc)
        if( config.USER_ID_LEVEL == 4):
            for tc in time_cols:
                i_cols.append('userID4_'+tc)
            for cc in cat_cols:
                i_cols.append('userID4_'+cc)
        if( config.USER_ID_LEVEL == 5):
            for tc in time_cols:
                i_cols.append('userID5_'+tc)
            for cc in cat_cols:
                i_cols.append('userID5_'+cc)
        if( config.USER_ID_LEVEL == 6):
            for tc in time_cols:
                i_cols.append('userID6_'+tc)
            for cc in cat_cols:
                i_cols.append('userID6_'+cc)
        if( config.USER_ID_LEVEL == 7):
            for tc in time_cols:
                i_cols.append('userID7_'+tc)
            for cc in cat_cols:
                i_cols.append('userID7_'+cc)
        if( config.USER_ID_LEVEL == 8):
            for tc in time_cols:
                i_cols.append('userID8_'+tc)
            for cc in cat_cols:
                i_cols.append('userID8_'+cc)
        if( config.USER_ID_LEVEL == 9):
            for tc in time_cols:
                i_cols.append('userID9_'+tc)
            for cc in cat_cols:
                i_cols.append('userID9_'+cc)
        if( config.USER_ID_LEVEL == 10):
            for tc in time_cols:
                i_cols.append('userID10_'+tc)
            for cc in cat_cols:
                i_cols.append('userID10_'+cc)

        for col in tqdm(i_cols):
            for agg_type in ['mean','std','median']:
                new_col_name = col+'_TransactionAmt_'+agg_type
                temp_df = both.copy()
                temp_df = temp_df.groupby([col])['TransactionAmt'].agg([agg_type]).reset_index().rename(
                                                    columns={agg_type: new_col_name})
        
                temp_df.index = list(temp_df[col])
                temp_df = temp_df[new_col_name].to_dict()   
    
                both[new_col_name] = both[col].map(temp_df)
                
        for col in tqdm(i_cols):
            new_col_name = col + '_TransactionAmt_value_counts'
            temp_dict = dict(both.groupby([col])['TransactionAmt'].value_counts())
            both[new_col_name] = both[[col,'TransactionAmt']].apply(tuple,axis=1)
            both[new_col_name] = both[new_col_name].map(temp_dict)
        
    return train, submission, both

def UserID( train, submission, both ):
    
    if( not config.KAGGLE_CHEATS ):
        
        train['userArea'] = train['addr1'].astype(str)+'_'+train['addr2'].astype(str)
        train['userID1'] = train['card1'].astype(str)+'_'+train['card2'].astype(str)
        train['userID2'] = train['userID1'].astype(str)+'_'+train['card3'].astype(str)+'_'+train['card5'].astype(str)
        train['userID3'] = train['userID2'].astype(str)+'_'+train['card4'].astype(str)
        train['userID4'] = train['userID3'].astype(str)+'_'+train['addr1'].astype(str)+'_'+train['addr2'].astype(str)
        train['userID5'] = train['userID4'].astype(str)+'_'+train['P_emaildomain_1'].astype(str)
        train['userID4_R_email'] = train['userID4'].astype(str)+'_'+train['R_emaildomain_1'].astype(str)
        train['userID4_ProductCD'] = train['userID4'].astype(str)+'_'+train['ProductCD'].astype(str)
        train['userID4_D9'] = train['userID4'].astype(str)+'_'+train['D9'].astype(str)
        #train['userID4_D16'] = train['userID4'].astype(str)+'_'+train['D16'].astype(str)
        train['userID4_browser'] = train['userID4'].astype(str)+'_'+train['browser'].astype(str)
        train['userID4_manufacturer'] = train['userID4'].astype(str)+'_'+train['manufacturer'].astype(str)
        train['userID4_e2e'] = train['userID4'].astype(str)+'_'+train['e2e_1'].astype(str)
        train['userID4_daysPassed'] = train['userID4'].astype(str)+'_'+train['daysPassed'].astype(str)
        train['userID4_weeksPassed'] = train['userID4'].astype(str)+'_'+train['weeksPassed'].astype(str)
        train['userID4_2weeksPassed'] = train['userID4'].astype(str)+'_'+train['2weeksPassed'].astype(str)
        train['userID4_monthsPassed'] = train['userID4'].astype(str)+'_'+train['monthsPassed'].astype(str)
        train['userID4_C13'] = train['userID4'].astype(str)+'_'+train['C13'].astype(str)
        

        train['userID5_ProductCD'] = train['userID5'].astype(str)+'_'+train['ProductCD'].astype(str)
        train['userID5_D9'] = train['userID5'].astype(str)+'_'+train['D9'].astype(str)
        train['userID5_browser'] = train['userID5'].astype(str)+'_'+train['browser'].astype(str)
        train['userID5_manufacturer'] = train['userID5'].astype(str)+'_'+train['manufacturer'].astype(str)
        train['userID5_daysPassed'] = train['userID5'].astype(str)+'_'+train['daysPassed'].astype(str)
        train['userID5_weeksPassed'] = train['userID5'].astype(str)+'_'+train['weeksPassed'].astype(str)
        train['userID5_2weeksPassed'] = train['userID5'].astype(str)+'_'+train['2weeksPassed'].astype(str)
        train['userID5_monthsPassed'] = train['userID5'].astype(str)+'_'+train['monthsPassed'].astype(str)
        train['userID5_C13'] = train['userID5'].astype(str)+'_'+train['C13'].astype(str)

        
        submission['userArea'] = submission['addr1'].astype(str)+'_'+submission['addr2'].astype(str)
        submission['userID1'] = submission['card1'].astype(str)+'_'+submission['card2'].astype(str)
        submission['userID2'] = submission['userID1'].astype(str)+'_'+submission['card3'].astype(str)+'_'+submission['card5'].astype(str)
        submission['userID3'] = submission['userID2'].astype(str)+'_'+submission['card4'].astype(str)
        submission['userID4'] = submission['userID3'].astype(str)+'_'+submission['addr1'].astype(str)+'_'+submission['addr2'].astype(str)
        submission['userID5'] = submission['userID4'].astype(str)+'_'+submission['P_emaildomain_1'].astype(str)
        submission['userID4_R_email'] = submission['userID4'].astype(str)+'_'+submission['R_emaildomain_1'].astype(str)
        submission['userID4_ProductCD'] = submission['userID4'].astype(str)+'_'+submission['ProductCD'].astype(str)
        submission['userID4_D9'] = submission['userID4'].astype(str)+'_'+submission['D9'].astype(str)
        #submission['userID4_D16'] = submission['userID4'].astype(str)+'_'+submission['D16'].astype(str)
        submission['userID4_browser'] = submission['userID4'].astype(str)+'_'+submission['browser'].astype(str)
        submission['userID4_manufacturer'] = submission['userID4'].astype(str)+'_'+submission['manufacturer'].astype(str)
        submission['userID4_e2e'] = submission['userID4'].astype(str)+'_'+submission['e2e_1'].astype(str)
        submission['userID4_daysPassed'] = submission['userID4'].astype(str)+'_'+submission['daysPassed'].astype(str)
        submission['userID4_weeksPassed'] = submission['userID4'].astype(str)+'_'+submission['weeksPassed'].astype(str)
        submission['userID4_2weeksPassed'] = submission['userID4'].astype(str)+'_'+submission['2weeksPassed'].astype(str)
        submission['userID4_monthsPassed'] = submission['userID4'].astype(str)+'_'+submission['monthsPassed'].astype(str)
        submission['userID4_C13'] = submission['userID4'].astype(str)+'_'+submission['C13'].astype(str)

        submission['userID5_ProductCD'] = submission['userID5'].astype(str)+'_'+submission['ProductCD'].astype(str)
        submission['userID5_D9'] = submission['userID5'].astype(str)+'_'+submission['D9'].astype(str)
        submission['userID5_browser'] = submission['userID5'].astype(str)+'_'+submission['browser'].astype(str)
        submission['userID5_manufacturer'] = submission['userID5'].astype(str)+'_'+submission['manufacturer'].astype(str)
        submission['userID5_daysPassed'] = submission['userID5'].astype(str)+'_'+submission['daysPassed'].astype(str)
        submission['userID5_weeksPassed'] = submission['userID5'].astype(str)+'_'+submission['weeksPassed'].astype(str)
        submission['userID5_2weeksPassed'] = submission['userID5'].astype(str)+'_'+submission['2weeksPassed'].astype(str)
        submission['userID5_monthsPassed'] = submission['userID5'].astype(str)+'_'+submission['monthsPassed'].astype(str)
        submission['userID5_C13'] = submission['userID5'].astype(str)+'_'+submission['C13'].astype(str)

        
    else:
        
        if( config.USER_ID_LEVEL > 0):
            both['userArea'] = both['addr1'].astype(str)+'_'+both['addr2'].astype(str)
            both['userID1'] = both['card1'].astype(str)+'_'+both['card2'].astype(str)
        if( config.USER_ID_LEVEL > 1):
            both['userID2'] = both['userID1'].astype(str)+'_'+both['card3'].astype(str)+'_'+both['card5'].astype(str)
        if( config.USER_ID_LEVEL > 2):
            both['userID3'] = both['userID2'].astype(str)+'_'+both['card4'].astype(str)
        if( config.USER_ID_LEVEL > 3):
            both['userID4'] = both['userID3'].astype(str)+'_'+both['addr1'].astype(str)+'_'+both['addr2'].astype(str)
        if( config.USER_ID_LEVEL > 4):
            both['userID5'] = both['userID4'].astype(str)+'_'+both['P_emaildomain_1'].astype(str)
        if( config.USER_ID_LEVEL > 5):
            both['userID6'] = both['userID5'].astype(str)+'_'+both['C13'].astype(str)
        if( config.USER_ID_LEVEL > 6):
            both['userID7'] = both['userID5'].astype(str)+'_'+both['C1+C2+C6+C11'].astype(str)
        if( config.USER_ID_LEVEL > 7):
            both['userID8'] = both['userID5'].astype(str)+'_'+both['C2'].astype(str)
        if( config.USER_ID_LEVEL > 8):
            both['userID9'] = both['userID5'].astype(str)+'_'+both['C14'].astype(str)
        if( config.USER_ID_LEVEL > 9):
            both['userID10'] = both['userID5'].astype(str)+'_'+both['C11'].astype(str)
        
        if( config.USER_ID_LEVEL == 1 ):
            both['userID1_R_email'] = both['userID1'].astype(str)+'_'+both['R_emaildomain_1'].astype(str)
            both['userID1_ProductCD'] = both['userID1'].astype(str)+'_'+both['ProductCD'].astype(str)
            both['userID1_D9'] = both['userID1'].astype(str)+'_'+both['D9'].astype(str)
            both['userID1_D16'] = both['userID1'].astype(str)+'_'+both['D16'].astype(str)
            both['userID1_browser'] = both['userID1'].astype(str)+'_'+both['browser'].astype(str)
            both['userID1_manufacturer'] = both['userID1'].astype(str)+'_'+both['manufacturer'].astype(str)
            
            both['userID1_10minPassed'] = both['userID1'].astype(str)+'_'+both['10minPassed'].astype(str)
            both['userID1_20minPassed'] = both['userID1'].astype(str)+'_'+both['20minPassed'].astype(str)
            both['userID1_30minPassed'] = both['userID1'].astype(str)+'_'+both['30minPassed'].astype(str)
            both['userID1_hoursPassed'] = both['userID1'].astype(str)+'_'+both['hoursPassed'].astype(str)
            both['userID1_8hoursPassed'] = both['userID1'].astype(str)+'_'+both['8hoursPassed'].astype(str)
            both['userID1_daysPassed'] = both['userID1'].astype(str)+'_'+both['daysPassed'].astype(str)
            both['userID1_weeksPassed'] = both['userID1'].astype(str)+'_'+both['weeksPassed'].astype(str)
            both['userID1_2weeksPassed'] = both['userID1'].astype(str)+'_'+both['2weeksPassed'].astype(str)
            both['userID1_monthsPassed'] = both['userID1'].astype(str)+'_'+both['monthsPassed'].astype(str)
        
        if( config.USER_ID_LEVEL == 2 ):
            both['userID2_R_email'] = both['userID2'].astype(str)+'_'+both['R_emaildomain_1'].astype(str)
            both['userID2_ProductCD'] = both['userID2'].astype(str)+'_'+both['ProductCD'].astype(str)
            both['userID2_D9'] = both['userID2'].astype(str)+'_'+both['D9'].astype(str)
            both['userID2_D16'] = both['userID2'].astype(str)+'_'+both['D16'].astype(str)
            both['userID2_browser'] = both['userID2'].astype(str)+'_'+both['browser'].astype(str)
            both['userID2_manufacturer'] = both['userID2'].astype(str)+'_'+both['manufacturer'].astype(str)
            
            both['userID2_10minPassed'] = both['userID2'].astype(str)+'_'+both['10minPassed'].astype(str)
            both['userID2_20minPassed'] = both['userID2'].astype(str)+'_'+both['20minPassed'].astype(str)
            both['userID2_30minPassed'] = both['userID2'].astype(str)+'_'+both['30minPassed'].astype(str)
            both['userID2_hoursPassed'] = both['userID2'].astype(str)+'_'+both['hoursPassed'].astype(str)
            both['userID2_8hoursPassed'] = both['userID2'].astype(str)+'_'+both['8hoursPassed'].astype(str)
            both['userID2_daysPassed'] = both['userID2'].astype(str)+'_'+both['daysPassed'].astype(str)
            both['userID2_weeksPassed'] = both['userID2'].astype(str)+'_'+both['weeksPassed'].astype(str)
            both['userID2_2weeksPassed'] = both['userID2'].astype(str)+'_'+both['2weeksPassed'].astype(str)
            both['userID2_monthsPassed'] = both['userID2'].astype(str)+'_'+both['monthsPassed'].astype(str)
        
        if( config.USER_ID_LEVEL == 3 ):
            both['userID3_R_email'] = both['userID3'].astype(str)+'_'+both['R_emaildomain_1'].astype(str)
            both['userID3_ProductCD'] = both['userID3'].astype(str)+'_'+both['ProductCD'].astype(str)
            both['userID3_D9'] = both['userID3'].astype(str)+'_'+both['D9'].astype(str)
            both['userID3_D16'] = both['userID3'].astype(str)+'_'+both['D16'].astype(str)
            both['userID3_browser'] = both['userID3'].astype(str)+'_'+both['browser'].astype(str)
            both['userID3_manufacturer'] = both['userID3'].astype(str)+'_'+both['manufacturer'].astype(str)
            
            both['userID3_10minPassed'] = both['userID3'].astype(str)+'_'+both['10minPassed'].astype(str)
            both['userID3_20minPassed'] = both['userID3'].astype(str)+'_'+both['20minPassed'].astype(str)
            both['userID3_30minPassed'] = both['userID3'].astype(str)+'_'+both['30minPassed'].astype(str)
            both['userID3_hoursPassed'] = both['userID3'].astype(str)+'_'+both['hoursPassed'].astype(str)
            both['userID3_8hoursPassed'] = both['userID3'].astype(str)+'_'+both['8hoursPassed'].astype(str)
            both['userID3_daysPassed'] = both['userID3'].astype(str)+'_'+both['daysPassed'].astype(str)
            both['userID3_weeksPassed'] = both['userID3'].astype(str)+'_'+both['weeksPassed'].astype(str)
            both['userID3_2weeksPassed'] = both['userID3'].astype(str)+'_'+both['2weeksPassed'].astype(str)
            both['userID3_monthsPassed'] = both['userID3'].astype(str)+'_'+both['monthsPassed'].astype(str)
        
        if( config.USER_ID_LEVEL == 4 ):
            both['userID4_R_email'] = both['userID4'].astype(str)+'_'+both['R_emaildomain_1'].astype(str)
            both['userID4_ProductCD'] = both['userID4'].astype(str)+'_'+both['ProductCD'].astype(str)
            both['userID4_D9'] = both['userID4'].astype(str)+'_'+both['D9'].astype(str)
            both['userID4_D16'] = both['userID4'].astype(str)+'_'+both['D16'].astype(str)
            both['userID4_browser'] = both['userID4'].astype(str)+'_'+both['browser'].astype(str)
            both['userID4_manufacturer'] = both['userID4'].astype(str)+'_'+both['manufacturer'].astype(str)
            
            both['userID4_10minPassed'] = both['userID4'].astype(str)+'_'+both['10minPassed'].astype(str)
            both['userID4_20minPassed'] = both['userID4'].astype(str)+'_'+both['20minPassed'].astype(str)
            both['userID4_30minPassed'] = both['userID4'].astype(str)+'_'+both['30minPassed'].astype(str)
            both['userID4_hoursPassed'] = both['userID4'].astype(str)+'_'+both['hoursPassed'].astype(str)
            both['userID4_8hoursPassed'] = both['userID4'].astype(str)+'_'+both['8hoursPassed'].astype(str)
            both['userID4_daysPassed'] = both['userID4'].astype(str)+'_'+both['daysPassed'].astype(str)
            both['userID4_weeksPassed'] = both['userID4'].astype(str)+'_'+both['weeksPassed'].astype(str)
            both['userID4_2weeksPassed'] = both['userID4'].astype(str)+'_'+both['2weeksPassed'].astype(str)
            both['userID4_monthsPassed'] = both['userID4'].astype(str)+'_'+both['monthsPassed'].astype(str)
           
        
        if( config.USER_ID_LEVEL == 5 ):
            both['userID5_R_email'] = both['userID4'].astype(str)+'_'+both['R_emaildomain_1'].astype(str)
            both['userID5_ProductCD'] = both['userID5'].astype(str)+'_'+both['ProductCD'].astype(str)
            both['userID5_D9'] = both['userID5'].astype(str)+'_'+both['D9'].astype(str)
            both['userID5_D16'] = both['userID5'].astype(str)+'_'+both['D16'].astype(str)
            both['userID5_browser'] = both['userID5'].astype(str)+'_'+both['browser'].astype(str)
            both['userID5_manufacturer'] = both['userID5'].astype(str)+'_'+both['manufacturer'].astype(str)
            
            both['userID5_10minPassed'] = both['userID5'].astype(str)+'_'+both['10minPassed'].astype(str)
            both['userID5_20minPassed'] = both['userID5'].astype(str)+'_'+both['20minPassed'].astype(str)
            both['userID5_30minPassed'] = both['userID5'].astype(str)+'_'+both['30minPassed'].astype(str)
            both['userID5_hoursPassed'] = both['userID5'].astype(str)+'_'+both['hoursPassed'].astype(str)
            both['userID5_8hoursPassed'] = both['userID5'].astype(str)+'_'+both['8hoursPassed'].astype(str)
            both['userID5_daysPassed'] = both['userID5'].astype(str)+'_'+both['daysPassed'].astype(str)
            both['userID5_weeksPassed'] = both['userID5'].astype(str)+'_'+both['weeksPassed'].astype(str)
            both['userID5_2weeksPassed'] = both['userID5'].astype(str)+'_'+both['2weeksPassed'].astype(str)
            both['userID5_monthsPassed'] = both['userID5'].astype(str)+'_'+both['monthsPassed'].astype(str)
            
        if( config.USER_ID_LEVEL == 6):
        
            both['userID6_R_email'] = both['userID4'].astype(str)+'_'+both['R_emaildomain_1'].astype(str)
            both['userID6_ProductCD'] = both['userID6'].astype(str)+'_'+both['ProductCD'].astype(str)
            both['userID6_D9'] = both['userID6'].astype(str)+'_'+both['D9'].astype(str)
            both['userID6_D16'] = both['userID6'].astype(str)+'_'+both['D16'].astype(str)
            both['userID6_browser'] = both['userID6'].astype(str)+'_'+both['browser'].astype(str)
            both['userID6_manufacturer'] = both['userID6'].astype(str)+'_'+both['manufacturer'].astype(str)
        
            both['userID6_10minPassed'] = both['userID6'].astype(str)+'_'+both['10minPassed'].astype(str)
            both['userID6_20minPassed'] = both['userID6'].astype(str)+'_'+both['20minPassed'].astype(str)
            both['userID6_30minPassed'] = both['userID6'].astype(str)+'_'+both['30minPassed'].astype(str)
            both['userID6_hoursPassed'] = both['userID6'].astype(str)+'_'+both['hoursPassed'].astype(str)
            both['userID6_8hoursPassed'] = both['userID6'].astype(str)+'_'+both['8hoursPassed'].astype(str)
            both['userID6_daysPassed'] = both['userID6'].astype(str)+'_'+both['daysPassed'].astype(str)
            both['userID6_weeksPassed'] = both['userID6'].astype(str)+'_'+both['weeksPassed'].astype(str)
            both['userID6_2weeksPassed'] = both['userID6'].astype(str)+'_'+both['2weeksPassed'].astype(str)
            both['userID6_monthsPassed'] = both['userID6'].astype(str)+'_'+both['monthsPassed'].astype(str)
        
        if( config.USER_ID_LEVEL == 7):
        
            both['userID7_R_email'] = both['userID4'].astype(str)+'_'+both['R_emaildomain_1'].astype(str)
            both['userID7_ProductCD'] = both['userID7'].astype(str)+'_'+both['ProductCD'].astype(str)
            both['userID7_D9'] = both['userID7'].astype(str)+'_'+both['D9'].astype(str)
            both['userID7_D16'] = both['userID7'].astype(str)+'_'+both['D16'].astype(str)
            both['userID7_browser'] = both['userID7'].astype(str)+'_'+both['browser'].astype(str)
            both['userID7_manufacturer'] = both['userID7'].astype(str)+'_'+both['manufacturer'].astype(str)
        
            both['userID7_10minPassed'] = both['userID7'].astype(str)+'_'+both['10minPassed'].astype(str)
            both['userID7_20minPassed'] = both['userID7'].astype(str)+'_'+both['20minPassed'].astype(str)
            both['userID7_30minPassed'] = both['userID7'].astype(str)+'_'+both['30minPassed'].astype(str)
            both['userID7_hoursPassed'] = both['userID7'].astype(str)+'_'+both['hoursPassed'].astype(str)
            both['userID7_8hoursPassed'] = both['userID7'].astype(str)+'_'+both['8hoursPassed'].astype(str)
            both['userID7_daysPassed'] = both['userID7'].astype(str)+'_'+both['daysPassed'].astype(str)
            both['userID7_weeksPassed'] = both['userID7'].astype(str)+'_'+both['weeksPassed'].astype(str)
            both['userID7_2weeksPassed'] = both['userID7'].astype(str)+'_'+both['2weeksPassed'].astype(str)
            both['userID7_monthsPassed'] = both['userID7'].astype(str)+'_'+both['monthsPassed'].astype(str)
            
        if( config.USER_ID_LEVEL == 8):
        
            both['userID8_R_email'] = both['userID4'].astype(str)+'_'+both['R_emaildomain_1'].astype(str)
            both['userID8_ProductCD'] = both['userID8'].astype(str)+'_'+both['ProductCD'].astype(str)
            both['userID8_D9'] = both['userID8'].astype(str)+'_'+both['D9'].astype(str)
            both['userID8_D16'] = both['userID8'].astype(str)+'_'+both['D16'].astype(str)
            both['userID8_browser'] = both['userID8'].astype(str)+'_'+both['browser'].astype(str)
            both['userID8_manufacturer'] = both['userID8'].astype(str)+'_'+both['manufacturer'].astype(str)
        
            both['userID8_10minPassed'] = both['userID8'].astype(str)+'_'+both['10minPassed'].astype(str)
            both['userID8_20minPassed'] = both['userID8'].astype(str)+'_'+both['20minPassed'].astype(str)
            both['userID8_30minPassed'] = both['userID8'].astype(str)+'_'+both['30minPassed'].astype(str)
            both['userID8_hoursPassed'] = both['userID8'].astype(str)+'_'+both['hoursPassed'].astype(str)
            both['userID8_8hoursPassed'] = both['userID8'].astype(str)+'_'+both['8hoursPassed'].astype(str)
            both['userID8_daysPassed'] = both['userID8'].astype(str)+'_'+both['daysPassed'].astype(str)
            both['userID8_weeksPassed'] = both['userID8'].astype(str)+'_'+both['weeksPassed'].astype(str)
            both['userID8_2weeksPassed'] = both['userID8'].astype(str)+'_'+both['2weeksPassed'].astype(str)
            both['userID8_monthsPassed'] = both['userID8'].astype(str)+'_'+both['monthsPassed'].astype(str)
            
        if( config.USER_ID_LEVEL == 9):
        
            both['userID9_R_email'] = both['userID4'].astype(str)+'_'+both['R_emaildomain_1'].astype(str)
            both['userID9_ProductCD'] = both['userID9'].astype(str)+'_'+both['ProductCD'].astype(str)
            both['userID9_D9'] = both['userID9'].astype(str)+'_'+both['D9'].astype(str)
            both['userID9_D16'] = both['userID9'].astype(str)+'_'+both['D16'].astype(str)
            both['userID9_browser'] = both['userID9'].astype(str)+'_'+both['browser'].astype(str)
            both['userID9_manufacturer'] = both['userID9'].astype(str)+'_'+both['manufacturer'].astype(str)
        
            both['userID9_10minPassed'] = both['userID9'].astype(str)+'_'+both['10minPassed'].astype(str)
            both['userID9_20minPassed'] = both['userID9'].astype(str)+'_'+both['20minPassed'].astype(str)
            both['userID9_30minPassed'] = both['userID9'].astype(str)+'_'+both['30minPassed'].astype(str)
            both['userID9_hoursPassed'] = both['userID9'].astype(str)+'_'+both['hoursPassed'].astype(str)
            both['userID9_8hoursPassed'] = both['userID9'].astype(str)+'_'+both['8hoursPassed'].astype(str)
            both['userID9_daysPassed'] = both['userID9'].astype(str)+'_'+both['daysPassed'].astype(str)
            both['userID9_weeksPassed'] = both['userID9'].astype(str)+'_'+both['weeksPassed'].astype(str)
            both['userID9_2weeksPassed'] = both['userID9'].astype(str)+'_'+both['2weeksPassed'].astype(str)
            both['userID9_monthsPassed'] = both['userID9'].astype(str)+'_'+both['monthsPassed'].astype(str)
            
        if( config.USER_ID_LEVEL == 10):
        
            both['userID10_R_email'] = both['userID4'].astype(str)+'_'+both['R_emaildomain_1'].astype(str)
            both['userID10_ProductCD'] = both['userID10'].astype(str)+'_'+both['ProductCD'].astype(str)
            both['userID10_D9'] = both['userID10'].astype(str)+'_'+both['D9'].astype(str)
            both['userID10_D16'] = both['userID10'].astype(str)+'_'+both['D16'].astype(str)
            both['userID10_browser'] = both['userID10'].astype(str)+'_'+both['browser'].astype(str)
            both['userID10_manufacturer'] = both['userID10'].astype(str)+'_'+both['manufacturer'].astype(str)
        
            both['userID10_10minPassed'] = both['userID10'].astype(str)+'_'+both['10minPassed'].astype(str)
            both['userID10_20minPassed'] = both['userID10'].astype(str)+'_'+both['20minPassed'].astype(str)
            both['userID10_30minPassed'] = both['userID10'].astype(str)+'_'+both['30minPassed'].astype(str)
            both['userID10_hoursPassed'] = both['userID10'].astype(str)+'_'+both['hoursPassed'].astype(str)
            both['userID10_8hoursPassed'] = both['userID10'].astype(str)+'_'+both['8hoursPassed'].astype(str)
            both['userID10_daysPassed'] = both['userID10'].astype(str)+'_'+both['daysPassed'].astype(str)
            both['userID10_weeksPassed'] = both['userID10'].astype(str)+'_'+both['weeksPassed'].astype(str)
            both['userID10_2weeksPassed'] = both['userID10'].astype(str)+'_'+both['2weeksPassed'].astype(str)
            both['userID10_monthsPassed'] = both['userID10'].astype(str)+'_'+both['monthsPassed'].astype(str)
        
    return train, submission, both

def time( train, submission, both ):
    
    if ( not config.KAGGLE_CHEATS ):
        
        train['minutesPassed'] = train['TransactionDT'] //(60)
        train['10minPassed'] = train['TransactionDT'] //(60*10)
        train['20minPassed'] = train['TransactionDT'] //(60*20)
        train['30minPassed'] = train['TransactionDT'] //(60*30)
        train['hoursPassed'] = train['TransactionDT'] //(3600)
        train['8hoursPassed'] = train['TransactionDT'] //(8*3600)
        train['daysPassed'] = train['TransactionDT'] //(3600*24)
        train['weeksPassed'] = train['TransactionDT'] // (3600*24) // 7
        train['2weeksPassed'] = train['TransactionDT'] // (3600*24) // 14
        train['monthsPassed'] = train['TransactionDT'] // (3600*24) // 30
    
        submission['minutesPassed'] = submission['TransactionDT'] //(60)
        submission['10minPassed'] = submission['TransactionDT'] //(60*10)
        submission['20minPassed'] = submission['TransactionDT'] //(60*20)
        submission['30minPassed'] = submission['TransactionDT'] //(60*30)
        submission['hoursPassed'] = submission['TransactionDT'] //(3600)
        submission['8hoursPassed'] = submission['TransactionDT'] //(8*3600)
        submission['daysPassed'] = submission['TransactionDT'] //(3600*24)
        submission['weeksPassed'] = submission['TransactionDT'] // (3600*24) // 7
        submission['2weeksPassed'] = submission['TransactionDT'] // (3600*24) // 14
        submission['monthsPassed'] = submission['TransactionDT'] // (3600*24) // 30
    
    else:

        both['minutesPassed'] = both['TransactionDT'] //(60)
        both['10minPassed'] = both['TransactionDT'] //(60*10)
        both['20minPassed'] = both['TransactionDT'] //(60*20)
        both['30minPassed'] = both['TransactionDT'] //(60*30)
        both['hoursPassed'] = both['TransactionDT'] //(3600)
        both['8hoursPassed'] = both['TransactionDT'] //(8*3600)
        both['daysPassed'] = both['TransactionDT'] //(3600*24)
        both['weeksPassed'] = both['TransactionDT'] // (3600*24) // 7
        both['2weeksPassed'] = both['TransactionDT'] // (3600*24) // 14
        both['monthsPassed'] = both['TransactionDT'] // (3600*24) // 30
    
    return train, submission, both

def TransactionAmt_time( train, submission, both ):
    
    if ( not config.KAGGLE_CHEATS ):
        
        train['TransactionAmt_minus_daysAvg'] = train['TransactionAmt'] - train['daysPassed_TransactionAmt_mean']
        train['TransactionAmt_minus_weeksAvg'] = train['TransactionAmt'] - train['weeksPassed_TransactionAmt_mean']
        train['TransactionAmt_minus_2weeksAvg'] = train['TransactionAmt'] - train['2weeksPassed_TransactionAmt_mean']
        train['TransactionAmt_minus_MonthsAvg'] = train['TransactionAmt'] - train['monthsPassed_TransactionAmt_mean']
    
        submission['TransactionAmt_minus_daysAvg'] = submission['TransactionAmt'] - submission['daysPassed_TransactionAmt_mean']
        submission['TransactionAmt_minus_weeksAvg'] = submission['TransactionAmt'] - submission['weeksPassed_TransactionAmt_mean']
        submission['TransactionAmt_minus_2weeksAvg'] = submission['TransactionAmt'] - submission['2weeksPassed_TransactionAmt_mean']
        submission['TransactionAmt_minus_MonthsAvg'] = submission['TransactionAmt'] - submission['monthsPassed_TransactionAmt_mean']    
    else:
    
        both['TransactionAmt_minus_8hoursAvg'] = both['TransactionAmt'] - both['8hoursPassed_TransactionAmt_mean']
        both['TransactionAmt_minus_daysAvg'] = both['TransactionAmt'] - both['daysPassed_TransactionAmt_mean']
        both['TransactionAmt_minus_weeksAvg'] = both['TransactionAmt'] - both['weeksPassed_TransactionAmt_mean']
        both['TransactionAmt_minus_2weeksAvg'] = both['TransactionAmt'] - both['2weeksPassed_TransactionAmt_mean']
        both['TransactionAmt_minus_MonthsAvg'] = both['TransactionAmt'] - both['monthsPassed_TransactionAmt_mean']  
        
    return train, submission, both


def TransactionAmt_UserID_Avg( train, submission, both ):
    
    if ( not config.KAGGLE_CHEATS ):
        
        train['TransactionAmt_minus_userID1Avg'] = train['TransactionAmt'] - train['userID1_TransactionAmt_mean']
        train['TransactionAmt_minus_userID2Avg'] = train['TransactionAmt'] - train['userID2_TransactionAmt_mean']
        train['TransactionAmt_minus_userID3Avg'] = train['TransactionAmt'] - train['userID3_TransactionAmt_mean']
        train['TransactionAmt_minus_userID4Avg'] = train['TransactionAmt'] - train['userID4_TransactionAmt_mean']
        train['TransactionAmt_minus_userID5Avg'] = train['TransactionAmt'] - train['userID5_TransactionAmt_mean']
        train['TransactionAmt_minus_userID5_ProductCD'] = train['TransactionAmt'] - train['userID5_ProductCD_TransactionAmt_mean']
        train['TransactionAmt_minus_userID5_browser'] = train['TransactionAmt'] - train['userID5_browser_TransactionAmt_mean']
        train['TransactionAmt_minus_userID5_D9'] = train['TransactionAmt'] - train['userID5_D9_TransactionAmt_mean']
        train['TransactionAmt_minus_userID5_daysPassed'] = train['TransactionAmt'] - train['userID5_daysPassed_TransactionAmt_mean']
        train['TransactionAmt_minus_userID5_weeksPassed'] = train['TransactionAmt'] - train['userID5_weeksPassed_TransactionAmt_mean']
        train['TransactionAmt_minus_userID5_2weeksPassed'] = train['TransactionAmt'] - train['userID5_2weeksPassed_TransactionAmt_mean']
        train['TransactionAmt_minus_userID5_monthsPassed'] = train['TransactionAmt'] - train['userID5_monthsPassed_TransactionAmt_mean']
    
        submission['TransactionAmt_minus_userID1Avg'] = submission['TransactionAmt'] - submission['userID1_TransactionAmt_mean']
        submission['TransactionAmt_minus_userID2Avg'] = submission['TransactionAmt'] - submission['userID2_TransactionAmt_mean']
        submission['TransactionAmt_minus_userID3Avg'] = submission['TransactionAmt'] - submission['userID3_TransactionAmt_mean']
        submission['TransactionAmt_minus_userID4Avg'] = submission['TransactionAmt'] - submission['userID4_TransactionAmt_mean']
        submission['TransactionAmt_minus_userID5Avg'] = submission['TransactionAmt'] - submission['userID5_TransactionAmt_mean']
        submission['TransactionAmt_minus_userID5_ProductCD'] = submission['TransactionAmt'] - submission['userID5_ProductCD_TransactionAmt_mean']
        submission['TransactionAmt_minus_userID5_browser'] = submission['TransactionAmt'] - submission['userID5_browser_TransactionAmt_mean']
        submission['TransactionAmt_minus_userID5_D9'] = submission['TransactionAmt'] - submission['userID5_D9_TransactionAmt_mean']
        submission['TransactionAmt_minus_userID5_daysPassed'] = submission['TransactionAmt'] - submission['userID5_daysPassed_TransactionAmt_mean']
        submission['TransactionAmt_minus_userID5_weeksPassed'] = submission['TransactionAmt'] - submission['userID5_weeksPassed_TransactionAmt_mean']
        submission['TransactionAmt_minus_userID5_2weeksPassed'] = submission['TransactionAmt'] - submission['userID5_2weeksPassed_TransactionAmt_mean']
        submission['TransactionAmt_minus_userID5_monthsPassed'] = submission['TransactionAmt'] - submission['userID5_monthsPassed_TransactionAmt_mean']

    else:
    
        both['TransactionAmt_minus_userArea'] = both['TransactionAmt'] - both['userArea_TransactionAmt_mean']
        
        if( config.USER_ID_LEVEL == 1 ):
            both['TransactionAmt_minus_userID1_R_emailAvg'] = both['TransactionAmt'] - both['userID1_R_email_TransactionAmt_mean']
            both['TransactionAmt_minus_userID1_ProductCDAvg'] = both['TransactionAmt'] - both['userID1_ProductCD_TransactionAmt_mean']
            both['TransactionAmt_minus_userID1_browserAvg'] = both['TransactionAmt'] - both['userID1_browser_TransactionAmt_mean']
            both['TransactionAmt_minus_userID1_D9Avg'] = both['TransactionAmt'] - both['userID1_D9_TransactionAmt_mean']
            both['TransactionAmt_minus_userID1_D16Avg'] = both['TransactionAmt'] - both['userID1_D16_TransactionAmt_mean']
            both['TransactionAmt_minus_userID1_daysPassedAvg'] = both['TransactionAmt'] - both['userID1_daysPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID1_10minPassedAvg'] = both['TransactionAmt'] - both['userID1_10minPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID1_20minPassedAvg'] = both['TransactionAmt'] - both['userID1_20minPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID1_30minPassedAvg'] = both['TransactionAmt'] - both['userID1_30minPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID1_hoursPassedAvg'] = both['TransactionAmt'] - both['userID1_hoursPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID1_8hoursPassedAvg'] = both['TransactionAmt'] - both['userID1_8hoursPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID1_daysPassedAvg'] = both['TransactionAmt'] - both['userID1_daysPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID1_weeksPassedAvg'] = both['TransactionAmt'] - both['userID1_weeksPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID1_2weeksPassedAvg'] = both['TransactionAmt'] - both['userID1_2weeksPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID1_monthsPassedAvg'] = both['TransactionAmt'] - both['userID1_monthsPassed_TransactionAmt_mean']
            
        if( config.USER_ID_LEVEL == 2 ):
            both['TransactionAmt_minus_userID2_R_emailAvg'] = both['TransactionAmt'] - both['userID2_R_email_TransactionAmt_mean']
            both['TransactionAmt_minus_userID2_ProductCDAvg'] = both['TransactionAmt'] - both['userID2_ProductCD_TransactionAmt_mean']
            both['TransactionAmt_minus_userID2_browserAvg'] = both['TransactionAmt'] - both['userID2_browser_TransactionAmt_mean']
            both['TransactionAmt_minus_userID2_D9Avg'] = both['TransactionAmt'] - both['userID2_D9_TransactionAmt_mean']
            both['TransactionAmt_minus_userID2_D16Avg'] = both['TransactionAmt'] - both['userID2_D16_TransactionAmt_mean']
            both['TransactionAmt_minus_userID2_daysPassedAvg'] = both['TransactionAmt'] - both['userID2_daysPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID2_10minPassedAvg'] = both['TransactionAmt'] - both['userID2_10minPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID2_20minPassedAvg'] = both['TransactionAmt'] - both['userID2_20minPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID2_30minPassedAvg'] = both['TransactionAmt'] - both['userID2_30minPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID2_hoursPassedAvg'] = both['TransactionAmt'] - both['userID2_hoursPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID2_8hoursPassedAvg'] = both['TransactionAmt'] - both['userID2_8hoursPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID2_daysPassedAvg'] = both['TransactionAmt'] - both['userID2_daysPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID2_weeksPassedAvg'] = both['TransactionAmt'] - both['userID2_weeksPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID2_2weeksPassedAvg'] = both['TransactionAmt'] - both['userID2_2weeksPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID2_monthsPassedAvg'] = both['TransactionAmt'] - both['userID2_monthsPassed_TransactionAmt_mean']
        
        if( config.USER_ID_LEVEL == 3 ):
            both['TransactionAmt_minus_userID3_R_emailAvg'] = both['TransactionAmt'] - both['userID3_R_email_TransactionAmt_mean']
            both['TransactionAmt_minus_userID3_ProductCDAvg'] = both['TransactionAmt'] - both['userID3_ProductCD_TransactionAmt_mean']
            both['TransactionAmt_minus_userID3_browserAvg'] = both['TransactionAmt'] - both['userID3_browser_TransactionAmt_mean']
            both['TransactionAmt_minus_userID3_D9Avg'] = both['TransactionAmt'] - both['userID3_D9_TransactionAmt_mean']
            both['TransactionAmt_minus_userID3_D16Avg'] = both['TransactionAmt'] - both['userID3_D16_TransactionAmt_mean']
            both['TransactionAmt_minus_userID3_daysPassedAvg'] = both['TransactionAmt'] - both['userID3_daysPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID3_10minPassedAvg'] = both['TransactionAmt'] - both['userID3_10minPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID3_20minPassedAvg'] = both['TransactionAmt'] - both['userID3_20minPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID3_30minPassedAvg'] = both['TransactionAmt'] - both['userID3_30minPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID3_hoursPassedAvg'] = both['TransactionAmt'] - both['userID3_hoursPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID3_8hoursPassedAvg'] = both['TransactionAmt'] - both['userID3_8hoursPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID3_daysPassedAvg'] = both['TransactionAmt'] - both['userID3_daysPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID3_weeksPassedAvg'] = both['TransactionAmt'] - both['userID3_weeksPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID3_2weeksPassedAvg'] = both['TransactionAmt'] - both['userID3_2weeksPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID3_monthsPassedAvg'] = both['TransactionAmt'] - both['userID3_monthsPassed_TransactionAmt_mean']
        
        if( config.USER_ID_LEVEL == 3 ):
            both['TransactionAmt_minus_userID4_R_emailAvg'] = both['TransactionAmt'] - both['userID4_R_email_TransactionAmt_mean']
            both['TransactionAmt_minus_userID4_ProductCDAvg'] = both['TransactionAmt'] - both['userID4_ProductCD_TransactionAmt_mean']
            both['TransactionAmt_minus_userID4_browserAvg'] = both['TransactionAmt'] - both['userID4_browser_TransactionAmt_mean']
            both['TransactionAmt_minus_userID4_D9Avg'] = both['TransactionAmt'] - both['userID4_D9_TransactionAmt_mean']
            both['TransactionAmt_minus_userID4_D16Avg'] = both['TransactionAmt'] - both['userID4_D16_TransactionAmt_mean']
            both['TransactionAmt_minus_userID4_daysPassedAvg'] = both['TransactionAmt'] - both['userID4_daysPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID4_10minPassedAvg'] = both['TransactionAmt'] - both['userID4_10minPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID4_20minPassedAvg'] = both['TransactionAmt'] - both['userID4_20minPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID4_30minPassedAvg'] = both['TransactionAmt'] - both['userID4_30minPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID4_hoursPassedAvg'] = both['TransactionAmt'] - both['userID4_hoursPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID4_8hoursPassedAvg'] = both['TransactionAmt'] - both['userID4_8hoursPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID4_daysPassedAvg'] = both['TransactionAmt'] - both['userID4_daysPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID4_weeksPassedAvg'] = both['TransactionAmt'] - both['userID4_weeksPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID4_2weeksPassedAvg'] = both['TransactionAmt'] - both['userID4_2weeksPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID4_monthsPassedAvg'] = both['TransactionAmt'] - both['userID4_monthsPassed_TransactionAmt_mean']
            
        if( config.USER_ID_LEVEL == 4 ):
            both['TransactionAmt_minus_userID4_R_emailAvg'] = both['TransactionAmt'] - both['userID4_R_email_TransactionAmt_mean']
            both['TransactionAmt_minus_userID4_ProductCDAvg'] = both['TransactionAmt'] - both['userID4_ProductCD_TransactionAmt_mean']
            both['TransactionAmt_minus_userID4_browserAvg'] = both['TransactionAmt'] - both['userID4_browser_TransactionAmt_mean']
            both['TransactionAmt_minus_userID4_D9Avg'] = both['TransactionAmt'] - both['userID4_D9_TransactionAmt_mean']
            both['TransactionAmt_minus_userID4_D16Avg'] = both['TransactionAmt'] - both['userID4_D16_TransactionAmt_mean']
            both['TransactionAmt_minus_userID4_daysPassedAvg'] = both['TransactionAmt'] - both['userID4_daysPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID4_10minPassedAvg'] = both['TransactionAmt'] - both['userID4_10minPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID4_20minPassedAvg'] = both['TransactionAmt'] - both['userID4_20minPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID4_30minPassedAvg'] = both['TransactionAmt'] - both['userID4_30minPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID4_hoursPassedAvg'] = both['TransactionAmt'] - both['userID4_hoursPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID4_8hoursPassedAvg'] = both['TransactionAmt'] - both['userID4_8hoursPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID4_daysPassedAvg'] = both['TransactionAmt'] - both['userID4_daysPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID4_weeksPassedAvg'] = both['TransactionAmt'] - both['userID4_weeksPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID4_2weeksPassedAvg'] = both['TransactionAmt'] - both['userID4_2weeksPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID4_monthsPassedAvg'] = both['TransactionAmt'] - both['userID4_monthsPassed_TransactionAmt_mean']
        
        if( config.USER_ID_LEVEL == 5 ):
            both['TransactionAmt_minus_userID5_R_emailAvg'] = both['TransactionAmt'] - both['userID5_R_email_TransactionAmt_mean']
            both['TransactionAmt_minus_userID5_ProductCDAvg'] = both['TransactionAmt'] - both['userID5_ProductCD_TransactionAmt_mean']
            both['TransactionAmt_minus_userID5_browserAvg'] = both['TransactionAmt'] - both['userID5_browser_TransactionAmt_mean']
            both['TransactionAmt_minus_userID5_D9Avg'] = both['TransactionAmt'] - both['userID5_D9_TransactionAmt_mean']
            both['TransactionAmt_minus_userID5_D16Avg'] = both['TransactionAmt'] - both['userID5_D16_TransactionAmt_mean']
            both['TransactionAmt_minus_userID5_daysPassedAvg'] = both['TransactionAmt'] - both['userID5_daysPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID5_10minPassedAvg'] = both['TransactionAmt'] - both['userID5_10minPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID5_20minPassedAvg'] = both['TransactionAmt'] - both['userID5_20minPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID5_30minPassedAvg'] = both['TransactionAmt'] - both['userID5_30minPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID5_hoursPassedAvg'] = both['TransactionAmt'] - both['userID5_hoursPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID5_8hoursPassedAvg'] = both['TransactionAmt'] - both['userID5_8hoursPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID5_daysPassedAvg'] = both['TransactionAmt'] - both['userID5_daysPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID5_weeksPassedAvg'] = both['TransactionAmt'] - both['userID5_weeksPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID5_2weeksPassedAvg'] = both['TransactionAmt'] - both['userID5_2weeksPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID5_monthsPassedAvg'] = both['TransactionAmt'] - both['userID5_monthsPassed_TransactionAmt_mean']
            
        if( config.USER_ID_LEVEL == 6 ):
            both['TransactionAmt_minus_userID6_R_emailAvg'] = both['TransactionAmt'] - both['userID6_R_email_TransactionAmt_mean']
            both['TransactionAmt_minus_userID6_ProductCDAvg'] = both['TransactionAmt'] - both['userID6_ProductCD_TransactionAmt_mean']
            both['TransactionAmt_minus_userID6_browserAvg'] = both['TransactionAmt'] - both['userID6_browser_TransactionAmt_mean']
            both['TransactionAmt_minus_userID6_D9Avg'] = both['TransactionAmt'] - both['userID6_D9_TransactionAmt_mean']
            both['TransactionAmt_minus_userID6_D16Avg'] = both['TransactionAmt'] - both['userID6_D16_TransactionAmt_mean']
            both['TransactionAmt_minus_userID6_daysPassedAvg'] = both['TransactionAmt'] - both['userID6_daysPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID6_10minPassedAvg'] = both['TransactionAmt'] - both['userID6_10minPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID6_20minPassedAvg'] = both['TransactionAmt'] - both['userID6_20minPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID6_30minPassedAvg'] = both['TransactionAmt'] - both['userID6_30minPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID6_hoursPassedAvg'] = both['TransactionAmt'] - both['userID6_hoursPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID6_8hoursPassedAvg'] = both['TransactionAmt'] - both['userID6_8hoursPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID6_daysPassedAvg'] = both['TransactionAmt'] - both['userID6_daysPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID6_weeksPassedAvg'] = both['TransactionAmt'] - both['userID6_weeksPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID6_2weeksPassedAvg'] = both['TransactionAmt'] - both['userID6_2weeksPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID6_monthsPassedAvg'] = both['TransactionAmt'] - both['userID6_monthsPassed_TransactionAmt_mean']
            
        if( config.USER_ID_LEVEL == 7 ):
            both['TransactionAmt_minus_userID7_R_emailAvg'] = both['TransactionAmt'] - both['userID7_R_email_TransactionAmt_mean']
            both['TransactionAmt_minus_userID7_ProductCDAvg'] = both['TransactionAmt'] - both['userID7_ProductCD_TransactionAmt_mean']
            both['TransactionAmt_minus_userID7_browserAvg'] = both['TransactionAmt'] - both['userID7_browser_TransactionAmt_mean']
            both['TransactionAmt_minus_userID7_D9Avg'] = both['TransactionAmt'] - both['userID7_D9_TransactionAmt_mean']
            both['TransactionAmt_minus_userID7_D16Avg'] = both['TransactionAmt'] - both['userID7_D16_TransactionAmt_mean']
            both['TransactionAmt_minus_userID7_daysPassedAvg'] = both['TransactionAmt'] - both['userID7_daysPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID7_10minPassedAvg'] = both['TransactionAmt'] - both['userID7_10minPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID7_20minPassedAvg'] = both['TransactionAmt'] - both['userID7_20minPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID7_30minPassedAvg'] = both['TransactionAmt'] - both['userID7_30minPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID7_hoursPassedAvg'] = both['TransactionAmt'] - both['userID7_hoursPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID7_8hoursPassedAvg'] = both['TransactionAmt'] - both['userID7_8hoursPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID7_daysPassedAvg'] = both['TransactionAmt'] - both['userID7_daysPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID7_weeksPassedAvg'] = both['TransactionAmt'] - both['userID7_weeksPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID7_2weeksPassedAvg'] = both['TransactionAmt'] - both['userID7_2weeksPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID7_monthsPassedAvg'] = both['TransactionAmt'] - both['userID7_monthsPassed_TransactionAmt_mean']
            
        if( config.USER_ID_LEVEL == 8 ):
            both['TransactionAmt_minus_userID8_R_emailAvg'] = both['TransactionAmt'] - both['userID8_R_email_TransactionAmt_mean']
            both['TransactionAmt_minus_userID8_ProductCDAvg'] = both['TransactionAmt'] - both['userID8_ProductCD_TransactionAmt_mean']
            both['TransactionAmt_minus_userID8_browserAvg'] = both['TransactionAmt'] - both['userID8_browser_TransactionAmt_mean']
            both['TransactionAmt_minus_userID8_D9Avg'] = both['TransactionAmt'] - both['userID8_D9_TransactionAmt_mean']
            both['TransactionAmt_minus_userID8_D16Avg'] = both['TransactionAmt'] - both['userID8_D16_TransactionAmt_mean']
            both['TransactionAmt_minus_userID8_daysPassedAvg'] = both['TransactionAmt'] - both['userID8_daysPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID8_10minPassedAvg'] = both['TransactionAmt'] - both['userID8_10minPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID8_20minPassedAvg'] = both['TransactionAmt'] - both['userID8_20minPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID8_30minPassedAvg'] = both['TransactionAmt'] - both['userID8_30minPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID8_hoursPassedAvg'] = both['TransactionAmt'] - both['userID8_hoursPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID8_8hoursPassedAvg'] = both['TransactionAmt'] - both['userID8_8hoursPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID8_daysPassedAvg'] = both['TransactionAmt'] - both['userID8_daysPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID8_weeksPassedAvg'] = both['TransactionAmt'] - both['userID8_weeksPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID8_2weeksPassedAvg'] = both['TransactionAmt'] - both['userID8_2weeksPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID8_monthsPassedAvg'] = both['TransactionAmt'] - both['userID8_monthsPassed_TransactionAmt_mean']
            
        if( config.USER_ID_LEVEL == 9 ):
            both['TransactionAmt_minus_userID9_R_emailAvg'] = both['TransactionAmt'] - both['userID9_R_email_TransactionAmt_mean']
            both['TransactionAmt_minus_userID9_ProductCDAvg'] = both['TransactionAmt'] - both['userID9_ProductCD_TransactionAmt_mean']
            both['TransactionAmt_minus_userID9_browserAvg'] = both['TransactionAmt'] - both['userID9_browser_TransactionAmt_mean']
            both['TransactionAmt_minus_userID9_D9Avg'] = both['TransactionAmt'] - both['userID9_D9_TransactionAmt_mean']
            both['TransactionAmt_minus_userID9_D16Avg'] = both['TransactionAmt'] - both['userID9_D16_TransactionAmt_mean']
            both['TransactionAmt_minus_userID9_daysPassedAvg'] = both['TransactionAmt'] - both['userID9_daysPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID9_10minPassedAvg'] = both['TransactionAmt'] - both['userID9_10minPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID9_20minPassedAvg'] = both['TransactionAmt'] - both['userID9_20minPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID9_30minPassedAvg'] = both['TransactionAmt'] - both['userID9_30minPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID9_hoursPassedAvg'] = both['TransactionAmt'] - both['userID9_hoursPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID9_8hoursPassedAvg'] = both['TransactionAmt'] - both['userID9_8hoursPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID9_daysPassedAvg'] = both['TransactionAmt'] - both['userID9_daysPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID9_weeksPassedAvg'] = both['TransactionAmt'] - both['userID9_weeksPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID9_2weeksPassedAvg'] = both['TransactionAmt'] - both['userID9_2weeksPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID9_monthsPassedAvg'] = both['TransactionAmt'] - both['userID9_monthsPassed_TransactionAmt_mean']
            
        if( config.USER_ID_LEVEL == 10 ):
            both['TransactionAmt_minus_userID10_R_emailAvg'] = both['TransactionAmt'] - both['userID10_R_email_TransactionAmt_mean']
            both['TransactionAmt_minus_userID10_ProductCDAvg'] = both['TransactionAmt'] - both['userID10_ProductCD_TransactionAmt_mean']
            both['TransactionAmt_minus_userID10_browserAvg'] = both['TransactionAmt'] - both['userID10_browser_TransactionAmt_mean']
            both['TransactionAmt_minus_userID10_D9Avg'] = both['TransactionAmt'] - both['userID10_D9_TransactionAmt_mean']
            both['TransactionAmt_minus_userID10_D16Avg'] = both['TransactionAmt'] - both['userID10_D16_TransactionAmt_mean']
            both['TransactionAmt_minus_userID10_daysPassedAvg'] = both['TransactionAmt'] - both['userID10_daysPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID10_10minPassedAvg'] = both['TransactionAmt'] - both['userID10_10minPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID10_20minPassedAvg'] = both['TransactionAmt'] - both['userID10_20minPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID10_30minPassedAvg'] = both['TransactionAmt'] - both['userID10_30minPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID10_hoursPassedAvg'] = both['TransactionAmt'] - both['userID10_hoursPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID10_8hoursPassedAvg'] = both['TransactionAmt'] - both['userID10_8hoursPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID10_daysPassedAvg'] = both['TransactionAmt'] - both['userID10_daysPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID10_weeksPassedAvg'] = both['TransactionAmt'] - both['userID10_weeksPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID10_2weeksPassedAvg'] = both['TransactionAmt'] - both['userID10_2weeksPassed_TransactionAmt_mean']
            both['TransactionAmt_minus_userID10_monthsPassedAvg'] = both['TransactionAmt'] - both['userID10_monthsPassed_TransactionAmt_mean']
            
        
        
    return train, submission, both


def TransactionDT_UserID_timediff(train, submission, both):
    
    user_id_names = []
    for i in range(config.USER_ID_LEVEL):
        user_id_names.append('userID'+str(i+1))
    
    for uin in user_id_names:
             new_col_name = uin+'_'+'TransactionSecTimeDiff'
             both[new_col_name] = both.groupby(uin)['TransactionDT'].transform(pd.Series.diff)
             both[new_col_name].fillna(-999,inplace=True)
    
    return train, submission, both

    
def Frequent_Categorical_Encoder( train, submission, both, CATEG ):
    
    for feat in CATEG:
        
        temp_dict = both[feat].value_counts().to_dict()
        both[feat] = both[feat].map(temp_dict)
    
    return train, submission, both

def D_normalized(train, submission, both):
    
    D_cols = [ 'D15','D14','D13','D12','D11','D10','D8','D6','D4','D3','D2','D1']
    
    not_needed_features = []
    for feat in tqdm(D_cols):
        for agg_type in ['mean','std']:
                new_col_name = feat+'_'+agg_type
                temp_dict = both.groupby('10minPassed')[feat].agg(agg_type).to_dict()
                both[new_col_name] = both['10minPassed'].map(temp_dict)
                not_needed_features.append(new_col_name)
                
        both[feat] = (both[feat] - both[feat+'_mean'])/both[feat+'_std']
        
    both.drop(not_needed_features,axis=1,inplace=True)
        
    return train, submission, both

def fill_pairs(train, submission, both, pairs):
    
    if( not config.KAGGLE_CHEATS):
        
        for pair in pairs:
            
            unique_train = []
            unique_submission = []
            
            for value in train[pair[0]].unique():
                unique_train.append(train[pair[1]][train[pair[0]] == value].value_counts().shape[0])
                
            for value in submission[pair[0]].unique():
                unique_submission.append(submission[pair[1]][submission[pair[0]] == value].value_counts().shape[0])
                
            pair_values_train = pd.Series(data=unique_train, index = train[pair[0]].unique())
            pair_values_submission = pd.Series(data=unique_submission, index=submission[pair[0]].unique())
            
            for value in pair_values_train[pair_values_train == 1].index:
                train.loc[train[pair[0]] == value, pair[1]] == train.loc[train[pair[0]] == value, pair[1]].value_counts().index[0]
                
            for value in pair_values_submission[pair_values_submission == 1].index:
                submission.loc[submission[pair[0]] == value, pair[1]] == submission.loc[submission[pair[0]] == value, pair[1]].value_counts().index[0]
                
    else:
    
        for pair in tqdm(pairs):
            
            unique_both = []
            
            for value in both[pair[0]].unique():
                unique_both.append(both[pair[1]][both[pair[0]] == value].value_counts().shape[0])
                           
            pair_values_both = pd.Series(data=unique_both, index = both[pair[0]].unique())
            
            for value in pair_values_both[pair_values_both == 1].index:
                both.loc[both[pair[0]] == value, pair[1]] == both.loc[both[pair[0]] == value, pair[1]].value_counts().index[0]
                    
    return train, submission, both

def fill_unique_value_pairs(train, submission, both):
    
    pairs = [('card1','card2'), ('card1','card3'), ('card1','card4'), ('card1','card5'), ('card1','card6'),
            ('card1','addr2'), ('card1','addr1')
            ]
        
    train, submission, both = fill_pairs(train, submission, both, pairs)
        
    return train, submission, both

def correlated_features( train, submission, both):
    
   both['C10+C8'] = both['C10'].fillna(-1) + both['C8'].fillna(-1)
   both.drop(['C10','C8'], axis=1, inplace = True )
    
   both['C1+C2+C6+C11'] = both['C1'].fillna(-1) + both['C2'].fillna(-1) + both['C6'].fillna(-1) + both['C11'].fillna(-1)
   both.drop(['C1','C2','C6','C11'], axis=1, inplace = True )
    
   both['C7+C12'] = both['C7'].fillna(-1) + both['C12'].fillna(-1)
   both.drop(['C7','C12'], axis=1, inplace = True )
      
   both.drop(['D5','D7'], axis=1, inplace=True)
   
   both['V15+V16'] = both['V15'].fillna(-1) + both['V16'].fillna(-1)
   both.drop(['V15','V16'], axis=1, inplace = True)
   
   both['V17+V18'] = both['V17'].fillna(-1) + both['V18'].fillna(-1)
   both.drop(['V17','V18'], axis=1, inplace = True)
   
   both['V27+V28+V89'] = both['V27'].fillna(-1) + both['V28'].fillna(-1) + both['V89'].fillna(-1)
   both.drop(['V27','V28','V89'], axis=1, inplace = True)
   
   both['V31+V32'] = both['V31'].fillna(-1) + both['V32'].fillna(-1)
   both.drop(['V31','V32'], axis=1, inplace = True)
   
   both['V95+V101+V143+V167+V177+V279+V293+V322'] = both['V95'].fillna(-1) + both['V101'].fillna(-1) + both['V143'].fillna(-1) + both['V167'].fillna(-1) +\
   both['V177'].fillna(-1) + both['V279'].fillna(-1) + both['V293'].fillna(-1) + both['V322'].fillna(-1)
   both.drop(['V95','V101','V143','V167','V177','V279','V293','V322'], axis=1, inplace = True )
    
   both['V96+V97+V102+V103+V179+V280+V295+V323+V324'] = both['V96'].fillna(-1) + both['V97'].fillna(-1) + both['V102'].fillna(-1) + both['V103'].fillna(-1) +\
   both['V179'].fillna(-1) + both['V280'].fillna(-1) + both['V295'].fillna(-1) + both['V323'].fillna(-1) + both['V324'].fillna(-1)
   both.drop(['V96','V97','V102','V103','V179','V280','V295','V323','V324'], axis=1, inplace = True )
   
   both['V99+V326'] = both['V99'].fillna(-1) + both['V326'].fillna(-1)
   both.drop(['V99','V326'], axis=1, inplace = True)
   
   both['V100+V327'] = both['V100'].fillna(-1) + both['V327'].fillna(-1)
   both.drop(['V100','V327'], axis=1, inplace = True)
   
   both['V129+V266+V269+V309+V334'] = both['V129'].fillna(-1) + both['V266'].fillna(-1) + both['V269'].fillna(-1) + both['V309'].fillna(-1) +\
   both['V334'].fillna(-1) 
   both.drop(['V129','V266','V269','V309','V334'], axis=1, inplace = True )
   
   both['V153+V154'] = both['V153'].fillna(-1) + both['V154'].fillna(-1)
   both.drop(['V153','V154'], axis=1, inplace = True)
   
   both['V180+V182'] = both['V180'].fillna(-1) + both['V182'].fillna(-1)
   both.drop(['V180','V182'], axis=1, inplace = True)
   
   both['V190+V199+V246+V257'] = both['V190'].fillna(-1) + both['V199'].fillna(-1) + both['V246'].fillna(-1) + both['V257'].fillna(-1)
   both.drop(['V190','V199','V246','V257'], axis=1, inplace = True)
   
   both['V215+V216+V277+V278'] = both['V215'].fillna(-1) + both['V216'].fillna(-1) + both['V277'].fillna(-1) + both['V278'].fillna(-1)
   both.drop(['V215','V216','V277','V278'], axis=1, inplace = True)
   
   both['V217+V231'] = both['V217'].fillna(-1) + both['V231'].fillna(-1)
   both.drop(['V217','V231'], axis=1, inplace = True)
   
   both['V219+V232+V233'] = both['V219'].fillna(-1) + both['V232'].fillna(-1) + both['V233'].fillna(-1)
   both.drop(['V219','V232','V233'], axis=1, inplace = True)
   
   both['V234+V236'] = both['V234'].fillna(-1) + both['V236'].fillna(-1)
   both.drop(['V234','V236'], axis=1, inplace = True)
   
   both['V296+V298+V329'] = both['V105'].fillna(-1) + both['V296'].fillna(-1) + both['V298'].fillna(-1) + both['V329'].fillna(-1)
   both.drop(['V105','V296','V298','V329'], axis=1, inplace = True)
   
   both['V106+V299+V330'] = both['V106'].fillna(-1) + both['V299'].fillna(-1) + both['V330'].fillna(-1)
   both.drop(['V106','V299','V330'], axis=1, inplace = True)
      
   both['V273+V275'] = both['V273'].fillna(-1) + both['V275'].fillna(-1)
   both.drop(['V273','V275'], axis=1, inplace = True)
   
   both['V320+V321'] = both['V320'].fillna(-1) + both['V321'].fillna(-1)
   both.drop(['V320','V321'], axis=1, inplace = True)
   
   both['V338+V339'] = both['V338'].fillna(-1) + both['V339'].fillna(-1)
   both.drop(['V338','V339'], axis=1, inplace = True)
   
   return train, submission, both
   
   
   
   
   