# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 19:30:52 2019

@author: striguei
"""

import config
import pandas as pd
import features as ft
import feat_selection as fs
from data_management import load_data
import preprocessor as pp
import models as md
from time import time
import datetime


if __name__ == '__main__':
    
    # Set seed in driver
    program_start_time = time()
    config.seed_everything(config.RANDOM_STATE)
    
    # load the data from the different files and store them in
    # pandas dataframes
    train, submission = load_data()
    print('data has been loaded.')
   
    train, y = ft.get_features(train, submission )
    print("train and target have been partitioned.")
    
    both = pd.concat([train, submission])
    
    train, submission, both = ft.feature_generator( train, submission, both )
    print('New features have been added.')
    
    NUMERIC, NUMERIC_NA, CATEG, CATEG_NA, DISCRETE, DISCRETE_NA = ft.partition_features( train, submission, both )
    print('features have been partitioned.')
    
    if( not config.KAGGLE_CHEATS ):
        
        if( config.DISC_IS_CATEG ):
            train[CATEG + DISCRETE] = train[CATEG + DISCRETE].astype('O')
            submission[CATEG + DISCRETE ] = submission[CATEG + DISCRETE ].astype('O')
        else:
            train[CATEG] = train[CATEG].astype('O')
            submission[CATEG ] = submission[CATEG].astype('O')
    
    else:
        
        if( config.DISC_IS_CATEG ):
            both[CATEG+DISCRETE] = both[CATEG+DISCRETE].astype('O')
        else:
            both[CATEG] = both[CATEG].astype('O')
    
    train, submission, both = pp.NaN_features(train, submission, both, NUMERIC_NA)
    print('missing indicator features have been added.')
   
    train, submission, both = pp.NaN_impute(train, submission, both, NUMERIC_NA, CATEG_NA, DISCRETE_NA)
    print('missing values have been imputed.')
 
    train, submission, both = pp.cat_encode(train, submission, both, y, CATEG, DISCRETE )
    print('categorical variables have been encoded.')
    
    train, submission, both = ft.feature_generator2( train, submission, both, CATEG)
     
    
    if( not config.KAGGLE_CHEATS ):
        
        train, features_to_keep = fs.feature_selection(train, submission, y)
        
    else:
        
        features_to_remove = []
        #both, constant_features_to_remove = fs.remove_constant(both)
        #features_to_remove += constant_features_to_remove
    
        #both, quasi_constant_features_to_remove = fs.remove_quasi_constant(both)
        #features_to_remove += quasi_constant_features_to_remove
    
        both, duplicate_features_to_remove = fs.remove_duplicates(both)
        features_to_remove += duplicate_features_to_remove

        #train.drop(features_to_remove, axis=1, inplace = True )
        #submission.drop(features_to_remove, axis=1, inplace = True )
        
        #train, rfe_features_to_remove = fs.rfe_lgb(train,y)
        #features_to_remove += rfe_features_to_remove
        #train.drop(rfe_features_to_remove, axis=1, inplace = True )
        #submission.drop(rfe_features_to_remove, axis=1, inplace = True )
        
        #features_to_remove = fs.p_value_elimination( train, submission)
    
    #both[CATEG].astype(str)
    #bothh = both.copy()
    #both = bothh.copy()
    #both.drop(features_to_remove, axis=1, inplace = True)
    
    if(  not config.KAGGLE_CHEATS):
        train.drop(duplicate_features_to_remove + config.RFE_FEATURES_TO_REMOVE + config.NEED_TO_REMOVE, axis=1, inplace = True)
        submission.drop(duplicate_features_to_remove + config.RFE_FEATURES_TO_REMOVE + config.NEED_TO_REMOVE, axis=1, inplace = True)
    else:
        both.drop(duplicate_features_to_remove +config.RFE_FEATURES_TO_REMOVE + config.NEED_TO_REMOVE, axis=1, inplace = True)  
        train = both.iloc[:len(train), :]
        submission = both.iloc[len(train):, :]
    
    y_pred = md.run_lgb( train, submission, y)
    #y_pred = md.run_xgb( train, submission, y)
    
    sample_submission = pd.read_csv('data/sample_submission.csv', index_col='TransactionID')
    sample_submission['isFraud'] = y_pred
    sample_submission.to_csv('lgb_time_series_cv.csv')
    print('finished')
    print('Total run time is {}'.format(str(datetime.timedelta(seconds=time() - program_start_time))))