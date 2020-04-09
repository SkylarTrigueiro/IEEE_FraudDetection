# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 18:59:47 2019

@author: striguei
"""
import numpy as np
import pandas as pd
import config
from feature_engine import missing_data_imputers as mdi
from feature_engine import categorical_encoders as ce
from feature_engine import discretisers as dt
import features as ft

def cat_encode( train, submission, both, y_train, CATEG, DISCRETE ):
    
    
    
    if( config.CATEGORICAL_ENCODE_METHOD == 'frequency' ):
        
        train, submission, both = freq_enc( train, submission, both, CATEG )
        
    if( config.CATEGORICAL_ENCODE_METHOD == 'mean' ):
        
        train, submission, both = rare_enc( train, submission, both, CATEG + DISCRETE )
        train, submission, both = mean_enc( train, submission, both, CATEG, y_train )
        
    if( config.CATEGORICAL_ENCODE_METHOD == 'ohe' ):
        
        train, submission, both = rare_enc( train, submission, both, CATEG + DISCRETE )
        lotso_labels = [ var for var in CATEG if train[var].nunique() > config.MAX_CAT_CARDINALITY ]
        train, submission, both = mean_enc( train, submission, both, lotso_labels, y_train )
        train, submission, both = tree_disc( train, submission, both, y_train, lotso_labels )
        train[CATEG + DISCRETE] = train[CATEG + DISCRETE].astype('O')
        submission[CATEG + DISCRETE] = submission[CATEG + DISCRETE].astype('O')
        train, submission, both = ft.top_x( train, submission, both, CATEG + DISCRETE)
    
    return train, submission, both

def tree_disc( train, submission, both, y_train, VAR):
    
    disc = dt.DecisionTreeDiscretiser(cv = 5, variables = VAR )
    disc.fit(train, y_train)
    train = disc.transform(train)
    submission = disc.transform(submission)
    
    return train, submission, both

def rare_enc( train, submission, both, CATEG ):
    
    if( not config.KAGGLE_CHEATS ):
        encoder = ce.RareLabelCategoricalEncoder( tol = config.RARE_LABEL_TOL, n_categories=1, variables = CATEG )
        encoder.fit(train)
        train = encoder.transform(train)
        submission = encoder.transform(submission)
    else:
        encoder = ce.RareLabelCategoricalEncoder( tol = config.RARE_LABEL_TOL, n_categories=1, variables = CATEG )
        encoder.fit(both)
        both = encoder.transform(both)

    
    return train, submission, both


def mean_enc( train, submission, both, CATEG, y_train ):
    
    encoder = ce.MeanCategoricalEncoder( variables=CATEG )
    encoder.fit(train, y_train)
    train = encoder.transform(train)
    submission = encoder.transform(submission)
    
    return train, submission, both

def freq_enc( train, submission, both, CATEG ):
    
    if( not config.KAGGLE_CHEATS ):
        encoder = ce.CountFrequencyCategoricalEncoder(variables = CATEG)
        encoder.fit(train)
        train = encoder.transform(train)
        submission = encoder.transform(submission)
    else:
        encoder = ce.CountFrequencyCategoricalEncoder(variables = CATEG)
        encoder.fit(both)
        both = encoder.transform(both)
        
    return train, submission, both

def NaN_impute( train, submission, both, NUMERIC_NA, CATEG_NA, DISCRETE_NA ):
    
    if( not config.DISC_IS_CATEG ):
        NUMERIC_NA += DISCRETE_NA
    
    if( config.NUMERICAL_IMPUTE_METHOD in ['mean', 'median']):
        
        train, submission, both = mmimputer(train, submission, both, NUMERIC_NA )
            
    if( config.NUMERICAL_IMPUTE_METHOD == 'arbitrary' ):
        
        train, submission, both = arbimputer( train, submission, both, NUMERIC_NA  )
        
    if( config.NUMERICAL_IMPUTE_METHOD == 'random sample'):
        
        train, submission, both = ranimputer( train, submission, both, NUMERIC_NA )
        
    if( config.CATEGORICAL_IMPUTE_METHOD == 'missing' ):
        
        if( config.DISC_IS_CATEG ):
            train, submission, both = catimputer( train, submission, both, CATEG_NA + DISCRETE_NA )
        else:
            train, submission, both = catimputer( train, submission, both, CATEG_NA )
        
    if( config.CATEGORICAL_IMPUTE_METHOD == 'frequent' ):
        
        if( config.DISC_IS_CATEG ):
            train, submission, both = freqimputer( train, submission, both, CATEG_NA + DISCRETE_NA )
        else:
            train, submission, both = freqimputer( train, submission, both, CATEG_NA )
    
    return train, submission, both

def freqimputer( train, submission, both, CATEG_NA):
    
    if( not config.KAGGLE_CHEATS ):
        imputer = mdi.FrequentCategoryImputer(variables = CATEG_NA)
        imputer.fit(train)
        train = imputer.transform(train)
        submission = imputer.transform(submission)
    else:
        imputer = mdi.FrequentCategoryImputer(variables = CATEG_NA)
        imputer.fit(both)
        both = imputer.transform(both)       
    
    return train, submission, both

def catimputer( train, submission, both, CATEG_NA):
   
    if( not config.KAGGLE_CHEATS ):
        imputer = mdi.CategoricalVariableImputer(variables = CATEG_NA)
        imputer.fit(train)
        train = imputer.transform(train)
        submission = imputer.transform(submission)
    else:
        imputer = mdi.CategoricalVariableImputer(variables = CATEG_NA)
        imputer.fit(both)
        both = imputer.transform(both)
    
    return train, submission, both

def ranimputer( train, submission, both, NUMERIC_NA ):
    
    if( not config.KAGGLE_CHEATS ):
        imputer = mdi.RandomSampleImputer(variables = NUMERIC_NA, random_state = config.RANDOM_STATE)
        imputer.fit(train)
        train = imputer.transform(train)
        submission = imputer.transform(submission)
    else:
        imputer = mdi.RandomSampleImputer(variables = NUMERIC_NA, random_state = config.RANDOM_STATE)
        imputer.fit(both)
        both = imputer.transform(both)
        
    
    return train, submission, both

def arbimputer( train, submission, both, NUMERIC_NA ):
    
    if( not config.KAGGLE_CHEATS ):
        imputer = mdi.ArbitraryNumberImputer(arbitrary_number=config.NUMERICAL_IMPUTE_ARB_VAL, variables=NUMERIC_NA)
        imputer.fit(train)
        train = imputer.transform(train)
        submission = imputer.transform(submission)
    else:
        imputer = mdi.ArbitraryNumberImputer(arbitrary_number=config.NUMERICAL_IMPUTE_ARB_VAL, variables=NUMERIC_NA)
        imputer.fit(both)
        both = imputer.transform(both)
        
    
    return train, submission, both

def mmimputer(train, submission, both, NUMERIC_NA):
    
    if( not config.KAGGLE_CHEATS ):
        imputer = mdi.MeanMedianImputer(imputation_method= config.NUMERICAL_IMPUTE_METHOD, variables = NUMERIC_NA )
        imputer.fit(train)
        train = imputer.transform(train)
        submission = imputer.transform(submission)
    else:
        imputer = mdi.MeanMedianImputer(imputation_method= config.NUMERICAL_IMPUTE_METHOD, variables = NUMERIC_NA )
        imputer.fit(both)
        both = imputer.transform(both)        

    return train, submission, both

def NaN_features( train, submission, both, var_to_impute):
    
    if( not config.KAGGLE_CHEATS ):
        train = add_NaN_feat(train, var_to_impute )
        submission = add_NaN_feat(submission, var_to_impute )
    else:
        both = add_NaN_feat(both, var_to_impute )
    
    return train, submission, both

def add_NaN_feat(df, var_to_impute ):
    
    imputer = mdi.AddNaNBinaryImputer( variables = var_to_impute )
    imputer.fit(df)
    df = imputer.transform(df)
    
    return df