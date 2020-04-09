# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 08:15:46 2019

@author: skyst
"""
import config
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.metrics import roc_auc_score
from time import time
import datetime
import xgboost as xgb
import lightgbm as lgb
from tqdm import tqdm


def feature_selection(df, y):
    
    features_to_remove = list()
    df, new_features_to_remove = remove_constant(df)
    features_to_remove.append(new_features_to_remove)
    df, new_features_to_remove = remove_quasi_constant(df)
    features_to_remove.append(new_features_to_remove)
    df, new_features_to_remove = remove_duplicates(df)
    features_to_remove.append(new_features_to_remove)
    df, new_features_to_remove = rfe_lgb(df)
    
    
    return df, features_to_remove

def remove_constant( df ):
    
    # remove constant features
    constant_features = [
    feat for feat in df.columns if df[feat].std() == 0
    ]
    
    return df, constant_features

def remove_quasi_constant( df):
    
    sel = VarianceThreshold(
    threshold=0.01)  # 0.1 indicates 99% of observations approximately
    sel.fit(df)
        
    features = list(df.columns)
    features_to_keep = df.columns[sel.get_support()]
    features_to_drop = [feat for feat in features if( (feat in features) and (feat not in features_to_keep))]
    
    return df, features_to_drop

def remove_duplicates( df ):
    
    duplicated_list = []
    for col in df.columns:
        if( '_na' in col):
            duplicated_list.append(col)
    
    duplicated_feat = []
    for i in tqdm(range(0, len(df.columns))):

        col_1 = df.columns[i]
        if( col_1 in duplicated_list):
 
            for col_2 in df.columns[i + 1:]:
                if( col_2 in duplicated_list):
                    if df[col_1].equals(df[col_2]):
                        duplicated_feat.append(col_2)
                    
    duplicated_feat = list(set(duplicated_feat))

    
    duplicated_feat = list(set(duplicated_feat))
    
    return df, duplicated_feat
    
def rfe_lgb( df, y ):
    
    feature_importance = pd.read_csv('feature_importances.csv')
    feature_importance.index = feature_importance.feature
    features = feature_importance['average'].copy()
    features.sort_values(ascending=True, inplace=True)
    features = list(features.index)
    
    # we initialise a list where we will collect the
    # features we should remove
    features_to_remove = []
 
    # set a counter to know how far ahead the loop is going
    count = 1
    
    X_train = df.iloc[:int(np.round(0.7*len(df))),:]
    X_test = df.iloc[int(np.round(0.3*len(df))):,:]
    y_train = y[:int(np.round(0.7*len(df)))]
    y_test = y[int(np.round(0.3*len(df))):]
    
    # initialise model
    params = {
                    'objective':'binary',
                    'boosting_type':'gbdt',
                    'metric':'auc',
                    'n_jobs':-1,
                    'learning_rate':0.01,
                    'num_leaves': 2**8,
                    'max_depth':-1,
                    'tree_learner':'serial',
                    'colsample_bytree': 0.5,
                    'subsample_freq':1,
                    'subsample':0.7,
                    'n_estimators':10000,
                    'max_bin':255,
                    'verbose':-1,
                    'seed': config.RANDOM_STATE,
                }

    classifier = lgb.LGBMClassifier(**params, n_jobs = 6)
    classifier.fit( X_train, y_train)
    y_pred = classifier.predict_proba(X_test)[:,1]
    
    auc_score_best = roc_auc_score(y_test, y_pred)
    print('All features ROC AUC = {}'.format( auc_score_best) )
    
    features = [ feat for feat in features if ( feat in df.columns ) ]
    # now we loop over all the features, in order of importance:
    # remember that features is the list of ordered features
    # by importance
    for feature in features:
        
        print()
        print('testing feature: ', feature, ' which is feature ', count,
              ' out of ', len(features))
        count = count + 1
    
        auc_current = list()
        
        classifier = lgb.LGBMClassifier( **params, n_jobs = 6)
        classifier.fit(X_train.drop( features_to_remove + [feature], axis=1 ),y_train)
        
        y_pred = classifier.predict_proba( X_test.drop( features_to_remove + [feature], axis=1))[:,1]
        
        auc_current = roc_auc_score(y_test, y_pred)

        print('New Test ROC AUC={}'.format((auc_current)))
 
        # print the original roc-auc with all the features
        print('All features Test ROC AUC={}'.format((auc_score_best)))
 
        # compare the drop in roc-auc with the tolerance
        # we set previously
        if auc_current >= auc_score_best:
            print('New best ROC AUC={}'.format(auc_current))
            print('drop: ', feature)
            print
            
            # if the drop in the roc is small and we remove the
            # feature, we need to set the new roc to the one based on
            # the remaining features
            auc_score_best = auc_current
            
            # and append the feature to remove to the collecting
            # list
            features_to_remove.append(feature)
            
        else:
            print('Drop in ROC AUC={}'.format(auc_score_best - auc_current))
            print('keep: ', feature)
            print
 
    # now the loop is finished, we evaluated all the features
    print('DONE!!')
    print('total features to remove: ', len(features_to_remove))
 
    # determine the features to keep (those we won't remove)
    features_to_keep = [x for x in features if x not in features_to_remove]
    print('total features to keep: ', len(features_to_keep))
    
    return df, features_to_remove

def p_value_elimination( train, test):
    
    from scipy.stats import ks_2samp
    
    list_p_value = []
    
    for feat in tqdm(train.columns):
        list_p_value.append( ks_2samp( test[feat], train[feat] )[1])
        
    Se = pd.Series( list_p_value, index = train.columns).sort_values()
    list_discarded = list( Se[Se < 0.000000001].index )
    
    return list_discarded