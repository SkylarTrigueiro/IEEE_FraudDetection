# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 07:39:38 2019

@author: skyst
"""
import config
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import BaggingClassifier
import lightgbm as lgb
from time import time
import datetime

def run_xgb( train, submission, y):
    
    params = {'bagging_fraction': 0.8993155305338455, 'colsample_bytree': 0.7463058454739352, 
                   'feature_fraction': 0.7989765808988153, 'gamma': 0.6665437467229817, 'eval_metric': 'auc',
                   'learning_rate': 0.013887824598276186, 'max_depth': 16, 'min_child_samples': 170,
                   'num_leaves': 220, 'reg_alpha': 0.39871702770778467, 'reg_lambda': 0.24309304355829786,
                   'subsample': 0.7, 'missing':-999}
    
    folds = TimeSeriesSplit(n_splits=config.NUM_FOLDS)
    
    aucs = list()


    training_start_time = time()
    for fold, (train_index, test_index) in enumerate(folds.split(train, y)):
        start_time = time()
        print('Training on fold {}'.format(fold + 1))
    
        X_train = train.iloc[train_index] 
        y_train = y.iloc[train_index]
        X_test = train.iloc[test_index]
        y_test = y.iloc[test_index]
        classifier = xgb.XGBClassifier(**params, n_estimators = 10000, n_jobs = 6 )
        classifier.fit( X_train, y_train , early_stopping_rounds=500, eval_set= [(X_test,y_test)] , verbose = 1000 )

        aucs.append(classifier.best_score)
    
        print('Fold {} finished in {}'.format(fold + 1, str(datetime.timedelta(seconds=time() - start_time))))
    
    print('-' * 30)
    print('Training has finished.')
    print('Total training time is {}'.format(str(datetime.timedelta(seconds=time() - training_start_time))))
    print('Mean AUC:', np.mean(aucs))
    print('-' * 30)

    best_iter = classifier.best_iteration
    
    classifier = xgb.XGBClassifier( **params, n_estimators=best_iter, n_jobs = 6 )
    classifier.fit( train, y)
    
    print('Train set')
    pred = classifier.predict_proba(train)[:,1]
    print('XGB roc-auc: {}'.format(roc_auc_score(y, pred)))
    
    y_pred = classifier.predict_proba(submission)[:,1]
    
    return y_pred

def run_lgb( train, submission, y):
    
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
                    'max_bin':1023,
                    'verbose':-1,
                    'seed': config.RANDOM_STATE,
                } 
    
    params2 = {
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
                    'max_bin':1023,
                    'verbose':-1,
                    'seed': config.RANDOM_STATE,
                } 
    
    folds = TimeSeriesSplit(n_splits=config.NUM_FOLDS)
    
    #aucs = list()
    feature_importances = pd.DataFrame()
    feature_importances['feature'] = train.columns
    
    X_train = train.iloc[:int(np.round((7/10)*len(train))),:]
    X_test = train.iloc[int(np.round((3/10)*len(train))):,:]
    y_train = y[:int(np.round((7/10)*len(train)))]
    y_test = y[int(np.round((3/10)*len(train))):]


    #train_data = lgb.Dataset(X_train, label=y_train)
    #validation_data = lgb.Dataset(X_test, label=y_test)
    #booster = lgb.train(params, train_data, valid_sets = [train_data, validation_data], verbose_eval=1000, early_stopping_rounds=500)
    
    '''
    training_start_time = time()
    for fold, (train_index, test_index) in enumerate(folds.split(train, y)):
        start_time = time()
        print('Training on fold {}'.format(fold + 1))
    
        train_data = lgb.Dataset(train.iloc[train_index], label=y.iloc[train_index])
        validation_data = lgb.Dataset(train.iloc[test_index], label=y.iloc[test_index])
        classifier = lgb.train(params, train_data, valid_sets = [train_data, validation_data], verbose_eval=1000, early_stopping_rounds=500)

        feature_importances['fold_{}'.format(fold + 1)] = classifier.feature_importance()
        aucs.append(classifier.best_score['valid_1']['auc'])
    
        print('Fold {} finished in {}'.format(fold + 1, str(datetime.timedelta(seconds=time() - start_time))))
    
    print('-' * 30)
    print('Training has finished.')
    print('Total training time is {}'.format(str(datetime.timedelta(seconds=time() - training_start_time))))
    print('Mean AUC:', np.mean(aucs))
    print('-' * 30)
    
    feature_importances['average'] = feature_importances[['fold_{}'.format(fold + 1) for fold in range(folds.n_splits)]].mean(axis=1)
    feature_importances.to_csv('feature_importances.csv')
    '''
    
    classifier = lgb.LGBMClassifier( **params2, n_estimators = 2616)
    classifier.fit(train, y)
    print('Train set')
    pred = classifier.predict_proba(train)[:,1]
    print('LGB roc-auc: {}'.format(roc_auc_score(y, pred)))
    
    y_pred = classifier.predict_proba(submission)[:,1]
    
    return y_pred