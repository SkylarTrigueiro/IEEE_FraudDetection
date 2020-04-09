# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 19:00:03 2019

@author: striguei
"""
import pandas as pd
import numpy as np
import config
from sklearn.model_selection import train_test_split

def load_data():
    
    # importing all of the data sets.

    train_trans_data = pd.read_csv("data/train_transaction.csv")
    train_id_data = pd.read_csv("data/train_identity.csv")
    test_trans_data = pd.read_csv("data/test_transaction.csv")
    test_id_data = pd.read_csv("data/test_identity.csv")
    
    train = train_trans_data.merge(train_id_data, on='TransactionID', how='left')
    submission = test_trans_data.merge(test_id_data, on='TransactionID', how='left')

    return train, submission