import os
import gc 
import sys
import time
import copy
import joblib
import numpy as np
import pandas as pd
import warnings 
if not sys.warnoptions:
    warnings.simplefilter("ignore")

from pathlib import Path 
from sklearn import preprocessing 
from sklearn.base import TransformerMixin 
from sklearn.pipeline import Pipeline 
from sklearn.feature_extraction import FeatureHasher 
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.utils.metaestimators import if_delegate_has_method 
stderr = sys.stderr 
sys.stderr = open(os.devnull, 'w')
import keras 
from keras import backend as kerasbackend 
from keras.wrappers.scikit_learn import KerasClassifier 
from keras.wrappers.scikit_learn import KerasRegressor 
sys.stderr = stderr 
import _utils as utils 

class PersistentModel:
    """
    A general class to manage persistent models 
    """
    def __init__(self):
        self.name = None
        self.state = None
        self.state_timestamp = None
        self.using_keras = False
    def save(self, name, path, overwrite=True, compress=3, locked_timeout=2):
        f = path + name + '.joblib'
        f_lock = f + '.lock'
        try:
            Path(path).mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            pass 
        
