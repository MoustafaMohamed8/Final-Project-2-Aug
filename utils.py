## Main Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import warnings
import missingno
warnings.filterwarnings('ignore')
## sklearn -- preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn_features.transformers import DataFrameSelector
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

#import GridSearchCV
from sklearn.model_selection import GridSearchCV

## sklearn -- models
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import xgboost as xgb


## skelarn -- metrics
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



TRAIN_DATA_PATH = os.path.join(os.getcwd(), 'heart.csv')
df = pd.read_csv(TRAIN_DATA_PATH)

df.head()



df.drop_duplicates(inplace=True)


age_bins = [0, 39, 59, float('inf')]  
age_labels = ['Young', 'Middle-Aged', 'Senior']  
df['age_category'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, include_lowest=True)




## to features and target
X = df.drop(columns=['output'], axis=1)
y = df['output']


## split to train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=45, stratify=y)

print('X_train.shape \t', X_train.shape)
print('y_train.shape \t', y_train.shape)
print('**'*20)
print('X_test.shape \t', X_test.shape)
print('y_test.shape \t', y_test.shape)

categ_cols = ['sex','exng','caa','cp','fbs','restecg','slp','thall','age_category']
num_cols = ["age","trtbps","chol","thalachh","oldpeak"]



num_pipline = Pipeline(steps=[
                ('selector', DataFrameSelector(num_cols)),
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

## Categorical
categ_pipline = Pipeline(steps=[
                 ('selector', DataFrameSelector(categ_cols)),
                 ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(drop='first', sparse_output=False))
])

all_pipeline = FeatureUnion(transformer_list=[
                        ('num', num_pipline),
                        ('categ', categ_pipline)
                    ])

## apply
_ = all_pipeline.fit_transform(X_train)


def process_new(x_new):
    df_new=pd.DataFrame([x_new],columns=X_train.columns)
    

    ##Adjust the datatypes
    df_new['age']=df_new['age'].astype('int64')
    df_new['sex']=df_new['sex'].astype('int64')
    df_new['cp']=df_new['cp'].astype('int64')
    df_new['trtbps']=df_new['trtbps'].astype('int64')
    df_new['chol']=df_new['chol'].astype('int64')
    df_new['fbs']=df_new['fbs'].astype('int64')
    df_new['restecg']=df_new['restecg'].astype('int64')
    df_new['thalachh']=df_new['thalachh'].astype('int64')
    df_new['exng']=df_new['exng'].astype('int64')
    df_new['oldpeak']=df_new['oldpeak'].astype('float64')
    df_new['slp']=df_new['slp'].astype('int64')
    df_new['caa']=df_new['caa'].astype('int64')
    df_new['thall']=df_new['thall'].astype('int64')
    df_new['age_category']=df_new['age_category'].astype('category')
   

    ## Apply the pipeline
    X_processed=all_pipeline.transform(df_new)


    return X_processed