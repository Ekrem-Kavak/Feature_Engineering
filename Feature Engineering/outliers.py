# FEATURE ENGINEERING & DATA PRE - PROCESSING

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: '%.3f' % x)
pd.set_option("display.width", 500)

def load():
    data = pd.read_csv("Feature Engineering/datasets/titanic.csv")
    return data

df = load()
df.head()

# Outliers (aykırı değerler)

# Yaş değişkenine göre aykırı değerlere bakalım

q1 = df["Age"].quantile(0.25) # birinci çeyreklik
q3 = df["Age"].quantile(0.75) # üçüncü çeyreklik

iqr = q3 - q1
up = q3 + 1.5 * iqr # pozitif aykırı değer sınırı
low = q1 - 1.5 * iqr # negati aykırı değer sınırı

df[(df["Age"] < low) | (df["Age"] > up)] # aykırı değerler (outliers)

df[(df["Age"] < low) | (df["Age"] > up)].index # aykırı değerlerin indeksleri

df[(df["Age"] < low) | (df["Age"] > up)].any(axis = None) # İçinde en az 1 tane aykırı değer varsa True döndürür.

df[~((df["Age"] < low) | (df["Age"] > up))].any(axis = None) # İçinde en az tane aykırı olmayan değer varsa True döndürür.

df[(df["Age"] < low)].any(axis = None) # False (yaş değişkeninde eksi değer olmadığı için aykırı değerlerin "low" kısmı geçersizdir.)

# FONKSİYONLAŞTIRMA

def outlier_threshold(dataframe, col_name, q1 = 0.25, q3 = 0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquartile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquartile_range
    low_limit = quartile1 - 1.5 * interquartile_range
    return low_limit, up_limit


outlier_threshold(df, "Age")
outlier_threshold(df, "Fare")

low, up = outlier_threshold(df, "Age")

df[(df["Fare"] < low) | (df["Fare"] > up)].head()
df[(df["Fare"] < low)].head()
df[(df["Fare"] > up)].head()


df[(df["Fare"] < low) | (df["Fare"] > up)].head().index

# outlier değer olup olmadığını bulan fonkisyon

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_threshold(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis = None):
        return True
    else:
        return False

check_outlier(df, "Age") # True (aykırı değer var)
check_outlier(df, "Pclass") # False (aykırı değer yok)

