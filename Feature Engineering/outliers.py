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

def load_application_train():
    data = pd.read_csv("Feature Engineering/datasets/application_train.csv")
    return data

dff = load_application_train()
df.head()


# Outliers (aykırı değerler)

# Yaş değişkenine göre aykırı değerlere bakalım

q1 = df["Age"].quantile(0.25) # birinci çeyreklik
q3 = df["Age"].quantile(0.75) # üçüncü çeyreklik

iqr = q3 - q1
up = q3 + 1.5 * iqr # pozitif aykırı değer sınırı
low = q1 - 1.5 * iqr # negatif aykırı değer sınırı

df[(df["Age"] < low) | (df["Age"] > up)] # aykırı değerler (outliers)

df[(df["Age"] < low) | (df["Age"] > up)].index # aykırı değerlerin indeksleri

df[(df["Age"] < low) | (df["Age"] > up)].any(axis = None) # İçinde en az 1 tane aykırı değer varsa True döndürür.

df[~((df["Age"] < low) | (df["Age"] > up))].any(axis = None) # İçinde en az 1 tane aykırı olmayan değer varsa True döndürür.

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
outlier_threshold(df, "Pclass")

low, up = outlier_threshold(df, "Age")

df[(df["Fare"] < low) | (df["Fare"] > up)].head()
df[(df["Fare"] < low)].head()
df[(df["Fare"] > up)].head()
df[(df["Pclass"] > up) |(df["Fare"] < low).head()]

df[(df["Fare"] < low) | (df["Fare"] > up)].head().index

# outlier değer olup olmadığını bulan fonksiyon

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_threshold(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis = None):
        return True
    else:
        return False

check_outlier(df, "Age") # True (aykırı değer var)
check_outlier(df, "Pclass") # False (aykırı değer yok)

# grab_col_names

dff = load_application_train()
dff.head()

def grab_col_names(dataframe, cat_th = 10, car_th = 20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardianl değişkenlerin
    isimlerini verir.
    NOT: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ----------
    dataframe: dataframe
    cat_th: int, optional
        numerik fakat kategorik olan değişkenler için sınıf eşik değeri
    car_th: int, optional
        kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    -------
    cat_cols: list
        kategorik değişken listesi
    num_cols: list
        numerik değişken listesi

    """

    # cat_cols, cut_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th
                   and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th
                   and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")
    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)

# "PassengerId" işimize yaramayan bir değişken olduğu için atabiliriz. (yorum)

num_cols = [col for col in num_cols if col not in "PassengerId"]

# Aykırı değer (outliers) olup olmadığını bulma

for col in num_cols:
    print(col, check_outlier(df, col))

# Diğer veri setimiz için uygulayalım.

cat_cols, num_cols, cat_but_car = grab_col_names(dff)

for col in num_cols:
    print(col, check_outlier(dff, col))

# "SK_ID_CURR" değişkenini atabiliriz. (yorum)

num_cols = [col for col in num_cols if col not in "SK_ID_CURR"]


# AYKIRI DEĞERLERE ERİŞMEK

def grab_outliers(dataframe, col_name, index = False):
    low, up = outlier_threshold(dataframe, col_name)
    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] > low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

check_outlier(df, "Age") # true
grab_outliers(df, "Age")
grab_outliers(df, "Age", True)

# AYKIRI DEĞER PROBLEMİNİ ÇÖZME

# 1- Aykırı değerleri silme

low, up = outlier_threshold(df, "Fare")
df.shape

df[~((df["Fare"] < low) | (df["Fare"] > up))].shape

def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_threshold(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers

cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if col not in "PasengerId"]

df.shape

for col in num_cols:
    new_df = remove_outlier(df, col)

df.shape[0] - new_df.shape[0] # Toplam 116 aykırı değer vardır.


# 2- Baskılama Yöntemi (re-assigment with thresholds)
# Aykırı değerler üst ve alt sınırı eşitlenir.

low, up = outlier_threshold(df, "Fare")

df[((df["Fare"] < low) | (df["Fare"] > up))]["Fare"]

df.loc[((df["Fare"] > low) | (df["Fare"] > up)), "Fare"]

df.loc[(df["Fare"] > up), "Fare"] = up

df.loc[((df["Fare"] < low), "Fare")] = low

# Yukarıdaki işlemleri fonksiyonlaştıralım.

def replace_with_threholds(dataframe, variable):
    low_limit, up_limit = outlier_threshold(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

df = load()
cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]

for col in num_cols:
    print(col, check_outlier(df, col)) # True (Aykırı değer var)

for col in num_cols:
    replace_with_threholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col)) # False (Artık aykırı değer yok)


# ÇOK DEĞİŞKENLİ AYKIRI DEĞER ANALİZİ: LOCAL OUTLIER FACTOR

"""
Local Outlier Factor (LOF) algoritması, aykırı değerleri tespit etmek için
yerel çevresel yoğunluğu değerlendirir. Veri noktasının çevresindeki komşu noktalarının
yoğunluğunu inceleyerek, bir veri noktasının genel veri setine göre ne kadar anormal
olduğunu belirlemeye çalışır. Eğer bir veri noktasının çevresindeki komşu noktalarının
yoğunluğu düşükse, bu nokta muhtemelen bir aykırı değer olarak kabul edilir.  
"""

df = sns.load_dataset("diamonds")
df = df.select_dtypes(include = ["float64", "int64"])
df = df.dropna()
df.head()

for col in df.columns:
    print(col, check_outlier(df, col))

low, up = outlier_threshold(df, "carat")

df[((df["carat"] < low) | (df["carat"] > up))].shape

low, up = outlier_threshold(df, "depth")

df[((df["depth"] < low) | (df["depth"] > up))].shape

clf = LocalOutlierFactor(n_neighbors = 20) # 20 komşu varsayılandır.
clf.fit_predict(df) # -1 aykırı değerleri ifade eder, 1 normal değerleri ifade eder.

df_scores = clf.negative_outlier_factor_
df_scores[0:5] # En aykırı 5 değer

np.sort(df_scores)[0:5] # en negatif skordan (en düşük), en pozitif (en yüksek) skora gider.

scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked = True, xlim = [0, 20], style = ".-")
plt.show()

# Tabloyu inceledik ve ilk 3 indeksin aykırı değer olduğuna karar verdik.
th = np.sort(df_scores)[3]

df[df_scores < th]

df[df.scores < th].shape

# Aykırı değerleri yorumlayalım.
df.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T


df[df_scores < th].index

# Aykırı değerleri silelim.

df[df.scores < th].drop(axis = 0, labels = df[df_scores < th].index)



