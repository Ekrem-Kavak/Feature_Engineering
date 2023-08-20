# FEATURE EXTRACTION PRACTICE
# TİTANİK VERİ SETİ

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
df.shape

# Değişken isimlerinin standartlaştırılması
df.columns = [col.upper() for col in df.columns]
df.head()

# 1- Feature Engineering

# "Cabin" değişkeninin olup olmadığı hakkında bir değişken ekledik.
df["NEW_CABIN_BOOL"] = df["CABIN"].notnull().astype("int")

# "Name" değişkeninin harf sayısı hakkında bir değişken ekledik.
df["NEW_NAME_COUNT"] = df["NAME"].str.len()

# "Name" değişkeninin kelime sayısı hakkında bir değişken ekledik.
df["NEW_NAME_WORD_COUNT"] = df["NAME"].apply(lambda x: len(str(x).split(" ")))

# "Name değişkeni içinde doktor olanları içeren bir değişken ekledik.
df["NEW_NAME_DR"] = df["NAME"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))

# "Name" değişkeni içindeki unvanları barındıran bir değişken ekledik.
df["NEW_TITLE"] = df.NAME.str.extract(" ([A-Za-z]+)\.", expand = False)

df["NEW_FAMILY_SIZE"] = df["SIBSP"] + df["PARCH"] + 1

# age * pclass değişkeninin sonuclarını bir değişkene atadık.
df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]

# "SibSp" ve "Parch" değişkenlerine göre yalnız olup olmama durumunu tespit eden değişken ekledik.
df.loc[((df["SIBSP"] + df["PARCH"]) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df["SIBSP"] + df["PARCH"]) == 0), "NEW_IS_ALONE"] = "YES"

# Yaşlara göre sınıflandırma yapan değişkeni ekledik.
df.loc[(df["AGE"] < 18), "NEW_AGE_CAT"] = "young"
df.loc[(df["AGE"] >= 18) & (df["AGE"] < 56), "NEW_AGE_CAT"] = "mature"
df.loc[(df["AGE"] >= 56), "NEW_AGE_CAT"] = "senior"

# Cinsiyet ve yaş değişkenlerinin ilişkisini tanımlayan bir değişken ekledik.
df.loc[(df["SEX"] == "male") & (df["AGE"] <= 21), "NEW_SEX_CAT"] = "youngmale"
df.loc[(df["SEX"] == "male") & ((df["AGE"] > 21) & (df["AGE"]) <= 50), "NEW_SEX_CAT"] = "maturemale"
df.loc[(df["SEX"] == "male") & (df["AGE"] > 50), "NEW_SEX_CAT"] = "seniormale"
df.loc[(df["SEX"] == "female") & (df["AGE"] <= 21), "NEW_SEX_CAT"] = "youngfemale"
df.loc[(df["SEX"] == "female") & ((df["AGE"] > 21) & (df["AGE"]) <= 50), "NEW_SEX_CAT"] = "maturefemale"
df.loc[(df["SEX"] == "female") & (df["AGE"] > 50), "NEW_SEX_CAT"] = "seniorfemale"

# Kategorik, sayısal, kategorik ama sayısal ve sayısal ama kardinal olanları listeleyelim.

def grab_col_names(dataframe, cat_th= 10, car_th = 20):
    cat_cols = [col for col in df.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in df.columns if dataframe[col].nunique() < cat_th
                    and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in df.columns if dataframe[col].nunique() > car_th
                   and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in df.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if "PASSENGERID" not in col]

# 2- Outliers (aykırı değerler)

# Aykırı değerlerin üst ve alt sınırını bulduk.

def outlier_threshold(dataframe, col_name, q1 = 0.25, q3 = 0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquartile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquartile_range
    low_limit = quartile1 + 1.5 * interquartile_range
    return up_limit, low_limit

# Aykırı değer olup olmadığını tespit ettik.

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_threshold(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis = None):
        return True
    else:
        return False


# Var olan aykırı değerleri baskılama yöntemiyle normalleştirelim.

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_threshold(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# Aykırı değerleri çağırdık.
for col in num_cols:
    print(col, check_outlier(df, col))

# Baskılama yöntemi ile aykırı değerleri yok ettik.
for col in num_cols:
    replace_with_thresholds(df, col)

# Aykırı değer kalmadı.
for col in num_cols:
    print(col, check_outlier(df, col))


# 3- MISSING VALUES (EKSİK DEĞERLER)

# Eksik değerlere sahip olan değişkenleri tespit ettik.

def missing_values_table(dataframe, na_name = False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending = False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending = False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis = 1, keys = ["n_miss", "ratio"])
    print(missing_df, end = "\n")

    if na_name:
        return na_columns

missing_values_table(df)
# Eksik değere sahip değişkenler: CABIN, AGE, NEW_AGE_PCLASS, NEW_AGE_CAT, EMBARKED

# "NEW_CABIN_BOOL" adında bir değişkenimiz olduğu için "CABIN" değişkenini sildik.

df.drop("CABIN", inplace = True, axis = 1)
df.head()

# "TICKET" VE "NAME" değişkenini içeren teni değişkenler oluşturduğumuz için bunları da sildik.
remove_cols = ["TICKET", "NAME"]
df.drop(remove_cols, inplace = True, axis = 1)

# "AGE" değişkenindeki eksik değerlerini medyan ile doldurduk.
df["AGE"].median() # 46.9375
df["AGE"] = df["AGE"].fillna(df.groupby("NEW_TITLE")["AGE"].transform("median"))

# "NEW_AGE_CAT" ve "NEW_AGE_PCLASS" değişkenlerini de ilgili değişkenler ile doldurduk.
df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]

df.loc[(df["AGE"] < 18), "NEW_AGE_CAT"] = "young"
df.loc[(df["AGE"] >= 18) & (df["AGE"] < 56), "NEW_AGE_CAT"] = "mature"
df.loc[(df["AGE"] >= 56), "NEW_AGE_CAT"] = "senior"

df.loc[(df["SEX"] == "male") & (df["AGE"] <= 21), "NEW_SEX_CAT"] = "youngmale"
df.loc[(df["SEX"] == "male") & ((df["AGE"] > 21) & (df["AGE"]) <= 50), "NEW_SEX_CAT"] = "maturemale"
df.loc[(df["SEX"] == "male") & (df["AGE"] > 50), "NEW_SEX_CAT"] = "seniormale"
df.loc[(df["SEX"] == "female") & (df["AGE"] <= 21), "NEW_SEX_CAT"] = "youngfemale"
df.loc[(df["SEX"] == "female") & ((df["AGE"] > 21) & (df["AGE"]) <= 50), "NEW_SEX_CAT"] = "maturefemale"
df.loc[(df["SEX"] == "female") & (df["AGE"] > 50), "NEW_SEX_CAT"] = "seniorfemale"

# "EMBARKED" değişkenindeki eksik değerleri mod değeri ile doldurduk.

df = df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis = 0)

missing_values_table(df) # Eksik değer kalmadı

# 4- LABEL ENCODING (ETİKET KODLAMASI)

# İlgili kategorik değişkenleri 0 ve 1 şeklinde binary haline getirelim.

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]


for col in binary_cols:
    df = label_encoder(df, col)

# 5- RARE ENCODING

# 0.01 oranından düşük nadir olan değişken sınıflarını birleştirelim.


def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "TARGET", cat_cols)

def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == "O"
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis = None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), "Rare", temp_df[var])

    return var

df = rare_encoder(df, 0.01)
df["NEW_TITLE"].value_counts()

# 6- ONE-HOT ENCODING

def one_hot_encoder(dataframe, categorical_cols, drop_first = True):
    dataframe = pd.get_dummies(dataframe, columns = categorical_cols, drop_first = drop_first)
    return dataframe

ohe_cols = [col for col in df.columns if  10 >= df[col].nunique() > 2]
one_hot_encoder(df, ohe_cols).head()

rare_analyser(df,"SURVIVED", cat_cols)

useless_cols = [col for col in df.columns if df[col].nunique() == 2 and
                (df[col].value_counts() / len(df) < 0.01).any(axis = None)]


