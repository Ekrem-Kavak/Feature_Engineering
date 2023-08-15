# MISSING VALUES (EKSİK DEĞERLER)

"""

Veri setleri içerisinde eksik değerler bulunabilir. Eksik değer sorununu
çözmek için 3 farklı yöntem uygulanabilir.
1- Eksik verileri silme
2- Temel istatistiksel işlemler yaparak değer atama
3- Gelişmiş istatistiksel işlemler ve makine öğrenmesi metotları kullanarak tahmine dayalı yöntemlerle değer atama

"""

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: '%.3f' % x)
pd.set_option("display.width", 500)


def load():
    data = pd.read_csv("Feature Engineering/datasets/titanic.csv")
    return data

df = load()
df.head()

# Eksik değerlerin olup olmadığının tespiti
df.isnull().values.any()

# Değişkenlerdeki eksik değer sayısı
df.isnull().sum()

# Değiğşkenlerdeki eksik olmayan değer sayısı
df.notnull().sum()

# Veri setindeki toplam eksik değer sayısı
df.isnull().sum().sum()

# En az 1 eksik değere sahip olan gözlem birimleri
df[df.isnull().any(axis = 1)]

# Eksik olmayan gözlem birimleri
df[df.notnull().all(axis = 1)]

# Değişkenlerdeki eksik değerlere göre azalan şekilde
df.isnull().sum().sort_values(ascending = False)

# Eksik değerin toplam gözlem birimine oranı
(df.isnull().sum() / df.shape[0] * 100).sort_values(ascending = False)

# Eksik değere sahip değişkenlerin isimleri

na_cols = [col for col in df.columns if df[col].isnull().sum() > 0]

# Yukarıda yaptığımız işlemleri fonksiyonlaştıralım.

def missing_values_table(dataframe, na_name = False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending = False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending = False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis = 1, keys = ["n_miss", "ratio"])
    print(missing_df, end = "\n")

    if na_name:
        return na_columns

missing_values_table(df, True)

# 1- Eksik değerleri silmek

df.dropna().shape

# 2- Temel istatistiksel işlemler yaparak değer atama

# "age" değişkenindeki eksik değerler için ortalamayı kullanalım.

df["Age"].fillna(df["Age"].mean())
df["Age"].fillna(df["Age"].median()).isnull().sum() # Eksik değer kalmadığı için 0 döndürür.

# Aynı değişkenn için medyan kullanalım.
df["Age"].fillna(df["Age"].median())

# 0 gibi sabit bir sayıyı da eksik değerler yerine yerleştirebiliriz.

df["Age"].fillna(0).isnull().sum()

# Tüm değişkenlerdeki eksik değerleri doldurma

# İlgili değişkenler sayısal ise ortalamasını ekle, değilse NaN değer olarak kalsın.
dff = df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis = 0).head()

dff.isnull().sum().sort_values(ascending = False)

# Kategorik değişkenlerde eksik değerleri doldurma
# Kategorik değişkenlerde genelde mod kullanılır.

df["Embarked"].fillna(df["Embarked"].mode()[0]).isnull().sum()

# Sabit bir string değeri de atanabilir.

df["Embarked"].fillna("missing")

# 10 ve daha az sınıfı olan (kardinal olmayan) eksik kategorik değişkenlerin yerine modunu yazalım.
df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis = 0).isnull().sum()


# Kategorik Değişken Kırılımında Değer Atama

# Cinsiyete göre yaş ortalamaları
df.groupby("Sex")["Age"].mean()

# Eksik değerlerin cinsiyet yaş ortalamalarına göre doldurulması

df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean"))

# Kadın ve erkekleri loc kullanarak ayrı ayrı da doldurabiliriz.

df.loc[(df["Age"].isnull()) & (df["Sex"] == "female"), "Age"] = df.groupby("Sex")["Age"].mean()["female"]

df.loc[(df["Age"].isnull()) & (df["Sex"] == "male"), "Age"] = df.groupby("Sex")["Age"].mean()["male"]

df.isnull().sum()

# 3- Tahmine Dayalı Atama ile Doldurma (makine öğrenmesi tekniği ile)

df = load()

def grab_col_names(dataframe, cat_th = 10, car_th = 20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th
                   and dataframe[col].dtypes == "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th
                   and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

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
num_cols = [col for col in num_cols if col not in "PassengerId"]

dff = pd.get_dummies(df[cat_cols + num_cols], drop_first = True)
dff.head()

# Değişkenlerin standartlaştırılması

from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

scaler = MinMaxScaler()
dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
dff.head()

# KNN algoritmasının uygulanması

from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors = 5) # en yakın 5 komşunun yaş ortalamasını verir.
dff = pd.DataFrame(imputer.fit_transform(dff), columns = dff.columns)
dff.head()

# Standartlaştırma işlemini geriye de alabiliriz.
dff = pd.DataFrame(scaler.inverse_transform(dff), columns = dff.columns)

# Eksik değerler yerine hangi değerlerin atandığını bulmak için
df["age_imputed_knn"] = dff[["Age"]]
df.loc[df["Age"].isnull(), ["Age", "age_imputed_knn"]]

# Tüm değerleri gözlemleme
df.loc[df["Age"].isnull()].head(20)


df = df.drop(["age_inputed_knn"], axis = 1, inplace = True)

# Eksik Veri Yapısının İncelenmesi

import missingno as msno

# Değişkenlerdeki eksik değer sayısını sütunlar halinde gösterir.
msno.bar(df)
plt.show()

# Eksik değerlerin yoğunlukları gösterilir.
msno.matrix(df)
plt.show()

# Eksik değerlerin birbirleriyle kolarasyonunu gösterir.
msno.heatmap(df)
plt.show()

# Eksik değerlerin Bağımlı Değişken İle İlişkisinin İncelenmesi

missing_values_table(df, True)
na_cols = missing_values_table(df, True)

def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()

    for col in na_columns:
        temp_df[col + "NA_FLAG"] = np.where(temp_df[col].isnull(), 1, 0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end = "\n\n\n")

missing_vs_target(df, "Survived", na_cols)

