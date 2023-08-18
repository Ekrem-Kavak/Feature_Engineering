# ENCODING SCALING (KODLAMA ÖLÇEKLENDİRME)
"""
Label Encoding: Kategorik veya sınıfsal verileri sayısal değerlere dönüştürmek için kullanılan
bir yöntemi ifade eder. Label encoding, verilerin arasında sıralama veya derecelendirme
varsa anlamlı hale gelir.

Binary Encoding: Kategorik değerleri ikili formatta (0 ve 1) temsil eder. Her
kategori, ikili bir sayı dizisi olarak ifade edilir. Bu yöntem, her sütunun belirli
bir bit konumunu temsil ettiği bir kodlama sistemidir.
"""

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

df["Sex"].head()

# Label encoding ile kategorik olan cinsiyet değişkenini sayısallaştıralım.
le = LabelEncoder()

# fit_transform ile yeni haline dönüştürelim.
le.fit_transform(df["Sex"])[0:5]
# Sayılar dğeişkenlerin isimlerine göre alfabetik olarak atanır.

# inverse_tranform ile hangi sayıya hengi değişkenin atandığını bulalım.
le.inverse_transform([0, 1])

# Yukarıdaki işlemleri fonksiyonlaştıralım.

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

label_encoder(df, "Sex").head(10)

# Veri seti içindeki 2 sınıfı olan tüm değişkenler için binary encoding işlemini uygulayalım.

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
              and df[col].nunique() == 2]

for col in binary_cols:
    label_encoder(df, col)

df[binary_cols].head()

# Daha büyük bir veri setinde uygulayalım.

def load_application_train():
    data = pd.read_csv("Feature Engineering/datasets/application_train.csv")
    return data

df = load_application_train()
df.shape

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
              and df[col].nunique() == 2]

df[binary_cols].head()

for col in binary_cols:
    label_encoder(df, col)

df = load()
df["Embarked"].value_counts()
df["Embarked"].nunique()
len(df["Embarked"].unique())

# ONE-HOT ENCODING
"""
Her kategori için ayrı bir sütun oluşturarak yapılır. Her örnek için sadece
ilgili kategoriye ait sütun değeri 1 olarak işaretlenirken, diğer kategori
sütunları 0 olarak kalır. Bu şekilde her kategori benzersiz bir sayısal temsil ile
ifade edilir.
"""

df = load()
df.head()
df["Embarked"].value_counts()

# get_dummies metodu ile "Embarked" değişkeni içindeki sınıfları değişkene çevirdik.
pd.get_dummies(df, columns = ["Embarked"]).head()

# "drop_first" ile yeni oluşturulan ilk sınıfı silinir.
pd.get_dummies(df, columns = ["Embarked"]).head()

# "Embarked" değişkeni içindeki eksik değerleri için de bir değişken oluşturalım.
pd.get_dummies(df, columns = ["Embarked"], dummy_na= True).head()

# "Sex" değişkeni içinde aynı işlemi uygularsak kadın ve erkek olarak iki farklı değişken üretir.
pd.get_dummies(df, columns = ["Sex"]).head()

# "drop_first" ile yeni eklenen ilk değişkeni silmesini sağladık. (alfabetik öncelik)
pd.get_dummies(df, columns = ["Sex"], drop_first= True).head()

# get_dummies ile "Sex" ve "Embarked" değişkenlerinin sınıflarını değişkene çevirdik ve
# "drop_first" ile oluşturulan ilk değişkenlei sildik.

pd.get_dummies(df, columns = ["Sex", "Embarked"], drop_first = True).head()

# Yukarıdaki işlemleri fonksiyonlaştıralım
def one_hot_encoder(dataframe, categorical_cols, drop_first = True):
    dataframe = pd.get_dummies(dataframe, columns = categorical_cols, drop_first = drop_first)
    return dataframe

# Veri seti içinde bulunan değişkenlerin 2'den büyük 11'den küçük olanlara
# one-hot encoding işlemini uygulayalım.

df = load()
ohe_cols = [col for col in df.columns if  10 >= df[col].nunique() > 2]
one_hot_encoder(df, ohe_cols).head()
"""
NOT: Örneğin bir değişken 3 farklı kategoriye sahipse one-hot encoding ile 3 ayrı
sütun oluşturulur ancak tüm sütunlar oluşturulduktan sonra ilk sütun dahil edilirse, 
aslında bu kategori setinin tamamını temsil eden bir sütun gereksiz yere eklenmiş olur.
Bu yüzden drop_first kullanılır.  
"""

# RARE ENCODING
"""
Rare encoding, nadir olarak karşılaşılan kategorik değerleri gruplayarak ve daha
az sayıda kategoriye dönüştürerek veri analizi ve ve makine öğrenimi modelleri için
daha etkili hale getirme amaçlı bir veri ön işleme tekniğidir. 
"""

# 1- Kategorik değişkenlerin azlık - çokluk durumunun analiz edilmesi

df = load_application_train()
df["NAME_EDUCATION_TYPE"].value_counts()

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

def cat_summary(dataframe, col_name, plot = False):
    print(pd.DataFrame({col_name : dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("----")
    if plot:
        sns.countplot(x = dataframe[col_name], data = dataframe)
        plt.show()

cat_cols, num_cols, cat_but_car = grab_col_names(df)

for col in cat_cols:
    cat_summary(df, col)

# "rare" kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi.

df["NAME_INCOME_TYPE"].value_counts()

df.groupby("NAME_INCOME_TYPE")["TARGET"].mean()

# Yukarıdaki işlemleri fonksiyonlaştıralım.

def rare_analyser(dataframe, target, cat_cols):
        for col in cat_cols:
            print(col, ":", len(dataframe[col].value_counts()))
            print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                                "RATIO": dataframe[col].value_counts() / len(dataframe),
                                "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end = "\n\n\n")

rare_analyser(df, "TARGET", cat_cols)

def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == "O"
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis = None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), "Rare", temp_df[var])

    return temp_df

new_df = rare_encoder(df, 0.01)
new_df.head()

rare_analyser(new_df, "TARGET", cat_cols)


# FEATURE SCALING (Özellik Ölçekleme)
"""
Bir veri setindeki değişkenlerin aralığını eşitlemek için kullanılan bir veri 
ön işleme tekniğidir. Bu, makine öğrenmesi modellerinin daha iyi performans 
göstermesine yardımcı olur. İki şekilde gerçekleştirilebilir. 
1- Standartlaştırma: Her değişkenin ortalaması 0 ve standart sapması 1 olacak 
şekilde ölçeklendirilir. 
2- Min - max ölçeklendirme: Her değişkenin aralığı 0 ile 1 arasında olacak
şekilde ölçeklendirilir.   
"""

# StandartScaler: Klasik standartlaştırma olarak bilinir.
# Tüm değerlerden ortalamayı çıkar ve standart sapmaya böl. z = (x - u) / s

df = load()
ss = StandardScaler()
df["Age_standard_scaler"] = ss.fit_transform(df[["Age"]])
df.head()

# RobustScaler: Tüm değerlerden medyan çıkarılıp iqr'a bölünür. (iqr = q3 - q1)

rs = RobustScaler()
df["Age_robust_scaler"] = rs.fit_transform(df[["Age"]])
df.describe().T

# Min - Max Scaler

# X_std = (X - X.min(axis = 0)) / (X.max(axis = 0) - X.min(axis = 0))
# X_scaled = X_std * (max - min) + min

mns = MinMaxScaler()
df["Age_min_max_scaler"] = mns.fit_transform(df[["Age"]])
df.describe().T

age_cols = [col for col in df.columns if "Age" in col]

def num_summary(dataframe, numerical_col, plot = False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins = 20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block = True)

for col in age_cols:
    num_summary(df, col, True)

# Sayısal değişkenleri qcut ile kategorik değişkenlere çevirebiliriz.
df["Age_qcut"] = pd.cut(df["Age"], 5)

