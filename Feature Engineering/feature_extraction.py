# FEATURE EXTRAXTION (ÖZELLİK ÇIKARIMI)

"""
- Feature extraction, bir veri kümesinde bulunan karmaşık ve yüksek
boyutlu verileri daha düşük boyutlu ve daha anlamlı özelliklerle temsil etme sürecidir.
Makine öğrenimi ve veri madenciliği gibi alanlarda sıkça kullanılır.
- Feature extraction, veri kümesinin daha sade ve anlamlı bir temsilini
oluşturarak bu sorunları ele alır. Özellikle, veri örneklerini daha düşük boyutlu
bir özellik uzayında ifade eden yeni özellikler oluşturur. Bu yeni özellikler,
verinin önemli örüntüleri, yapıları ve ilişkileri hakkında daha fazla bilgi
taşıyabilir.
- Örneğin, resim işlemede evrişimli sinir ağları (CNN'ler) ile özellik çıkarma
yapılabilir. Metin verileri için, kelime gömme (word embedding) gibi yöntemlerle
özellik çıkarma gerçekleştirilebilir.
- Sonuç olarak, veri analizini daha etkili ve verimli hale getirmek için yüksek
boyutlu verileri daha anlamlı ve komptakt özelliklere dönüştüren bir işlemdir.
"""


# Binary Features

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

# "Cabin" değişkeni için NaN değerleri 1, olmayanları 1 olarak nitelendiren bir değişken oluşturalım.
df["NEW_CABIN_BOOL"] = df["Cabin"].notnull().astype("int")

# Kabin değişkeninin olup olmamasının hayatta kalma oranı ile ilişkisi
df.groupby("NEW_CABIN_BOOL").agg({"Survived" : "mean"})
"""
proportion z_test, bir popülasyonun iki kategorik özellik arasındaki farkın
istatistiksel olarak anlamlı olup olma durumunu değerlendiren bir hipotez testidir. 
Özellikle iki bağımsız popülasyonun arasındaki farkı test etmek amacıyla kullanılır.
"""

from statsmodels.stats.proportion import proportions_ztest

test_stat, pvalue = proportions_ztest(count = [df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].sum(),
                                               df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].sum()],

                                      nobs =[df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].shape[0],
                                             df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].shape[0]])

print("Test Stat = %.4f, p_value = %.4f" % (test_stat, pvalue))
# Test Stat = 9.4597, p-value = 0.0000
# p-value = 0.05'Den küçük olduğu için hipotez reddedildi yani ikisi arasında anlamlı bir fark bulunmaktadır.


# "SisSp" (yakın akraba), "Parch" (uzak akraba) değişkenlerinin hayatta kalma oranları ile ilişkisini içeren yeni bir değişken oluşturalım.

df.loc[((df["SibSp"] + df["Parch"]) > 0), "NEW_IS_ALONE"] = "NO" # Yalnız olmayanlar
df.loc[((df["SibSp"] + df["Parch"]) == 0), "NEW_IS_ALONE"] = "YES" # Yalnız olanlar

df.groupby("NEW_IS_ALONE").agg({"Survived": "mean"})

test_stat, pvalue = proportions_ztest(count = [df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].sum(),
                                               df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].sum()],

                                      nobs = [df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].shape[0],
                                              df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].shape[0]])

print("Test Stat = %4f, p-value = %.4f" % (test_stat, pvalue))


# Text'ler Üzerinden Özellik Türetmek

# Letter Count

# İsimlerin kaç harfli olduğunu içeren bir değişken oluşturalım.
df["NEW_NAME_COUNT"] = df["Name"].str.len()
df.head()

# Word Count

# İsmin kaç kelimeden olduğunu içerne bir değişken oluşturalım.
df["NEW_NAME_WORD_COUNT"] = df["Name"].apply(lambda x: len(str(x).split(" ")))

# Özel yapıları yakalamak

# Doktor olanları yeni bir değişkene atayalım.
df["NEW_NAME_DR"] = df["Name"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))

df.groupby("NEW_NAME_DR").agg({"Survived": ["mean", "count"]})

# Regex İle Değişken üretmek

df["NEW_TITLE"] = df.Name.str.extract(" ([A-Za-z]+)\.", expand = False)
df.head()

df[["NEW_TITLE", "Survived", "Age"]].groupby(["NEW_TITLE"]).agg({"Survived": "mean","Age": ["count","mean"]})

# Feature Interactions (Özellik Etkileşimleri)

df["NEW_AGE_PCLASS"] = df["Age"] * df["Pclass"]

df["NEW_FAMILY_SIZE"] = df["SibSp"] + df["Parch"] + 1

df.loc[(df["Sex"] == "male") & (df["Age"] <= 21), "NEW_SEX_CAT"] = "youngmale"

df.loc[(df["Sex"] == "male") & ((df["Age"] > 21) & (df["Age"]) <= 50), "NEW_SEX_CAT"] = "maturemale"

df.loc[(df["Sex"] == "male") & (df["Age"] > 50), "NEW_SEX_CAT"] = "youngmale"

df.loc[(df["Sex"] == "female") & (df["Age"] <= 21), "NEW_SEX_CAT"] = "youngmale"

df.loc[(df["Sex"] == "female") & ((df["Age"] > 21) & (df["Age"]) <= 50), "NEW_SEX_CAT"] = "maturefemale"

df.loc[(df["Sex"] == "female") & (df["Age"] > 50), "NEW_SEX_CAT"] = "seniorfemale"

df.head()

df.groupby("NEW_SEX_CAT")["Survived"].mean()

