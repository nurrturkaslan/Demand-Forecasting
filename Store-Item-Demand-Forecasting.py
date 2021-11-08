# Store Item Demand Forecasting #

# 1. İş Problemi
# Bir mağaza zinciri, 10 farklı mağazası ve 50 farklı ürünü için 3 aylık bir talep tahmini istemektedir.

# 2. Veri Seti Hikayesi
# Bu veri seti farklı zaman serisi tekniklerini denemek için sunulmuştur.
# Bir mağaza zincirinin 5 yıllık verilerinde 10 farklı mağazası ve 50 farklı ürünün bilgileri yer almaktadır.

# 3. Değişkenler
# date – Satış verilerinin tarihi
# ~~Tatil efekti veya mağaza kapanışı yoktur.

# Store – Mağaza ID’si
# ~~Her bir mağaza için eşsiz numara.

# Item – Ürün ID’si
# ~~Her bir ürün için eşsiz numara.

# Sales – Satılan ürün sayıları,
# ~~Belirli bir tarihte belirli bir mağazadan satılan ürünlerin sayısı.

# GÖREV

# Aşağıdaki zaman serisi ve makine öğrenmesi tekniklerini kullanarak ilgili mağaza
# zinciri için 3 aylık bir talep tahmin modeli oluşturunuz.
# ▪ Random Noise
# ▪ Lag/Shifted Features
# ▪ Rolling Mean Features
# ▪ Exponentially Weighted Mean Features
# ▪ Custom Cost Function (SMAPE)
# ▪ LightGBM ile Model Validation

#Importing Libraries
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import time
import lightgbm as lgb
import warnings

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
warnings.filterwarnings('ignore')

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

train = pd.read_csv("Weeks/WEEK_09/train.csv", parse_dates=["date"])
test = pd.read_csv("Weeks/WEEK_09/test.csv", parse_dates=["date"])
sample_sub = pd.read_csv("Weeks/WEEK_09/sample_submission.csv")
df = pd.concat([train, test], sort=False)

# Exploratory Data Analysis
df["date"].min()
df["date"].max()

check_df(train)
check_df(test)
check_df(sample_sub)
check_df(df)

df["sales"].describe([0.10, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95, 0.99]) #satış dağılımı

df[["store"]].nunique() #kaç benzersiz mağaza var

df[["item"]].nunique() #kaç benzersiz item var

df.groupby(["store"])["item"].nunique() #mağazalardaki benzerisz item sayısı

df.groupby(["store", "item"]).agg({"sales": ["sum"]}) #mağzalardaki yapılan sales toplamları

df.groupby(["store", "item"]).agg({"sales": ["sum", "mean", "median", "std"]}) #mağaza-item satış istatistikleri

# Feature Engineering
def create_date_features(df):
    df['month'] = df.date.dt.month  # hangi ayda
    df['day_of_month'] = df.date.dt.day  # ayın hangi gününde
    df['day_of_year'] = df.date.dt.dayofyear  # yılın hangi gününde
    df['week_of_year'] = df.date.dt.weekofyear  # yılın hangi haftasında
    df['day_of_week'] = df.date.dt.dayofweek    # haftanın hangi gününde
    df['year'] = df.date.dt.year              # hangi yılda
    df["is_wknd"] = df.date.dt.weekday // 4   # haftasonu mu değil mi
    df['is_month_start'] = df.date.dt.is_month_start.astype(int)   # ayın başlangıcı mı
    df['is_month_end'] = df.date.dt.is_month_end.astype(int)      # ayın bitişi mi

    df['quarter'] = df.date.dt.quarter
    df['is_quarter_start'] = df.date.dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df.date.dt.is_quarter_end.astype(int)
    df['days_in_month'] = df.date.dt.daysinmonth  # bulunduğu ay kaç çekiyor.
    df['is_year_end'] = df.date.dt.is_year_end.astype(int)
    # haftanın günleri
    df['is_Mon'] = np.where(df['day_of_week'] == 1, 1, 0)
    df['is_Tue'] = np.where(df['day_of_week'] == 2, 1, 0)
    df['is_Wed'] = np.where(df['day_of_week'] == 3, 1, 0)
    df['is_Thu'] = np.where(df['day_of_week'] == 4, 1, 0)
    df['is_Fri'] = np.where(df['day_of_week'] == 5, 1, 0)
    df['is_Sat'] = np.where(df['day_of_week'] == 6, 1, 0)
    df['is_Sun'] = np.where(df['day_of_week'] == 7, 1, 0)

    df["isFirst"] = [1 if i<=15 else 0 for i in df['day_of_month']]

    #0: Winter - 1: Spring - 2: Summer - 3: Fall
    df.loc[df["month"]  == 12, "season"] = 0
    df.loc[df["month"]  == 1, "season"] = 0
    df.loc[df["month"]  == 2, "season"] = 0

    df.loc[df["month"]  == 3, "season"] = 1
    df.loc[df["month"]  == 4, "season"] = 1
    df.loc[df["month"]  == 5, "season"] = 1

    df.loc[df["month"]  == 6, "season"] = 2
    df.loc[df["month"]  == 7, "season"] = 2
    df.loc[df["month"]  == 8, "season"] = 2

    df.loc[df["month"]  == 9, "season"] = 3
    df.loc[df["month"]  == 10, "season"] = 3
    df.loc[df["month"]  == 11, "season"] = 3

    return df

df = create_date_features(df)

df.groupby(["store", "item", "month"]).agg({"sales": ["sum", "mean", "median", "std"]})



# Random Noise
def random_noise(dataframe):
    return np.random.normal(scale=1.6, size=(len(dataframe)))

# yeni featureların üzerine rastgele gürültü ekliyoruz. genellenebilirliği arttırmak için

# Lag/Shifted Features
# geçmiş gerçek değerleri barındıracak


df.sort_values(by=['store', 'item', 'date'], axis=0, inplace=True)

check_df(df)
# satışın ilk 10 gözlemine bakalım:
df["sales"].head(10)

# Birinci gecikme
df["sales"].shift(1).values[0:10]

# İkinci gecikme
df["sales"].shift(2).values[0:10]

# Üçüncü gecikme
df["sales"].shift(3).values[0:10]

# gecikme bilgisi

def lag_features(dataframe, lags):
    for lag in lags:
        dataframe['sales_lag_' + str(lag)] = dataframe.groupby(["store", "item"])['sales'].transform(
            lambda x: x.shift(lag)) + random_noise(dataframe)
    return dataframe

# 3 aylık tahminler beklendiği için

df = lag_features(df, [91, 98, 105, 112, 119, 126, 182, 364, 546, 728]) # 3 ayın sonrasını tahmin edeceğimiz için
check_df(df)

# Rolling Mean Features - hareketli ortalama

#rolling kaydırmak
#window - adım sayısı

def roll_mean_features(dataframe, windows):
    for window in windows:
        dataframe['sales_roll_mean_' + str(window)] = dataframe.groupby(["store", "item"])['sales']. \
                                                          transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=10, win_type="triang").mean()) + random_noise(
            dataframe)
    return dataframe

df = roll_mean_features(df, [365, 546,730])

# Exponentially Weighted Mean Features
def ewm_features(dataframe, alphas, lags):
    for alpha in alphas:
        for lag in lags:
            dataframe['sales_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe.groupby(["store", "item"])['sales'].transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe


alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
lags = [91, 98, 105, 112, 180, 270, 365, 546, 728]

df = ewm_features(df, alphas, lags)
check_df(df)

# One-Hot Encoding

df = pd.get_dummies(df, columns=['store', 'item', 'day_of_week', 'month', "isFirst"])

# Converting sales to log(1+sales)

df['sales'] = np.log1p(df["sales"].values)
check_df(df)

# Model

# Custom Cost Function

# kaggle'ın istediği şekil

def smape(preds, target):
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds - target)
    denom = np.abs(preds) + np.abs(target)
    smape_val = (200 * np.sum(num / denom)) / n
    return smape_val

# tahmin edilen değerlerin eklenmesi
def lgbm_smape(preds, train_data):
    labels = train_data.get_label()
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return 'SMAPE', smape_val, False


# Time-Based Validation Sets
# 2017'nin başına kadar (2016'nın sonuna kadar) train seti.
train = df.loc[(df["date"] < "2017-01-01"), :]

# 2017'nin ilk 3'ayı validasyon seti.
val = df.loc[(df["date"] >= "2017-01-01") & (df["date"] < "2017-04-01"), :]

cols = [col for col in train.columns if col not in ['date', 'id', "sales", "year"]]

Y_train = train['sales']
X_train = train[cols]

Y_val = val['sales']
X_val = val[cols]

# kontrol
Y_train.shape, X_train.shape, Y_val.shape, X_val.shape

# LightGBM Model

# LightGBM parameters
lgb_params = {'metric': {'mae'},
              'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'num_boost_round': 1000, #iterasyon sayısı
              'early_stopping_rounds': 200,
              'nthread': -1}


lgbtrain = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)
lgbval = lgb.Dataset(data=X_val, label=Y_val, reference=lgbtrain, feature_name=cols)

model = lgb.train(lgb_params, lgbtrain,
                  valid_sets=[lgbtrain, lgbval],
                  num_boost_round=lgb_params['num_boost_round'],
                  early_stopping_rounds=lgb_params['early_stopping_rounds'],
                  feval=lgbm_smape,
                  verbose_eval=200)

y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)

smape(np.expm1(y_pred_val), np.expm1(Y_val))



# Değişken önem düzeyleri

def plot_lgb_importances(model, plot=False, num=10):

    gain = model.feature_importance('gain')
    feat_imp = pd.DataFrame({'feature': model.feature_name(),
                             'split': model.feature_importance('split'),
                             'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    if plot:
        plt.figure(figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x="gain", y="feature", data=feat_imp[0:25])
        plt.title('feature')
        plt.tight_layout()
        plt.show()
    else:
        print(feat_imp.head(num))


plot_lgb_importances(model, num=30)
plot_lgb_importances(model, num=30, plot=True)

lgb.plot_importance(model, max_num_features=20, figsize=(10, 10), importance_type="gain")
plt.show()


# Final Model

train = df.loc[~df.sales.isna()]
Y_train = train['sales']
X_train = train[cols]

test = df.loc[df.sales.isna()]
X_test = test[cols]

lgb_params = {'metric': {'mae'},
              'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'nthread': -1,
              "num_boost_round": model.best_iteration}

# LightGBM dataset
lgbtrain_all = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)

model = lgb.train(lgb_params, lgbtrain_all, num_boost_round=model.best_iteration)
test_preds = model.predict(X_test, num_iteration=model.best_iteration)

# Create submission
submission_df = test.loc[:, ['id', 'sales']]
submission_df['sales'] = np.expm1(test_preds)
submission_df['id'] = submission_df.id.astype(int)
submission_df.to_csv('submission_demand_2.csv', index=False)
submission_df.head(20)