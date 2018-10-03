# HousePrice

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

#データの読み込み
df_train = pd.read_csv('all/train.csv')
df_test = pd.read_csv('all/test.csv')

print(df_train.shape)
print(df_test.shape)

# Id，　Saleprice以外の全データを結合(左端: Id, 右端: SalePrice)
df_all = pd.concat((df_train.loc[:,'MSSubClass':'SaleCondition'], df_test.loc[:,'MSSubClass':'SaleCondition']))

# 変数の整理
#築年数
df_all['YearBuilt'] = 2018 - df_all['YearBuilt']
df_train['YearBuilt'] = 2018 - df_train['YearBuilt']

# 売れてからの年数
df_all['YrSold'] = 2018 - df_all['YrSold']
df_train['YrSold'] = 2018 - df_train['YrSold']

#ガレージの築年数
df_all['GarageYrBlt'] = 2018 - df_all['GarageYrBlt']
df_train['GarageYrBlt'] = 2018 - df_train['GarageYrBlt']

#似た変数をまとめる
df_all['TotalHousePorchSF'] = df_all['EnclosedPorch'] + df_all['OpenPorchSF'] +df_all['3SsnPorch']+df_all['ScreenPorch']
df_train['TotalHousePorchSF'] = df_train['EnclosedPorch'] + df_train['OpenPorchSF'] +df_train['3SsnPorch']+df_train['ScreenPorch']
df_all['TotalHouseSF'] = df_all['1stFlrSF'] + df_all['2ndFlrSF'] + df_all['TotalBsmtSF']
df_train['TotalHouseSF'] = df_train['1stFlrSF'] + df_train['2ndFlrSF'] + df_train['TotalBsmtSF']

#Low Qualityな面積を全体の面積から引く
df_all['TotalHouseSFHighQuality'] = df_all['TotalHouseSF'] - df_all['LowQualFinSF']
df_train['TotalHouseSFHighQuality'] = df_train['TotalHouseSF'] - df_train['LowQualFinSF']

#相関係数行列の可視化
fig, ax = plt.subplots(1, 1, figsize=(30, 30))
sns.heatmap(df_train.corr(), vmax=1, vmin=-1, center=0, annot=True, ax=ax)

# SalePriceとの相関が低い変数・多重共線性を引き起こす変数を削除
df_all.drop([ 'YearBuilt', '1stFlrSF', 'GarageArea', 'TotRmsAbvGrd', 'GarageYrBlt', 'GarageCars', 'TotalHouseSF', 'TotalHouseSFHighQuality'], axis=1, inplace=True)

# データの欠損値
df_all.isnull().sum()[df_all.isnull().sum()>0]

#One Hot Encoding
df_all = pd.get_dummies(df_all)

#欠損値を平均値で補完
df_all = df_all.fillna(df_all.mean())

df_train["SalePrice"].hist(bins=30)

df_train["SalePrice"] = np.log1p(df_train["SalePrice"])
#対数変換後の分布を確認
df_train["SalePrice"].hist(bins=30)

#学習データ、テストデータに分割
X = df_all[:df_train.shape[0]]
X_for_test = df_all[df_train.shape[0]:]
y = df_train.SalePrice
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

reg = Lasso(alpha=0.0004)
reg.fit(X_train, y_train)

y_pred_lasso = reg.predict(X_test)

# データセットを生成
lgb_train = lgb.Dataset(X_train,y_train)

# LigthGBMのハイパーパラメータ
lgbm_params = {'task': 'train','boosting_type': 'gbdt','objective': 'regression','metric': {'l2'},'num_leaves': 256,
'learning_rate': 0.01,'num_iterations':2000,'feature_fraction': 0.4,'bagging_fraction': 0.7,'bagging_freq': 5}

# 上記のパラメータでモデルを学習
model = lgb.train(lgbm_params, lgb_train, num_boost_round=1500)

# テストデータを予測
y_pred_lgbm = model.predict(X_test, num_iteration=model.best_iteration)

#全データで学習
reg.fit(X, y)
lgb_train_full = lgb.Dataset(X,y)
model = lgb.train(lgbm_params, lgb_train_full, num_boost_round=1500)

# ラッソ・LightGBMの予測及びスタッキング
pred = np.expm1(reg.predict(X_for_test))
pred2 = np.expm1(model.predict(X_for_test))
pred3 = (pred*0.7+pred2*0.3)

submission = pd.DataFrame({"id":df_test.Id, "SalePrice":pred})
submission.to_csv("houseprice21.csv", index = False)
