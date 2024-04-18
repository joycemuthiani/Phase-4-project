#%%
'''
    It fill the missing rating using MICE (Multivariate Imputation by Chained Equations)
    imputer method with RF Regressor as an estimator.
    
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

ratings = pd.read_csv('data/raw/ratings.csv')
movies = pd.read_csv('data/raw/movies.csv')

df = pd.pivot_table(ratings, values='rating', index='movieId', columns='userId')


users_rating = ratings.groupby('movieId')['rating'].agg('count')
movies_rating = ratings.groupby('userId')['rating'].agg('count')

f,ax = plt.subplots(1,1,figsize=(16,4))
plt.scatter(users_rating.index,users_rating,color='green')
plt.axhline(y=10,color='r')
plt.xlabel('MovieId')
plt.ylabel('No. of users voted')
plt.show()

df_m = df.loc[users_rating[users_rating > 10].index,:]
df_m # The number of movies reduced from 9724 to 2121

f,ax = plt.subplots(1,1,figsize=(16,4))
plt.scatter(movies_rating.index,movies_rating,color='green')
plt.axhline(y=50,color='r')
plt.xlabel('UserId')
plt.ylabel('No. of votes by user')
plt.show()

users_threshold = movies_rating[movies_rating > 50]
df_u = df_m.loc[:,movies_rating[movies_rating > 50].index]
df_u # the number of users reduced from 610 to 378
df_final = df_u.copy()

imp = IterativeImputer(estimator=RandomForestRegressor(), 
                               initial_strategy='mean',
                               max_iter=10, tol=1e-10, random_state=0)
df_filled = imp.fit_transform(df_final)
df_filled

df_clean= pd.DataFrame(data=df_filled[0:,0:],
            index=[i for i in range(df_filled.shape[0])],
            columns=['U_'+str(i) for i in range(df_filled.shape[1])])

df_clean.index = df_filled.index

df_final= df_clean.T

df_final.to_csv('data/preprocessed/df_final.csv', index= False)