import logging
import pandas as pd
import pickle
from time import sleep
from sklearn.decomposition import NMF

logging.basicConfig(#filename='RecommenderLog.log',
                    format='%(asctime)s:%(levelname)s: %(message)s')


def update_model(df):

    R = pd.DataFrame(df, index=df.index, columns=df.columns).values


    model = NMF(n_components=126, init='random', random_state=1, max_iter=100000, solver='cd', alpha_W=0.01, alpha_H=0.01, l1_ratio=0.5, verbose=True)

    fit_model = model.fit(R)

    return fit_model, R 

if __name__ == '__main__':

    while True:
        df_final = pd.read_csv('../data/preprocessed/df_final.csv')
        nmf, R_nmf = update_model(df_final)
        with open('models/NMF_model.pickle','wb') as f:
            pickle.dump(nmf, f)
        #logging.warning('New version of the NMF trained model saved in the "models" folder.')
        with open('models/NMF_R.pickle','wb') as f2:
            pickle.dump(R_nmf, f2)
        #logging.warning('New version of the R matrix for the NMF model saved in the "models" folder.')

        sleep(60*60*12)


#%%
