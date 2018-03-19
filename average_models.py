import numpy as np
import pandas as pd


def logistic(x):
    return 1 / (1 + np.exp(-x))


def logit(x):
    return np.log(x/(1-x))


df_all = pd.read_csv('test.all.pred', sep=' ', header=None, names=['id', 'proball'])
langs = ['fr_en', 'en_es', 'es_en']
for lang in langs:
    df = pd.read_csv('test.'+lang+'.pred', sep=' ', header=None, names=['id', 'prob'])
    df = df.merge(df_all, on='id')
    df['probcomb'] = logistic((.5*logit(df.prob)+.5*logit(df.proball)))
    df[['id', 'probcomb']].to_csv('test.'+lang+'.predcomb', header=False,
                                  index=False, sep=" ")
