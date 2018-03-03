from processing import build_data
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import numpy as np
from scipy import sparse


# load data
data = build_data('en', 'data/data_en_es/en_es.slam.20171218.train.new',
                  'data/data_en_es/en_es.slam.20171218.dev.new',
                  'data/data_en_es/en_es.slam.20171218.dev.key', n_users=300)
train_x, train_ids, train_y, dev_x, dev_ids, dev_y = data

# store user id's separately
train_users = np.array([d.pop('user') for d in train_x])
dev_users = np.array([d.pop('user') for d in dev_x])

# put data in scipy sparse matrix
dv = DictVectorizer()
train_x_sparse = dv.fit_transform(train_x)
dev_x_sparse = dv.transform(dev_x)

# standardize predictor scales
scalable_preds = [i for i in dv.feature_names_ if '=' not in i]
scalable_preds_idx = [dv.vocabulary_[i] for i in scalable_preds]
pred_ranges = (train_x_sparse[:, scalable_preds_idx].max(axis=0).A -
               train_x_sparse[:, scalable_preds_idx].min(axis=0).A).flatten()
for i in range(len(pred_ranges)):
    if pred_ranges[i] == 0:
        pred_ranges[i] = 1
for i in range(len(pred_ranges)):
    train_x_sparse[:, scalable_preds_idx[i]] = (
        train_x_sparse[:, scalable_preds_idx[i]] / pred_ranges[i]
    )
    dev_x_sparse[:, scalable_preds_idx[i]] = (
        dev_x_sparse[:, scalable_preds_idx[i]] / pred_ranges[i]
    )

print('data shape:', train_x_sparse.shape)

# train linear model
print('training all-users linear')
model = LogisticRegression(C=.5)
model.fit(train_x_sparse, train_y)
train_df = model.decision_function(train_x_sparse)
dev_df = model.decision_function(dev_x_sparse)
dev_predicted = model.predict_proba(dev_x_sparse)[:, 1]
dev_predictions_df = pd.DataFrame({
    'instance': dev_ids,
    'prediction': dev_predicted
})
# print auc score
print('all-users auc:', roc_auc_score(dev_y, dev_predicted))

print('training per-user linear')
train_y = np.array(train_y)
train_x_withdf = sparse.hstack([train_x_sparse, 100*train_df[:, None]], format='csr')
dev_x_withdf = sparse.hstack([dev_x_sparse, 100*dev_df[:, None]], format='csr')
dev_predicted_indiv = np.zeros(dev_predicted.shape)
for user in np.unique(train_users):
    trainmask = (train_users == user)
    devmask = (dev_users == user)
    if sum(devmask) > 0:
        usermodel = LogisticRegression(C=.1)
        usermodel.fit(train_x_withdf[trainmask, :], train_y[trainmask])
        dev_predicted_indiv[devmask] = usermodel.predict_proba(dev_x_withdf[devmask, :])[:, 1]

print('per-user auc:', roc_auc_score(dev_y, dev_predicted_indiv))


