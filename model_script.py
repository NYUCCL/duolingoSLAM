from processing import build_data
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import lightgbm as lgb


# load data
data = build_data('data/data_en_es/en_es.slam.20171218.train.new',
                  'data/data_en_es/en_es.slam.20171218.dev.new',
                  'data/data_en_es/en_es.slam.20171218.dev.key', n_users=None)
train_x, train_ds, train_y, dev_x, dev_ids, dev_y = data

# put data in scipy sparse matrix
dv = DictVectorizer()
train_x_sparse = dv.fit_transform(train_x)
dev_x_sparse = dv.transform(dev_x)

# train linear model
model = LogisticRegression(C=.1)
model.fit(train_x_sparse, train_y)
dev_predicted = model.predict_proba(dev_x_sparse)[:, 1]
dev_predictions_df = pd.DataFrame({
    'instance': dev_ids,
    'prediction': dev_predicted
})
# print auc score
print(roc_auc_score(dev_y, dev_predicted))
dev_predictions_df.to_csv('sklearn_logreg.pred',
                          header=False, index=False, sep=" ")

# train light gradient boosting machine model
d_train = lgb.Dataset(train_x_sparse, label=train_y)
d_valid = lgb.Dataset(dev_x_sparse, label=dev_y)
params = {
    'application': 'binary',
    'metric': 'auc',
    'learning_rate': .1,
    'num_leaves': 128,
    'min_data_in_leaf': 20,
}
bst = lgb.train(params, d_train, valid_sets=[d_train, d_valid],
                valid_names=['train', 'valid'],
                num_boost_round=1000, verbose_eval=10)
dev_predicted = bst.predict(dev_x_sparse)
# print auc score
print(roc_auc_score(dev_y, dev_predicted))
dev_predictions_df = pd.DataFrame({
    'instance': dev_ids,
    'prediction': dev_predicted
})
dev_predictions_df.to_csv('lightgbm.pred', header=False, index=False, sep=" ")
