from starter_code.baseline import load_data
from starter_code.eval import load_labels
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import lightgbm as lgb


# load data using starter code
train_data, train_labels = load_data('data/data_en_es/en_es.slam.20171218.train')
dev_data, dev_labels = (load_data('data/data_en_es/en_es.slam.20171218.dev'),
                        load_labels('data/data_en_es/en_es.slam.20171218.dev.key'))

# put data in scipy sparse matrix, put labels in list
dv = DictVectorizer()
train_data_dicts = [i.to_features() for i in train_data]
train_data_dicts = dv.fit_transform(train_data_dicts)
train_labels_list = [train_labels[i.instance_id] for i in train_data]
dev_data_dicts = [i.to_features() for i in dev_data]
dev_data_dicts = dv.transform(dev_data_dicts)
dev_labels_list = [dev_labels[i.instance_id] for i in dev_data]

# train linear model
model = LogisticRegression(C=.1)
model.fit(train_data_dicts, train_labels_list)
dev_predicted = model.predict_proba(dev_data_dicts)[:, 1]
dev_predictions_df = pd.DataFrame({
    'instance': [i.instance_id for i in dev_data],
    'prediction': dev_predicted
})
# print auc score
print(roc_auc_score(dev_labels_list, dev_predicted))
dev_predictions_df.to_csv('sklearn_logreg.pred',
                          header=False, index=False, sep=" ")

# train light gradient boosting machine model
d_train = lgb.Dataset(train_data_dicts, label=train_labels_list)
d_valid = lgb.Dataset(dev_data_dicts, label=dev_labels_list)
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
dev_predicted = bst.predict(dev_data_dicts)
# print auc score
print(roc_auc_score(dev_labels_list, dev_predicted))
dev_predictions_df = pd.DataFrame({
    'instance': [i.instance_id for i in dev_data],
    'prediction': dev_predicted
})
dev_predictions_df.to_csv('lightgbm.pred', header=False, index=False, sep=" ")
