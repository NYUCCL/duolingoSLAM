import os
from processing import build_data
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lang', default='en_es')
parser.add_argument('--users', default='100')

# use this to change language pair trained on
args = vars(parser.parse_args())
lang = args['lang']
users = args['users']
if users == 'all':
    n_users = None
else:
    n_users = int(users)
print('using ' + lang + ' dataset, ' + users + ' users')
# lightgbm parameters for each model. Different ones might be better for
# different language pairs
params = {
    'fr_en': {
        'application': 'binary',
        'metric': 'auc',
        'learning_rate': .1,
        'num_leaves': 64,
        'min_data_in_leaf': 20,
        'num_boost_round': 600,
        'cat_smooth': 100,
    },
    'en_es': {
        'application': 'binary',
        'metric': 'auc',
        'learning_rate': .1,
        'num_leaves': 64,
        'min_data_in_leaf': 20,
        'num_boost_round': 600,
        'cat_smooth': 100,
    },
    'es_en': {
        'application': 'binary',
        'metric': 'auc',
        'learning_rate': .1,
        'num_leaves': 64,
        'min_data_in_leaf': 20,
        'num_boost_round': 600,
        'cat_smooth': 100,
    },
    'all': {
        'application': 'binary',
        'metric': 'auc',
        'learning_rate': .1,
        'num_leaves': 128,
        'min_data_in_leaf': 20,
        'num_boost_round': 1000,
        'cat_smooth': 100,
        'max_cat_threshold': 64
    }
}

# load data
if lang == 'all':
    data = build_data(
        'all',
        [
            'data/data_{0}/{0}.slam.20171218.train.new'.format('en_es'),
            'data/data_{0}/{0}.slam.20171218.train.new'.format('fr_en'),
            'data/data_{0}/{0}.slam.20171218.train.new'.format('es_en')
        ],
        [
            'data/data_{0}/{0}.slam.20171218.dev.new'.format('en_es'),
            'data/data_{0}/{0}.slam.20171218.dev.new'.format('fr_en'),
            'data/data_{0}/{0}.slam.20171218.dev.new'.format('es_en')
        ],
        labelfiles=[
            'data/data_{0}/{0}.slam.20171218.dev.key'.format('en_es'),
            'data/data_{0}/{0}.slam.20171218.dev.key'.format('fr_en'),
            'data/data_{0}/{0}.slam.20171218.dev.key'.format('es_en')
        ],
        n_users=n_users)
else:
    data = build_data(
        lang[:2],
        ['data/data_{0}/{0}.slam.20171218.train.new'.format(lang)],
        ['data/data_{0}/{0}.slam.20171218.dev.new'.format(lang)],
        labelfiles=['data/data_{0}/{0}.slam.20171218.dev.key'.format(lang)],
        n_users=n_users)
train_x, train_ids, train_y, test_x, test_ids, test_y = data


cat_features = ['token', 'root', 'user',
                'prev_token', 'next_token', 'parseroot_token']
for key in cat_features:
    val_dict = {}
    val_idx = 0
    for d in train_x + test_x:
        t = d[key]
        if t in val_dict:
            d[key] = val_dict[t]
        else:
            val_dict[t] = val_idx
            d[key] = val_idx
            val_idx += 1

# put data in scipy sparse matrix
dv = DictVectorizer()
train_x_sparse = dv.fit_transform(train_x)
test_x_sparse = dv.transform(test_x)

names = dv.feature_names_

# train light gradient boosting machine model
d_train = lgb.Dataset(train_x_sparse, label=train_y)
d_valid = lgb.Dataset(test_x_sparse, label=test_y)
bst = lgb.train(params[lang], d_train, valid_sets=[d_train, d_valid],
                valid_names=['train', 'valid'],
                feature_name=names,
                categorical_feature=cat_features,
                num_boost_round=params[lang]['num_boost_round'],
                verbose_eval=10)
if not os.path.exists('models'):
    os.makedirs('models')
bst.save_model('models/dev.{}.bst'.format(lang))
test_predicted = bst.predict(test_x_sparse)
# print auc score
print(roc_auc_score(test_y, test_predicted))
test_predictions_df = pd.DataFrame({
    'instance': test_ids,
    'prediction': test_predicted
})
test_predictions_df.to_csv('dev.{}.pred'.format(lang), header=False,
                           index=False, sep=" ")
