import os.path
import sys
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lesion', default='none')
args = vars(parser.parse_args())
lesion_type = args['lesion']

max_bytes = 2**31 - 1

# read
bytes_in = bytearray(0)
input_size = os.path.getsize('alldata.p')
with open('alldata.p', 'rb') as f_in:
    for _ in range(0, input_size, max_bytes):
        bytes_in += f_in.read(max_bytes)
data = pickle.loads(bytes_in)

train_x, train_ids, train_y, test_x, test_ids, test_y = data


print('loaded data')

if lesion_type == 'none':
    print('using all features')
    cat_features = ['token', 'root', 'user',
                    'prev_token', 'next_token', 'parseroot_token']
elif lesion_type == 'neighbors':
    print('lesioning neighbor features')
    cat_features = ['token', 'root', 'user']
    for d in train_x + test_x:
        keys = [key for key in d]
        for key in keys:
            if key in ['prev_token', 'next_token', 'parseroot_token']:
                d.pop(key)
            if '_pos:' in key:
                d.pop(key)
elif lesion_type == 'external':
    print('lesioning external word features')
    cat_features = ['token', 'root', 'user',
                    'prev_token', 'next_token', 'parseroot_token']
    remove = ['frequency', 'levenshtein', 'leven_frac', 'aoa']
    for d in train_x + test_x:
        for key in remove:
            d.pop(key)
else:
    print('Unknown lesion type')
    sys.exit()


# put data in scipy sparse matrix
dv = DictVectorizer()
train_x_sparse = dv.fit_transform(train_x)
test_x_sparse = dv.transform(test_x)
names = dv.feature_names_

print('features:')
print(names)

print('built lesioned data set')

params = {
   'application': 'binary',
   'metric': 'auc',
   'learning_rate': .05,
   'num_leaves': 1024,
   'min_data_in_leaf': 100,
   'num_boost_round': 750,
   'cat_smooth': 200,
   'max_cat_threshold': 64,
   'feature_fraction': .7,
}


# train light gradient boosting machine model
d_train = lgb.Dataset(train_x_sparse, label=train_y)
d_valid = lgb.Dataset(test_x_sparse, label=test_y)
bst = lgb.train(params, d_train, valid_sets=[d_train, d_valid],
                valid_names=['train', 'valid'],
                feature_name=names,
                categorical_feature=cat_features,
                num_boost_round=params['num_boost_round'],
                early_stopping_rounds=50,
                verbose_eval=10)

test_predicted = bst.predict(test_x_sparse)
auc = roc_auc_score(test_y, test_predicted)
print("auc:", auc)

with open('lesion_auc_{}.txt'.format(lesion_type), 'w') as f:
    f.write('{}, {}\n'.format(lesion_type, auc))
