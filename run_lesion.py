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

elif lesion_type == 'word':
    print('lesioning word feats')
    cat_features = ['user']
    remove = ['token', 'root', 'prev_token', 'next_token', 'parseroot_token']
    for d in train_x + test_x:
        keys = [key for key in d]
        for key in keys:
            if '_pos:' in key:
                d.pop(key)
            elif 'morphological_feature' in key:
                d.pop(key)
            elif 'dependency_label' in key:
                d.pop(key)
            elif 'part_of_speech' in key:
                d.pop(key)
            elif key in remove:
                d.pop(key)
elif lesion_type == 'word_ids':
    print('lesioning word ids')
    cat_features = ['user']
    remove = ['token', 'root', 'prev_token', 'next_token', 'parseroot_token']
    for d in train_x + test_x:
        for key in remove:
            d.pop(key, None)
elif lesion_type == 'word_otherfeats':
    print('lesioning non-id word features')
    cat_features = ['token', 'root', 'user',
                    'prev_token', 'next_token', 'parseroot_token']
    for d in train_x + test_x:
        keys = [key for key in d]
        for key in keys:
            if '_pos:' in key:
                d.pop(key)
            elif 'morphological_feature' in key:
                d.pop(key)
            elif 'dependency_label' in key:
                d.pop(key)
            elif 'part_of_speech' in key:
                d.pop(key)
elif lesion_type == 'external':
    print('lesioning external word features')
    cat_features = ['token', 'root', 'user',
                    'prev_token', 'next_token', 'parseroot_token']
    remove = ['frequency', 'levenshtein', 'leven_frac', 'aoa']
    for d in train_x + test_x:
        for key in remove:
            d.pop(key, None)
elif lesion_type == 'user':
    print('lesioning user features')
    cat_features = ['token', 'root',
                    'prev_token', 'next_token', 'parseroot_token']
    remove = ['user', 'entropy', 'burst_length', 'mean_burst_duration', 'median_burst_duration']
    for d in train_x + test_x:
        for key in remove:
            d.pop(key)
elif lesion_type == 'user_id':
    print('lesioning user id')
    cat_features = ['token', 'root',
                    'prev_token', 'next_token', 'parseroot_token']
    for d in train_x + test_x:
        d.pop('user')
elif lesion_type == 'user_otherfeats':
    print('lesioning user other features')
    cat_features = ['token', 'root', 'user',
                    'prev_token', 'next_token', 'parseroot_token']
    remove = ['entropy', 'burst_length', 'mean_burst_duration', 'median_burst_duration']
    for d in train_x + test_x:
        for key in remove:
            d.pop(key)
elif lesion_type == 'temporal':
    print('lesioning temporal features')
    cat_features = ['token', 'root', 'user',
                    'prev_token', 'next_token', 'parseroot_token']
    key_starters = ['token:', 'root:']
    key_enders = ['encounters', 'time_since_last_encounter', 'time_since_last_label',
                  'encounters_lab', 'encounters_unlab', 'first_encounter',
                  'erravg0', 'erravg1', 'erravg2', 'erravg3']
    for d in train_x + test_x:
        for s in key_starters:
            for e in key_enders:
                d.pop(s+e, None)
elif lesion_type == 'ids':
    print('lesioning ids')
    cat_features = []
    remove = ['token', 'root', 'user',
                    'prev_token', 'next_token', 'parseroot_token']
    for d in train_x + test_x:
        for key in remove:
            d.pop(key, None)
elif lesion_type == 'nonids':
    print('lesioning all but ids')
    cat_features = ['token', 'root', 'user',
                    'prev_token', 'next_token', 'parseroot_token']
    for d in train_x + test_x:
        keys = [key for key in d]
        for key in keys:
            if key not in cat_features:
                d.pop(key)
elif lesion_type == 'exercise':
    print('lesioning exercise features')
    cat_features = ['token', 'root', 'user',
                    'prev_token', 'next_token', 'parseroot_token']
    for d in train_x + test_x:
        keys = [key for key in d]
        for key in keys:
            if 'client' in key:
                d.pop(key)
            elif 'format' in key:
                d.pop(key)
            elif 'session' in key:
                d.pop(key)
            elif key == 'time':
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
