from processing import build_data
import pickle


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
    ])
train_x, train_ids, train_y, test_x, test_ids, test_y = data


word_feat = 'token'
word_stats = {}
langlist = ['en_es', 'fr_en', 'es_en']
for l in langlist:
    with open('data/'+l+'_wordwordfeats.txt', 'r') as f:
        for line in f.readlines():
            line = line.split(',')
            # add language identifier tag to end of word,
            # as is done in features
            word_stats[line[0].lower()+'_'+l[:2]] = {
                'frequency': float(line[2]),
                'levenshtein': int(line[3]),
                'leven_frac': float(line[4]),
                'aoa': float(line[5])
            }

for d in train_x + test_x:
    word = d[word_feat].lower()
    if word in word_stats:
        stats = word_stats[word]
        d['frequency'] = stats['frequency']
        d['levenshtein'] = stats['levenshtein']
        d['leven_frac'] = stats['leven_frac']
        d['aoa'] = stats['aoa']

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

data = (train_x, train_ids, train_y, test_x, test_ids, test_y)

max_bytes = 2**31 - 1

bytes_out = pickle.dumps(data)
with open('alldata.p', 'wb') as f_out:
    for idx in range(0, len(bytes_out), max_bytes):
        f_out.write(bytes_out[idx:idx+max_bytes])

