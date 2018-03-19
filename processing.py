"""
Classes and functions to load and preprocess duolingo data sets.

The main function is build_data, which takes train and test file names and
returns lists of dictionaries containing instance features, as well as lists of
instance id's and labels.

The feature set is builtt in a nested manner. First, a User object is created
when a new user id is found in the data, then Exercise objects are added as
more exercises for the user are found. The Exercise object in turn creates a
set of Instance objects corresponding to each word in the Exercise.

Features that are directly available from the data set are added to their
respective objects upon creation. Other features that need to be calculated
across Instances or Exercises can be created by adding more functions to the
Instance and Exercise objects. New features should be added to the respective
object's "features" attribute, and the User class's "build_all" method should
be augmented with code to call the needed functions.

After all features are built, the build_data function will call "to_features"
on each User, which will combine user-level, exercise-level, and instance-level
features and return a flat list of dictionaries representing all features for
each instance.
"""

from collections import OrderedDict
import numpy as np
import copy
import hashlib

from collections import Counter
from scipy.stats import entropy
from scipy.stats.stats import pearsonr

def build_data(language, train_datafiles, test_datafiles, labelfiles=[],
               n_users=None, featurized=True):
    """
    This function loads and returns data and labels from a training and
    testfile, which are assumed to contain the same users. If a test

    Parameters:
        language: 2-letter code for the language being learned.
        train_datafile: the file name of the training data file.
        test_datafile: the file name of the test data file.
        test_labelsfile (optional): the file name of the labels for the test
            data.
        n_users (optional): only collect data from first n_users users in the
            training/test files. Can be helpful for quicker testing.
        featurized (optional, default True): If True, return data features as a
            list of dictionaries, which can easily be fed into models. If
            False, return the list of User objects as the first return value.
            Can be helpful for development.
    Returns:
        train_x: list of dictionaries of training instance features
            (unless featurized=False).
        train_ids: list of instance id's for training instances.
        train_y: list of labels for training instances.
        test_x: see above.
        test_ids: see above.
        test_y: see above.
    """

    users = OrderedDict()
    print('loading data files')
    for tf in train_datafiles:
        _load_file(tf, users, False, n_users)
    for tf in test_datafiles:
        _load_file(tf, users, True, n_users)
    print('retrieving labels')
    for label_file in labelfiles:
        labels = dict()
        with open(label_file, 'rt') as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                else:
                    line = line.split()
                instance_id = line[0]
                label = float(line[1])
                labels[instance_id] = label
        for u in users.values():
            u.propagate_labels(labels)
    print('building features')
    for u in users.values():
        u.build_all()
    print('retrieving features')
    train_ids = [l for u in users.values() for l in u.get_ids(False)]
    train_y = [l for u in users.values() for l in u.get_labels(False)]
    test_ids = [l for u in users.values() for l in u.get_ids(True)]
    test_y = [l for u in users.values() for l in u.get_labels(True)]
    if featurized:
        train_x = [f for u in users.values() for f in u.to_features(False)]
        test_x = [f for u in users.values() for f in u.to_features(True)]
    else:
        # return list of user objects as train_x - can be useful for
        # development/testing
        train_x = list(users.values())
        test_x = None
    return train_x, train_ids, train_y, test_x, test_ids, test_y


# helper function for load_data; handles loading of a single data file
def _load_file(datafile, users, test, n_users):
    lang = datafile.split('/')[-1].split('_')[0][-2:]
    with open(datafile, 'r') as f:
        usernum = 0
        exercise_lines = []
        line_user = ''
        for line in f:
            line = line.strip()
            if len(line) == 0 and len(exercise_lines) == 0:
                continue
            elif len(line) == 0:
                if test and line_user not in users:  
                    break
                elif usernum == n_users and line_user not in users:  # stop if we get too many users
                    break
                elif line_user not in users:  # create a new user we encountered
                    users[line_user] = User(line_user, lang) 
                    usernum += 1
                users[line_user].add_exercise(exercise_lines, test) # add the exercise lines following the user record
                exercise_lines = [] # reset exercise lines
            elif line[0] == "#":
                # add language to end of user for combined training in case
                # some users are in multiple languages (don't know if this
                # actually occurs)
                line_user = line[7:15] + lang 
                exercise_lines.append(line)
            else:
                exercise_lines.append(line)


class User:
    """
    This class stores all information relating to a single duolingo user,
    including all exercises and instances (i.e., words) completed.
    """
    def __init__(self, id, lang):
        # attribute to store features for later analysis
        self.features = dict()
        self.id = id
        self.lang = lang
        self.exercises = []
        self.instances = []
        self.features['user'] = self.id
        self.features['lang'] = self.lang

    def add_exercise(self, textlist, test):
        # add new Exercise object to process chunk of input lines
        # each time this is called it is the entire block
        # for a particular entry
        self.exercises.append(Exercise(textlist, self, test, self.lang))

    def add_instance(self, instance):
        # link instance to current User object
        self.instances.append(instance)

    def to_features(self, test):
        # return flattened list of features from all Exercises/Instances
        chosen_exercises = [e for e in self.exercises if e.test == test]
        instance_features = [{**self.features, **i}
                             for e in chosen_exercises
                             for i in e.to_features()]
        return instance_features

    def get_labels(self, test):
        # return flattened list of labels from all Instances
        chosen_exercises = [e for e in self.exercises if e.test == test]
        labels = [i.label for e in chosen_exercises for i in e.instances]
        return labels

    def get_ids(self, test):
        # return flattened list of instance Ids
        chosen_exercises = [e for e in self.exercises if e.test == test]
        ids = [i.id for e in chosen_exercises for i in e.instances]
        return ids

    def propagate_labels(self, labels):
        # propagate dictionary of labels out to instances (used when labels
        # reside in a separate key file)
        for e in self.exercises:
            e.propagate_labels(labels)

    def compute_usage_entropy(self):
        # by Anselm:
        x_days = [e.days%1.0 for e in self.exercises]
        x_bins = [round(x * 24 * 3) for x in x_days]  # 20-minutes bins
        freq = Counter(x_bins)
        rel_freq = [freq[key]/len(x_bins) for key in freq]
        self.entropy = entropy(rel_freq, base=2)
        self.features['entropy'] = self.entropy

    def compute_motivation(self):
        # by Todd:
        x_days = [e.days for e in self.exercises]
        sessions = [[]]
        sindex = 0
        t_minus = x_days[0]
        sessions[sindex].append(t_minus)
        for t in x_days[1:]:
            if t-t_minus > (1./24.): # if more than a day passed
                sessions.append([])
                sindex+=1
            sessions[sindex].append(t)
            t_minus = t
        bursts = np.array([len(i) for i in sessions])
        self.features['burst_length'] = len(bursts)
        self.features['mean_burst_duration'] = np.mean(bursts)
        self.features['median_burst_duration'] = np.median(bursts)

    def build_all(self):
        # create all higher-order features for this user, and its Exercises and
        # Instances. This function can be modified to build more features!

        self.compute_usage_entropy()
        self.compute_motivation()

        self.n_train = sum([not e.test for e in self.exercises])
        self.n_test = sum([e.test for e in self.exercises])
        self.n_total = len(self.exercises)
        stats = []
        for e in self.exercises:
            e.build_temporal_stats(stats)
        for i, e in enumerate(self.exercises):
            e.encode_temporal_stats(stats, i)
        for e in self.exercises:
            e.set_others_pos()

        episode_counts = {}
        for e in self.exercises: # set a feature for the number of times the context has been repeated
            ex = e.instances
            words = ','.join([i.token for i in ex])
            hash_object = hashlib.md5(words.encode())
            mykey = hash_object.hexdigest()
            if mykey in episode_counts:
                episode_counts[mykey]+=1
            else:
                episode_counts[mykey]=1
            e.set_context(episode_counts[mykey])


class Exercise:
    """
    This class processes and stores all information relating to a single
    Duolingo exercise
    """
    def __init__(self, textlist, user, test, lang):
        # initialize with a list of lines, of which the first should contain
        # exercise information and the rest should contain instance
        # information, along with the creating User and whether this exercise
        # is in the test set.
        self.features = dict()
        self.user = user
        self.test = test
        self.instances = []
        self.textlist = textlist
        for t in textlist[1:]:
            self.instances.append(Instance(t, self, user, lang))
        line = textlist[0][2:].split()
        for parameter in line:
            [key, value] = parameter.split(':')
            if key == 'countries':
                value = value.split('|')
            elif key == 'days':
                value = float(value)
            elif key == 'time':
                if value == 'null':
                    value = None
                else:
                    assert '.' not in value
                    value = int(value)
                    if value < 1:
                        value = None
            if key != 'user':
                setattr(self, key, value)
        self.features['format:' + self.format] = 1.0
        self.features['days'] = self.days
        self.features['session:' + self.session] = 1.0
        self.features['client:' + self.client] = 1.0
        self.features['exercise_length'] = len(self.instances)
        if self.time is not None:
            self.features['time'] = self.time

    def set_others_pos(self):
        for idx, instance in enumerate(self.instances):
            if idx == 0:
                prev_inst = None
            else:
                prev_inst = self.instances[idx-1]
            if idx == len(self.instances)-1:
                next_inst = None
            else:
                next_inst = self.instances[idx+1]
            root_idx = instance.dependency_edge_head - 1
            if root_idx < 0 or root_idx >= len(self.instances):
                root_inst = None
            else:
                root_inst = self.instances[root_idx]
            instance.set_others_pos(prev_inst, next_inst, root_inst)

    def to_features(self):
        instance_features = [{**self.features, **i.to_features()}
                             for i in self.instances]
        return instance_features

    def propagate_labels(self, labels):
        for i in self.instances:
            i.propagate_labels(labels)

    def set_context(self, freq):
        setattr(self,'context_rep',freq)
        self.features['context_rep'] = freq

    def build_temporal_stats(self, stats):
        # stats holds a list of dictionaries. Each element in the list is a
        # snapshot of the learning history with every encountered word at a
        # specific exercise index
        idx = len(stats)
        if idx == 0:
            ex_stats = {}
            stats.append(ex_stats)
        else:
            ex_stats = copy.copy(stats[-1])
            stats.append(ex_stats)
        to_add = {}
        for i in self.instances:
            i.build_temporal_stats(to_add)
        err_lr_list = [.3, .1, .03, .01]
        for key, value in to_add.items():
            # create a new dictionary for each word that
            # occurs in the exercise
            if key not in ex_stats:
                ws = {
                    'erravg': [0] * 4,
                    'date': self.days,
                    'encounters': value['encounters'],
                    'idx': idx,
                }
                ex_stats[key] = ws
            else:
                ws = copy.deepcopy(ex_stats[key])
                ex_stats[key] = ws
                ws['date'] = self.days
                ws['encounters'] += value['encounters']
                ws['idx'] = idx
            # update error trackers
            for i, lr in enumerate(err_lr_list):
                if self.test:
                    lr = 0
                errnew = value['outcome'] / value['encounters']
                ws['erravg'][i] = (
                    ws['erravg'][i] +
                    lr * (errnew - ws['erravg'][i])
                )

    def encode_temporal_stats(self, stats, idx):
        self.features['exercise_num'] = idx
        for i in self.instances:
            i.encode_temporal_stats(stats, idx)

class Instance:
    """
    This class processes and stores all information relating to a single
    Duolingo instance (i.e., word)
    """
    def __init__(self, text, exercise, user, lang):
        # initialized using the line of text, the creating Exercise, and the
        # creating User
        self.label = None
        self.lang = lang
        self.features = dict()
        self.exercise = exercise
        self.user = user
        user.add_instance(self)
        line = text.split()
        self.id = line[0]
        self.token = line[1].lower() + '_' + lang
        self.root_token = line[2].lower() + '_' + lang
        self.part_of_speech = line[3]
        self.morphological_features = dict()
        for l in line[4].split('|'):
            [key, value] = l.split('=')
            self.morphological_features[key] = value
        self.dependency_label = line[5]
        self.dependency_edge_head = int(line[6])
        if len(line) == 8:
            self.label = float(line[7])
        # add initial features to feature dictionary
        self.features['token'] = self.token
        self.features['word_length'] = len(self.token)
        self.features['root'] = self.root_token
        self.features['part_of_speech:' + self.part_of_speech] = 1.0
        for key, value in self.morphological_features.items():
            self.features['morphological_feature:' + key + '_' + value] = 1.0
        self.features['dependency_label:' + self.dependency_label] = 1.0

    def to_features(self):
        return self.features

    def propagate_labels(self, labels):
        if self.id in labels:
            self.label = labels[self.id]

    def build_temporal_stats(self, to_add):
        keys = ['token:' + self.token, 'root:' + self.root_token]
        for key in keys:
            if key not in to_add:
                to_add[key] = {'outcome': 0.0, 'encounters': 0}
            ta = to_add[key]
            ta['encounters'] += 1
            if not self.exercise.test:
                ta['outcome'] += self.label
            else:
                ta['outcome'] += 0.0

    def encode_temporal_stats(self, stats, idx):
        # stats holds a list of dictionaries. Each element in the list is a
        # snapshot of the learning history with every encountered word at a
        # specific exercise index.
        # For creating features, we imagine that a random number of recent exercises
        # in the range [1, number of test items] were test items, and don't look at
        # the error information for those items.
        if self.exercise.test:
            last_labeled_idx = self.user.n_train - 1
        else:
            n_back = int(np.ceil(np.random.random() * (self.user.n_test-1)))
            last_labeled_idx = idx - n_back
        if last_labeled_idx < 0:
            last_labeled_idx = -1
            ex_stats_lab = {}
        else:
            ex_stats_lab = stats[last_labeled_idx]
        ex_stats = stats[max([idx - 1, 0])]
        keys = [('token:' + self.token, 'token'),
                ('root:' + self.root_token, 'root')]
        for key in keys:
            if key[0] in ex_stats:
                ws = ex_stats[key[0]]
                self.features[key[1] + ':encounters'] = ws['encounters']
                self.features[key[1] + ':time_since_last_encounter'] = (self.exercise.days - ws['date'])
                if key[0] in ex_stats_lab:
                    ws_lab = ex_stats_lab[key[0]]
                    self.features[key[1] + ':time_since_last_label'] = (self.exercise.days - ws_lab['date'])
                    self.features[key[1] + ':encounters_lab'] = ws_lab['encounters']
                    for i, err in enumerate(ws_lab['erravg']):
                        self.features[key[1] + ':erravg'+str(i)] = err
                    self.features[key[1] + ':encounters_unlab'] = ws['encounters'] - ws_lab['encounters']
                else:
                    self.features[key[1] + ':time_since_last_label'] = -99
                    self.features[key[1] + ':encounters_unlab'] = ws['encounters']
            else:
                self.features[key[1] + ':first_encounter'] = 1.0
                self.features[key[1] + ':time_since_last_encounter'] = -99
                self.features[key[1] + ':time_since_last_label'] = -99

    def set_others_pos(self, prev_inst, next_inst, root_inst):
        for inst, name in [(prev_inst, 'prev'), (next_inst, 'next'), (root_inst, 'parseroot')]:
            if inst is None:
                self.features[name + '_pos:None'] = 1.0
                self.features[name + '_token'] = '_NONE_'
            else:
                self.features[name + '_pos:' + inst.part_of_speech] = 1.0
                self.features[name + '_token'] = inst.token
