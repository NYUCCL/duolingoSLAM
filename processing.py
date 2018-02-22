"""
Classes and functions to load and preprocess duolingo data sets.

The main function is build_data, which takes train and test file names and
returns lists of dictionaries containing instance features, as well as lists of
instance id's and labels.

The feature set is build in a nested manner. First, a User object is created
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


def build_data(train_datafile, test_datafile, test_labelsfile=None,
               n_users=None, featurized=True):
    """
    This function loads and returns data and labels from a training and
    testfile, which are assumed to contain the same users. If a test

    Parameters:
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
    _load_file(train_datafile, users, False, n_users)
    _load_file(test_datafile, users, True, n_users)
    print('building features')
    for u in users.values():
        u.build_all()
    print('retrieving features')
    if featurized:
        train_x = [f for u in users.values() for f in u.to_features(False)]
        test_x = [f for u in users.values() for f in u.to_features(True)]
    else:
        # return list of user objects as train_x - can be useful for
        # development/testing
        train_x = list(users.values())
        test_x = None
    print('retrieving labels')
    train_ids = [l for u in users.values() for l in u.get_ids(False)]
    train_y = [l for u in users.values() for l in u.get_labels(False)]
    test_ids = [l for u in users.values() for l in u.get_ids(True)]
    if test_labelsfile:
        labels = dict()
        with open(test_labelsfile, 'rt') as f:
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

        test_y = [l for u in users.values() for l in u.get_labels(True)]
    else:
        test_y = None
    return train_x, train_ids, train_y, test_x, test_ids, test_y


# helper function for load_data; handles loading of a single data file
def _load_file(datafile, users, test, n_users):
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
                elif usernum == n_users and line_user not in users:
                    break
                elif line_user not in users:
                    users[line_user] = User(line_user)
                    usernum += 1
                users[line_user].add_exercise(exercise_lines, test)
                exercise_lines = []
            elif line[0] == "#":
                line_user = line[7:15]
                exercise_lines.append(line)
            else:
                exercise_lines.append(line)


class User:
    """
    This class stores all information relating to a single duolingo user,
    including all exercises and instances (i.e., words) completed.
    """
    def __init__(self, id):
        # attribute to store features for later analysis
        self.features = dict()
        self.id = id
        self.exercises = []
        self.instances = []
        self.features['user:' + self.id] = 1.0

    def add_exercise(self, textlist, test):
        # add new Exercise object to process chunk of input lines
        self.exercises.append(Exercise(textlist, self, test))

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

    def build_all(self):
        # create all higher-order features for this user, and its Exercises and
        # Instances. This function can be modified to build more features!
        pass


class Exercise:
    """
    This class processes and stores all information relating to a single
    Duolingo exercise
    """
    def __init__(self, textlist, user, test):
        # initialize with a list of lines, of which the first should contain
        # exercise information and the rest should contain instance
        # information, along with the creating User and whether this exercise
        # is in the test set.
        self.features = dict()
        self.user = user
        self.test = test
        self.instances = []
        for t in textlist[1:]:
            self.instances.append(Instance(t, self, user))
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
                    if value < 0:
                        value = None
            setattr(self, key, value)
        self.features['format:' + self.format] = 1.0

    def to_features(self):
        instance_features = [{**self.features, **i.to_features()}
                             for i in self.instances]
        return instance_features

    def propagate_labels(self, labels):
        for i in self.instances:
            i.propagate_labels(labels)


class Instance:

    """
    This class processes and stores all information relating to a single
    Duolingo instance (i.e., word)
    """
    def __init__(self, text, exercise, user):
        # initialized using the line of text, the creating Exercise, and the
        # creating User
        self.features = dict()
        self.exercise = exercise
        self.user = user
        user.add_instance(self)
        line = text.split()
        self.id = line[0]
        self.token = line[1]
        self.part_of_speech = line[2]
        self.morphological_features = dict()
        for l in line[3].split('|'):
            [key, value] = l.split('=')
            if key == 'Person':
                value = int(value)
            self.morphological_features[key] = value
        self.dependency_label = line[4]
        self.dependency_edge_head = int(line[5])
        if len(line) == 7:
            self.label = float(line[6])
        # add initial features to feature dictionary
        self.features['token'] = self.token.lower()
        self.features['part_of_speech:' + self.part_of_speech] = 1.0
        for morphological_feature in self.morphological_features:
            self.features['morphological_feature:' + morphological_feature] = 1.0
        self.features['dependency_label:' + self.dependency_label] = 1.0

    def to_features(self):
        return self.features

    def propagate_labels(self, labels):
        if self.id in labels:
            self.label = labels[self.id]
