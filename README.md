# Duolingo Shared Task on Second Language Acquisition Modeling

## Acquiring the data

Download from
[here](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/8SWHNO)
and unzip in the "data" folder

## Working on the project

Make sure you download the "new" data files, found in the dropbox link I posted on slack.

To run the model and test on the dev set, use `lightgbm_dev.py`. The gradient
boosting solution really works much better than the linear model right now. For
fastest training, use the `fr_en` data set since it's smallest, or adjust
`n_users` when calling `build_data`.

Unfortunately installing lightgbm isn't completely trivial: [Instructions are here.](http://lightgbm.readthedocs.io/en/latest/Installation-Guide.html#macos**
If there aren't any install errors then it isn't so bad. You'll need homebrew, which is here https://brew.sh/
If there are errors I'll try to help if I can!

**EDIT**: It looks like you might be able to install via pip without installing
from source. Try `pip install lightgbm`, or `pip install wheel` then `pip
install lightgbm`.

Hopefully if you can't install it then the linear model, found at `model_script.py`,
will also give you a general idea if added features are helping or not.

## To do

There are a few next steps that come to mind regarding the current approach (of
course, we could also create a whole new model like a neural network, but I
won't include that here).

For the ideas related to creating new features, this generally involves adding
new methods to the Exercise and/or Instance classes that calculate new features
and add them to the object's `features` dictionary, and then updating the User
class to iterate through the Exercises and call the new method in `build_all`.

### features about time course of exercises
It could be useful to look at the time course of exercises for a user, and
calculate something like number of exercises in last hour/day/week. Maybe people
need some time to "warm up" during a session. Maybe they also get tired if
they've done *too* many exercises recently. **Once you see how the current code
works this should be a pretty straight-forward one.**

### Features about text of the word itself.
Right now all the words a inserted into the model as categorical variables, but
perhaps added features like the number of letters in a word could be helpful.
This one could potentially be straightforward, though I could imagine going
further with this as well!

### add features on nearby words/related words
The probability of a word being mistaken is probably affected by nearby words in
some way.

You could add features to each instance based on the words
before or after, or the word's root word. I'm not sure what features would be
best here-maybe just the neighboring word's part of speach as a start? or its
morphological features?

**This kind of thing will likely be extra important for languages where word
gender needs to align, but solving this for gender in particular might be
tricky.**

### fiddling with lightgbm parameters
The parameters for the lightgbm haven't been fully optimized. There are a
[lot](http://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html) to choose
from. But it's kind of a rabbit hole.

By the way, if you want to look at the feature importances of a lightgbm model,
you can load the saved model and then use the `feature_importance()` and
`feature_names()` methods to find out the names of the features and how many
tree splits each feature was used in.
