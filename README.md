# Duolingo Shared Task on Second Language Acquisition Modeling

This repository contains code for running the 2nd place (Spanish-to-English) and
3rd place (English-to-Spanish and French-to-English) model in the 
[Duolingo SLAM competition](http://sharedtask.duolingo.com/). 
The paper describing our approach can be found [here](https://psyarxiv.com/r93wc/).

## Acquiring the data

Download from
[here](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/8SWHNO)
and unzip in the "data" folder

## Running the model

To preprocess the data, run `reprocess_syntax.py` on each data file. See the
file's docstring for more details on getting google SyntaxNet set up. Then run
`translate_frequency.py` to generate external word-frequency features.

The model can then be trained to produce predictions on the `dev` set using
`lightgbm_dev.py` or on the `test` set using `lightgbm_script.py`. The language
trained on (`en_es`, `fr_en`, `es_en`, or `all`) and the number of user trained
on can be controlled using the `--lang` and `--users` flags.

Models trained on each individual language can be averaged with a model trained
on all languages using the `average_models.py` script.

## Testing model lesions

To test the effects of removing different feature sets, first run
`preprocess_to_pickle.py` to create a pickled version of the data and cut down
on preprocessing time across different lesions. Then run `run_lesion.py`, using
the `--lesion` flag to choose the lesion experiment to conduct. See code or
paper for list of options.

The results of the lesions can be plotted using `graph_lesions.r` (in R, not python).
