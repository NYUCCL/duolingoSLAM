# Duolingo SLAM Shared Task

This archive contains a baseline model and evaluation script for Duolingo's 2018 Shared Task on Second Language Acquisition Modeling (SLAM). 
The model is L2-regularized logistic regression, trained with SGD weighted by frequency.   

## Setup

This baseline model is written in Python. It depends on the `future` library for compatibility with both Python 2 and 3,
which on many machines may be obtained by executing `pip install future` in a console.

In order to run the baseline model and evaluate your predictions, perform the following:

* Download and extract the contents of the file to a local directory.
* To train the baseline model: 
  * Open a console and `cd` into the directory where `baseline.py` is stored
  * Execute: 
    
    ```bash
    python baseline.py --train path/to/train/data.train 
                       --test path/to/dev_or_test/data.dev
                       --pred path/to/dump/predictions.pred
    ``` 
    to create predictions for your chosen track. Note that we use `test` interchangeably for the dev and test sets because both are test sets.
* To evaluate the baseline model:
  * Execute     
  
    ```bash
    python eval.py --pred path/to/your/predictions.pred
                   --key path/to/dev_or_test/labels.dev.key
    ```
    to print a variety of metrics for the baseline predictions to the screen.