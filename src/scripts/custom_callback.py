import sys
import optuna
from tqdm import tqdm
from functools import partial
import datetime

class TqdmCallback:
    def __init__(self, tqdm_object, metric_name="value"):
        self._tqdm = tqdm_object
        self._metric_name = metric_name
        
    def __call__(self, study, trial):
        self._tqdm.update(1)
        self._tqdm.set_postfix({self._metric_name: trial.value})
