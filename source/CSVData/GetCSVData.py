"""Class containing equations."""

import numpy as np
import random
import pandas as pd
# import os

def get_csv_training_and_holdout(
    csv_name : str
):
    """Calculate the travel time between two points including wait time at the end node."""
    # for reproduction
    s = 0
    random.seed(s)
    np.random.seed(s)

    # read in the data to pandas
    # print(os.listdir("source/CSVData/data/."))
    # print(os.listdir("PokemonPython-main/source/CSVData/data/."))
    RatioData = pd.read_csv("source/CSVData/data/" + csv_name)

    RatioData.describe()

    msk = np.random.rand(len(RatioData)) < 0.8
    train = RatioData[msk]
    holdout = RatioData[~msk]
    # check the number of records we'll validate our MSE with
    holdout.describe()
    # check the number of records we'll train our algorithm with
    train.describe()
    return train, holdout
