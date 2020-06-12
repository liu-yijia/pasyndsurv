#mcandrew, Team Member's Name 1, Team Member's Name 2, Team Member's Name 3

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":

    # pull data from our git repo into python's local memory
    allData = pd.read_csv("../../data/cases/PATrainingDataCases.csv")

    # for now, lets subset our data to the most recent training week and a single county
    mostrecentweek = allData.trainingweek.max()
    singleCounty = allData[ (allData.fips == 42095) & (allData.trainingweek==mostrecentweek)  ]

    # Looks like a few NANS show up in the data. I'm not sure why yet, but those details can be sorted out after we
    # have our model running. For now, let's set NAN to 0
    singleCounty = singleCounty.replace(np.nan,0.)

    # ---------------------------------------------------------------------------
    # TRAIN MODEL Below
    # Your model should take in data from Single County and output forecasts (predictions) for at least 1 week ahead. 
    # ---------------------------------------------------------------------------
 
