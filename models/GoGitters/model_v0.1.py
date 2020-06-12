#mcandrew, Eunice and Michael

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.arima_model import ARIMA # note the import here (you may need to pip3 install statsmodel)

if __name__ == "__main__":

    # pull data from our git repo into python's local memory
    allData = pd.read_csv("../../data/cases/PATrainingDataCases.csv")

    # for now, lets subset our data to the most recent training week and a single county
    mostrecentweek = allData.trainingweek.max()
    singleCounty = allData[ (allData.fips == 42091) & (allData.trainingweek==mostrecentweek)  ]

    # Looks like a few NANS show up in the data. I'm not sure why yet, but those details can be sorted out after we
    # have our model running. For now, let's set NAN to 0
    singleCounty = singleCounty.replace(np.nan,0.)

    # ---------------------------------------------------------------------------
    # TRAIN MODEL Below
    # Your model should take in data from Single County and output forecasts (predictions) for at least 1 week ahead. 
    # ---------------------------------------------------------------------------

    # Documentation for ARIMA from Statmodels is here
    # https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima_model.ARIMA.html
    
    trainingData = singleCounty[['modelweek','dohweb__numnewpos']].set_index('modelweek')
    
    GoGittersModel = ARIMA( trainingData, order=(2,0,0) )
    Fitted_GoGittersModel = GoGittersModel.fit()

    # predictions for the next 4 weeks
    lastWeekOfData = trainingData.index.max()
    forecasts = Fitted_GoGittersModel.predict(start=len(trainingData),end=len(trainingData)+4) # some issues will arise here when we are in december

    fig,ax = plt.subplots()
    
    # Lets plot the model weeks (the training data index) against our true number of cases.
    ax.plot(trainingData,color='b', label = "True cases") 

    # The ARIMA model outputs estimated values (called fitted values) of the true number of COVID cases for the time period we observed data.
    # Let's see what they look like.
    ax.plot(Fitted_GoGittersModel.fittedvalues,color='orange',label="Estimated")

    # These are our predictions, estimates for values not yet observed.
    ax.plot( np.arange(trainingData.index.max(), trainingData.index.max()+4+1) , forecasts, color='green', label = "Prediction") 

    ax.legend() # add in a legend

    ax.set_xlabel("Model weeks (the number of epidemic weeks from 1970W01)") # add an xlabel
    ax.set_ylabel("The number of new COVID-19 cases") # add in a ylabel too. Who dosn't love a good ylabel?
    
    plt.show() # show us our awesomee work!
   

    # TODO
    # The predictions in green are POINT estimates. We need to output, for every week ahead, a probability distribution over the number of new cases.
    # Let's discuss after you spend some time looking at the literature and at home the ARIMA model may be able to output a prob dist. 


    
