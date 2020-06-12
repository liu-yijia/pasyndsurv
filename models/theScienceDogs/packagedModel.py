#mcandrew

import sys
import numpy as np
import pandas as pd

class sciencedogsmodel(object):
    # Copy this section 
    def __init__(self,trainingdata):
        self.trainingdata = trainingdata
    
    # Copy this function 
    def makeForecasts(self):
        forecastsForAllRegions = pd.DataFrame()
        for fip in self.trainingdata.fips.unique():
            if np.isnan(fip):
                continue
            
            sys.stdout.write('\rForecasting FIPS = {:05d}\r'.format(int(fip)))
            sys.stdout.flush()
            
            regiondata = self.trainingdata[self.trainingdata.fips==fip]

            if regiondata.shape[0]==0:
                continue
            
            if np.all(np.isnan(regiondata.dohweb__numnewpos)):
                continue
            
            forecastData = self.makeForecastForOneRegion(regiondata)
            
            forecastData['fips'] = fip
            
            forecastsForAllRegions = forecastsForAllRegions.append(forecastData)
        return forecastsForAllRegions
    
    # -------------------------------This is your model code------------------------
    def makeForecastForOneRegion(self,regiondata):
        import scipy
        from sklearn.linear_model import LinearRegression

        # Make prediction 1 week into the future
        # Covariates needed for model
        covariates2predictCases = ['cdcili__ili','covidtracker__numnewpos']
        target = ['dohweb__numnewpos']

        # Setup "X" data
        Xvariables = regiondata.loc[:,['modelweek']+covariates2predictCases]
        Xvariables.loc[:,'modelweekPlusOne'] = Xvariables.modelweek.values+1
        Xvariables = Xvariables.drop(columns=['modelweek'])

        # extract target variable 
        targetVariable = regiondata.loc[:,['modelweek']+target]

        # build data for model
        XandTarget = Xvariables.merge(targetVariable, left_on=['modelweekPlusOne'], right_on = ['modelweek'])

        #build model
        XandTarget = XandTarget.dropna()
        model      = LinearRegression().fit(X=XandTarget[covariates2predictCases] , y= XandTarget[target])
        beta       = model.coef_[0]
        intercept  = model.intercept_

        # make probabilistic predictions
        mostRecentDataOnXandTarget = XandTarget.iloc[-1]

        # Linear regression assumes a normal distribution
        def computeSigma(XandTarget,target):
            predictions = model.predict(XandTarget[covariates2predictCases])
            target = XandTarget[target].values

            return sum((target - predictions)**2) / len(predictions)-1

        variance = computeSigma(XandTarget,target)
        mean = intercept + beta[0]*mostRecentDataOnXandTarget.cdcili__ili + beta[1]*mostRecentDataOnXandTarget.covidtracker__numnewpos

        # assign probabilities to the number of new cases
        predictiveDistCDF = scipy.stats.norm(loc=mean, scale=variance).cdf

        forecastData = {'numnewcases_leftbin':[],'numnewcases_rightbin':[],'prob':[]}

        stepsize=5
        for numOfNewCases in np.arange(0,10**3,stepsize):
            probability = float( predictiveDistCDF(numOfNewCases+stepsize) - predictiveDistCDF(numOfNewCases) )

            forecastData['numnewcases_leftbin'].append(numOfNewCases)
            forecastData['numnewcases_rightbin'].append(numOfNewCases+stepsize)
            forecastData['prob'].append(probability)
        forecastData = pd.DataFrame(forecastData)

        # i restricted the range of values to forecast between 0 and 1000.
        forecastData['prob'] = forecastData.prob/forecastData.prob.sum()

        return forecastData  

if __name__ == "__main__":

    # test packaged model

    # read data
    data = pd.read_csv('../../data/cases/PATrainingDataCases.csv')

    # subset to specific training week and use only epidemic weeks in 2020
    specificTrainingWeek = data[(data.trainingweek==202020) & (data.epiweek>201952)]

    # Startup model
    forecastingModel = sciencedogsmodel( specificTrainingWeek )

    # Forecast for all regions
    forecastsForAllRegions = forecastingModel.makeForecasts()
