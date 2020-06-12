#mcandrew

import sys
sys.path.append('../')

import numpy as np
import pandas as pd

import os
from glob import glob

from downloadHelper.downloadtools import timestamp
 
class dataSource(object):
    datasource=np.nan
    def __init__(self,datafolder,region):
        self.datafolder = datafolder
        self.region     = region
        self.data       = self.pullMostRecentData()
        
        self.clean()
        
        self.formatDate()
        
        self.addEpiWeek().addModelWeek()
        self.data['datasource'] = self.datasource

    def clean(self):
        pass

    def addFractionOfPositives(self):
        data = self.data
        data['fractionPos'] = 1.*data.newPos/(data.newPos + data.newNeg)
        self.data = data
        
    def pullMostRecentData(self):
        files = glob(os.path.join(self.datafolder,"*csv"))
        dt= pd.to_datetime('1970-01-01')
        for n,fil in enumerate(files):
            timestamp = pd.to_datetime( (fil.split('_')[-1].split('.')[0]) )
            if timestamp > dt:
                dt=timestamp
                index_of_most_recent_file = n
        data = pd.read_csv(files[n])

        fldr = self.datafolder.split('/')[1]
        print("{:s} data as of {:04d}-{:02d}-{:02d}-{:02d}".format(fldr,timestamp.year,timestamp.month,timestamp.day,timestamp.hour))
        return data

    def addDataSource2variables(self,vars):
        data = self.data
        fromOldName2NewName = {}
        for var in vars:
            fromOldName2NewName[var] = "{:s}__{:s}".format(self.datasource,var)
        data.rename(columns = fromOldName2NewName, inplace=True)
        self.data = data

    def has_multiple_regions(self):
        return True if len(self.data.region.unique())>1 else False

    def formatDate(self):
        self.data['date'] = pd.to_datetime(self.data.date.astype('str'))

    def addEpiWeek(self):
        if 'epiweek' in self.data:
            self.data.epiweek = self.data.epiweek.astype(str)
            return self
        from epiweeks import Week, Year
        epiweeks = []
        for dt in self.data.date:
            yr,mnth,day = dt.year,dt.month,dt.day
            epiweek = Week.fromdate(yr,mnth,day)
            epiweeks.append( "{:04d}{:02d}".format(epiweek.year,epiweek.week) )
        self.data['epiweek'] = epiweeks
        return self

    def fromEpiWeek2ModelWeek(self,week):
        from epiweeks import Week, Year
        if type(week) is Week:
            pass
        elif type(week) is str:
            week = Week(int(week[:4]),int(week[4:]))
        elif type(week) is int:
            week = str(week)
            week = Week(int(week[:4]),int(week[4:]))
            
        numWeeks=0
        w = Week(1970,1)
        while True:
            if w.year < week.year:
                numWeeks+=Year(w.year).totalweeks
                w = Week(w.year+1,1)
            else:
                break
        while w < week:
            numWeeks+=1
            w+=1
        return numWeeks
    
    def addModelWeek(self):
        d = self.data
        modelWeeks = []
        for n,week in enumerate(d.epiweek):
            modelWeeks.append(self.fromEpiWeek2ModelWeek(week))
        d['modelweek'] = modelWeeks
        self.data = d

    class MultipleRegionsError(Exception):
        pass
    
class covidtrackermanag(dataSource):
    datasource='covidtracker'
    def clean(self):
        self.data = self.data[['date','state','positiveIncrease','negativeIncrease','totalTestResultsIncrease']]
        
        fromOldName2NewName = {}
        for oldname,newname in zip(['positiveIncrease','negativeIncrease','totalTestResultsIncrease','state']
                                  ,['numnewpos','numnewneg','numnewtest','region']):
            fromOldName2NewName[oldname] = "{:s}".format(newname)
        self.data.rename(columns = fromOldName2NewName, inplace=True)
        
        if self.has_multiple_regions():
            raise self.MultipleRegionsError("Multiple regions in this dataset.")
 
        self.data['region'] = self.region.lower()
        
        self.formatDate()

    def groupByWeek(self):
        def addUpColumns(subset):
            return subset[['numnewpos','numnewneg','numnewtest']].replace(np.nan,0).apply(sum,0)
        self.data = self.data.groupby(['epiweek','modelweek','region']).apply(addUpColumns).reset_index()
        return self

class jhuCSSEmanag(dataSource):
    datasource='jhucsse'
    def clean(self):
        d = self.data
        self.data.rename(columns= {'Province_State':'region'},inplace=True)

        if self.has_multiple_regions():
            raise self.MultipleRegionsError("Multiple regions in this dataset.")
        self.data['region'] = self.region.lower()

    def addNumNewPos(self):
        def addNN(x):
            x = x.sort_values('date')    
            x['numnewpos'] = x['count'].diff()
            x=x.replace(np.nan,0)
            return x.drop(columns=['FIPS'])
        d = self.data.groupby(['FIPS']).apply(addNN).reset_index()
        self.data = d
        
    def groupByWeek(self):
        self.data = self.data.groupby(['epiweek','modelweek','region','FIPS']).apply( lambda x: pd.Series({'numnewpos':sum(x.numnewpos)}) ).reset_index()
        return self
        
class cdcILImanag(dataSource):
    datasource = "cdcfluview"
    def clean(self):
        if self.has_multiple_regions():
            raise self.MultipleRegionsError("Multiple regions in this dataset.")
    def formatDate(self):
        pass

class dohWebsiteManag(dataSource):
    datasource='dohwebsite'
    def clean(self):
        pass

    def groupByWeek(self):
        self.data = self.data.groupby(['epiweek','modelweek','region','fips']).apply( lambda x: pd.Series({'numnewpos':sum(x.numnewpos)}) ).reset_index()
        return self

def addDataSource2variables(d,datasource,vrs):
    fromOldName2NewName = {}
    for var in vrs:
        fromOldName2NewName[var] = "{:s}__{:s}".format(datasource,var)
    d = d.rename(columns = fromOldName2NewName)
    return d
    
if __name__ == "__main__":

    #-----------------------------------------------------------------------------------------------------------------
    # JHU
    jhuCSSE = jhuCSSEmanag("./jhuCSSE",'PA')
    
    jhuCSSE.addNumNewPos()
    jhuCSSE.groupByWeek()
    
    jhuData = jhuCSSE.data
    jhuData = addDataSource2variables(jhuData,"jhucsse",['numnewpos'])
    jhuData = jhuData.rename(columns={'FIPS':'fips'})
    #-----------------------------------------------------------------------------------------------------------------
    # covidtracker
    covidtracker = covidtrackermanag("./covidtracking",'PA')
    covidtracker.groupByWeek()
    covidtrackerData = covidtracker.data

    covidtrackerData = addDataSource2variables(covidtrackerData,"covidtracker",['numnewpos','numnewneg','numnewtest'])
    #-----------------------------------------------------------------------------------------------------------------
    # ili
    cdcILI       = cdcILImanag("./CDCili",'PA')
    cdcILIdata   = cdcILI.data
    cdcILIdata   = cdcILIdata.drop(columns=['lag','release_date','datasource'])

    cdcILIdata   = addDataSource2variables(cdcILIdata,"cdcili",['num_patients','num_providers','wili','ili'])
    #-----------------------------------------------------------------------------------------------------------------
    # Website DOH data
    dohWebsiteData = dohWebsiteManag("./LehighDOHdata/","PA")
    dohWebsiteData.groupByWeek()
    
    dohWebsiteData   = addDataSource2variables(dohWebsiteData.data,"dohweb",['numnewpos'])
    #-----------------------------------------------------------------------------------------------------------------
    datasources = [jhuData,covidtrackerData,cdcILIdata,dohWebsiteData]
    allData = datasources[0]
    for data in datasources[1:]:
        try:
            allData = allData.merge(data,on=['epiweek','modelweek','region','fips'],how='outer')
        except KeyError:
            allData = allData.merge(data,on=['epiweek','modelweek','region'],how='outer')
    allData = allData.sort_values('epiweek')

    # ---------------------------------------------------------------------------
    # Census Data

    censusData = pd.read_csv("../populationEstimates/PApopdata.csv")
    censusData = censusData[["countyfips","POP"]]
    censusData = censusData.rename( columns = {"countyfips":"fips", "POP":"census"} )

    allData = allData.merge(censusData,on=["fips"])
    allData.to_csv('./PAcasesData_{:s}.csv'.format( timestamp() ))
