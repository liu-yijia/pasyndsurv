import matplotlib
import pandas
import scipy
import numpy as np
import sklearn
import tkinter as tk
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from numpy import array
from numpy import reshape
from random import seed
from random import randint
from statsmodels.tsa.holtwinters import ExponentialSmoothing


r = tk.Tk() 
r.title('Covid County Finder')
r.geometry('300x300')
v = tk.StringVar()
v.set("Type Here")
tbox = tk.Entry(r,width=10,textvariable=v)    #sets up start screen
s = v.get()
tbox.grid(column=1, row=2)
txt = tk.Label(r, text = "Please enter county below.")
txt.grid(column=0, row=0)


url = "https://raw.githubusercontent.com/jeremymack-LU/covid19/master/covid19_pa_counties.csv"
names = ['County','Date','Cases','Deaths']
dataset = read_csv(url, names=names)     #imports data
daysmeasured = 98


countylist = []
counties = dataset['County']
for x in range(0,66):
    h = x*daysmeasured+3
    countylist.append(counties[h])
    
def onClick():
    entry = tbox.get()
    num = countylist.index(entry)   #defines buttons and what they do
    getgraph(num)
    
def onClick2():
    plotdata()

button = tk.Button(r, text='Enter', width=5, bg = 'purple', fg = 'white', command=onClick) 
button.grid(column=1, row=3)

button = tk.Button(r, text='Done', width=5, bg = 'red', fg = 'white', command=onClick2) 
button.grid(column=2, row=3) 



print(dataset.groupby('County').size())

arr = []
for x in range(1,daysmeasured+1):
    arr.append(x)


def getgraph(num):
    s = daysmeasured*num+1
    for x in range(1, 2):
        countydata = dataset[s:s+daysmeasured]
        #print(countydata.describe())
        countycases = countydata['Cases']
        
        
        #print(countycases.head(100))
        dates = countydata['Date']
        county = countydata['County']
        thiscounty = county[s]
        
        ploty = countycases.copy()
        
        for x in range(s, s+daysmeasured):
            ploty[x] = int(ploty[x])
            
        plt.plot(arr, ploty, label = thiscounty)
        s = s+daysmeasured+1
        plt.ylabel('Number of Cases')
        plt.xlabel('Day since 3/05/20')
        plt.title('Cases in '+thiscounty)
        
        #machineanalysis(countydata, thiscounty)
        #machineanalysisgraph(countydata, thiscounty)
        HoltWinters(ploty)
        plt.legend() 

def plotdata():
    plt.show()


def HoltWinters(cases):

    cases.astype('int64')
    
    
    
    data = pandas.DataFrame(cases)
    
    
    train = data.iloc[:85, 0]
    test = data.iloc[84:, 0]

    model = ExponentialSmoothing(train, trend = 'add', damped=False, seasonal='add', seasonal_periods=3).fit()
    pred = model.predict(start=test.index[0], end=test.index[0])
    
    plt.figure(figsize = (22,8))
    plt.plot(train.index, train, label = 'train')
    plt.plot(test.index, test, label = 'test')
    plt.plot(pred.index, pred, label = 'prediction')
    
    plt.legend()
    plt.show()
    








def machineanalysis(dataset, county):
    casesdata = dataset['Cases']
    casesarr = array(casesdata)
    casesarr2D = casesarr.reshape((casesarr.shape[0], 1))
    global arr
    arrarr = array(arr)
    arrarr2D = arrarr.reshape((arrarr.shape[0], 1))
    day = arrarr2D
    numcases = casesarr2D
    

    model = LinearRegression()
    model2 = LogisticRegression()
    model.fit(day,numcases)
    model2.fit(day,numcases)
    
    global daysmeasured
    predicted_day = 100
    daysdiff = predicted_day-daysmeasured
    
    
    print('Predicted cases in '+str(daysdiff)+' days in '+county+' county:')
    print('Linear Regression: '+str(model.predict([[predicted_day]])))
    print('Logistic Regression: '+str(model2.predict([[predicted_day]])))
    
    if predicted_day<daysmeasured:
        sortarr = []
        linpred = int(model.predict([[predicted_day]]))  #makes best estimate based on what model is closest to actual value-more to help me determine which model is better at each point in time
        logpred = int(model2.predict([[predicted_day]]))
        linlogavg = (linpred+logpred)/2
        
        lindiff = abs(int(casesdata[predicted_day:predicted_day+1])-linpred)
        logdiff = abs(int(casesdata[predicted_day:predicted_day+1])-logpred)
        linlogdiff = abs(int(casesdata[predicted_day:predicted_day+1])-linlogavg)
        
        sortarr.append(lindiff)
        sortarr.append(logdiff)
        sortarr.append(linlogdiff)
        sortarr.sort()
        
        if lindiff == sortarr[0]:
            print('Best prediction: '+str(linpred))
        elif logdiff == sortarr[0]:
            print('Best prediction: '+str(logpred))
        elif linlogdiff == sortarr[0]:
            print('Best prediction: '+str(linlogavg))
            


def machineanalysisgraph(dataset, county):
    casesdata = dataset['Cases']
    casesarr = array(casesdata)
    casesarr2D = casesarr.reshape((casesarr.shape[0], 1))
    global arr
    arrarr = array(arr)
    arrarr2D = arrarr.reshape((arrarr.shape[0], 1))
    day = arrarr2D
    numcases = casesarr2D
    

    model = LinearRegression()
    model2 = LogisticRegression()
    model.fit(day,numcases)
    model2.fit(day,numcases)
    
    global daysmeasured
    predmodel = []
    for x in range(0, daysmeasured):
        sortarr = []
        predicted_day = x
        linpred = int(model.predict([[predicted_day]]))  #makes best estimate based on what model is closest to actual value-more to help me determine which model is better at each point in time
        logpred = int(model2.predict([[predicted_day]]))
        linlogavg = (linpred+logpred)/2
        
        lindiff = abs(int(casesdata[predicted_day:predicted_day+1])-linpred)
        logdiff = abs(int(casesdata[predicted_day:predicted_day+1])-logpred)
        linlogdiff = abs(int(casesdata[predicted_day:predicted_day+1])-linlogavg)
        
        sortarr.append(lindiff)
        sortarr.append(logdiff)
        sortarr.append(linlogdiff)
        sortarr.sort()
        
        BestPred = 0
        if lindiff == sortarr[0]:
            BestPred = linpred
            print(str(x)+' Used linpred!')
        elif logdiff == sortarr[0]:
            BestPred = logpred
            print(str(x)+' Used logpred!')
        elif linlogdiff == sortarr[0]:
            BestPred = linlogavg
            print(str(x)+' Used linlogavg!')
        predmodel.append(BestPred)
    plt.plot(arr, predmodel, label = county+'model')
r.mainloop()