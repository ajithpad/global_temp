import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from sklearn import datasets, linear_model

data_path = '/Users/ajith/bigdata/data/GlobalLandTemperatures/'

city_temp = pd.read_csv(data_path + 'GlobalLandTemperaturesByCity.csv', sep = ',', parse_dates = True)



regr = linear_model.LinearRegression()

#Function to get the city's name and plot its temperature for all the months across all years

def par_city_temp(city, month):
    par_temp = city_temp[city_temp['City']== city]
    par_temp['dt'] = pd.to_datetime(par_temp['dt'])
    par_temp_mon = par_temp[par_temp['dt'].dt.month == month]
    nonull_temp = par_temp_mon[pd.notnull(par_temp_mon['AverageTemperature'])]
    regr.fit(nonull_temp['dt'].dt.year.values.reshape(-1,1), nonull_temp['AverageTemperature'].values.reshape(-1,1))
    plt.plot(par_temp_mon['dt'].dt.year, par_temp_mon['AverageTemperature'],'.')
    #    plt.plot(nonull_temp['dt'].dt.year.values.reshape(-1,1), regr.predict(nonull_temp['dt'].dt.year.values.reshape(-1,1)), lw = 3.)
    low_year = np.min(nonull_temp['dt'].dt.year.values)
    plt.plot(np.arange(low_year,2051).reshape(-1,1), regr.predict(np.arange(low_year,2051).reshape(-1,1)), lw = 3.)
    return par_temp_mon
    
