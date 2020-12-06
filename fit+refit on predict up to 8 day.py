# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 22:02:41 2020

@author: jr
"""

#Import ganeral libs
import pandas as pd
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import dateparser
import seaborn as sns

# getting name csv and country name

import os

directory = os.path.join("c:\\","path")
mypath = os.getcwd()
csv_file = []
for root,dirs,files in os.walk(mypath):
    for file in files:       
        if file.endswith(".csv"):           
           csv_file.append(file)   
#print (csv_file)           
link = csv_file

#read data
def get_right_df(link):
    from datetime import date
    df = pd.read_csv(link)
    df_date = dateparser.parse(df['Date'][0], settings={'DATE_ORDER': 'YMD'}).date()
    start_date = date(2020,1,22)
    if df_date != start_date:
        df_copy = pd.DataFrame(np.zeros(df.shape))
        df_copy.index = df.index
        df_copy.columns = df.columns
        df_copy[['Country']] = df[['Country']]
        dates = [ (start_date + timedelta(days=x)).strftime("%m-%d-%Y") for x in range(0,len(df['Date'])) ]
        df_copy['Date'] = dates
        df_work=df_copy.head(int(abs(start_date-df_date).days))
        
        #print (df_work) #if need to check df_work, by the way it is zeros in all value
    else:
        df_work = df
    df = pd.concat([df_work[['Country', 'Date', 'Confirmed', 'Recovered', 'Deaths', 'Active','New_infected']],df[['Country', 'Date', 'Confirmed', 'Recovered', 'Deaths', 'Active', 'New_infected']]], ignore_index=True)
    df.reset_index()
    #Next 3 lines use to see result of Data
    #print (df.head(50))  
    #print(df.info())
    #print (df[['Date','Confirmed',  'Recovered',  'Deaths',   'Active',  'New_infected']])
    return df
#check the Data
#for l in link:
    #print(get_right_df(l)) 

def smoothing_vector(link, n=2):
    infected = get_right_df(link)['New_infected'].tolist()    
    infectedma = [0]
    x=1
    while x < len(infected):
        if x//n == x/n: #there is term of smoothing: 7 mean that week 
            infectedma.append(int(sum(infected[x-n:x])//n))
        x+=1
    return infectedma



def get_date(link,n=2):
    date = get_right_df(link[1])['Date'].tolist()
    datema = []
    for d in range(len(date)):
        if d//n == d/n:
            datema.append(date[d])
    return datema        



def real_plot(date, infected, pd, pi):
    fig = plt.figure()
    scatter1 = plt.scatter(date, infected)
    
    graph1 = plt.plot(pd, pi,color='r')
    graph1 = plt.plot(date, infected)
    
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))

    plt.tick_params(axis = 'x',which = 'major',labelrotation = -45)
    print('Plot: ', len(graph1), graph1)
    plt.show()    
#real_plot(datema, infectedma) #this line just for example




#get Dataframe accoarding countries


    

def collect_df(link):
    columns = [] 
    for l in link:
        columns.append(get_right_df(l)['Country'].iloc[0])
    df_countries_alt = pd.DataFrame(columns = columns)
    for l in link:
        df = get_right_df(l)
        #print (df)

        c = get_right_df(l)['Country']
        df_countries_alt[c[0]] = df['New_infected']
    return df_countries_alt    

#print ('========================collect_df====================')
#print (collect_df(link))
#print ('========================collect_df====================')



def df_for_predict(link):
    df_countries = pd.DataFrame()
    for l in link:
        c = get_right_df(l)['Country']
        df_countries[c[0]] = smoothing_vector(l, n=2)
    return df_countries
print (df_for_predict(link)) 

# Making time series DF
def ts_df(df):
    #turn_df = df[country]
    columns=np.arange(0,8)
    
    data = []
    for num in range(len(df)):
        if num >= 8:
            data.append(df[num-8:num].tolist())
    #print (data)
            
    
    ts = pd.DataFrame(data, columns=columns)
    ts.fillna(0, inplace=True)
    return ts



def split_for_test(df, country):
    fit = df.iloc[:len(df)]
    test = df.iloc[len(df)-9:len(df)]
    #print ('=============================fit==================================')
    #print (fit)
    #print ('=============================fit==================================')
    #print ('=============================test==================================')
    #print (test)
    #print ('=============================test==================================')
    q = input('what do you want fit(1) or test(2): ')
    if q=='1':
        return fit
    else:
        return test





#print ('========================ts_df====================')
#print (ts_df(df_for_predict(link), choise))
#print ('========================ts_df====================')


#print ('========================df_for_predict(link)====================')
#print (df_for_predict(link))
#print ('========================df_for_predict(link)====================')

# View Data
#print (sns.pairplot(ts_df(df_for_predict(link)))) #-it is look like strong regression

#Predict
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#split
choise = input('choose country for train and predict, please:')
nday = int(input('choose day in Future:'))
fit_df = split_for_test(df_for_predict(link), choise)
#print (fit_df.loc[50:90])
pre_country_df = fit_df[choise]
country_df = ts_df(pre_country_df )
#print ('==========================country_df============================')
#print (country_df.head(50))
#print ('==========================country_df============================')

x = country_df.drop([7],axis=1)
y = country_df[7]
#print ('x for train = ')
#print (x)
#print ('y for train = ')
#print (y)


#Split with train_test_split no shuffle 
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42, shuffle=False)

#fit
from sklearn.svm import LinearSVC
#model = LinearSVC(C=0.01, max_iter=10000, dual = False).fit(x, y)

model = LinearRegression().fit(x, y)
#print ('score of train = ', model.score(x_train,y_train))

#predict_for_std = model.predict(x)
stdtry = np.std(pre_country_df)
print ('==================stdtry===================')
print (stdtry)
print ('==================stdtry===================')

#General Test but not last

#predict
gros_test = split_for_test(df_for_predict(link), choise) 
pre_gros_test_country = gros_test[choise]
print ('==================pre_gros_test_country=============')
print (pre_gros_test_country)
print ('==================pre_gros_test_country=============')


d = len(pre_country_df)-1
new_predicts =[]
gros_test_country = ts_df(pre_gros_test_country)
print ('=========================first predict=======================')
print (gros_test_country.drop([0],axis=1))
print ('=========================first predict=======================')

gros_predict = np.round(model.predict(gros_test_country.drop([0],axis=1)),0)
print ('=========================gros_predict========================')
print (gros_predict)
print ('=========================gros_predict========================')

d+=1
new_predicts.append(pd.Series(gros_predict).values[0])
pre_country_df[d] = pd.Series(gros_predict).values[0]
#print (pre_country_df)
loop = 0
while loop < nday:
    gros_test_country = ts_df(pre_country_df)
    #print (gros_test_country)
    gros_x = gros_test_country.drop([7],axis=1)
    gros_y = gros_test_country[7]
    #x_train, x_test, y_train, y_test = train_test_split(gros_x, gros_y, test_size=0.33, random_state=42, shuffle=False)
    
    model.fit(gros_x,gros_y)
    print ('score of train = ', model.score(gros_x,gros_y))

    
    
    gros_test_country = ts_df(pre_country_df.iloc[len(pre_country_df)-9:])
    print (gros_test_country)
    

    gros_predict = np.round(model.predict(gros_test_country.drop([0],axis=1)),0)
    new_predicts.append(pd.Series(gros_predict).values[0]) 
    d+=1
    pre_country_df[d] = pd.Series(gros_predict).values[0]
    loop+=1

#print ('==================pre_country_df=============')
#print (pre_country_df.loc[90:])    
#print ('==================pre_country_df=============')

new_df = pd.DataFrame()
new_df['real'] = df_for_predict(link)[choise].loc[len(df_for_predict(link)[choise]):len(df_for_predict(link)[choise])+nday].values
new_preds = []

new_df['predict'] = new_predicts
new_df['mistake'] = abs(new_df['real'] - new_df['predict'])
for num in range(len(new_predicts)):
    if new_df['mistake'].loc[num] > stdtry:
        new_preds.append(new_predicts[num]+stdtry)
    else:
        new_preds.append(new_predicts[num])
new_df['predict + stdtry'] = new_preds
new_df['mistake + stdtry'] = new_df['real'] - new_df['predict + stdtry']
print ()
print ('=======================new_df======================')
print (new_df.head(30))
print ('=======================new_df======================')
newstd = np.std(new_predicts)
print (newstd)
print (new_preds)

datepred = pd.date_range(get_date(link, n=2)[-1], periods=8).strftime("%m-%d-%Y")
print (datepred)
#datepred = get_date(link, n=2)[len(df_for_predict(link)[choise]):len(df_for_predict(link)[choise])+1+nday]
dataold = get_date(link, n=2)
print (dataold)
real_plot (dataold,df_for_predict(link)[choise].tolist()[:len(dataold)],
           datepred,new_preds) 

plt.plot(datepred, new_preds, color='r')
#plt.plot(datepred, new_df['real'].tolist(), color='b')
ax = plt.gca()
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
plt.tick_params(axis = 'x',which = 'major',labelrotation = -45)
plt.show()




