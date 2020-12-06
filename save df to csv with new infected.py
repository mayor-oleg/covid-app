# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 20:31:45 2020

@author: jr
"""

from mycovidparser import parsercovid as parser
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import time
from datetime import date
from datetime import timedelta





def count_new_infected(country,parser):
    
    perday = []
    inf = 0
    s= 0
    x=0
    date = []
    infected = [0]
    infectedma = [0]
    datema =[0]
    df_pred = pd.DataFrame()
    for df in parser():
        
        if country != 'All':
            newdf = df.loc[df['Country'] == country]
        
        
            work_newdf = newdf.groupby(['Country']).sum()
            work_newdf['Date'] = df.index[0][0]
        
        
            df_pred = pd.concat([df_pred, work_newdf[['Confirmed',  'Recovered',  'Deaths',  'Active','Date']]])#,ignore_index=True)
        #print (df_pred)
        
            s = work_newdf['Confirmed'].sum() 
        else:
            
            s = df['Confirmed'].sum()
        
        #s = newdf['Active'].sum()
        if s != 0: #old
            perday.append(s)
        date.append((df.index[0])[0])
        #df_pred['Date'] = df.index[0][0]
        #if x == 30: #for test
        #    break
        x +=1
   
    #print (df_pred) #Df with out new infected
    
    
    x=1
    #print (len(perday))
    while x < len(perday):
        
        if perday[x] == perday[x-1]:
            infected.append(inf)
        else:
            inf = abs(perday[x]-perday[x-1])
            infected.append(inf)

        
        
  #old      
        #if perday[x]>perday[x-1]:
           #infected.append(perday[x]-perday[x-1])
           #print(perday[x]-perday[x-1])
           #df_pred['New_infected'] = perday[x]-perday[x-1]
        #else:
           #infected.append(perday[x-1]-perday[x])
           #df_pred['New_infected'] = perday[x-1]-perday[x]
    
        x+=1
       
    
    

    #print (infected) #new infected this line for test
    
    #print (len(infected)) #How many days from start
    df_pred['New_infected'] = infected
    #df_pred['New_infected_ma'] = infectedma
    #print (df_pred)
    df_pred.to_csv('df_pred'+country[:3]+'.csv')
    #print (df_pred) #DF with new infected 
    return df_pred
#for all countries in the World uncoment below
#for df in parser():
#    countries = df['Country'].unique().tolist()
#    break
    
#for prototip
def country_list():
    today = date.today()
    l = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/'+((today-timedelta(5)).strftime("%m-%d-%Y"))+'.csv'
    db = pd.read_csv(l)
    coun = db['Country_Region'].unique().tolist()
    coun.append('All')
    return coun
countries = country_list()
for country in countries:
    print (country)
    count_new_infected(country,parser)
#print (count_new_infected(parser())) 