
#my imports
from datetime import date
from datetime import timedelta
#Import ganeral libs
import pandas as pd
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import dateparser
import seaborn as sns
import os
from sklearn.linear_model import LinearRegression

# dash imports
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

#viz imports
from plotly import graph_objs as go


# getting name csv and country name

link = ['df_predAfg.csv', 'df_predAlb.csv']  



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



def get_date(link,n=7):
    date = get_right_df(link[1])['Date'].tolist()
    datema = []
    for d in range(len(date)):
        if d//n == d/n:
            datema.append(date[d])
    return datema 

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
        print (l)
        c = get_right_df(l)['Country']
        df_countries[c[0]] = smoothing_vector(l, n=2)
    return df_countries
#print (df_for_predict(link).iloc[100:150]) 

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




#print ('========================ts_df====================')
#print (ts_df(df_for_predict(link), choise))
#print ('========================ts_df====================')


#print ('========================df_for_predict(link)====================')
#print (df_for_predict(link))
#print ('========================df_for_predict(link)====================')

# View Data
#print (sns.pairplot(ts_df(df_for_predict(link)))) #-it is look like strong regression






#style
PLOTLY_THEME ='simple_white'
STYLE = [dbc.themes.FLATLY]

#building server
app = dash.Dash('covid_predict_app', external_stylesheets=STYLE)
server = app.server

#list of countries
def country():
    today = date.today()
    l = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/'+((today-timedelta(5)).strftime("%m-%d-%Y"))+'.csv'
    db = pd.read_csv(l)
    coun = db['Country_Region'].unique().tolist()
    coun.append('All')
    return coun
    


#controls
controls =  dbc.Card(
        [
             dbc.Row([
                 dbc.Col(dbc.FormGroup(
                 [
                     dbc.Label("Choose a Country:"),
                     dcc.Dropdown(
                             id = "country-selector",
                             options = [{"label":x, "value":x} for x in country()],
                             value ="All"#, multi = True
                     )
                 ]
                 )),
              ], align = 'center')  
        ],
        body = True
)
                 
   
# inicialisation Graph                 
graph = dcc.Graph (id = 'graph')


#general layout
app.layout = dbc.Container(
    [
     html.H1("Covid19 reserch"),
     html.Hr(),
     dbc.Row([dbc.Col(controls, width = 6)]),
     dbc.Row([dbc.Col(html.Div("Wait a few minutes, we collect data"),width = 6)]),
     dbc.Row([dbc.Col(graph, width = 6)]
              #dbc.Col(qtygraph, width = 6)]
                , align =  "center")
     
    ], fluid = True
)


@app.callback(Output (component_id = 'graph', component_property = 'figure'),
              [Input(component_id = 'country-selector', component_property = 'value')])
def update_date_graph (country):
    print (country)
    #Predict

    #split
    choise = country
    nday = 7
    fit_df = df_for_predict(link).iloc[:len(df_for_predict(link))]
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
    #from sklearn.svm import LinearSVC
#model = LinearSVC(C=0.01, max_iter=10000, dual = False).fit(x, y)

    model = LinearRegression().fit(x, y)
#print ('score of train = ', model.score(x_train,y_train))

#predict_for_std = model.predict(x)
    stdtry = np.std(pre_country_df)


#General Test but not last

#predict
    gros_test = df_for_predict(link).iloc[len(df_for_predict(link))-9:len(df_for_predict(link))]
    pre_gros_test_country = gros_test[choise]
#print ('==================pre_gros_test_country=============')
#print (pre_gros_test_country)
#print ('==================pre_gros_test_country=============')


    d = len(pre_country_df)-1
    new_predicts =[]
    gros_test_country = ts_df(pre_gros_test_country)
#print ('=========================first predict=======================')
#print (gros_test_country.drop([0],axis=1))
#print ('=========================first predict=======================')

    gros_predict = np.round(model.predict(gros_test_country.drop([0],axis=1)),0)
#print ('=========================gros_predict========================')
#print (gros_predict)
#print ('=========================gros_predict========================')

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
    #new_df['real'] = df_for_predict(link)[choise].loc[110:110+nday].values
    new_preds = []

    #new_df['predict'] = new_predicts
    #new_df['mistake'] = abs(new_df['real'] - new_df['predict'])
    #for num in range(len(new_predicts)):
    #    if new_df['mistake'].loc[num] > stdtry:
    #       new_preds.append(new_predicts[num]+stdtry)
    #    else:
    #        new_preds.append(new_predicts[num])
    #new_df['predict + stdtry'] = new_preds
    #new_df['mistake + stdtry'] = new_df['real'] - new_df['predict + stdtry']
#print ()
#print ('=======================new_df======================')
#print (new_df.head(30))
#print ('=======================new_df======================')
#newstd = np.std(new_predicts)
#print (newstd)
    datepred = pd.date_range(get_date(link, n=2)[-1], periods=8).strftime("%m-%d-%Y")
#print (datepred)
    dataold = get_date(link, n=2)
#print (dataold)   
    fig = go.Figure(layout=go.Layout(height=400, width=1024))
    fig.add_trace(go.Scatter(x = dataold,
                             y = df_for_predict(link)[choise].tolist(),
                             fill = None, mode = 'lines+markers',
                             name = 'new infected per day', line = {'color':'green'}))
    fig.add_trace(go.Scatter(x = datepred,
                             y = new_predicts,
                             fill = None, mode = 'lines+markers',
                             name = 'predict next 7 days', line = {'color':'blue'}))
    fig.update_traces(opacity=1)
    fig.update_layout(#template=PLOT_THEME,
            title_text = "daily infected", xaxis_title = 'date',
                      yaxis_title = 'infected')
    return fig

if __name__ =="__main__":
    app.run_server()
    #(debug = True, host = '127.0.0.1', port = 8050)
                 
