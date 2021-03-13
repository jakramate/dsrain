# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [START gae_python38_app]
from flask import Flask, url_for, render_template, request, send_file
#from urllib.parse import urlparse
#from dateutil import parser, tz
from datetime import datetime, timedelta
from flask_wtf import FlaskForm
from wtforms import DateField, SelectField
from wtforms.validators import DataRequired
import html, requests

# datastores
from google.cloud import datastore
datastore_client = datastore.Client(project='dsrain')

# graphing
import plotly
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import json

# for machine learning 
from sklearn import ensemble
from joblib import load

# motuclient and netCDF for fetching SLA data
import sys
import netCDF4 as nc
from motuclient import *

# for getting rain info which takes a really long time
import threading

# If `entrypoint` is not defined in app.yaml, App Engine will look for an app
# called `app` in `main.py`.
app = Flask(__name__)


class ProvinceForm(FlaskForm):
    provinces = ['ChiangMai','Bangkok',
         'KamphaengPhet',
         'ChaiNat',
         'NakhonNayok',
         'NakhonPathom',
         'NakhonSawan',
         'Nonthaburi',
         'PathumThani',
         'PhraNakhonSiAyutthaya',
         'Phichit',
         'Phitsanulok',
         'LopBuri',
         'SamutPrakan',
         'SamutSongkhram',
         'SamutSakhon',
         'Saraburi',
         'SingBuri',
         'SuphanBuri',
         'Sukhothai',
         'UthaiThani',
         'AngThong',
         'Phetchabun',
         'Chanthaburi',
         'Chachoengsao',
         'ChonBuri',
         'Trat',
         'PrachinBuri',
         'Rayong',
         'SaKaeo',
         'Kanchanaburi',
         'Tak',
         'PrachuapKhiriKhan',
         'Ratchaburi',
         'Phetchaburi',
         'Nan',
         'Phayao',
         'Lampang',
         'Lamphun',
         'Uttaradit',
         'ChiangRai',
         'Phrae',
         'MaeHongSon',
         'Kalasin',
         'KhonKaen',
         'Chaiyaphum',
         'NakhonPhanom',
         'NakhonRatchasima',
         'BuriRam',
         'MahaSarakham',
         'Mukdahan',
         'Yasothon',
         'RoiEt',
         'SiSaKet',
         'SakonNakhon',
         'Surin',
         'NongKhai',
         'NongBuaLamPhu',
         'AmnatCharoen',
         'UdonThani',
         'UbonRatchathani',
         'Loei',
         'Krabi',
         'Chumphon',
         'Trang',
         'NakhonSiThammarat',
         'Narathiwat',
         'Pattani',
         'Phang-nga',
         'Phatthalung',
         'Phuket',
         'Yala',
         'Ranong',
         'Songkhla',
         'Satun',
         'SuratThani']
    province = SelectField('Pick a province', choices=provinces, validators = [DataRequired()])

class DateForm(FlaskForm):
    start = DateField('Pick a Date',  validators=[DataRequired()])
    end = DateField('Pick a Date',  validators=[DataRequired()])


def storeSLA(date, item): # storing sealevel average on datastore
    entity = datastore.Entity(key=datastore_client.key('date', date))
    entity.update(item)
    datastore_client.put(entity)

def storeRain(stnDate, item): # storing rain info on datastore
    entity = datastore.Entity(key=datastore_client.key('stndate', stnDate))
    entity.update(item)
    datastore_client.put(entity)

def storeRainPred(provDate, item): # storing rain prediction on datastore
    entity = datastore.Entity(key=datastore_client.key('provdate', provDate))
    entity.update(item)
    datastore_client.put(entity)

def predictRain(dates, bob, smt, province='ChiangMai'):
    # predicting rainfail for all province on 'date'
    #print(date, bob, smt)
    reg = load('static/gboost.joblib')
    stationFile = 'static/thailand_metstation.csv'
    provinces = pd.read_csv(stationFile, dtype={"province": str, "region":str, "stncode":str})

    if province == 'all':
        # we are working on nationwide prediction
        rainPredict = [0] * len(provinces)
        for i, p in provinces.iterrows():
            lat = p.lat 
            lng = p.lng
            pred = reg.predict(np.array([''.join(dates[0].split('-')[1:]), bob[0], smt[0], lat, lng], dtype='float').reshape(1,-1))[0]
            if pred < 0:
                rainPredict[i] = 0
            else:
                rainPredict[i] = pred
            storeRainPred(dates[0]+'-'+p.province, {'date':dates[0], 'province':p.province, 'pred':rainPredict[i]})
    else:
        # working on province-wise prediction 
        rainPredict = [0] * len(dates)
        lat = provinces[provinces.province == province].lat
        lng = provinces[provinces.province == province].lng
        for i in range(len(dates)):
            pred = reg.predict(np.array([''.join(dates[i].split('-')[1:]), bob[i], smt[i], lat, lng], dtype='float').reshape(1,-1))[0]
            if pred < 0:
                rainPredict[i] = 0
            else:
                rainPredict[i] = pred
            # store prediction for future usage 
            storeRainPred(dates[i]+'-'+province, {'date':dates[i], 'province':province, 'pred':rainPredict[i]})
        # we are working from endDate to startDate so reverse the list before returning
        rainPredict = rainPredict[::-1]

    return rainPredict

def fetchRainPred(startDate, endDate, province):
    # if there's already a prediction on datastore fetch and return them
    query = datastore_client.query(kind='provdate')
    if province != 'all': # if nation-wide is needed, do not specify province
        query.add_filter('province', '=', province)
    query.add_filter('date','>', startDate)
    query.add_filter('date','<', endDate)
    rainPredQuery = query.fetch()

    # checking number of records in query results
    rainPreds = []
    for r in rainPredQuery:
        rainPreds.append(r['pred'])

    if province == 'all' and len(rainPreds) < 76:
        # there should be 76 records for this type of query if not, make new predictions
        # query and populating SLA info into numpy arrays
        print("cache missed nationwide", startDate, endDate)
        slas = fetchSLA(startDate, endDate)
        bneg = []
        bpos = []
        sneg = []
        spos = []
        dates = []
        for sla in slas:
            bneg.append(sla['bneg'])
            bpos.append(sla['bpos'])
            sneg.append(sla['sneg'])
            spos.append(sla['spos'])
            dates.append(sla['date'])
        
        # calculating BoB and Sumatra index
        bob = np.array(bpos) - np.array(bneg)
        smt = np.array(spos) - np.array(sneg)
        
        rainPreds = predictRain([dates[0]], [bob[0]], [smt[0]], 'all')

    elif province != 'all' and len(rainPreds) < 178:
        # for province specific we should have 178 predictions for about 6 months if not make new predictions
        # query and populating SLA info into numpy arrays
        print("cache missed ", province, len(rainPreds))
        slas = fetchSLA(startDate, endDate)
        bneg = []
        bpos = []
        sneg = []
        spos = []
        dates = []
        for sla in slas:
            bneg.append(sla['bneg'])
            bpos.append(sla['bpos'])
            sneg.append(sla['sneg'])
            spos.append(sla['spos'])
            dates.append(sla['date'])
        
        # calculating BoB and Sumatra index
        bob = np.array(bpos) - np.array(bneg)
        smt = np.array(spos) - np.array(sneg)


        rainPreds = predictRain(dates, bob, smt, province)

    return rainPreds  # make sure that rainPred is a list

def fetchRain(startDate, endDate, province):
    #print(province, startDate, endDate)
    # performing province to stationID conversion, a bit costly
    stationID = '327501'
    stationFile = 'static/thailand_metstation.csv'
    df = pd.read_csv(stationFile, dtype={"province": str, "region":str, "stncode":str})
    df = df[df["province"] == province]
    for idx, row in df.iterrows():
        stationID = row.stncode
        break

    #print('station id', stationID)
    query = datastore_client.query(kind='stndate')
    query.add_filter('stn', '=', stationID)
    query.add_filter('date','>', startDate)
    query.add_filter('date','<', endDate)
    rain = query.fetch()
    return rain

def fetchSLA(startDate, endDate):
    query = datastore_client.query(kind='date')
    query.add_filter('date','>', startDate)
    query.add_filter('date','<', endDate)
    query.order = ['-date']
    slas = query.fetch()
    return slas


@app.route('/download/bob')
def download_bob():
    return send_file('/tmp/bob.csv',
                     mimetype='text/csv',
                     attachment_filename='bob.csv',
                     as_attachment=True)

@app.route('/download/smt')
def download_smt():
    return send_file('/tmp/smt.csv',
                     mimetype='text/csv',
                     attachment_filename='smt.csv',
                     as_attachment=True)

@app.route('/download/rain')
def download_rain():
    return send_file('/tmp/rain.csv',
                     mimetype='text/csv',
                     attachment_filename='rain.csv',
                     as_attachment=True)
            

# a function which creates graphjson
def create_plot(time, index, legend='data', province='ChiangMai'):
    #N = len(data)  # the eength of the data 
    #x = np.linspace(0, 1, N)
    x = time
    y = index
    df = pd.DataFrame({'date': x, 'index_value': y}) # creating a sample dataframe

    # saving df to csv
    filename = '/tmp/'+legend+'.csv'
    df.to_csv(filename)

    fig = go.Figure()
    if 'BoB' in legend:
        fig.add_trace(go.Bar(x=df['date'],y=df['index_value'], name=legend, marker_color='cornflowerblue'))
        fig.update_layout(title_text='BoB index')
        fig.update_yaxes(title_text='Index value')
    elif 'SMT' in legend:
        fig.add_trace(go.Bar(x=df['date'],y=df['index_value'], name=legend, marker_color='goldenrod'))
        fig.update_layout(title_text='Sumatra index')
        fig.update_yaxes(title_text='Index value')
    elif 'Actual' in legend:
        fig.add_trace(go.Bar(x=df['date'],y=df['index_value'], name=legend, marker_color='green'))
        fig.update_layout(title_text='Actual precipication in '+province, xaxis_nticks=25)
        fig.update_yaxes(title_text='Precipication (mm)')
    else:
        fig.add_trace(go.Bar(x=df['date'],y=df['index_value'], name=legend, marker_color='royalblue'))
        fig.update_layout(title_text='Predicted precipication for '+province, xaxis_nticks=25)
        fig.update_yaxes(title_text='Precipication (mm)')


    fig.update_xaxes(title_text='Date')

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return graphJSON


# a function to create Thailand's map using geoJSON
def create_map(rainData):
    
    geoFile = 'static/thailand_geo.json'
    stationFile = 'static/thailand_metstation.csv'

    with open(geoFile) as fin:
        provinces = json.load(fin)

    df = pd.read_csv(stationFile, dtype={"province": str, "region":str, "stncode":str})

    # adding 'rainmm' col to existing dataframe
    df["rainmm"] = rainData

    # check month of year and set appropriate limits
    fig = go.Figure()
    fig.add_trace(go.Choroplethmapbox(geojson=provinces, locations=df.province, z=df.rainmm, 
                                  featureidkey="properties.CHA_NE",
                                  colorscale='blues', zmin=0, zmax=10,
                                  colorbar={'title':'Precipitation (mm)', 'tickvals':[0,2,4,6,8,10]}
                                  ))
    fig.update_layout(mapbox_style="carto-positron", mapbox_zoom=4.2, 
                  mapbox_center= {"lat": 13, "lon": 100.9925},
                  margin={"r":20,"t":20,"l":20,"b":20})
    
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return graphJSON

@app.route('/info')
def info():
    return render_template('info.html')

# adding route for weather index
@app.route('/data', methods=['POST','GET'])
def data():

    form = DateForm(meta={'csrf': False})
    #if form.validate_on_submit():  # if form submitted
    if form.is_submitted():
        startDate = datetime.datetime.strptime(form.start.raw_data[0], '%m/%d/%Y') - timedelta(1)
        endDate = datetime.datetime.strptime(form.end.raw_data[0], '%m/%d/%Y') + timedelta(1)
        #print('selected', startDate, endDate)
        slas = fetchSLA(startDate.strftime('%Y-%m-%d'), endDate.strftime('%Y-%m-%d'))
    else:
        startDate = (datetime.datetime.now() - timedelta(365)).strftime('%Y-%m-%d')
        endDate = (datetime.datetime.now()).strftime('%Y-%m-%d')
        slas = fetchSLA(startDate, endDate)
        #print('default', startDate, endDate)

    # retrieve data from data store upto 1 year and construct the graph
    bneg = []
    bpos = []
    sneg = []
    spos = []
    dates = []
    for sla in slas:
        bneg.append(sla['bneg'])
        bpos.append(sla['bpos'])
        sneg.append(sla['sneg'])
        spos.append(sla['spos'])
        dates.append(sla['date'])
    
    bob = np.array(bpos) - np.array(bneg)
    smt = np.array(spos) - np.array(sneg)

    bar = []
    bar.append(create_plot(dates, bob, 'BoB'))
    bar.append(create_plot(dates, smt, 'SMT'))

    return render_template('data.html', plot1=bar[0], plot2=bar[1], form=form)


@app.route('/', methods=['POST','GET'])
def forecast():
    # selecting a province to view detailed forecasts
    form = ProvinceForm(meta={'csrf': False})

    # the default is ChiangMai
    province = 'ChiangMai'
    if form.is_submitted():
        province = form.province.raw_data[0]

    # setting up important dates
    yesterday = (datetime.datetime.now() - timedelta(2))
    tomorrow  = (datetime.datetime.now() + timedelta(1))
    startDate = (datetime.datetime.now() - timedelta(180))
    endDate   = (datetime.datetime.now() - timedelta(1))

    # query actual rainfall from datastore
    dateRange = pd.date_range(startDate+timedelta(1), endDate-timedelta(1), freq='d')
    dates = list(dateRange.strftime('%Y-%m-%d'))
    rainQuery = fetchRain(startDate.strftime('%Y-%m-%d'), endDate.strftime('%Y-%m-%d'), province)
    rainmm = [0] * len(dates)
    for r in rainQuery:
        rainmm[dates.index(r['date'])] = r['rainmm']
    
    # predicts rainfalls for all provinces for tomorrow based on today's indexes, and returns nation-wide predictions
    # the startDate must be yesterday and endDate is tomorrow because datastore only
    # supports '<' and '>', in this way we would get the indexes for today as features for the regression model
    nationwidePred = fetchRainPred(yesterday.strftime('%Y-%m-%d'), tomorrow.strftime('%Y-%m-%d'), 'all')
    
    # query prediction results for the last 6 months for the selected province 
    provincePred = fetchRainPred(startDate.strftime('%Y-%m-%d'), endDate.strftime('%Y-%m-%d'), province)

    bar = []
    bar.append(create_map(nationwidePred))
    bar.append(create_plot(dates, np.array(rainmm), 'Actual', province))
    bar.append(create_plot(dates, np.array(provincePred), 'Predicted', province))
    return render_template('forecast.html', wmap=bar[0], plot1=bar[1], plot2=bar[2], form=form, 
            date=tomorrow.strftime('%d %B %Y'), province=province)

if __name__ == '__main__':
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be)) configured by adding an `entrypoint` to app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)
# [END gae_python38_app]
