from flask import Flask, render_template, request
import pickle
import numpy as np
import requests
import json
import pandas as pd
import math

app = Flask(__name__)

def stationlocator(API, Latitude, Longitude):
    dt = requests.get('https://api.waqi.info/feed/geo:'+str(Latitude)+';'+str(Longitude)+'/?token='+API)
    doc = dt.text
    j = json.loads(doc)
    data = j['data'] ## Contains AQI value for each of the pollutants
    nearest_station = data['city']['name'] ## Contains name of the monitoring station
    geo_loc = data['city']['geo'] ## Contains geolocation of the monitoring station
    time_of_retreival = data['time']['s'] ## Time of monitoring
    list_of_poll = list(data['iaqi'].keys()) ## List of pollutants captured
    station_details = {'data':data, 'nearest_station':nearest_station, 'time_of_retreival':time_of_retreival,'list_of_poll':list_of_poll, 'geo':geo_loc}
    print()
    print('Information successfully fetched from the nearest station')
    return station_details ## Output as dictionary


## Function to extract pollutant concentration
def poll_conc(data, list_of_poll):
    criteria_poll = ['pm10', 'pm25', 'so2','no2', 'o3','co'] ## List of criteria pollutants notified (NH3 is not included)
    poll_conc =[]

    for i in criteria_poll:
        if i in list_of_poll:
            val = data['iaqi'][i]['v']
            sheet = pd.read_excel('AQI_breakpoint.xlsx', sheet_name=i+'_US') ## The sheet with conversion factors
            req_row = sheet.loc[(sheet['Lower AQI']<= val) & (sheet['Upper AQI']>= val),:].reset_index()
            ## Let us convert AQI to Pollution concentration
            step_1 = (req_row['Upper AQI'][0] - req_row['Lower AQI'][0])/ (req_row['Upper Conc'][0] - req_row['Lower Conc'][0])
            step_2 = (int(val) - req_row['Lower AQI'][0])/step_1
            actual_conc = (step_2+ req_row['Lower Conc'][0]) *req_row['Conversion_const'][0]
            poll_conc.append(actual_conc)
        else:
            poll_conc.append('No Information available')
    poll_outs = {'poll_conc': poll_conc, 'criteria_poll': criteria_poll}
    return poll_outs

def aqi(poll_outs): ## Takes output of step 2 as input
    AQI = []
    
    for poll, val in zip(poll_outs['criteria_poll'], poll_outs['poll_conc']):
        if val !='No Information available':
            sheet = pd.read_excel('AQI_breakpoint.xlsx', sheet_name=poll+'_IND')
            ## Ceiling the value just not to missed out the intermediate values (eg: PM10 =50.5,101.6 ug/m3,... etc )
            ciel_val = math.ceil(val)
            req_row = sheet.loc[(sheet['Lower Conc']<= ciel_val) & (sheet['Upper Conc']>= ciel_val),:].reset_index()
            ## Let us now get the AQI at India scale for the pollutant
            step_1 = (req_row['Upper AQI'][0] - req_row['Lower AQI'][0])/ (req_row['Upper Conc'][0] - req_row['Lower Conc'][0])
            step_2 = step_1 * (val - req_row['Lower Conc'][0])
            AQI_vals = step_2+ req_row['Lower AQI'][0]
            AQI.append(AQI_vals)
            
        else:
            AQI.append('No Information available') ## Only in case of the pollutant details being not monitored/captured by the station
    return AQI

def mainFunction(lat, lon):
    station_details=stationlocator("5583344fe93b06116e9569799570df579a1ff3b4",lat, lon)
    data = station_details['data']
    list_of_poll=station_details['list_of_poll']
    poll_outs=poll_conc(data,list_of_poll)
    AQI=aqi(poll_outs)
    return AQI

def load_model(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

file_path1 = 'newwtf.pkl'
file_path2 = '5params.pkl'

data1 = load_model(file_path1)
data2 = load_model(file_path2)

model1=data1
model2=data2

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/result', methods=['GET', 'POST'])
def result():
    PM10= int(float(request.form['PM10']))
    PM2_5 = int(float(request.form['PM2.5']))
    NO2 = int(float(request.form['NO2']))
    NH3 = int(float(request.form['NH3']))
    CO = int(float(request.form['CO']))
    SO2 = int(float(request.form['SO2']))
    O3 = int(float(request.form['O3']))
    data = np.array([[PM2_5,PM10,NO2,NH3,SO2,CO,O3]])
    output = model1.predict(data)
    return render_template("result.html", prediction=output)

@app.route('/select_location')
def select_location():
    return render_template("new_location.html")

@app.route('/after_location/<float:lat>/<float:lon>')
def after_location(lat, lon):
    AQI=mainFunction(lat, lon)
    print(AQI)
    data = np.array([[AQI[0],AQI[1],AQI[2],AQI[3],AQI[4],AQI[5]]])
    output = model2.predict(data)
    return render_template("result2.html", prediction=output)

if __name__ == '__main__':
    app.run(debug=True)
