from urllib.parse import urlparse
from dateutil import parser, tz
from datetime import datetime, timedelta
import html, requests

import pandas as pd

# datastores
from google.cloud import datastore
datastore_client = datastore.Client(project='rareyetem')

def storeRain(stnDate, item): # storing rain info on datastore
    entity = datastore.Entity(key=datastore_client.key('stndate', stnDate))
    entity.update(item)
    datastore_client.put(entity)

def getrain():
    headers = {'User-Agent': 'Mozilla/5.0'}
    URL = "https://www.tmd.go.th/cis/freedata.php?method=fd_SearchDaily"
    # open station file and looping thru each station
    metStnFile = '/Users/jkm/rareyetem/static/thailand_metstation.csv'
    df = pd.read_csv(metStnFile, dtype={"province": str, "region":str, "stncode":str})

    today = (datetime.now()).strftime('%-d %-m %Y').split()

    # set to range(0) for day by day fetching
    for window in range(2):
        today = (datetime.now() - timedelta(window)).strftime('%-d %-m %Y').split()
        #print(window, today)
        today2format = (datetime.now() - timedelta(window)).strftime('%Y-%m-%d')
        for idx, stn in df.iterrows():
            print(stn['stncode'])
            # constructing a payload
            payload = {
                "ISTATION_ID": stn['stncode'],
                "IDAY": today[0],
                "IMONTH": today[1],
                "IYEAR": today[2]
            }
            # making an HTTP request to tmd's website
            session = requests.session()
            r = session.post(URL, headers=headers, data=payload)

            # parsing precipitation info
            pos = r.text.find("ปริมาณ")   # offsetting <td> tag
            dataText = r.text[pos:].split('>')
            # process the output until there's nothing to be done
            tmdData = {}
            tmdData["stn"]  = stn['stncode']
            tmdData["province"] = stn['province']
            tmdData["date"] = today2format  # second format which match those in database

            # extractin rain information
            rainmm = dataText[-1].rstrip()
            try:
                tmdData["rainmm"] = float(rainmm)
                #print(tmdData)
                if tmdData["rainmm"] > 0:
                    storeRain(tmdData["stn"] + '-' + tmdData["date"], tmdData)
                #else:
                #    print("No insertion due to zero rainfall")
            except:
                #print("Non numerical value found, skipping")
                pass

if __name__ == '__main__':
    print("[INFO] Start fetching rainfall data")
    getrain()
    print("[INFO] Fetching rainfall data finished")
