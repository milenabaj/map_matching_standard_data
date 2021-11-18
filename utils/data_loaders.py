"""
@author: Milena Bajic (DTU Compute)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import psycopg2 # pip install psycopg2==2.7.7 or pip install psycopg2==2.7.7
from json import loads
import sys,os, glob
from datetime import datetime
import pickle
from json import loads

def get_trips_info(task_ids = None, only_GM = False, select_friction = False):
    
   conn = psycopg2.connect() 
   
   if task_ids:
        task_ids=str(tuple(task_ids))
        quory = 'SELECT * FROM public."Trips" WHERE ("Trips"."Fully_Imported"=\'True\' AND "Trips"."TaskId" IN {0}) ORDER BY "TaskId" ASC'.format(task_ids)
   else:
        quory = 'SELECT * FROM public."Trips" WHERE "Trips"."Fully_Imported"=\'True\' ORDER BY "TaskId" ASC'
    
   # Set cursor
   cursor = conn.cursor()
    
   d = pd.read_sql(quory, conn, coerce_float = True) 
   d['Datetime']=pd.to_datetime(d['Created_Date'])
    
   if only_GM:
       d = d[d['TaskId']!=0]
   
   # Condition?
   if select_friction:
       cond = d.apply(lambda row: 'Friction' in row['StartPositionDisplay'], axis=1)
       d = d[cond]

    
   # Close the connection
   cursor.close()
   conn.close()  
    
   return d

def filter_keys(msg, remove_gm = True):
    if remove_gm:
        remove_list= ['id', 'start_time_utc', 'end_time_utc','start_position_display',
                      'end_position_display','device','duration','distanceKm','tag', 
                      'personal', '@ts','@uid', '@t','obd.whl_trq_est', '@rec']
    else:
        remove_list = []
    msg = {k : v for k,v in msg.items() if k not in remove_list}
    return msg
 

def extract_string_column(sql_data, col_name = 'message'):
    # if json
    try: 
        sql_data[col_name] = sql_data[col_name].apply(lambda message: loads(message))
    except:
        pass
    keys = sql_data[col_name].iloc[0].keys()
    n_keys =  len(keys)
    for i, key in enumerate(keys):
        print('Key {0}/{1}'.format(i, n_keys))
        sql_data[key] = sql_data[col_name].apply(lambda col_data: col_data[key])
        
    sql_data.drop(columns=[col_name],inplace=True,axis=1)
    return sql_data
    
def check_nans(sql_data, is_aran = False, exclude_cols = []):   
    n_rows = sql_data.shape[0]
    for col in  sql_data.columns:
        if col in exclude_cols:
            continue
        n_nans = sql_data[col].isna().sum()
        n_left = n_rows - n_nans
        print('Number of nans in {0}: {1}/{2}, left: {3}/{2}'.format(col, n_nans, n_rows, n_left ))
    return


def load_viafrik_data(trip_id, all_sensors = True, load_nrows = -1):
    
    # Set up connection
     #==============#
    print("\nConnecting to PostgreSQL database to load GM")
    conn = psycopg2.connect()
   
    quory = 'SELECT "DRDMeasurementId","TS_or_Distance","T","lat","lon","message" FROM "DRDMeasurements" WHERE "FK_Trip"=\'{0}\' ORDER BY "TS_or_Distance" ASC'.format(trip_id)
      

    print(quory)
    cursor = conn.cursor()
    meas_data = pd.read_sql(quory, conn, coerce_float = True)
    meas_data.reset_index(inplace=True, drop=True) 
    
    # Extract message
    #=================# 
    meas_data['Message'] = meas_data.message.apply(lambda msg: filter_keys(loads(msg), remove_gm=False))
    meas_data.drop(columns=['message'],inplace=True,axis=1)
    meas_data.reset_index(inplace=True, drop=True)
    meas_data = meas_data[['TS_or_Distance','T', 'lat', 'lon','Message']]

    meas_data = pd.concat([meas_data, extract_string_column(meas_data,'Message')],axis=1)
     
    # Close connection
    #==============#
    if(conn):
        cursor.close()
        conn.close()
        print("PostgreSQL connection is closed")
        
    return meas_data


def filter_latlon(data, col_string, lat_min, lat_max, lon_min, lon_max):
    data = data[data['lat'].between(lat_min,lat_max)]
    data = data[data['lon'].between(lon_min,lon_max)]
    data.reset_index(inplace=True, drop=True)
    return data


def load_DRD_data(DRD_trip, is_p79 = False, is_ARAN = False, load_nrows = -1):
    '''
    
    Use this function to load and examin DRD data.
    
    Parameters
    ----------
    DRD_trip : string
        DRD trip id.
    is_p79 : TYPE, optional
        DESCRIPTION. The default is False.
    is_ARAN : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    sql_data : TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.
    trips : TYPE
        DESCRIPTION.

    '''
       
    # Set up connection
    print("\nConnecting to PostgreSQL database to load the DRD data")
    conn = psycopg2.connect()

    
    # Execute quory: get sensor data
    print('Selecting data')
    if is_ARAN:
        quory = 'SELECT * FROM "DRDMeasurements" WHERE "FK_Trip"=\'{0}\' ORDER BY "TS_or_Distance" ASC'.format(DRD_trip)
    elif is_p79:
        #quory = 'SELECT "DRDMeasurementId","TS_or_Distance","T","lat","lon","message" FROM "DRDMeasurements" WHERE ("FK_Trip"=\'{0}\' AND "lat"<={1} AND "lat">={2} AND "lon"<={3} AND "lon">={4}) ORDER BY "TS_or_Distance" ASC'.format(DRD_trip, lat_max, lat_min, lon_max, lon_min)
        quory = 'SELECT "DRDMeasurementId","TS_or_Distance","T","lat","lon","message" FROM "DRDMeasurements" WHERE "FK_Trip"=\'{0}\' ORDER BY "TS_or_Distance" ASC'.format(DRD_trip)
    else:
        print('Set either p79 or ARAN to true. Other vehicle trips not implemented yet.')
        sys.exit(0)
    
    # Set the number of rows to load
    if load_nrows!=-1:
        quory = '{0} LIMIT {1}'.format(quory, load_nrows)
        
    # Load and sort data
    cursor = conn.cursor()
    sql_data = pd.read_sql(quory, conn, coerce_float = True)

    # Sort also in pandas after conversion to float
    sql_data.TS_or_Distance = sql_data.TS_or_Distance.map(lambda raw: float(raw.replace(',','.')))
    sql_data['TS_or_Distance'] = sql_data['TS_or_Distance'].astype(float)
    sql_data.sort_values(by ='TS_or_Distance', inplace=True)
    sql_data.reset_index(drop = True, inplace=True)
    
    if is_ARAN:
        drop_cols = ['DRDMeasurementId', 'T', 'isComputed', 'FK_Trip', 'FK_MeasurementType', 'Created_Date',
       'Updated_Date','BeginChainage','EndChainage']
        extract_string_column(sql_data)
        sql_data.drop(drop_cols, axis=1, inplace = True)
         
    if is_p79:
        iri =  sql_data[sql_data['T']=='IRI']
        iri['IRI_mean'] = iri.message.apply(lambda message: (loads(message)['IRI5']+loads(message)['IRI21'])/2)
        iri.drop(columns=['message','DRDMeasurementId', 'T',],inplace=True,axis=1)
        
        # Filter iri
        iri = iri[(iri.lat>0) & (iri.lon>0)]
        iri.reset_index(drop=True, inplace=True)
    
    # Get information about the trip
    print('Getting trip information')
    quory = 'SELECT * FROM "Trips"'
    #quory = 'SELECT * FROM "Trips" WHERE "TaskId"=\'0\''
    cursor = conn.cursor()
    trips = pd.read_sql(quory, conn) 
    
    # Close connection
    if(conn):
        cursor.close()
        conn.close()
        print("PostgreSQL connection is closed")
        
    # Return
    if is_p79:   
        return sql_data, iri, trips
    elif is_ARAN:
        return sql_data, None, trips




  
