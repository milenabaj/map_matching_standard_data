"""
@author: Milena Bajic (DTU Compute)
e-mail: lenka.bajic@gmail.com
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

def load_DRD_data(DRD_trip, conn_data, prod_db = True, p79 = False, aran = False, viafrik = False, dev_mode = False, load_n_rows = 500):
    '''
    
    Use this function to load and examin DRD data.
    
    Parameters
    ----------
    DRD_trip : string
        DRD trip id.
    conn_data: dictionary
        Database connection information.
    prod_db: BOOL
        Use production database. If False, will use development database. The default is True.
    p79 : BOOL, optional
        DESCRIPTION. The default is False.
    aran : BOOL, optional
        DESCRIPTION. The default is False.
    dev_mode: BOOL, optional
        The code will load load_n_rows lines. The default is False.
    load_n_rows: INT, optinal
        DESCRIPTION. The default is 500.
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
    
    if prod_db:
        print("\nConnecting to production database")
        db_data = conn_data['prod']
    else:
        print("\nConnecting to development database")
        db_data = conn_data['dev']   
        
    db = db_data['database']
    username = db_data['user']
    password = db_data['password']
    host = db_data['host']
    port = db_data['port']

    # Connection    
    conn = psycopg2.connect(database=db, user=username, password=password, host=host, port=port)
    
    # Execute quory: get sensor data
    print('Selecting data for trip: {0}'.format(DRD_trip))
    if aran:
        quory = 'SELECT * FROM "DRDMeasurements" WHERE "FK_Trip"=\'{0}\' ORDER BY "TS_or_Distance" ASC'.format(DRD_trip)
    elif (p79 or viafrik):
        quory = 'SELECT "DRDMeasurementId","TS_or_Distance","T","lat","lon","message" FROM "DRDMeasurements" WHERE "FK_Trip"=\'{0}\' ORDER BY "TS_or_Distance" ASC'.format(DRD_trip)
    else:
        print('Set either p79 or aran or viafrik to True.')
        sys.exit(0)
    
    # Set the number of rows to load
    if dev_mode:
        quory = '{0} LIMIT {1}'.format(quory, load_n_rows)
        
    # Load and sort data
    cursor = conn.cursor()
    sql_data = pd.read_sql(quory, conn, coerce_float = True)

    # Sort also in pandas after conversion to float
    sql_data.TS_or_Distance = sql_data.TS_or_Distance.map(lambda raw: float(raw.replace(',','.')))
    sql_data['TS_or_Distance'] = sql_data['TS_or_Distance'].astype(float)
    sql_data.sort_values(by ='TS_or_Distance', inplace=True)
    sql_data.reset_index(drop = True, inplace=True)
    
    # Preparation depending on data type
    if aran:
        drop_cols = ['DRDMeasurementId', 'T', 'isComputed', 'FK_Trip', 'FK_MeasurementType', 'Created_Date',
       'Updated_Date','BeginChainage','EndChainage']
        extract_string_column(sql_data)
        sql_data.drop(drop_cols, axis=1, inplace = True)
    elif viafrik:
        sql_data['Message'] = sql_data.message.apply(lambda msg: filter_keys(loads(msg), remove_gm=False))
        sql_data.drop(columns=['message'],inplace=True,axis=1)
        sql_data.reset_index(inplace=True, drop=True)
        sql_data = sql_data[['TS_or_Distance','T', 'lat', 'lon','Message']]
        sql_data = pd.concat([sql_data, extract_string_column(sql_data,'Message')],axis=1)
    elif p79:
        iri =  sql_data[sql_data['T']=='IRI']
        iri['IRI_mean'] = iri.message.apply(lambda message: (loads(message)['IRI5']+loads(message)['IRI21'])/2)
        iri.drop(columns=['message','DRDMeasurementId', 'T',],inplace=True,axis=1)
        iri = iri[(iri.lat>0) & (iri.lon>0)]
        iri.reset_index(drop=True, inplace=True)
    
    # Get information about the trip
    print('Getting DRD trips information')
    quory = 'SELECT * FROM "Trips" WHERE "TaskId"=\'0\''
    cursor = conn.cursor()
    trips = pd.read_sql(quory, conn)         
        
    # Close connection
    if(conn):
        cursor.close()
        conn.close()
        print("PostgreSQL connection is closed")
        
    if p79:   
        return sql_data, iri, trips
    else:
        return sql_data, None, trips



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


def filter_DRDtrips_by_year(DRD_trips, sel_2021 = False, sel_2020 = False):
    DRD_trips['Datetime']=pd.to_datetime(DRD_trips['Created_Date'])
    if sel_2020: 
        DRD_trips['Year'] = DRD_trips['Datetime'].apply(lambda date: '2020')
        DRD_trips = DRD_trips[ DRD_trips['Year']=='2020']
    elif sel_2021:
        DRD_trips['Year'] = DRD_trips['Datetime'].apply(lambda date: '2021')
        DRD_trips[ DRD_trips['Year']=='2021']
    DRD_trips.drop(['Datetime','Year'], axis=1, inplace=True)
    return DRD_trips
    

def drop_duplicates(DRD_data, iri):
    # Drop duplicate columns (due to ocassical errors in database)
    DRD_data = DRD_data.T.drop_duplicates().T #
    if iri:
        iri = iri.T.drop_duplicates().T
    return DRD_data, iri