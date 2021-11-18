"""
@author: Milena Bajic (DTU Compute)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import ceil, floor
import datetime

def df_to_iri_seq(data, iri, var_name, out_dir):
    print('Matching Iri segments')
    # Filter needed
    iri = iri[['DRD_TS_or_Distance', 'DRD_IRI5', 'DRD_IRI21']]
    if var_name=='acc':
        data = data[['GM_Acceleration_x', 'GM_Acceleration_y','GM_Acceleration_z','GM_Acceleration_full', 'Date','Time','DRD_TS_or_Distance', 'DRD_lat', 'DRD_lon','lat','lon']]
    elif var_name=='speed':
        data = data[['GM_Speed', 'Date','Time','DRD_TS_or_Distance', 'DRD_lat', 'DRD_lon','lat','lon']]
    else:
        data = data[[var_name, 'Date','Time','DRD_TS_or_Distance', 'DRD_lat', 'DRD_lon','lat','lon']]
        
    data['Datetime'] =  data.apply(lambda row: datetime.datetime.combine(row.Date, row.Time), axis=1)
    data.drop(['Date','Time'],axis=1, inplace=True)
    data.rename(columns={'DRD_TS_or_Distance':'DRD_Distance'},inplace=True)
    
    # Min and max distance in matched data
    min_d =  data.DRD_Distance.min()
    max_d = data.DRD_Distance.max()    

    # Get the closest int larger than this num and divisable by 10
    min_iri_d = 10*ceil(min_d/10)
    max_iri_d = 10*floor(max_d/10)

    # Prepare dataframe for seq data
    segments = [] #all segments
    
    # Road segment
    for cur_iri in range(min_iri_d,max_iri_d,10): 
        print('Segment: {0}-{1}'.format(cur_iri,cur_iri+10))
        segment_data={}

        matching = data[data.DRD_Distance.between(cur_iri,cur_iri+10)]
        if matching.shape[0]==0:
            continue
        
        t0 = matching['Datetime'].min()
        matching['Time'] = matching['Datetime'].apply(lambda t: (t-t0).total_seconds())
        matching.drop(['Datetime'],axis=1, inplace=True)
        
        # Fill seq df
        for col in matching.columns:
            segment_data[col] = [matching[col].to_numpy()]
        for col in iri.columns:
            segment_data[col] = iri[iri.DRD_TS_or_Distance==cur_iri+10][col]
   
     
        # Skip segments with less than 10 points
        if segment_data['Time'][0].shape[0]<10:
            continue
            
        # Append this segment
        segments.append(pd.DataFrame(segment_data))
  
    sequence_df = pd.concat(segments)
    
    # Skip segments with opposite directions
    sequence_df = sequence_df[sequence_df['Time'].apply(lambda row: row[-1]-row[0]>0)]
    
    # Save
    sequence_df.reset_index(inplace=True, drop=True)
    sequence_df.drop(['DRD_Distance','DRD_TS_or_Distance'],axis=1, inplace=True)
    sequence_df.to_pickle('matched_iri.pkl')
    iri.to_excel(r'matched_iri.xlsx', index = False)
        
    return sequence_df
    
    