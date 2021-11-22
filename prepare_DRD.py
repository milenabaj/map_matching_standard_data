"""
@author: Milena Bajic (DTU Compute)
e-mail: lenka.bajic@gmail.com
"""

import sys, os, argparse
import pandas as pd
from utils.data_loaders import *
from utils.plotting import *
from utils.matching import *
from utils.helpers import *
import json

#=================================#
# SETTINGS
#=================================#
# Script arguments
parser = argparse.ArgumentParser(description='Please provide command line arguments.')

parser.add_argument('--route', default= None, help='Process all trips on this route, given in json file.')
parser.add_argument('--trip', default = None, type=int, help='Process this trip only. The route name will be loaded from jthe json file.')

parser.add_argument('--p79', action='store_true', help = 'If this is p79 data, pass true.')
parser.add_argument('--aran', action='store_true', help = 'If this is aran data, pass true.')
parser.add_argument('--viafrik', action='store_true', help = 'If this is Viafrik friction data, pass true.')

parser.add_argument('--map_match', action='store_true', help = 'To map match GPS coordinates, pass true. The default is False.')
parser.add_argument('--plot', action='store_true', help = 'To plot data on Open Streep Map, pass true. The default is False.')

parser.add_argument('--routes_file', default= "json/routes.json", help='Json file with route information.')
parser.add_argument('--conn_file', default= "json/.connection.json", help='Json file with connection information.')
parser.add_argument('--out_dir', default= "data", help='Output directory.')
parser.add_argument('--preload_map_matched', action='store_true', help = 'Preload map matched output if it exists.') 
parser.add_argument('--preload_plots', action='store_true', help = 'Preload plots if they exist.') 
parser.add_argument('--dev_mode', action='store_true', help = 'Development mode. Will process a limited number of lines. Use only for testing.') 

# Parse arguments
args = parser.parse_args()

# Route, trip
route = args.route
trip = args.trip

# Datatype
p79 = args.p79
aran = args.aran
viafrik = args.viafrik

# Map match
do_map_match = args.map_match

# Additional setup
plot = args.plot
routes_file = args.routes_file
conn_file = args.conn_file
out_dir = args.out_dir
preload_map_matched = args.preload_map_matched
preload_plots = args.preload_plots
dev_mode = args.dev_mode

# Temporaraly 
viafrik = False
aran = True
trip  = 7792
plot = False
preload_map_matched = True
do_map_match = False

#=========================#
# PREPARATION 
#=========================#
# Check if exactly one data type is chosen
datatypes_bool = [p79, aran, viafrik]
n_datatypes = datatypes_bool.count(True)
if n_datatypes!=1:
    print('Set exactly one data type to process. Choose p79 or aran or viafrik.')
    sys.exit(0)

# Exit if both route and trip are passed
if route and trip:
    print('Do not choose both route and trip. If a route is passed - all trips in the json file for this route will be used. If a trip is passed, only this trip will be used and the route name will be loaded from the json file.')
    sys.exit(0) 
    
# If none passed, also exit
if not route and not trip:
    print('Choose either a route or a trip. If a route is passed - all trips in the json file for this route will be used. If a trip is passed, only this trip will be used and the route name will be loaded from the json file.')
    sys.exit(0)

# Load route info file
with open(routes_file, "r") as f:
    route_data = json.load(f)
    
# Load connection info file
with open(conn_file, "r") as f:
    conn_data = json.load(f)
        
# If trip pased, find route for this trip
if trip:
   trips_thisroute = [trip]
   route = find_route(trip, route_data)
   
   # If route not found, set to unkown
   if not route:
       route= 'unknown'

# If route passed, use all trips
if route:
    if p79:
        trips_thisroute =  route_data[route]['P79_trips']
    elif aran:
        trips_thisroute =  route_data[route]['ARAN_trips']     
    elif viafrik:
        trips_thisroute =  route_data[route]['VIAFRIK_trips'] 
        
     # If no trips found for user passed route, then exit
    if not trips_thisroute:
        print('No trips found for this route in json file. Please add trips for this route.')
        sys.exit(0)

# Create output directory for this route
if p79:
    out_dir_route = '{0}/P79_processesed_data/{1}'.format(out_dir, route)
    datatype = 'p79'
elif aran:
    out_dir_route = '{0}/ARAN_processesed_data/{1}'.format(out_dir, route)
    datatype = 'aran'
elif viafrik:
    out_dir_route = '{0}/VIAFRIK_processesed_data/{1}'.format(out_dir, route)
    datatype = 'viafrik'
if not os.path.exists(out_dir_route):
    os.makedirs(out_dir_route)

# Create output subdirectory for route plots
out_dir_plots = '{0}/plots'.format(out_dir_route)
if not os.path.exists(out_dir_plots):
    os.makedirs(out_dir_plots)
 
#=========================#
# PROCESSING 
#=========================#        
# Process trips
for trip in trips_thisroute:
    
    # Map match filename
    file_suff =  'route-{0}_taskid-{1}_{2}'.format(route, trip, datatype)
    full_map_match_filename = '{0}/map_matched_data{1}.pickle'.format(out_dir_route, file_suff)
            
    # Load if asked to preload the map matched file and if it exists 
    if do_map_match and preload_map_matched:
        if os.path.exists(full_map_match_filename):
            print('Loading map matched result: {0}'.format(full_map_match_filename))
            DRD_data = pd.read_pickle(full_map_match_filename)
         
            # Plot corrected trajectory on OSRM
            if plot:
                plot_geolocation(map_matched_data['lon_map'],  map_matched_data['lat_map'], name = 'DRD_{0}_GPS_mapmatched_gpspoints'.format(trip), out_dir = out_dir_plots, plot_firstlast = 100, preload = preload_plots)
                
        else:
             print('Map match output file not found. Exiting')
             sys.exit(0)
             
    # Else do the pipeline
    else:
        '''
        # Load data
        if dev_mode:
            DRD_data, iri, DRD_trips = load_DRD_data(trip, p79 = p79, aran = aran, load_nrows = 10, preload = True) 
        else:
            if viafrik:
                DRD_data = load_viafrik_data(trips_thisroute[0])
                iri = None
            elif aran:
                DRD_data, iri, DRD_trips = load_DRD_data(trip, p79 = p79, aran = aran, dev_mode = dev_mode) 
                RD_data.dropna(subset=['lat','lon'], inplace=True)  # Drop nans
                iri = None
            else:
                DRD_data, iri, DRD_trips = load_DRD_data(trip, p79 = p79, aran = aran) 
        '''
        if viafrik:
            DRD_data = load_viafrik_data(trips_thisroute[0])
            iri = None
        else:
            DRD_data, iri, DRD_trips = load_DRD_data(trip, conn_data, p79 = p79, aran = aran, dev_mode = dev_mode) 
        sys.exit(0)
        
        # Select only trips from one year
        DRD_trips_filter = filter_DRDtrips_by_year(DRD_trips, sel_2021 = True)

        # d[ d['Datetime']<'2020-12-31']
   
        #DRD_trips_aran = DRD_trips[DRD_trips['aran']==True]
        #DRD_trips_aran_2021 =  DRD_trips_aran[DRD_trips_aran['Datetime']>'2021-01-01']
    
        # Drop duplicate columns (due to ocassical errors in database)
        DRD_data = DRD_data.T.drop_duplicates().T # Lat and Lon are stored twice for Viafrik?? Error in the database
        if iri:
            iri = iri.T.drop_duplicates().T
        
        if plot:
            plot_geolocation(DRD_data['lon'],  DRD_data['lat'], name = 'route6_{0}'.format(trip), out_dir = out_dir_plots, plot_firstlast = 100, preload = preload_plots)
          
    
        # Map match 
        if do_map_match:
        
            # Standard vehicle type map matching
            if aran:
                DRD_data.dropna(subset=['lat','lon'], inplace=True) 
                DRD_data = map_match_gps_data(DRD_data, lat_name= 'lat', lon_name = 'lon', is_GM = False, out_dir = out_dir_route , out_file_suff = file_suff)
            elif p79:
                DRD_data = map_match_gps_data(iri, is_GM = False, out_dir = out_dir_route , out_file_suff = file_suff)
            elif viafrik:
                DRD_data = map_match_gps_data(DRD_data, lat_name= 'Lat', lon_name = 'Lon', is_GM = False, out_dir = out_dir_route , out_file_suff = file_suff)
           
            # Plot corrected trajectory on OSRM
            if plot:
                plot_geolocation(DRD_data['lon_map'],  DRD_data['lat_map'], name = 'DRD_{0}_GPS_mapmatched_gpspoints'.format(trip), out_dir = out_dir_plots, plot_firstlast = 100, preload = preload_plots)
            
            
        
        