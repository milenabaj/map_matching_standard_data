"""
@author: Milena Bajic (DTU Compute)
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
parser.add_argument('--trip', default = None, type=int, help='Process this trip only. The route name will be loaded from json file.')
parser.add_argument('--p79', action='store_true', help = 'If this is p79 data, pass true.')
parser.add_argument('--aran', action='store_true', help = 'If this is aran data, pass true.')
parser.add_argument('--viafrik', action='store_true', help = 'If this is Viafrik friction data, pass true.')
parser.add_argument('--map_match', action='store_true', help = 'To map match GPS coordinates, pass true.')
parser.add_argument('--plot', action='store_true', help = 'To plot data on OSM, pass true.')
parser.add_argument('--json', default= "json/routes.json", help='Json file with route information.')
parser.add_argument('--out_dir', default= "data", help='Output directory.')
parser.add_argument('--preload', action='store_true', help = 'Preload files if they exist. Else, they will created.') 
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
map_match = map_match

# Additional
plot = args.plot
json_file = args.json
out_dir = args.out_dir
preload = args.preload
dev_mode = args.dev_mode

# Temp
viafrik = True
trip  = 7792
preload = True
plot = True
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

# Load json file
with open(json_file, "r") as f:
    route_data = json.load(f)
        
# If trip pased, find route for this trip
if trip:
   trips_thisroute = [trip]
   route = find_route(trip, route_data)
   
   # If route not found, set to unkown
   if not route:
       route= 'unkown'

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
elif aran:
    out_dir_route = '{0}/ARAN_processesed_data/{1}'.format(out_dir, route)
elif viafrik:
     out_dir_route = '{0}/VIAFRIK_processesed_data/{1}'.format(out_dir, route)
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
    
    # Load data
    if dev_mode:
        DRD_data, iri, DRD_trips = load_DRD_data(trip, p79 = p79, aran = aran, load_nrows = 10) 
    else:
        if viafrik:
            DRD_data = load_viafrik_data(trips_thisroute[0])
            iri = None
        elif aran:
            DRD_data, iri, DRD_trips = load_DRD_data(trip, p79 = p79, aran = aran) 
            RD_data.dropna(subset=['lat','lon'], inplace=True)  # Drop nans
            iri = None
        else:
            DRD_data, iri, DRD_trips = load_DRD_data(trip, p79 = p79, aran = aran) 
            
    
    # Drop duplicate columns (due to ocassical errors in database)
    DRD_data = DRD_data.T.drop_duplicates().T # Lat and Lon are stored twice for Viafrik?? Error in database
    if iri:
        iri = iri.T.drop_duplicates().T
    
    if plot:
        plot_geolocation(DRD_data['lon'],  DRD_data['lat'], name = 'route6_{0}'.format(trip), out_dir = out_dir_plots, plot_firstlast = 100, preload = False)
      

    # Map match 
    if map_match:
        file_suff =  'P79_route-{0}_taskid-{1}_full'.format(route, trip)
        full_filename = '{0}/map_matched_data{1}.pickle'.format(out_dir_route, file_suff)
        print(full_filename) 
        
        # Load if map matched file exists and not recreate
        if os.path.exists(full_filename) and preload:
            map_matched_data  = pd.read_pickle(full_filename)
            
        # Map match
        else:
            if aran:
                DRD_data.dropna(subset=['lat','lon'], inplace=True) 
                map_matched_data = map_match_gps_data(DRD_data, is_GM = False, out_dir = out_dir_route , out_file_suff = file_suff, preload = preload)
            elif p79:
                map_matched_data = map_match_gps_data(iri, is_GM = False, out_dir = out_dir_route , out_file_suff = file_suff, preload = preload)
            elif viafrik:
                map_matched_data = map_match_gps_data(DRD_data, lat_name= 'Lat', lon_name = 'Lon', is_GM = False, out_dir = out_dir_route , out_file_suff = file_suff, preload = preload)
       
        # Plot corrected trajectory on OSRM
        if plot:
            plot_geolocation(map_matched_data['lon_map'],  map_matched_data['lat_map'], name = 'DRD_{0}_GPS_mapmatched_gpspoints'.format(trip), out_dir = out_dir_plots, plot_firstlast = 100, preload = False)
        
        
        
        