"""
@author: Milena Bajic (DTU Compute)
e-mail: lenka.bajic@gmail.com
"""

def find_route(trip, route_data):
    """
    

    Parameters
    ----------
    trip : STRING
        Trip id.
    route_data : dictionary
        Dictionary file with routes information about the routes.

    Returns
    -------
    route : STRING
        Route name on which this trip is.

    """
    # Try to find the route name for the given trip
    route = None
    for route_cand in route_data.keys():
        if trip in route_data[route_cand]['GM_trips']:
            route = route_cand
            return route
        
    # If not found set to default name if predict mode
    if not route:
        print('Route not found.')
        return None
        
def set_original_columns(p79 = False, aran = False, viafrik = False):          
    if p79:
        lat_name = 'lat'
        lon_name = 'lon'
        datatype = 'p79'
    elif aran:
        lat_name = 'lat'
        lon_name = 'lon'
        datatype = 'aran'
    elif viafrik:
        lat_name = 'Lat'
        lon_name = 'Lon'
        datatype = 'viafrik'

    return datatype, lat_name, lon_name
    