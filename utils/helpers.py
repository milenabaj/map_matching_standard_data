#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 15:25:07 2021

@author: mibaj
"""

def find_route(trip, route_data):
    
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
        
             

    