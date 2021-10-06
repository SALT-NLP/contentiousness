import os
import sys

import pandas as pd

import numpy as np
from geopy import geocoders
from geopy.exc import GeocoderTimedOut
import pickle
import country_converter as coco
from tqdm import tqdm


if len(sys.argv) != 2:
    print("Usage: python get_locations.py data_file")

COMPLETE_DATA = sys.argv[1]

df = pd.read_pickle(COMPLETE_DATA)

def get_users(thread):
    users = [thread.author] if thread.author != '[deleted]' else []
    for comment in thread.comments:
        if comment['author'] != '[deleted]':
            users.append(comment['author'])
        
    return users
        
users = df.apply(get_users, axis=1)
print(users)

user_info_df = pd.read_csv('user_information.csv', sep=';')

def str_to_list(x):
    x = x.strip('][')
    x = x.replace(', ', ',')
    x = x.replace("'", '')
    return x.split(',')

user_info_df.places_lived = user_info_df.places_lived.apply(str_to_list)

user_dict = {}

def to_dict(user):
    user_dict[user.user_name] = user.places_lived

user_info_df.apply(to_dict, axis=1)

def get_unique_locations(x):
    locations = []
    for user in x:
        try:
            user_list = user_dict[user]
            if user_list != ['']:
                locations += user_list
        except:
            continue    
    return set(locations)

def get_locations(x):
    locations = []
    for user in x:
        try:
            user_list = user_dict[user]
            if user_list != ['']:
                locations += user_list
        except:
            continue    
    return locations

unique_locs = users.apply(get_unique_locations)
locs = users.apply(get_locations)
df['unique_locations'] = unique_locs
df['locations'] = locs

all_locations = []

def get_all_locations(x):
    for location in x:
        if location not in all_locations:
            all_locations.append(location)
            
df.locations.apply(get_all_locations)

def vectorize_locations(x):
    ret = np.zeros(len(all_locations))
    for location in x:
        ret[all_locations.index(location)] += 1
        
    return ret

def code_locations(x):
    gn = geocoders.Nominatim(user_agent='nlp-controversy', timeout=6000)
    places = []
    for location in x:        
        try:
            place = gn.geocode(location, addressdetails=True)
            place = place.raw['address']['country_code']
            places.append(place)
        except TypeError as e:
            print(e)
        except GeocoderTimedOut as e:
            print(e)
        except AttributeError as e:
            print(e)
    places = coco.convert(names=places, to='ISO3')
    if isinstance(places, list):
        return places
    else:
        return [places]

tqdm.pandas()
    
df['vectorized_locations'] = df.locations.apply(vectorize_locations)
df['coded_locations'] = df.locations.apply(code_locations)
df.coded_locations.progress_apply(get_all_locations)
df['vectorized_coded_locations'] = df.coded_locations.apply(vectorize_locations)
print(len(df.vectorized_coded_locations.values[0]))
with open('all_locations.pickle', 'wb') as f:
    pickle.dump(all_locations, f)
print(df.coded_locations)

df.to_pickle(COMPLETE_DATA)