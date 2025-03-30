#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on September 2022

@author: hendricks martin

Edited by:
@author: richter bettina
"""

import sys
sys.path.append('./scripts/')
from input_variables import process_smet_pro_forecast
from smet import read_smet
from read_profile import read_profile

import pandas as pd
import pickle

#%%

path_models = './scripts/RF_model_wetsnow/'

model_2001_2022 = pickle.load(open(path_models+'rf_2001_2022.sav', 'rb'))
features_model_2001_2022 = pd.read_csv(path_models+'rf_2001_2022.csv')
features_model_2001_2022 = list(features_model_2001_2022.features)

def run_rf_wet(df):
    df = df[list(set(features_model_2001_2022+['elevation']))+['station_code','aspect','profile_time','datum']].dropna(0)
    model_predicted_probability_2001_2022 = model_2001_2022.predict_proba(df[features_model_2001_2022])
    df['probability_wet_AvD_model_2001_2022'] = model_predicted_probability_2001_2022[:,1]
    return(df)
    
finput='./input/'
station='WFJ2'
smet = read_smet(finput+station+'.smet')
profile_dic = read_profile(finput+station+'.pro')

#%%
start = '2021-11-01'
til = '2022-06-01'
d_list =  pd.date_range(start=start, end=til)

#%%
D = []
for d in d_list:    
    date_str = d.strftime('%Y-%m-%d %H:%M:%S')
    try:
        data = process_smet_pro_forecast(smet,profile_dic,date_str,station)
    except:
        continue
    D = D + [data]
df = pd.concat(D)
#%%
df = run_rf_wet(df)

#%%
fout = './output/'
df.to_csv(fout+'/'+station+'.csv')
