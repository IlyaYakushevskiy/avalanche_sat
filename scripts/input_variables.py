import datetime
from datetime import timedelta
import numpy as np
import pandas as pd

#####################################################################
def get_time_wettest_profile(smet_df,datetime_str):
    
    # get the time during the day (date) at which the profile is the wettest
    # date: date in str format '%Y-%m-%d'
    # S5: LWC_index

    datetime_str_to = datetime.datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S') + datetime.timedelta(days=1)
    datetime_str = datetime.datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')

    sub = smet_df.loc[(smet_df.timestamp>=datetime_str)&(smet_df.timestamp<datetime_str_to)]

    profile_datetime = str(sub.loc[sub.hour=='15:00:00','timestamp'].max()) ### defalut profile time if no LWC

    if len(sub)>0:
        max_lwc = sub.S5.max()
        arg_max_lwc = sub.S5.idxmax()
        if max_lwc>0: profile_datetime = str(sub.loc[arg_max_lwc]['timestamp'])

    return profile_datetime

#####################################################################
def wet_feat_extract_live(pro_dic,d):			
    
    # compute LWC related input feature
    # profile as a dic (output of read_profile)
    # d: date and hour in str format '%Y-%m-%d %H:%M:%S'
    # the hour correspond to the time of the wettest profile of the day

    df = pd.DataFrame.from_dict(pro_dic['data']).T
    df.index = df.index.astype(str)
    a = df.loc[df.index== d]
    
    if a.isna().sum().sum() == len(list(a)):
        df = pd.DataFrame(columns=['max_lwc', 'water', 'prop_base', 'HS', 'prop_wet', 'prop_up', 
                       'depth_max_lwc_1','depth_max_lwc_2','max_depth_wet_layer','sum_up','sum_base'],index=[0])
    else:
        
    
        slice = pd.DataFrame(index = a.height.values[0],columns=['thickness','lwc'])
        slice.lwc = a.lwc.values[0]
        i = np.zeros(len(slice))
        i[0] = a.height.values[0][0]
        i[1:] = np.diff(a.height.values[0])
        slice.thickness = i

        mean_lwc = sum(slice.lwc * slice.thickness) / slice.index.max()
        max_lwc = a.lwc[0].max()
        water = sum(slice.lwc * slice.thickness)

        HS = a.height[0].max()
        wet = slice.loc[slice.lwc>0]
        if len(wet)>0: max_depth_wet_layer = (HS - wet.index).max()
        else: max_depth_wet_layer = 0
        depth_max_lwc_1 = HS - slice.loc[slice.lwc==max_lwc].index.max()
        depth_max_lwc_2 = HS - slice.loc[slice.lwc==max_lwc].index.min()



        base = slice.loc[(slice.index<=30)]
        up = slice.loc[(HS - slice.index)<=15]

        prop_base = base.loc[base.lwc>0,'thickness'].sum() / base.thickness.sum()
        prop_wet = slice.loc[(slice.lwc>0),'thickness'].sum() / slice.thickness.sum()
        prop_up = up.loc[up.lwc>0,'thickness'].sum() / up.thickness.sum()

        sum_up = sum(up.lwc * up.thickness) / up.thickness.sum()
        sum_base = sum(base.lwc * base.thickness) / base.thickness.sum()

        features = [mean_lwc,max_lwc, water, prop_base, HS, prop_wet, prop_up, 
                           depth_max_lwc_1,depth_max_lwc_2,max_depth_wet_layer,sum_up,sum_base]

        data = np.array(features).reshape(1,len(features))

        df = pd.DataFrame(data=data,columns=['mean_lwc','max_lwc', 'water', 'prop_base', 'HS', 'prop_wet', 'prop_up', 
                           'depth_max_lwc_1','depth_max_lwc_2','max_depth_wet_layer','sum_up','sum_base'],index=[0])
    return df

#####################################################################
def smet_features(smet_df,datetime_str,name,datetime_profile):

    datetime_str_to = datetime.datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S') + datetime.timedelta(days=1)    
    datetime_str = datetime.datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')

    date_15 = datetime.datetime.strptime(datetime_profile, '%Y-%m-%d %H:%M:%S')

     
    smet_df = smet_df.rename({'timestamp':'datum'},axis=1)
        
    ### smet features: daily and at 3 PM
  
    
    smet_df_date = smet_df.loc[(smet_df.datum>=datetime_str)&(smet_df.datum<datetime_str_to)]

    smet_df_date.index = np.zeros(len(smet_df_date))
    smet_daily = smet_df_date.groupby(smet_df_date.index).mean()

    column_name = [x+'_daily' for x in list(smet_daily)]
    smet_daily.columns = column_name
    smet_daily['datum'] = datetime_str
    smet_daily['station_code'] = name[0:4]
    smet_daily['aspect'] = name[4:5]
    smet_daily['smet_from'] = smet_df_date.datum.min()
    smet_daily['smet_to'] = smet_df_date.datum.max()
    
    ### smet at profile time

    smet_15 = smet_df.loc[smet_df.datum==date_15]
    smet_15.index = [str(datetime_str)]
    
    return smet_daily, smet_15



#####################################################################
def profile_features(profile,profile_datetime):

    
    date_15 = datetime.datetime.strptime(profile_datetime, '%Y-%m-%d %H:%M:%S')

    elevation = profile['info']['altitude']
    
    #### features extracted 0, 1, 2, 3 days before date_str
    
    period = [0,1,2,3]
    P = []
    for p in period:
        dt_day_ = date_15 - datetime.timedelta(days=p)
        R = wet_feat_extract_live(profile,str(dt_day_))
        
        R['datum'] = dt_day_.date()
        P = P +[R]
        
    P = pd.concat(P)
    P.index = pd.to_datetime(P.datum)
    P['elevation'] = elevation
    P =  P.resample('1d').first()
    feat = [x for x in list(R) if 'datum' not in x]    
    
    
    #### compute temporal changes
    
    for f in feat:
        P[f+'_'+ str(1) +'_diff'] = P[f]-P[f].shift(1)
        P[f+'_'+ str(2) +'_diff'] = P[f]-P[f].shift(2)
        P[f+'_'+ str(3) +'_diff'] = P[f]-P[f].shift(3)
   
    P = P.loc[P.index==profile_datetime[0:10]]
    
    return P


#####################################################################
def process_smet_pro_forecast(smet_df, profile_dic,datetime_str,name):
    
    smet_df['date'] = [str(x.date()) for x in smet_df.timestamp]
    smet_df['hour'] = [str(x.hour)+':00:00' for x in smet_df.timestamp]
    smet_df = smet_df.fillna(0)

    profile_datetime = get_time_wettest_profile(smet_df,datetime_str)
    
    smet_daily, smet_15 = smet_features(smet_df,datetime_str,name,profile_datetime)    #take dataframe and generate smet's features
    P = profile_features(profile_dic,profile_datetime)      #take dic and generate profile's features
    
    data = pd.concat([P.reset_index(drop=True),smet_daily.reset_index(drop=True),smet_15.reset_index(drop=True)],axis=1)
    data['profile_time'] = profile_datetime
    data = data.loc[:,~data.columns.duplicated()].copy()
    
    return data


#####################################################################
def concat_nowcast_forecast(smet_nowcast,smet_forecast,pro_nowcast,pro_forecast):
    
    smet_nowcast['nowcast'] = True
    smet_forecast['nowcast'] = False
    ### pro input format is dic ---> convert it to dataframe
    pro_nowcast_df = pd.DataFrame.from_dict(pro_nowcast)

    pro_nowcast_df['nowcast'] = True

    pro_forecast_df = pd.DataFrame.from_dict(pro_forecast)

    pro_forecast_df['nowcast'] = False

    ### remove header
    pro_forecast_df = pro_forecast_df[6:]


    ### remove overlaping timestamp
    smet_forecast = smet_forecast.loc[smet_forecast.timestamp>smet_nowcast.timestamp.max()]
    pro_forecast_df = pro_forecast_df.loc[pro_forecast_df.index>pro_nowcast_df.index[6:].max()]


    smet = pd.concat([smet_nowcast,smet_forecast])
    pro = pd.concat([pro_nowcast_df,pro_forecast_df])

    ### convert pro to dict (initial format)
    pro_dict = pro.to_dict()

    return smet, pro_dict