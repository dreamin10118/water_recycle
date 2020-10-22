import pandas as pd
import numpy as np
import datetime

def get_date(str_datetime):
    """
    從字串裡取出日期
    str_datetime: 字串，預期形式：2020-08-21 21:37:51.043661+08:00
    """
    arr_date = np.array(str_datetime.split(' ')[0].split('-')).astype('int')
    
    return datetime.date(arr_date[0], arr_date[1], arr_date[2])


def get_time(str_datetime):
    """
    從字串裡取出時間
    str_datetime: 字串，預期形式：2020-08-21 21:37:51.043661+08:00
    """
    arr_time = np.array(str_datetime.split(' ')[1].split('+')[0].split(':')).astype('int')
    
    return datetime.time(arr_time[0], arr_time[1], arr_time[2])

def time_delta(date, time1, time0):
    """計算2個時間點之間經過幾分鐘
    date: datetime.date，日期
    time0: datetime.time，時間點0
    time1: datetime.time，時間點1
    假設時間點1 > 時間點0
    """
    t0 = datetime.datetime.combine(date, time0)
    t1 = datetime.datetime.combine(date, time1)
    dt = t1 - t0
    dt = dt.seconds / 60
    
    return dt

    
def gen_stat_time(minute_step):
    """產生1天24小時的時間間隔
    minute_step: 整數，必需不高於60。
    例子: minute_step=10 表示每隔10分鐘去劃分24小時
    """
    arr_m = np.arange(0,60,step=minute_step)
    arr_h = np.arange(0,24,step=1)
    cartesian = np.transpose([np.tile(arr_h, len(arr_m)), np.repeat(arr_m, len(arr_h))])
    stat_time = [datetime.time(x, y, 0) for x,y in cartesian]
    stat_time.sort()
    
    return np.array(stat_time) 


def concat_equal_time_partition_df(stat_time, data, concat_days, method):
    """將資料以某個時間間隔切塊
    stat_time: gen_stat_time 的回傳
    data: pandas.DataFrame
    concat_days: datetime.date，表示要切塊的日期
    method: str，統計數，例如 "mean"、"max"，當輸入"tail"表示取最後一筆
    """
    
    df_list = [get_equal_time_partition_df(stat_time,  data[data['date']==day], method) for day in concat_days]
    concat_df = pd.concat(df_list,axis=0)
    concat_df['reg_time'] = concat_df[['date','stat_time']].apply(lambda x:datetime.datetime.combine(x[0], x[1]), axis=1)
    
    return concat_df

def get_equal_time_partition_df(stat_time, df, method):
    """將某一天資料以某個時間間隔切塊
    stat_time: gen_stat_time 的回傳
    df: pandas.DataFrame
    method: str，統計數，例如 "mean"、"max"，當輸入"tail"表示取最後一筆
    """
    
    t = stat_time
    str_stat_time = [f"datetime.time({t[i].hour},{t[i].minute})" for i in range(len(t))]   
    dfs = [df.query(f"time<={str_stat_time[i+1]}").query(f"time>={str_stat_time[i]}") for i in range(len(t)-1)]
    result = pd.DataFrame()
    
    df_len = np.array([len(x) for x in dfs])
    start_idx = np.where(df_len>0)[0].min()
    end_idx =  np.where(df_len>0)[0].max()
   
    first = dfs[start_idx]['time'].values.min() 
    first_stat_time = t[t>first].min()
    first_time_idx = np.where(t==first_stat_time)[0][0]

    int_id_diff = start_idx - first_time_idx
    
    for i in range(start_idx,end_idx+1):

        df_i = dfs[i]
        if len(df_i) > 0:
            df_i['stat_time'] = t[i-int_id_diff] 
            df_i = df_i.groupby('stat_time').agg(method).tail(1).reset_index()
            result = pd.concat([result,df_i],axis=0)
        else:
            df_i = result.tail(1).copy() 
            df_i['stat_time'] = t[i-int_id_diff] 
            result = pd.concat([result,df_i],axis=0)
            
    result['date'] = df['date'].unique()[0]

    table_last_time = result['stat_time'].values.max()
    last_row = df[df['time']>table_last_time].copy()
    
    if len(last_row) > 0:
        append_row = last_row
        append_row['stat_time'] = t[0]
        append_row = last_row.groupby('stat_time').agg(method).tail(1).reset_index()
        append_row['date'] = df['date'].unique()[0]+datetime.timedelta(days = 1)
        result = pd.concat([result, append_row], axis=0)
        
    result.reset_index(drop=True, inplace=True)
    
    return result

def get_velocity_acc_df(df, var, col_time, drop):
    """將選擇的資料欄位計算速度與加速度，已經假設資料按照時間先後排序
    df: pandas.DataFrame
    var: str， 欄位名稱
    col_time: str，紀錄時間的欄位名稱
    drop: bool，是否將計算過程產生的欄位去除
    """
    
    lag_var = f"lag_{var}"
    vel_var = f"{var}_velocity"
    lag_vel_var = f"lag_{var}_velocity"
    acc_var = f"{var}_acc"
    
    speed_df = df.copy()
    speed_df[lag_var] = speed_df[var].shift(1)
    speed_df['lag_time'] = speed_df[col_time].shift(1)
    speed_df = speed_df.dropna()
    speed_df['time_diff'] =speed_df[['date',col_time,'lag_time']].apply(lambda x: time_delta(x[0],x[1],x[2]), axis=1)
    speed_df[vel_var] = speed_df.eval(f"({var} - {lag_var})/time_diff")
    speed_df[lag_vel_var] = speed_df[vel_var].shift(1)
    speed_df = speed_df.dropna()
    speed_df[acc_var] = speed_df.eval(f"({vel_var} - {lag_vel_var})/time_diff")
    
    if drop == True:
        speed_df.drop([f"lag_{var}",f"lag_{var}_velocity",'lag_time','time_diff'],axis=1,inplace=True)
    return speed_df

def get_velocity_acc_equal_time_partition(stat_time, data, var, method, drop):
    """將資料切成相同時間間隔，並把選擇的資料欄位計算速度與加速度
    data: pandas.DataFrame
    var: str， 欄位名稱
    method: str，統計數，例如 "mean"、"max"，當輸入"tail"表示取最後一筆
    drop: bool，是否將計算過程產生的欄位去除
    """
    
    df_partition =  get_equal_time_partition_df(stat_time, data, method)
    df_partition['date'] = data['date'].unique()[0]
    
    return get_velocity_acc_df(df_partition, var,'stat_time', drop).reset_index(drop=True)
        

def concat_velocity_acc_df(stat_time, data, concat_days, var, method, drop=False):
    """再給定某天資料下，將相同時間間隔資料計算速度與加速度。最後把資料依照日期整合
    stat_time: gen_stat_time 的回傳
    data: pandas.DataFrame
    var: str， 欄位名稱
    method: str，統計數，例如 "mean"、"max"，當輸入"tail"表示取最後一筆
    drop: bool，是否將計算過程產生的欄位去除
    """
    
    df_list = [get_velocity_acc_equal_time_partition(stat_time, data[data['date']==day], var, method, drop) for day in concat_days]
    concat_df = pd.concat(df_list,axis=0)
    concat_df['reg_time'] = concat_df[['date','stat_time']].apply(lambda x:datetime.datetime.combine(x[0], x[1]), axis=1)
    
    return concat_df

def merge_many_dfs(df_list, join_col):
    """水平整併資料，以列數最多的資料表為主，將其他資料表整併
    df_list: list, 每個元素都是pandas.DataFrame
    join_col: str, 要整併的參照欄位
    """
    df_len = tuple(len(x) for x in df_list)
    max_row = max(df_len)
    max_row_df_idx = df_len.index(max_row)
    merge_dfs = [df_list[i] for i in range(len(df_list)) if i !=  max_row_df_idx]
    main_df = df_list[max_row_df_idx]
    
    for df in merge_dfs:
        main_df =  main_df.merge(df, on=join_col, how='left')
    
    return main_df
