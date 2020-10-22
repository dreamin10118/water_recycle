from open_model_lib import *
from open_data_func_lib import get_velocity_acc_df

# 讀取資料
all_data = pd.read_pickle('my_water_data.pkl')
# 選取要測試的預測日期
sel_dates =[datetime.date(2020, 9, 23), datetime.date(2020, 9, 30), datetime.date(2020, 10, 4), datetime.date(2020, 10, 7), datetime.date(2020, 10, 8), datetime.date(2020, 10, 9), datetime.date(2020, 10, 12)]

# 進行濃水量預測
t2flow_pred = pd.DataFrame()
for day_i in sel_dates:
    my_df = mydata[mydata['date']<=day_i]
    past_train_data = my_df[my_df['date']<=my_df['date'].unique()[:-6][-1]]

    past_model_i =  ReduceModel(past_train_data, 6, 't2flow_dif').get_reduce_model()
    latest_model_i =  ReduceModel(get_latest_5_day_df(my_df), 6, 't2flow_dif').get_reduce_model()
    t2flow_pred = pd.concat([t2flow_pred, obtain_best_pred(past_model_i, latest_model_i, True)], axis=0)

# 進行產水量預測
outflow_pred  = pd.DataFrame()
for day_i in sel_dates:
    my_df = mydata[mydata['date']<=day_i]
    past_train_data = my_df[my_df['date']<=my_df['date'].unique()[:-6][-1]]

    past_model_i =  ReduceModel(past_train_data, 6, 'outflow_t2').get_reduce_model()
    latest_model_i =  ReduceModel(get_latest_5_day_df(my_df), 6, 'outflow_t2').get_reduce_model()
    outflow_pred  = pd.concat([outflow_pred, obtain_best_pred(past_model_i, latest_model_i, True)], axis=0)

# 進行水回收率預測
water_rate_pred  = pd.DataFrame()
for day_i in sel_dates:
    my_df = mydata[mydata['date']<=day_i]
    past_train_data = my_df[my_df['date']<=my_df['date'].unique()[:-6][-1]]

    past_model_i =  ReduceModel(past_train_data, 6, 'water_prod_rate').get_reduce_model()
    latest_model_i =  ReduceModel(get_latest_5_day_df(my_df), 6, 'water_prod_rate').get_reduce_model()
    water_rate_pred  = pd.concat([water_rate_pred, obtain_best_pred(past_model_i, latest_model_i, False)], axis=0)

water_rate_pred.reset_index(drop=True, inplace=True)
outflow_pred.reset_index(drop=True, inplace=True)
t2flow_pred.reset_index(drop=True, inplace=True)

water_rate_pred.rename(columns={'pred':'water_prod_rate_pred'}, inplace=True)
water_rate_pred.drop(['actual','lag'], axis=1, inplace=True)
outflow_pred.rename(columns={'pred':'outflow_t2_pred'}, inplace=True)
outflow_pred.drop(['actual','lag'], axis=1, inplace=True)
t2flow_pred.rename(columns={'pred':'t2flow_pred'}, inplace=True)
t2flow_pred.drop(['actual','lag'], axis=1, inplace=True)

# 水平合併預測資料
pred_all = pd.concat([water_rate_pred, outflow_pred.drop(['reg_time'],axis=1), t2flow_pred.drop(['reg_time'],axis=1)], axis=1)

# 進行原水槽pH值異常檢測
pH_status = pd.DataFrame()
for day_i in sel_dates:
    pH_alert = WaterAlert(get_latest_5_day_df(mydata[mydata['date']<=day_i]).dropna(), 'date', 'reg_time')
    pH_status = pd.concat([pH_status, pH_alert.get_alert_data('pre_ph','pre_ph_velocity', 'pre_ph_acc', False)], axis=0)

pH_status['pH_alert'] = pH_status[['pre_ph', 'pre_ph_velocity', 'status']].apply(lambda x: 1 if x[0]>3 and x[1]>0 and x[2]==-1 else 0, axis=1)
pH_status.drop(['status'],axis=1,inplace=True)

# 進行脫鹽率異常檢測
de_salt_df = mydata[['reg_time','date','hour','de_salt_rate']].reset_index(drop=True)
de_salt_df['time'] = de_salt_df['hour'].apply(lambda x: datetime.time(x,0)).values
de_salt_speed =  get_velocity_acc_df(de_salt_df , 'de_salt_rate', 'time', True)
de_salt_speed.head()

de_salt_status = pd.DataFrame()
for day_i in sel_dates:
    de_salt_alert = WaterAlert(get_latest_5_day_df(de_salt_speed[de_salt_speed['date']<=day_i]).dropna(), 'date', 'reg_time')
    de_salt_status = pd.concat([de_salt_status, de_salt_alert.get_alert_data('de_salt_rate','de_salt_rate_velocity', 'de_salt_rate_acc', False)], axis=0)

de_salt_status['de_salt_alert'] = de_salt_status[['de_salt_rate_velocity','status']].apply(lambda x: 1 if x[0]<0 and x[1]==-1 else 0, axis=1)
de_salt_status.drop(['status'],axis=1,inplace=True)

# 合併預測資料與異常檢測資料
water_status = pH_status.merge(de_salt_status,on='reg_time')
pred_and_status = pred_all.merge(water_status,on='reg_time',how='left')
pred_and_status.head()