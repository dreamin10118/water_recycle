from open_data_func_lib import *

# 資料前處理
elct_df = pd.read_csv('電表.csv')
flow_df = pd.read_csv('流量.csv')
orp_df = pd.read_csv('導電度.csv')
ph_df = pd.read_csv('酸鹼值.csv')

for df in (elct_df, flow_df, orp_df, ph_df):
    if 'deleted' in df.columns:
        df.drop(['deleted'],axis=1,inplace=True)
    if 'created_at' in df.columns:
        df.drop(['created_at'],axis=1,inplace=True)
    df['date'] = df['uploaded_at'].apply(get_date)
    df['time'] = df['uploaded_at'].apply(get_time)
    df.sort_values(by=['date','time'], inplace=True)
    df.drop(['uploaded_at','id'],axis=1,inplace=True)
    df.reset_index(drop=True,inplace=True)
del df


flow_df = flow_df.query('date>datetime.date(2020,8,31)').reset_index(drop=True)
# 用1小時來做為時間間隔
time_interval = gen_stat_time(60)
inflow = flow_df.query("sensor=='進流流量計'").rename(columns={'fw':'inflow_fw','t1':'inflow_t1','t2':'inflow_t2'})
outflow = flow_df.query("sensor=='產水流量計'").rename(columns={'fw':'outflow_fw','t1':'outflow_t1','t2':'outflow_t2'})

inflow_daily = inflow.groupby(['date'])['inflow_fw'].sum().reset_index()
# 有進流產生的日期
days =  inflow_daily.query('inflow_fw>0').reset_index(drop=True)['date'].unique()
days =  inflow['date'].unique()

# 整理進流與產水資料
vec_acc_inflow = concat_velocity_acc_df(time_interval, inflow, days, 'inflow_t2', 'tail', True).reset_index(drop=True).drop(['date','stat_time'], axis=1)
vec_acc_outflow = concat_velocity_acc_df(time_interval, outflow, days, 'outflow_t2', 'tail',True).reset_index(drop=True).drop(['date','stat_time'], axis=1)
vec_acc_inflow.drop(['index','sensor','fw_unit','inflow_t1','t_unit','time'], axis=1, inplace=True)
vec_acc_outflow.drop(['index','sensor','fw_unit','outflow_t1','t_unit','time'], axis=1, inplace=True)
vec_acc_flow = merge_many_dfs([vec_acc_inflow, vec_acc_outflow],'reg_time')

# 整理電表資料
elct_days = elct_df[elct_df['date'].isin(days)]['date'].unique()
vec_acc_power = concat_velocity_acc_df(time_interval, elct_df, elct_days, 'kwh_tot','mean',True).reset_index(drop=True).drop(['date','stat_time'], axis=1)

# 整理導電度資料
orp_df['unit_name'] = orp_df['unit_name'].replace(["b'mS/cm '","b'uS/cm '"], ['mS/cm ', 'uS/cm '])
orp_df['uS_orp'] = orp_df[['unit_name','orp']].apply(lambda x: x[1]*1000 if x[0] == 'mS/cm ' else x[1], axis=1)
orp_df.drop(['status','temperature','orp'],axis=1,inplace=True)
orp_days = orp_df[orp_df['date'].isin(days)]['date'].unique()
orp_df.drop(['unit_name'],axis=1,inplace=True)

pre_orp = orp_df.query("sensor=='原水導電度計'").rename(columns={'uS_orp':'pre_uS_orp'})
prod_orp = orp_df.query("sensor=='產水導電度計'").rename(columns={'uS_orp':'prod_uS_orp'})
recy_orp = orp_df.query("sensor=='濃水導電度計'").rename(columns={'uS_orp':'recy_uS_orp'})

vec_acc_pre_orp = concat_velocity_acc_df(time_interval, pre_orp, orp_days, 'pre_uS_orp', 'mean',True).reset_index(drop=True).drop(['date','stat_time'], axis=1)
vec_acc_prod_orp = concat_velocity_acc_df(time_interval, prod_orp, orp_days, 'prod_uS_orp', 'mean',True).reset_index(drop=True).drop(['date','stat_time'], axis=1)
vec_acc_recy_orp = concat_velocity_acc_df(time_interval, recy_orp, orp_days, 'recy_uS_orp', 'mean',True).reset_index(drop=True).drop(['date','stat_time'], axis=1)

vec_acc_orp = merge_many_dfs([vec_acc_pre_orp, vec_acc_prod_orp, vec_acc_recy_orp],'reg_time')


# 整理pH值資料
ph_df.drop(['status','temperature','unit_name'],axis=1,inplace=True)
ph_days = ph_df[ph_df['date'].isin(days)]['date'].unique()

pre_ph = ph_df.query("sensor=='原水pH計'").rename(columns={'ph':'pre_ph'})
prod_ph = ph_df.query("sensor=='產水pH計'").rename(columns={'ph':'prod_ph'})
recy_ph = ph_df.query("sensor=='濃水pH計'").rename(columns={'ph':'recy_ph'})

vec_acc_pre_ph = concat_velocity_acc_df(time_interval, pre_ph, days, 'pre_ph', 'mean',True).reset_index(drop=True).drop(['date','stat_time'], axis=1)
vec_acc_prod_ph = concat_velocity_acc_df(time_interval, prod_ph, days, 'prod_ph', 'mean',True).reset_index(drop=True).drop(['date','stat_time'], axis=1)
vec_acc_recy_ph = concat_velocity_acc_df(time_interval, recy_ph, days, 'recy_ph', 'mean',True).reset_index(drop=True).drop(['date','stat_time'], axis=1)

vec_acc_ph = merge_many_dfs([vec_acc_pre_ph, vec_acc_prod_ph, vec_acc_recy_ph],'reg_time')

# 將所有整理好的資料水平整併
all_df = merge_many_dfs([vec_acc_power, vec_acc_flow, vec_acc_orp, vec_acc_ph], 'reg_time')
all_df['water_prod_rate'] = all_df.eval('outflow_t2/inflow_t2')
all_df['de_salt_rate'] = all_df.eval("(pre_uS_orp - prod_uS_orp)/pre_uS_orp")
all_df['inflow_equm'] = all_df[['inflow_fw','inflow_t2_velocity']].apply(lambda x: 1 if x[0]==x[1]==0 else 0, axis=1)
all_df['outflow_equm'] = all_df[['outflow_fw','outflow_t2_velocity']].apply(lambda x: 1 if x[0]==x[1]==0 else 0, axis=1)
all_df['date'] = all_df['reg_time'].apply(lambda x: x.date())
all_df['hour'] = all_df['reg_time'].apply(lambda x: x.hour)
all_df['t2flow_dif'] = all_df.eval('inflow_t2 - outflow_t2')

all_df.to_pickle("my_water_data.pkl")
