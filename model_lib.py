import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from numpy import fft
from sklearn import linear_model
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.preprocessing import StandardScaler 
from sklearn.ensemble import VotingRegressor, StackingRegressor, RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

def fourierExtrapolation(x, harm, n_predict):
    """利用快速傅立葉變換將資料進行頻譜分析，然後在轉換成預測值
    x: numpy.array，單變數時間序列歷史資料
    harm: int，諧波次數
    n_predict: int，要預測幾期；當n_predict=2表示要預測下連續2期之數值
    """
    # 參考來源: https://gist.github.com/tartakynov/83f3cd8f44208a1856ce
    n = x.size
    n_harm = harm                  
    x_freqdom = fft.fft(x) 
    f = fft.fftfreq(n)            
    indexes = list(range(n))
    indexes.sort(key = lambda i: np.absolute(f[i]))
 
    t = np.arange(0, n + n_predict)
    restored_sig = np.zeros(t.size)
    for i in indexes[:1 + n_harm * 2]:
        ampli = np.absolute(x_freqdom[i]) / n   
        phase = np.angle(x_freqdom[i])       
        restored_sig += ampli * np.cos(2 * np.pi * f[i] * t + phase)
    return restored_sig 

def fft_forecast(steps, ts_data, col_var):
    """快速傅立葉變換預測模型
    steps:int, 需要用到幾個期間的數列；steps=20表示每次預測都要往後選取20期的數列資料
    ts_data: pandas.DataFrame，要預測的資料表
    col_var: str，要預測的變數欄位名稱
    """
    tidx = pd.to_datetime(ts_data['reg_time'].values)
    tidx = pd.DatetimeIndex(tidx.values, freq = tidx.inferred_freq)
    ts = pd.Series(data=ts_data[col_var].values, index=tidx)
    fft = lambda x: fourierExtrapolation(ts[(x-steps):x],1,1)
    ts_data['idx'] = ts_data.index
    result = ts_data.query(f'idx >= {steps}').reset_index(drop=True).copy()
    result['fft'] = [fft(x)[0] for x in result['idx']]
    return result

def ts_CV_train(X, y, reg):
    """進行時間序列交叉驗證所使用的資料分割方式，採用TimeSeriesSplit，預設分成10等分
    X: numpy.array，特徵變數矩陣
    y: 標籤值矩陣
    reg: sklearn之迴歸物件"""
    error = []
    tscv = TimeSeriesSplit(n_splits=10)
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        reg.fit(X_train, y_train)
        error.append(mean_squared_error(y_test, reg.predict(X_test)))
    return np.mean([error])


def get_latest_5_day_df(data):
    """以今天為準，選取連續前5天資料來與今天資料合併
    data: pandas.DataFrame
    """
    date_arr = data['date'].unique()
    today_idx = np.where(date_arr == data.tail(1)['date'].values[0])[0][0]
    include_past_5_days = date_arr[(today_idx-5):today_idx+1]

    return data[data['date'].isin(include_past_5_days)].reset_index(drop=True)


class ExtratFeatures:
    """特徵選取模組。給定n個特徵下，選擇最好的特徵組合其特徵個數最多不超過10個
    X: pandas.DataFrame or numpy.array，特徵變數矩陣
    y: pandas.DataFrame or numpy.array，標籤值向量
    scaler: sklearn.preprocessing 之標準化處理物件
    """
    def __init__(self, X, y, scaler=None):
        self.X = X
        self.y = y
        self.scaler = scaler
    
    def elastic_net_features(self):
        """利用彈性網正規化剔除不重要的變數
        """
        reg = linear_model.ElasticNet(random_state=1110011, selection='random')
        sel = SelectFromModel(reg, threshold="mean")

        if self.scaler is None:
            sel.fit(self.X, self.y)
        else:
            X_sc = self.scaler.fit_transform(self.X)
            y_sc = self.scaler.fit_transform(self.y.values.reshape(-1, 1)).ravel()
            sel.fit(X_sc, y_sc)

        sel_features = self.X.columns[(sel.get_support())]
        return sel_features

    def ridge_RFE(self, sel_var):
        """利用脊迴歸搭配後退法逐步迴歸尋找最佳特徵變數組合
        sel_var: elastic_net_features的回傳結果
        scaler: sklearn.preprocessing 之標準化處理物件
        """
        model = linear_model.Ridge()
        sel = RFE(model, n_features_to_select=10, step=1)
        if self.scaler is None:
            sel.fit(self.X[sel_var],self.y)
        else:
            X_sc = self.scaler.fit_transform(self.X)
            y_sc = self.scaler.fit_transform(self.y.values.reshape(-1, 1)).ravel()
            sel.fit(X_sc, y_sc)

        sel_features = self.X.columns[(sel.support_)]
        return sel_features

    def select_features(self):
        """輸出最後篩選的最佳特徵變數組合"""
        elastic_sel = self.elastic_net_features()
        if len(elastic_sel) > 1:
            final_sel = self.ridge_RFE(elastic_sel)
        else:
            final_sel = elastic_sel
        return final_sel

    
class ReduceModel:
    """將變數進行lag operator之後，利用ExtratFeatures得到落後變數的最佳組合
    data: pandas.DataFrame，未經過lag operator的資料表
    lags: int，要連續取多少落後期
    col_y: str，標籤的欄位名稱
    """
    def __init__(self, data, lags, col_y=None):
        
        self.col_y = col_y
        self.data = data
        # 選取要進行lag operator之變數
        power_cols = ['kw_tot', 'kwh_tot_velocity', 'kwh_tot_acc']
        inflow_cols = ['inflow_t2', 'inflow_t2_velocity', 'inflow_t2_acc']
        outflow_cols = ['outflow_t2', 'outflow_t2_velocity', 'outflow_t2_acc']
        water_cols = ['water_prod_rate','de_salt_rate','inflow_equm','outflow_equm','t2flow_dif']
        pre_orp_cols = ['pre_uS_orp', 'pre_uS_orp_velocity', 'pre_uS_orp_acc']
        prod_orp_cols = ['prod_uS_orp', 'prod_uS_orp_velocity', 'prod_uS_orp_acc']
        recy_orp_cols = ['recy_uS_orp','recy_uS_orp_velocity','recy_uS_orp_acc']
        pre_ph_cols = ['pre_ph', 'pre_ph_velocity', 'pre_ph_acc']
        prod_ph_cols = ['prod_ph','prod_ph_velocity','prod_ph_acc']
        recy_ph_cols = ['recy_ph','recy_ph_velocity','recy_ph_acc']
        all_features = power_cols + inflow_cols + outflow_cols + water_cols + pre_orp_cols +\
                       prod_orp_cols + recy_orp_cols + pre_ph_cols + prod_ph_cols + recy_ph_cols +['hour','fft']
        self.fft_df = fft_forecast(lags, data, self.col_y).dropna().reset_index(drop=True)
        
        self.X = self.fft_df[all_features] 
        self.y = self.fft_df[['date','reg_time',self.col_y]]
        self.lags = lags
        
    def lag_data_model(self):
        """進行lag operator"""
        result_df = self.y
        for i in range(1, self.lags+1):
            lag = self.X.shift(i)
            lag.columns = [x+f"_{i}" for x in self.X.columns]
            result_df = pd.concat([result_df, lag], axis=1)
        return result_df
    
    def get_reduce_model(self, print_features=False):
        """取得最佳特徵變數組合所形成的資料表
        print_features: bool，是否將特徵變數組合列出並計算與標籤的相關係數
        """
        model = self.lag_data_model().dropna().reset_index(drop=True)
        X_model = model.iloc[:,3:]
        y_model = model.iloc[:,2]
        
        reduce_features = list(ExtratFeatures(X_model, y_model, scaler=StandardScaler()).select_features())
        reduce_model = model[['date','reg_time',self.col_y] + reduce_features]
        
        if print_features:
            print(reduce_model.corr()[self.col_y])
        return reduce_model


class EnsembleModel:
    """集成學習模組，採用sklearn兩模組 VotingRegressor 與 StackingRegressor。
    使用時可選取要VotingRegressor 或者 StackingRegressor來進行
    reduce_model: pandas.DataFrame，最佳特徵變數組合所形成的資料表
    """
    def __init__(self, reduce_model):
        
        self.reduce_model = reduce_model
        self.X_reduce = self.reduce_model.iloc[:,3:]
        self.y_reduce = self.reduce_model.iloc[:,2]
        # 0.8為訓練資料的占比，可調整。
        self.train_size = int(len(self.reduce_model)*0.8)

        self.X_train = self.X_reduce[:self.train_size]
        self.y_train =  self.y_reduce[:self.train_size]
        self.X_test = self.X_reduce[self.train_size:]
        self.y_test = self.y_reduce[self.train_size:]

    def pls_reg(self):
        """使用時間序列之TimeSeriesSplit資料分割方法，進行交叉驗證並選出偏最小平方迴歸最佳的成分個數"""
        parm_dict = {}
        for i in range(1,len(self.X_train.columns)):
            pls = PLSRegression(n_components=i)
            parm_dict[i] = ts_CV_train(self.X_train.values, self.y_train.values, pls)
        best_n_coms = min(parm_dict.keys(), key=(lambda k: parm_dict[k]))
        return PLSRegression(n_components=best_n_coms)
    
    def avg_model(self):
        """加權平均模式的集成學習，採用VotingRegressor模組"""
        estimators = [('knr', KNeighborsRegressor(n_neighbors=15, weights='distance')),
                     ('gbr',GradientBoostingRegressor(random_state=111159)),
                     ('rfr',RandomForestRegressor(random_state=111159)),
                     ('mlp',MLPRegressor(random_state=111159)),
                     ('linear',linear_model.LinearRegression())]
        reg = VotingRegressor(estimators=estimators)
        reg.fit(self.X_train, self.y_train)
        pls = self.pls_reg().fit(self.X_train, self.y_train)
        pred = (pls.predict(self.X_test).reshape(1,-1)[0] + reg.predict(self.X_test))/2 
        
        pred_df = pd.DataFrame({'reg_time':self.reduce_model.iloc[self.train_size:,1], \
                                'pred':pred, 'actual':self.y_test})
            
        return pred_df
    
    
    def stack_model(self):
        """疊堆式模式的集成學習，採用StackingRegressor模組"""
        estimators = [('knr', KNeighborsRegressor(n_neighbors=15, weights='distance')),
                     ('gbr',GradientBoostingRegressor(random_state=111159)),
                     ('rfr',RandomForestRegressor(random_state=111159)),
                     ('mlp',MLPRegressor(random_state=111159)),
                     ('linear',linear_model.LinearRegression()),
                     ('pls',self.pls_reg())]
    
        reg = StackingRegressor(estimators=estimators, final_estimator=linear_model.Lasso())
        reg.fit(self.X_train, self.y_train)
        pred = reg.predict(self.X_test)
        pred_df = pd.DataFrame({'reg_time':self.reduce_model.iloc[self.train_size:,1], \
                                'pred':pred, 'actual':self.y_test})
            
        return pred_df

    
class ModelPerformance:
    """計算模型表現，測試資料預測的判定係數與 Mean Squared Error
    pred_df: pandas.DataFrame，預測值資料表
    col_pred: 預測值欄位名稱
    col_actual: 實際值欄位名稱"""
    def __init__(self, pred_df, col_pred, col_actual):
        
        self.pred_df = pred_df
        self.col_pred = col_pred
        self.col_actual = col_actual
        self.y_pred = pred_df[col_pred].values
        self.y_actual = pred_df[col_actual].values
        self.model_r_square = round(r2_score(self.pred_df[col_actual].values, self.pred_df[col_pred].values),4)*100
        self.mse = round(mean_squared_error(self.pred_df[col_actual].values, self.pred_df[col_pred].values),4)

    def plot_pred_actual(self):
        """將預測值與實際值畫圖比較"""
        plt.figure(figsize=(10,5))
        ts = self.pred_df[['reg_time','actual','pred']]
        ts.index = ts['reg_time'].values
        ts.drop(['reg_time'],axis=1,inplace=True)
        sns.lineplot(data=ts)


def obtain_best_pred(past_data_model, latest_data_model):
    """根據集成學習模式在過去5天的表現，從VotingRegressor 與 StackingRegressor挑選一個做為今天要用的集成學習模式
    past_data_model: pandas.DataFrame，過去5天的資料中，最佳特徵變數組合所形成的資料
    latest_data_model: pandas.DataFrame，最近一天(含今天)的資料中，最佳特徵變數組合所形成的資料
    """
    past_ensemble = EnsembleModel(past_data_model)
    latest_ensemble = EnsembleModel(latest_data_model)

    avg_model_score =  ModelPerformance(past_ensemble.avg_model(),  'pred','actual').model_r_square
    stack_model_score = ModelPerformance(past_ensemble.stack_model(), 'pred','actual').model_r_square
    # 根據判定係數選擇集成學習模式
    if avg_model_score > stack_model_score:
        return latest_ensemble.avg_model()
    else:
        return latest_ensemble.stack_model()

class WaterAlert:
    """進行孤立森林並取得可能的異常點。利用過去5天資料的樣態來判斷今天的水質是否有異常
    water_data: pandas.DataFrame，水質資料
    col_date: str，日期欄位名稱
    col_time: str，時間欄位名稱
    """
    def __init__(self, water_data, col_date, col_time):
        
        self.water_data = water_data
        self.col_date = col_date
        self.col_time = col_time
        recent_df = get_latest_5_day_df(self.water_data).dropna()
        self.train = recent_df[recent_df[self.col_date]<recent_df.tail(1)[self.col_date].unique()[0]].reset_index(drop=True)
        self.test = recent_df[recent_df[self.col_date]==recent_df.tail(1)[self.col_date].unique()[0]].reset_index(drop=True)
        
    def get_alert_data(self, var, var_velocity, var_acceleration, plot=False):
        """取得可能的異常水質資料
        var: str，水質衡量變數之欄位名稱，例如pH
        var_velocity: str，水質變化速度之欄位名稱
        var_acceleration: str，水質變化加(減)速之欄位名稱
        plot: bool，是否將結果畫成圖表
        """
        x_train = self.train.dropna()[[var_velocity, var_acceleration]].values
        x_test = self.test.dropna()[[var_velocity, var_acceleration]].values
        clf = IsolationForest(random_state=1910141).fit(x_train)
        plot_test = self.test.dropna()[[self.col_time, var, var_velocity, var_acceleration]]
        plot_test = plot_test.assign(status=clf.predict(x_test)) 
        
        if plot:
            sns.scatterplot(x=var_velocity, y=var_acceleration, data=plot_test,hue='status')
            
        return plot_test
