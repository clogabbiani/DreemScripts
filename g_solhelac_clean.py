
# coding: utf-8

# In[ ]:


# Prediction of CBT-i efficiency on the ISI index (considering the delta over time) based on clinical, demographic and night data


# In[4]:


import pandas as pd
import datetime as dt
import math
pd.options.display.max_columns = 999
import simplejson as json
import boto3
import os
import uuid
import scipy.stats as ss
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np  
from numpy import random
import scipy
import sklearn
import matplotlib
import plotly.graph_objects as go
from plotly.colors import n_colors

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import statsmodels.api as sm
import statsmodels
from statsmodels.regression import linear_model

from numpy import arange

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.inspection import permutation_importance

from bayes_opt import BayesianOptimization

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import neighbors

plt.style.use('dark_background')


# In[ ]:


##############################

# USEFUL FUNCTIONS

def try_parse(x):
    try:
        return dateutil.parser.parse(x)
    except:
        return None

    
def get_weekstart(date):
    dt = datetime.strptime(date, '%Y-%m-%d')
    start = dt - timedelta(days=dt.weekday())
    return(start)


def to_sec(t):
    return (t.hour * 60 + t.minute) * 60 + t.second

def to_time(sec):
    h = int(sec//3600)
    m = int((sec-h*3600)//60)
    s = int(sec - h*3600 - m*60)
    return dt.time(hour = h, minute = m, second = s)

def circstdtime(secs):
    circ = ss.circstd(secs, high = 23*3600 + 59*60 +59, low = 0)
    circ = to_time(circ)
    return circ

def circmean(secs):
    circ = ss.circmean(secs, high = 23*3600 + 59*60 +59, low = 0)
    circ = to_time(circ)
    return circ

def calculate_age(born):
    today = dt.date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))

#function to populate null and 0 months and days (people born in year 1950 without any more info are considered born on 01/01/1950)
def manage_null(entry):
    if (entry != None) & (math.isnan(entry)==False):
        if int(entry) != 0:
            return entry
        else : 
            return 1
    else :
        return 1


# # General Operations on Dreemer usage and Spectral Power

# In[ ]:


#EXTRACT DREEMER AND QUESTIONNAIRE DATA


# In[ ]:


# Dreemer dataframe 
dreemerage = pd.DataFrame(DreemerDreemer.objects.filter(cohorts__all_customers=True).values('dreemer', 'year_of_birth', 'month_of_birth', 'day_of_birth', 'gender', 'created_date'))

#getting out the users without year of birth
dreemerage = dreemerage[dreemerage['year_of_birth'].isnull()==False]
dreemerage = dreemerage[dreemerage['year_of_birth']<2005]
dreemerage = dreemerage[dreemerage.year_of_birth>1920]
dreemerage['age'] = None
dreemerage['age'] = dreemerage.apply(lambda x: calculate_age(dt.datetime(year = int(x['year_of_birth']), month = int(manage_null(x['month_of_birth'])), day = int(manage_null(x['day_of_birth'])))), axis = 1)
dreemerage = dreemerage.query('18 < age < 100')
dreemerage = dreemerage[dreemerage.gender.notnull()]

# Load sleep questionnaire scores (creating tables dreemer x test_type filled with scores) and merging that to dreemerage df
sqc = pd.DataFrame(SleepQuestionnaireScore.objects.filter(dreemer__in=dreemerage['dreemer'], week = "onboarding")
                   .values('dreemer', 'test_type', 'score'))
sqc = sqc.drop_duplicates(subset = ['dreemer', 'test_type']).pivot(index = 'dreemer', columns = "test_type", values = "score")
dreemerage =  dreemerage.merge(sqc.reset_index(), how = 'left', on = 'dreemer')
dreemerage = dreemerage.rename(columns = {'SD' : 'subj_SD', 'SOL' : 'subj_SOL', 'WASO': 'subj_WASO'}).drop('SE', axis = 1)


# In[ ]:


#EXTRACT RECORD DATA

#all_recs = pd.DataFrame(RecordReportV2.objects.filter(dreem_record_endpoints_version='1.6.2+analytics').values('record', 'record__user', 'record__device','dreem_record_endpoints_version', 'endpoints', 'features'))
#all_recs.endpoints=all_recs.endpoints.apply(lambda x: json.loads(x))
#all_recs=pd.concat([all_recs,all_recs['endpoints'].apply(lambda x:pd.Series(x))],axis=1).drop('endpoints', axis = 1)
#all_recs.to_csv('all_recs_final.csv', sep = ",")

all_recs = pd.read_csv('all_recs_final.csv')
all_recs.drop("Unnamed: 0", axis = 1, inplace = True)

#filter out for quality
all_recs = all_recs[(all_recs.proportion_scorable>=0.8)&(all_recs.proportion_off_head<=0.1)]


# In[ ]:


# Filter to throw out nights split over multiple records
master_record_multiple_records = pd.DataFrame(RecordMasterRecord.objects.values('user', 'date', 'records'))
to_exclude = master_record_multiple_records.groupby(['user','date']).agg({'records' : 'count'})
to_exclude = to_exclude[to_exclude['records']>1].reset_index()
to_exclude = list(to_exclude[['user','date']].apply(tuple, axis = 1))
to_exclude = master_record_multiple_records[master_record_multiple_records[['user', 'date']].apply(tuple, axis = 1).isin(to_exclude)]
to_exclude = list(to_exclude['records'])
all_recs['record'] = all_recs['record'].map(uuid.UUID)
all_recs = all_recs[~all_recs.record.isin(to_exclude)]


# In[ ]:


# Filter to throw out records associated to testing protocols
testing = ProductTestingLinkedRecord.objects.all().values_list('record', flat = True)
all_recs = all_recs[~all_recs.record.isin(testing)]


# In[ ]:


# Filter to exclude records associated to research headbands and pharma
research_hbs=HeadbandCohorts.objects.filter(research_headbands=True).values_list('headband', flat = True)
pharma_hbs=HeadbandCohorts.objects.filter(entity__isnull=False).values_list('headband', flat = True)
all_recs['record__device'] = all_recs['record__device'].map(uuid.UUID)
all_recs = all_recs[~all_recs.record__device.isin(research_hbs)]
all_recs = all_recs[~all_recs.record__device.isin(pharma_hbs)]


# In[ ]:


# Filter to exclude pharma records - better safe than sorry
# disastrous mix of all records associated to dreemers from pharma admin dreemers restricted list and whatever we can find on Tramat
old = ProtocolTracking.objects.all().values_list('dreemer', flat = True)
tramat = PipelineAccount.objects.all().values_list('id', flat = True)
all_recs = all_recs[~all_recs.record__user.isin(old)]
all_recs = all_recs[~all_recs.record__user.isin(tramat)]


# In[ ]:


# Filter for duration (only night longer than 3 hours)
all_recs = all_recs[all_recs.tst>= 3*60]

# Filter to throw out some pesky nan values
all_recs = all_recs[all_recs.lights_on.notnull()]


# In[ ]:


all_recs.head()


# In[ ]:


# Add stim information
stims = pd.DataFrame(RecordReport.objects.all().values('record', 'activation_stims', "number_of_stimulations"))

# Merge stim to all_recs dataframe
all_recs = all_recs.merge(stims, how = "left", on = "record")
all_recs = all_recs[all_recs.activation_stims == 0.0]


# In[ ]:


# Count n of records for stimulated records and not stim records
all_recs.groupby('activation_stims')['record'].describe()


# In[ ]:


# DREEMER SCOPE NIGHT DATA

# CbtEfficiency: Tables summarizing the results of CBTi
# get first CBT efficiency value, with ISI
res_cbt = pd.DataFrame(CbtEfficiency.objects.filter(enlist__user__in=dreemerage['dreemer'], metric_name = "delta ISI (declared)")
                       .values('enlist', 'enlist__started_since', 'enlist__user', 'metric_name', 'metric_start_value', 'metric_end_value', 'start_time_name', 'end_time_name', 'metric_efficiency_value'))
res_cbt = res_cbt[(res_cbt.end_time_name == 'cbt end') & (res_cbt.start_time_name == 'onboarding')]
res_cbt['metric_efficiency_value'] = res_cbt['metric_efficiency_value'].astype(float)
res_cbt = res_cbt.loc[res_cbt.groupby('enlist__user')['enlist__started_since'].idxmin()]
dreemerage = dreemerage.merge(res_cbt[['enlist__user', 'metric_efficiency_value']]
                              .rename(columns={'enlist__user':'dreemer', 'metric_efficiency_value': 'ISI_evol'})
                              , how = 'left', on = 'dreemer')
dreemerage['gender'].iloc[dreemerage['gender'] == 0] = 1


# In[ ]:


all_recs_new = all_recs.rename(columns = {'record__user' : 'dreemer'})
all_recs_new['dreemer'] = all_recs_new['dreemer'].map(uuid.UUID)


# In[ ]:


# Keeping only the nights that occurred before the CBT started
all_recs_new = all_recs_new.merge(res_cbt[['enlist__user', 'enlist__started_since']]
                          .rename(columns={'enlist__user':'dreemer'}), how = 'left', on = "dreemer")
all_recs_new = all_recs_new[all_recs_new.enlist__started_since.notnull()]
all_recs_new = all_recs_new[all_recs_new.enlist__started_since > all_recs_new.lights_off]


# In[ ]:


# PLOT 1: each dreemer, how many records
test_count = all_recs_new.sort_values('lights_off', ascending = True).groupby('dreemer').agg({'record':'count'})
test_count = test_count.reset_index().groupby('record').agg({'dreemer':'nunique'})
test_count['total'] = test_count.dreemer.sum()

# cumsum: cumulative sum among columns
test_count['dreemers_with_less_than_x_nights'] = test_count['dreemer'].cumsum()
test_count['dreemers_included'] = test_count['total']  - test_count['dreemers_with_less_than_x_nights']

test_count = test_count[test_count.index<30]
sns.barplot(x=test_count.index, y= test_count['dreemers_included'])
plt.show()


# In[ ]:


# keeping only the first n (5) nights, and the users with at least n (5) nights
n = 5
all_recs_new = all_recs_new.sort_values('lights_off', ascending = True).groupby('dreemer').head(n)
count = all_recs_new.groupby('dreemer').agg({'record':'count'}).rename(columns= {'record': 'night_count'}).reset_index()
all_recs_new = all_recs_new[all_recs_new.dreemer.isin(list(count[count.night_count==n]['dreemer']))]


# In[ ]:


# Aggregating all sleep data to get a user-centric vision, computing mean and std for a set of variables
metrics_for_mean = ['micro_arousal_index', 'proportion_scorable', 'tst','sol_aasm', 'sol', 'waso_aasm', 'waso',
    'wake_duration', 'n1_duration', 'n2_duration','n3_duration', 'rem_duration', 'nrem_duration',
       'n1_percentage', 'n2_percentage', 'n3_percentage','rem_percentage', 'nrem_percentage',
        'n2_latency_aasm','n3_latency_aasm','rem_latency_aasm','rem_latency','awakenings', 'lps', 'lrem4','sleep_efficiency', 
       'proportion_off_head']
metrics_for_std =  ['micro_arousal_index', 'proportion_scorable', 'tst', 'sol_aasm', 'sol', 'waso_aasm', 'waso','wake_duration', 'n1_duration', 'n2_duration','n3_duration', 'rem_duration', 'nrem_duration', 'n1_percentage','n2_percentage', 'n3_percentage','rem_percentage', 'nrem_percentage', 'n2_latency_aasm', 'n3_latency_aasm', 'rem_latency_aasm',
       'rem_latency', 'awakenings', 'lps', 'lrem4', 'sleep_efficiency','proportion_off_head']

recs_mean = all_recs_new.groupby('dreemer')[metrics_for_mean].mean().add_prefix('mean_').reset_index()
recs_std = all_recs_new.groupby('dreemer')[metrics_for_std].std().add_prefix('std_').reset_index()

all_recs_new['lights_on'] = pd.to_datetime(all_recs_new['lights_on'])
all_recs_new['lights_off'] = pd.to_datetime(all_recs_new['lights_off'])
all_recs_new['lights_on_time'] = all_recs_new['lights_on'].map(to_sec)
all_recs_new['lights_off_time'] = all_recs_new['lights_off'].map(to_sec)

recs_circmean = all_recs_new.groupby('dreemer').agg({'lights_on_time' : circmean, 'lights_off_time' : circmean})
recs_circmean['mean_lights_on_sec'] = recs_circmean['lights_on_time'].map(to_sec)
recs_circmean['mean_lights_off_sec'] = recs_circmean['lights_off_time'].map(to_sec)
recs_circmean.rename(columns = {'lights_on_time' : 'mean_lights_on', 'lights_off_time' : 'mean_lights_off'}, inplace = True)

recs_circstd = all_recs_new.groupby('dreemer').agg({'lights_on_time' : circstdtime, 'lights_off_time' : circstdtime})
recs_circstd['std_lights_on_sec'] = recs_circstd['lights_on_time'].map(to_sec)
recs_circstd['std_lights_off_sec'] = recs_circstd['lights_off_time'].map(to_sec)
recs_circstd.rename(columns = {'lights_on_time' : 'std_lights_on', 'lights_off_time' : 'std_lights_off'}, inplace = True)

recs_circstd.reset_index(inplace = True)
recs_circmean.reset_index(inplace = True)

dreemerage = dreemerage.merge(recs_mean, how = "left", on = "dreemer")
dreemerage = dreemerage.merge(recs_std, how = "left", on = "dreemer")
dreemerage = dreemerage.merge(count, how = "left", on = "dreemer")
dreemerage = dreemerage.merge(recs_circmean, how = "left", on = "dreemer")
dreemerage = dreemerage.merge(recs_circstd, how = "left", on = "dreemer")


# In[ ]:


# Last round of cleaning
# just ensuring non specified gender are neutral during segmentation
dreemerage.loc[dreemerage.gender==3, "gender"] = 0.5
dreemerage = dreemerage[dreemerage['ISI_evol'].notnull()]
dreemerage = dreemerage[dreemerage.mean_tst.notnull()]

# inclusion criteria (ISI over 14)
dreemerage = dreemerage[dreemerage["ISI"]>=14]


# In[ ]:


dreemerage.groupby('gender').describe()


# In[ ]:


dreemerage["treatment_responders"] = dreemerage["ISI_evol"]<=-8
dreemerage["remissions"] = dreemerage["ISI"] + dreemerage["ISI_evol"]<8


# In[ ]:


len(dreemerage)


# In[ ]:


dreemerage["treatment_responders"].sum()


# In[ ]:


dreemerage["remissions"].sum()


# #### SPECTRAL POWER CLAUDIO

# In[ ]:


# Add spectral power information
recs = list(all_recs_new[all_recs_new.dreemer.isin(list(dreemerage['dreemer']))]['record'])
sp_pow = pd.DataFrame(RecordSpectralPower.objects.filter(record_id__in=recs).values('record_id', 'channel', 'sleep_stage', 'slow_wave_activity', 'delta', 'theta', 'alpha','sigma', 'timebins'))
wm = lambda x: np.average(x, weights=sp_pow.loc[x.index, "timebins"])
sp_pow = sp_pow.rename(columns={'record_id':'record'}).merge(all_recs_new[['record','dreemer']], how ='left', on = 'record')


# In[ ]:


# filter out epochs with hight slow waves activity
to_exclude = list(sp_pow[sp_pow.slow_wave_activity>1000000]['record'])
sp_pow = sp_pow[~sp_pow.record.isin(to_exclude)]


# In[ ]:


## FOR DIVIDING INTO REM N1 N2 N3 WAKE

'''
ss = {0:"wake", 1:"n1", 2:"n2", 3: "n3", 4:"rem"}
sp_pow_1 = sp_pow
sp_pow_1['sleep_stage'] = sp_pow_1['sleep_stage'].map(lambda x: ss[x])
'''


# In[ ]:


## FOR DIVINDING INTO REM, NREM AND WAKE

# assemble sleep stages 1, 2 and 3 into NREM stage using the weighted average wm
ss = {0:"WAKE", 1:"NREM", 2:"NREM", 3: "NREM", 4:"REM"}
sp_pow_1 = sp_pow
sp_pow_1['sleep_stage'] = sp_pow_1['sleep_stage'].map(lambda x: ss[x])
sp_pow_1.head()


# In[ ]:


# taking the weighted mean : here, we have a mean across all epochs of a given user for a given phase of sleep
sp_pow_1 = sp_pow_1.groupby(['dreemer', 'channel', 'sleep_stage']).agg({'slow_wave_activity':wm ,
                                                            'delta':wm,
                                                            'theta':wm,
                                                            'alpha':wm,
                                                            'sigma':wm,
                                                       'timebins':'sum'})
sp_pow_1.reset_index(inplace=True)

sp_pow_1['delta'] = sp_pow_1['delta'].map(np.log)
sp_pow_1['theta'] = sp_pow_1['theta'].map(np.log)
sp_pow_1['alpha'] = sp_pow_1['alpha'].map(np.log)
sp_pow_1['sigma'] = sp_pow_1['sigma'].map(np.log)
sp_pow_1['slow_wave_activity'] = sp_pow_1['slow_wave_activity'].map(np.log)

# saving TIMEBINS to keep information about how many epochs were used
timebins_df = sp_pow_1[['sleep_stage','timebins']]


# In[ ]:


# Channel 1
sp_pow_ch1 = sp_pow_1[sp_pow_1.channel=="eeg1"]
sp_pow_ch1 = sp_pow_ch1.drop(['channel', 'timebins'], axis = 1)


# In[ ]:


# Plotting spectral powers
to_plot = sp_pow_ch1.set_index(['dreemer',  'sleep_stage']).stack()
to_plot = to_plot.reset_index().rename(columns={0:"value", "level_2":"spectral_band"})
to_plot['sleep_stage'].unique()

colors = n_colors('rgb(5, 200, 200)', 'rgb(200, 10, 10)', 5, colortype='rgb')

fig = go.Figure()
for ss in to_plot['sleep_stage'].unique():
    fig.add_trace(go.Violin(x=to_plot[to_plot.sleep_stage==ss]['value'], y = to_plot[to_plot.sleep_stage==ss]['spectral_band'], name=ss))

fig.update_traces(orientation='h', side='positive', width=3, points=False)
fig.update_layout(xaxis_showgrid=False, xaxis_zeroline=False)

fig.update_layout(
    xaxis_title="Log Power",
    yaxis_title="Spectral bands"
)
    
fig.show()


# In[ ]:


# Mergining spectral power informations to 

sp_pow_ch1 = sp_pow_ch1.set_index(['dreemer',  'sleep_stage']).stack().unstack([2,1])
sp_pow_ch1.columns = [tup[0] + '_'+tup[1]+"_ch1" for tup in sp_pow_ch1.columns]
sp_pow_ch1.reset_index(inplace=True)

dreemerage = dreemerage.merge(sp_pow_ch1, how = "left", on ='dreemer')


# # Trying to predict ISI evolution using subjective and objective data, including spectral power data

# In[ ]:


dreemerage.columns


# In[ ]:


user_data = dreemerage[[#'dreemer',
 'gender',
 'age',
 'HADS-A',
 'HADS-D',
 #'ISI',     #### TO BE REMOVED FROM THE COLUMN LIST
 #'subj_SD',
 'subj_SOL',
 'subj_WASO',
 #"remissions",
 #"treatment_responders",
 'mean_micro_arousal_index',
 'mean_proportion_scorable',
 'mean_tst',
 'mean_sol_aasm',
 'mean_sol',
 'mean_waso_aasm',
 'mean_waso',
 'mean_wake_duration',
 'mean_n1_duration',
 'mean_n2_duration',
 'mean_n3_duration',
 'mean_rem_duration',
 'mean_nrem_duration',
 'mean_n1_percentage',
 'mean_n2_percentage',
 'mean_n3_percentage',
 'mean_rem_percentage',
 'mean_nrem_percentage',
 'mean_n2_latency_aasm',
 'mean_n3_latency_aasm',
 'mean_rem_latency_aasm',
 'mean_rem_latency',
 'mean_awakenings',
 'mean_lps',
 'mean_lrem4',
 'mean_sleep_efficiency',
 'mean_proportion_off_head',
 'std_micro_arousal_index',
 'std_proportion_scorable',
 'std_tst',
 'std_sol_aasm',
 'std_sol',
 'std_waso_aasm',
 'std_waso',
 'std_wake_duration',
 'std_n1_duration',
 'std_n2_duration',
 'std_n3_duration',
 'std_rem_duration',
 'std_nrem_duration',
 'std_n1_percentage',
 'std_n2_percentage',
 'std_n3_percentage',
 'std_rem_percentage',
 'std_nrem_percentage',
 'std_n2_latency_aasm',
 'std_n3_latency_aasm',
 'std_rem_latency_aasm',
 'std_rem_latency',
 'std_awakenings',
 'std_lps',
 'std_lrem4',
 'std_sleep_efficiency',
 'std_proportion_off_head',
 'mean_lights_on_sec',
 'mean_lights_off_sec',
 'std_lights_on_sec',
 'std_lights_off_sec',
    'delta_NREM_ch1', 
    'theta_NREM_ch1',
    'alpha_NREM_ch1', 
    'sigma_NREM_ch1', 
    'delta_REM_ch1', 
    'theta_REM_ch1', 
    'alpha_REM_ch1', 
    'sigma_REM_ch1',
    'delta_WAKE_ch1', 
    'theta_WAKE_ch1',
    'alpha_WAKE_ch1', 
    'sigma_WAKE_ch1',
   #'delta_n1_ch1', 
   #'theta_n1_ch1',
   #'alpha_n1_ch1', 
   #'sigma_n1_ch1', 
   #'delta_n2_ch1', 
   #'theta_n2_ch1',
   #'alpha_n2_ch1', 
   #'sigma_n2_ch1', 
   #'delta_n3_ch1', 
   #'theta_n3_ch1',
   #'alpha_n3_ch1', 
   #'sigma_n3_ch1', 
   #'delta_rem_ch1', 
   #'theta_rem_ch1', 
   #'alpha_rem_ch1', 
   #'sigma_rem_ch1',
   #'delta_wake_ch1',
   #'theta_wake_ch1',
   #'alpha_wake_ch1',
   #'sigma_wake_ch1',
'ISI_evol']]


# In[ ]:


sns.displot(user_data['ISI_evol'], kde = False)


# In[ ]:


# Checking for correlations between variables
for column in list(user_data.columns):
    if abs(np.corrcoef(user_data[[column,'ISI_evol']], rowvar = False)[0,1])>0.1:
        print(f"Correlation between CBT response and {column} is {np.corrcoef(user_data[[column,'ISI_evol']], rowvar = False)[0,1]}")
        sns.scatterplot(x=user_data['ISI_evol'], y = user_data[column])
        plt.show()


# In[ ]:


# SPLIT TRAINING AND TEST SET
user_data.shape[0]
    


# In[ ]:


# All data
user_data = user_data.dropna()
n = len(user_data.columns)-1

actual_error_random_forest = []
actual_error_random_forest_hyperparameters = []
actual_error_random_forest_hyperparameters_feature_importance = []

# Cross-Validation on the user_data with 5 folds
user_data 

for cv in range(9):
    
    # Splitting (random to be bootstrapped)
    s_scaler = StandardScaler()
    #train, test = train_test_split(user_data, test_size=0.2)
    low_indexes = cv * user_data.shape[0]
    high_indexes = (cv+1) * user_data.shape[0]
    test = user_data.loc[low_indexes:high_indexes]
    train = user_data[

    trainscaled = s_scaler.fit_transform(train)
    X_train = pd.DataFrame(trainscaled[:,:n])
    X_train.columns = user_data.columns[:-1]
    y_train = pd.Series(trainscaled[:,n], name = "ISI_evol")

    testscaled = s_scaler.transform(test)
    X_test = pd.DataFrame(testscaled[:,:n])
    X_test.columns = user_data.columns[:-1]
    y_test = pd.Series(testscaled[:,n], name = "ISI_evol")
    print('ISI train mean in bootstrap number ', boot, ' : ',train['ISI_evol'].mean())
    print('ISI train std in bootstrap number ', boot, ' : ',train['ISI_evol'].std())
       
    ########################
    
    # 1. RANDOM FOREST MODEL

    # Create regressor object
    regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
    # Fit the regressor with x and y data
    regressor.fit(train.drop('ISI_evol', axis = 1), train[['ISI_evol']])

    y_predicted = regressor.predict(test.drop('ISI_evol', axis = 1))

    actual_error =  np.array(test['ISI_evol']) - y_predicted
    actual_error.mean()
    actual_error_random_forest.append((actual_error**2).mean())
    
    ########################
    
    # 2. RANDOM FOREST WITH HYPERPARAMETERS MODEL

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestRegressor()
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = 1)
    # Fit the random search model
    rf_random.fit(train.drop('ISI_evol', axis = 1), np.array(train['ISI_evol']))
    rf_random.best_estimator_
    #RandomForestRegressor(max_depth=90, max_features='sqrt', min_samples_leaf=4, n_estimators=800)

    # Fitting Random Forest Regression to the dataset with the best parameters
     # create regressor object
    regressor = rf_random.best_estimator_
    # fit the regressor with x and y data
    regressor.fit(train.drop('ISI_evol', axis = 1), train[['ISI_evol']])  
    y_predicted = regressor.predict(test.drop('ISI_evol', axis = 1))
    actual_error =  np.array(test['ISI_evol']) - y_predicted
    print(actual_error.mean())
    actual_error_random_forest_hyperparameters.append((actual_error**2).mean())

    ########################
    
    # 2. RANDOM FOREST WITH HYPERPARAMETERS AND FEATURE IMPORTANCE MODEL

    result = permutation_importance(
        regressor, test.drop('ISI_evol', axis = 1), test['ISI_evol'], n_repeats=10, random_state=42, n_jobs=1)

    forest_importances = pd.Series(result.importances_mean, index=list(test.drop('ISI_evol', axis = 1).columns))

    forest_importances = forest_importances.sort_values(ascending = False)

    fig, ax = plt.subplots(figsize=(16, 12))
    forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
    ax.set_title("Feature importances using permutation on full model")
    ax.set_ylabel("Mean accuracy decrease")
    #fig.tight_layout()
    plt.show()

    features_to_keep = list(forest_importances[forest_importances>0].index)

    regressor = RandomForestRegressor(n_estimators = 800, min_samples_leaf = 4, max_features = 'sqrt', max_depth=90, random_state = 0)
    # fit the regressor with x and y data
    regressor.fit(train[features_to_keep], train[['ISI_evol']])
    y_predicted = regressor.predict(test[features_to_keep])
    actual_error =  np.array(test['ISI_evol']) - y_predicted
    print(actual_error.mean())
    actual_error_random_forest_hyperparameters_feature_importance.append((actual_error**2).mean())
    


# In[ ]:


# Create df to store all MSE result from the bootstrapping
df_MSE = pd.DataFrame(list(zip(actual_error_random_forest, actual_error_random_forest_hyperparameters, actual_error_random_forest_hyperparameters_feature_importance)),
               columns =['MSE_random_forest', 'MSE_random_forest_hyperparameters', 'MSE_random_forest_hyperparameters_feature_importance'])

df_MSE


# # Other ML techniques 

# ### Bayesian Optimisation

# In[ ]:


# RandomForest with Bayesian optimization
# Importing useful packages


# In[9]:


from basis.configurator import get_other_variables
from basis.core import model
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pytorch_lightning as pl
import torch
from datetime import timedelta, datetime
from timeit import default_timer
from dataset.to_mmap import h5_to_mmaps
# from dataset.checks import plot_night, plot_eeg
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import json
import uuid
from pytz import timezone
from dataset.checks import clean_mmap_directory
from basis.checks import clean_exp_directory


def objective(hyperparameters):
    print(hyperparameters)

    # Set the random number generator for reproducibility
    pl.seed_everything(hyperparameters['seed'], workers=True)

    # Get date and time
    dt_string = datetime.now(tz=timezone('Europe/Paris')).strftime("%b-%d-%Y -> %H:%M:%S")

    # Get a hash uuid for convenience
    exp_hash = str(uuid.uuid1())
    print("exp_hash\n", exp_hash)

    descriptor = {'dataset_date_time': dt_string,
                  'channels': ['C3', 'C4', 'LOC', 'ROC', 'A1', 'A2'],
                  'n_in': 4,
                  'n_out': 1,
                  'time_window': 90,
                  'n_ext_epochs': 3,
                  'sample_frequency': 64,
                  'normalize_iqr': True,
                  'clip_iqr': hyperparameters['clip_iqr'],
                  'standardizing_scale': hyperparameters['standardizing_scale'],
                  'filter': True,
                  'dataset_directory': mmap_directory,
                 }

    # edf_to_h5(edf_directory, h5_directory, N)
    # xml_to_h5(xml_directory, h5_directory, N)
    t1 = default_timer()
    descriptor['dataset_hash'] = h5_to_mmaps(h5_directory, descriptor, exp_hash, force=False, n_nights=N)
    t2 = default_timer()
    print("\nBuilding mmap took {:0>8} seconds.".format(str(timedelta(seconds=round(t2-t1)))))

    t1 = default_timer()
    # plot_night(mmap_directory, descriptor)
    # plot_eeg(mmap_directory, descriptor)
    # plot_features_from_mmap(mmap_directory, descriptor)
    # plot_features_from_h5(h5_directory, plot_directory, descriptor)
    t2 = default_timer()
    print("\nPlot took {:0>8} seconds.".format(str(timedelta(seconds=round(t2-t1)))))

    configurator = {'exp_date_time': dt_string,
                    'model': 'fno',
                    'n_features': 64,
                    'n_layers': 6,
                    'learning_rate': 1e-3,
                    'batch_size': int(2000/(90*20)),
                    'cutoff_frequency': 20,
                    'n_overlapping': 1,
                    'n_folds': 5,
                    'train_validation_ratio': 0.8,
                    'seed': hyperparameters['seed'],
                    'exp_hash': exp_hash,
                    'exp_directory': exp_directory,
                    }

    configurator = get_other_variables(descriptor, configurator)

    my_model = model(configurator)

    trainer = pl.Trainer(gpus=-1,
                        progress_bar_refresh_rate=0,
                        callbacks=[EarlyStopping(monitor='avg_validation_loss')],
                        )

    # Cross validation
    results = {} ; training_duration = [] ; f1 = [] ; loss_weights = []
    for k in range(configurator['n_folds']):
        
        my_model.cv_split(k)
        
        t1 = default_timer()
        trainer.fit(my_model)
        t2 = default_timer()
        duration = "{:0>8}".format(str(timedelta(seconds=round(t2-t1))))
        training_duration.append(duration)
        print("\nTraining the model took "+duration+" seconds.")

        trainer.test(my_model)

        results.update(my_model.results)
        f1.append(my_model.f1)
        loss_weights.extend(my_model.configurator['loss_weights'])

    configurator = my_model.configurator
    configurator['training_duration'] = training_duration
    configurator['f1'] = f1
    configurator['loss_weights'] = loss_weights

    # Save relevant data
    with open(configurator['exp_directory']+'/'+configurator['exp_hash']+"/configurator.json", "w") as f:
        json.dump(configurator, f, indent=3)

    with open(configurator['exp_directory']+'/'+configurator['exp_hash']+"/results.json", "w") as f:
        json.dump(results, f, indent=3)

    return {'loss': 1-sum(f1)/len(f1),
            'status': STATUS_OK}

space = {'seed': hp.randint('seed', 10),
         'time_window': hp.quniform('time_window', 1, 90, 1),
         'clip_iqr': hp.qnormal('clip_iqr', 2, 1, 0.5),
         'standardizing_scale': hp.choice('standardizing_scale', ['epoch','time_window','record']),
         'n_layers': hp.quniform('n_layers', 1, 6, 1),
         'cutoff_frequency': hp.uniform('cutoff_frequency', 1e-2, 20),
        }


trials = Trials()

best = fmin(objective,
            space=space,
            algo=tpe.suggest,
            max_evals=1,
            trials=trials)
            
print('best', best)

