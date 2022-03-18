
# coding: utf-8

# Two cohorts, each divided by gender and age classes : healthy subjects (no insomnia, no diagnosed sleep disorder, no pre existing radical sleep constraint) and insomniacs.
# 
# Filter nights to keep only those with >85% scorability.
# 
# Start by reproducing charts in [https://jcsm.aasm.org/doi/full/10.5664/jcsm.7036](https://jcsm.aasm.org/doi/full/10.5664/jcsm.7036) and go from there
# 
# Add data about positions - very important because of the proven link between position and apnea events in the literature
# 
# Possibly, add spectral power analysis?

# **Context**
# 
# As of today, sleep institutions such as the AASM don't provide clinicians with clear reference data to interpret clinical sleep parameters. Most data in the literature comes from restricted samples (~200 people), with limited reliability due to the very high interindividual and intraindividual variation within sleep variables across nights. 
# 
# Additionally, data in the literature comes from polysomnography in sleep clinics, which poses several limitations to the collection of natural data. Most notably, polysomnography requires a setup that often leads to uncomfortable sleep. Data collected in a lab setting tends to differ from sleep in a home environment in that temperature, bedding, subject comfort and sleeping positions are different.
# 
# This publication's goal is to give a first, accurate outlook of what sleep looks like "in the wild" : outside sleep clinics, with no cumbersome polysomnography that might make the subjects less comfortable, or the data biased.
# 
# Dreem is in a unique position to provide this outlook, as we have at our disposal data from a very high number of subjects, and longitudinal metrics that can help nuance how relevant the reference data is, both for healthy subjects and insomniacs.

# In[1]:


RecordRecord.objects.all().count()


# In[1]:


import pandas as pd
import datetime as dt
import math
pd.options.display.max_columns = 999
import simplejson as json
import boto3
import os
import numpy as np
import uuid
import dateutil.parser


# In[2]:


def try_parse(x):
    try:
        return dateutil.parser.parse(x)
    except:
        return None


# In[3]:


from django import db
db.close_old_connections()


# In[4]:


#extract demo and questionnaire related data


# In[5]:


def logmean(l):
    return np.log(np.mean(l))


# In[6]:


dreemerage = DreemerDreemer.objects.filter(cohorts__all_customers=True).values('dreemer', 'year_of_birth', 'month_of_birth', 'day_of_birth', 'gender', 'created_date')

#getting the age of each dreemer
dreemerage = pd.DataFrame(dreemerage)

#getting out the users without year of birth
dreemerage = dreemerage[dreemerage['year_of_birth'].isnull()==False]
dreemerage = dreemerage[dreemerage['year_of_birth']<2005]
dreemerage = dreemerage[dreemerage.year_of_birth>1920]
dreemerage['age'] = None

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
    
dreemerage['age'] = dreemerage.apply(lambda x: calculate_age(dt.datetime(year = int(x['year_of_birth']), month = int(manage_null(x['month_of_birth'])), day = int(manage_null(x['day_of_birth'])))), axis = 1)
    
dreemerage = dreemerage.query('18 < age < 100')


# In[7]:


dreemerage.iloc[0:2]


# In[8]:


#extract records
all_recs = pd.DataFrame(RecordReportV2.objects.filter(dreem_record_endpoints_version='1.6.2+analytics').values('record', 'record__user', 'record__device','dreem_record_endpoints_version', 'endpoints', 'features'))
all_recs.endpoints=all_recs.endpoints.apply(lambda x: json.loads(x))
all_recs=pd.concat([all_recs,all_recs['endpoints'].apply(lambda x:pd.Series(x))],axis=1).drop('endpoints', axis = 1)
all_recs.to_csv('all_recs_final.csv', sep = ",")
#all_recs = pd.read_csv('all_recs_final.csv')
#all_recs.drop("Unnamed: 0", axis = 1, inplace = True)
#all_recs['record__user'] = all_recs['record__user'].map(uuid.UUID)


# In[8]:


all_recs['lights_on'] = all_recs['lights_on'].map(try_parse)
all_recs['lights_off'] = all_recs['lights_off'].map(try_parse)


# In[9]:


#filter for quality
all_recs = all_recs[(all_recs.proportion_scorable>=0.8)&(all_recs.proportion_off_head<=0.1)]


# In[10]:


#filter to throw out nights split over multiple records
master_record_multiple_records = pd.DataFrame(RecordMasterRecord.objects.values('user', 'date', 'records'))
to_exclude = master_record_multiple_records.groupby(['user','date']).agg({'records' : 'count'})
to_exclude = to_exclude[to_exclude['records']>1].reset_index()
to_exclude = list(to_exclude[['user','date']].apply(tuple, axis = 1))
to_exclude = master_record_multiple_records[master_record_multiple_records[['user', 'date']].apply(tuple, axis = 1).isin(to_exclude)]
to_exclude = list(to_exclude['records'])
all_recs['record'] = all_recs['record'].map(uuid.UUID)
all_recs = all_recs[~all_recs.record.isin(to_exclude)]


# In[11]:


#filter to throw out records associated to testing protocols


# In[12]:


testing = ProductTestingLinkedRecord.objects.all().values_list('record', flat = True)
all_recs = all_recs[~all_recs.record.isin(testing)]


# In[13]:


#filter to exclude records associated to research headbands
research_hbs=HeadbandCohorts.objects.filter(research_headbands=True).values_list('headband', flat = True)
all_recs['record__device'] = all_recs['record__device'].map(uuid.UUID)
all_recs = all_recs[~all_recs.record__device.isin(research_hbs)]


# In[14]:


#filter to exclude pharma records
#disastrous mix of all records associated to dreemers from pharma admin dreemers restricted list and whatever we can find on Tramat
old = ProtocolTracking.objects.all().values_list('dreemer', flat = True)
tramat = PipelineAccount.objects.all().values_list('id', flat = True)
all_recs = all_recs[~all_recs.record__user.isin(old)]
all_recs = all_recs[~all_recs.record__user.isin(tramat)]


# In[15]:


#filter for duration
all_recs = all_recs[all_recs.tst>= 3*60]


# In[16]:


#filter to throw out some pesky nan values
all_recs = all_recs[all_recs.lights_on.notnull()]


# In[17]:


#add stim information
stims = pd.DataFrame(RecordReport.objects.all().values('record', 'activation_stims', "number_of_stimulations"))
all_recs = all_recs.merge(stims, how = "left", on = "record")


# In[18]:


#add sip information
sip = pd.DataFrame(SipOnOffV2.objects.all().values('record','sip_index_in_this_sip_use','duration_sip'))
sip = sip.drop_duplicates()
sip = sip[sip.record.isin(all_recs['record'])]
sip = sip.groupby('record').agg({'duration_sip':'sum'}).reset_index()
all_recs = all_recs.merge(sip, how = 'left', on ="record")
all_recs['duration_sip'] = all_recs['duration_sip'].fillna(0)


# In[19]:


len(all_recs)


# In[20]:


#get diagnoses
text = 'Please indicate if you have been diagnosed with any of the following sleep disorders'
answ = pd.DataFrame(AllQuestionnairesAnswersEn.objects.filter(question_text=text).values('dreemer', 'question_text', 'answer_text', 'fill_date'))


# In[21]:


answ['answer_text'].value_counts()


# In[22]:


not_healthy = list(answ[answ['answer_text']!="Never been diagnosed with these disorders"]['dreemer'])
healthy = list(answ[answ['answer_text']=="Never been diagnosed with these disorders"]['dreemer'])
insomniacs_diagnosed = list(answ[answ['answer_text']=="Insomnia"]['dreemer'])
really_healthy = [user for user in healthy if user not in not_healthy]


# In[23]:


dreemerage['supposed_healthy'] = (dreemerage.dreemer.isin(really_healthy))
dreemerage['diagnosed_insomniacs'] = (dreemerage.dreemer.isin(insomniacs_diagnosed))


# In[24]:


sqc = pd.DataFrame(SleepQuestionnaireScore.objects.filter(dreemer__in=dreemerage['dreemer'], week = "onboarding").values('dreemer', 'test_type', 'score'))
sqc = sqc.drop_duplicates(subset = ['dreemer', 'test_type']).pivot(index = 'dreemer', columns = "test_type", values = "score")
dreemerage=  dreemerage.merge(sqc.reset_index(), how = 'left', on = 'dreemer')
dreemerage = dreemerage[dreemerage['ISI'].notnull()]


# In[25]:


#get age class at first record


# In[26]:


def get_age_class(n):
    if n <= 30:
        return "19-30"
    elif n <= 40:
        return "31-40"
    elif n <= 50:
        return "41-50"
    elif n <= 60:
        return "51-60"
    elif n <= 70:
        return "61-70"
    else:
        return "70+"
    
dreemerage['birthday'] = dreemerage.apply(lambda x : dt.datetime(year = int(x['year_of_birth']), month = int(manage_null(x['month_of_birth'])), day = int(manage_null(x['day_of_birth']))), axis = 1)
first_recs = all_recs.groupby('record__user').agg({'lights_on':'min'})
first_recs = first_recs.reset_index().rename(columns={'lights_on':'date_first_record', 'record__user':'dreemer'})
dreemerage = dreemerage.merge(first_recs, how = 'left', on ='dreemer')
dreemerage['age_at_start'] = (dreemerage['date_first_record'] - dreemerage['birthday']).map(lambda x: x.days/365.25)
dreemerage = dreemerage[dreemerage.age_at_start.notnull()]
dreemerage['age_class'] = dreemerage['age_at_start'].map(get_age_class)


# In[27]:


# keep only completely healthy subjects
healthy = dreemerage[(dreemerage.supposed_healthy) & (dreemerage['HADS-A']<11) & (dreemerage['HADS-D']<11)& (dreemerage['ISI']<8)]


# In[28]:


# get only insomniacs
insomniacs = dreemerage[(dreemerage.supposed_healthy|dreemerage.diagnosed_insomniacs)&(dreemerage['ISI']>=15)]


# In[29]:


len(healthy)


# In[30]:


len(insomniacs)


# In[31]:


#keeping only the nights that occurred before the CBT started, or nights from users with no cbt
cbt = pd.DataFrame(ProgramEnlist.objects.filter(program__in=[7, 18, 20]).values('user', 'started_since'))
cbt = cbt.groupby('user').agg({'started_since':"min"}).reset_index().rename(columns=({'user':'record__user'}))


# In[32]:


all_recs = all_recs.merge(cbt, how = 'left', on = "record__user")
all_recs_no_cbt = all_recs[all_recs.started_since.isnull()]
all_recs_pre_cbt = all_recs[all_recs.started_since.map(lambda x: x.date()) > all_recs.lights_on.map(lambda x: x.date())]


# In[33]:


all_recs = pd.concat([all_recs_no_cbt, all_recs_pre_cbt], axis = 0)


# In[34]:


#selecting only a sample of records. First, figuring out how many users have how many records


# In[35]:


test_count = all_recs.sort_values('lights_off', ascending = True).groupby('record__user').agg({'record':'count'})


# In[36]:


test_count = test_count.reset_index().groupby('record').agg({'record__user':'nunique'})


# In[37]:


test_count['total'] = test_count.record__user.sum()


# In[38]:


test_count['dreemers_with_less_than_x_nights'] = test_count['record__user'].cumsum()
test_count['dreemers_included'] = test_count['total']  - test_count['dreemers_with_less_than_x_nights']


# In[39]:


test_count = test_count[test_count.index<30]


# In[40]:


import seaborn as sns
from matplotlib import pyplot as plt


# In[41]:


sns.barplot(x=test_count.index, y= test_count['dreemers_included'])
plt.show()


# **First choice of parameter : n, number of nights**

# In[42]:


#keeping only the first n ones, and the users with at least n nights
n = 30
all_recs = all_recs.sort_values('lights_off', ascending = True).groupby('record__user').head(n)
count = all_recs.groupby('record__user').agg({'record':'count'}).rename(columns= {'record': 'night_count'}).reset_index()
all_recs = all_recs[all_recs.record__user.isin(list(count[count.night_count==n]['record__user']))]


# In[43]:


#and now, we get started


# In[44]:


all_recs['cohort'] = None
all_recs.loc[all_recs.record__user.isin(healthy['dreemer']), 'cohort'] = 'healthy'
all_recs.loc[all_recs.record__user.isin(insomniacs['dreemer']), 'cohort'] = 'insomniacs'


# In[45]:


all_recs = all_recs[all_recs.cohort.notnull()]


# In[46]:


len(all_recs)


# In[47]:


all_recs.groupby('cohort').agg({'record__user':'nunique', 'record':"count"})


# In[48]:


#aggregating all sleep data to get a user-centric vision
metrics_for_agg = ['micro_arousal_index', 'tst','sol_aasm', 'waso_aasm',
       'n1_percentage', 'n2_percentage', 'n3_percentage','rem_percentage',
        'n2_latency_aasm','n3_latency_aasm','rem_latency_aasm','rem_latency','awakenings','sleep_efficiency']


# In[49]:


means = all_recs.groupby(['record__user', 'cohort']).agg({metric : 'mean' for metric in metrics_for_agg})


# In[50]:


means.columns = [col+"_mean" for col in means.columns]


# In[51]:


std = all_recs.groupby(['record__user', 'cohort']).agg({metric : 'std' for metric in metrics_for_agg})


# In[52]:


std.columns = [col+"_std" for col in std.columns]


# In[53]:


ref_dat = means.reset_index().merge(std.reset_index(), how = 'left', on = ["record__user", 'cohort'])


# In[54]:


ref_dat = ref_dat.rename(columns={'record__user':'dreemer'}).merge(dreemerage, how = 'left', on ="dreemer")


# In[55]:


ref_dat = ref_dat[~ref_dat.gender.isin([0,3])]
ref_dat = ref_dat[ref_dat.gender.notnull()]


# In[56]:


mf = {1:"Male", 2:"Female"}
ref_dat['gender']=ref_dat['gender'].map(lambda x: mf[x])


# In[165]:


means = ['micro_arousal_index_mean', 'tst_mean',
       'sol_aasm_mean', 'waso_aasm_mean', 'n1_percentage_mean',
       'n2_percentage_mean', 'n3_percentage_mean', 'rem_percentage_mean',
       'n2_latency_aasm_mean', 'n3_latency_aasm_mean', 'rem_latency_aasm_mean',
       'rem_latency_mean', 'awakenings_mean', 'sleep_efficiency_mean']
stds = ['micro_arousal_index_std', 'tst_std', 'sol_aasm_std', 'waso_aasm_std',
       'n1_percentage_std', 'n2_percentage_std', 'n3_percentage_std',
       'rem_percentage_std', 'n2_latency_aasm_std', 'n3_latency_aasm_std',
       'rem_latency_aasm_std', 'rem_latency_std', 'awakenings_std',
       'sleep_efficiency_std']


# In[166]:


ref_dat.groupby(['cohort', 'age_class', 'gender']).agg({'dreemer':'nunique'}).stack().unstack([1,2])


# In[167]:


#build matching sample by taking people as close as possible in gender and age
heal = ref_dat[ref_dat.cohort=="healthy"]
ins = ref_dat[ref_dat.cohort=="insomniacs"]
l = []
for i, r in heal.iterrows():
    ins['dist'] = (ins['age']-r['age']).map(abs)
    ins.sort_values('dist', ascending = True, inplace = True)
    l+=list(ins[(ins.age_class==r['age_class'])&(ins.gender==r['gender'])]['dreemer'].head(1))
    ins = ins[~ins.dreemer.isin(l)]


# In[168]:


len(l)


# In[169]:


ref_dat[ref_dat.cohort=="healthy"].dreemer.nunique()


# In[170]:


ref_dat = ref_dat[(ref_dat.cohort=="healthy")|(ref_dat.dreemer.isin(l))]
all_recs = all_recs[(all_recs.cohort=="healthy")|(all_recs.record__user.isin(l))]


# In[171]:


means_dat = ref_dat.groupby(['cohort', 'age_class', 'gender']).agg({mean : 'mean' for mean in means})
means_dat = means_dat.apply(lambda x: round(x,2), axis=1)
means_dat_s = means_dat.stack().unstack([1,2])
means_dat_s = means_dat_s.astype(str)
std_dat = ref_dat.groupby(['cohort', 'age_class', 'gender']).agg({mean : 'std' for mean in means})
std_dat = std_dat.apply(lambda x: round(x,2), axis=1)
std_dat_s = std_dat.stack().unstack([1,2])
std_dat_s= std_dat_s.astype(str)
std_dat_s = std_dat_s.apply(lambda x: '+/-'+x)
means_table = means_dat_s + std_dat_s


# In[127]:


means_table


# In[128]:


means_dat = ref_dat.groupby(['cohort', 'age_class', 'gender']).agg({std : 'mean' for std in stds})
means_dat = means_dat.apply(lambda x: round(x,2), axis=1)
means_dat_s = means_dat.stack().unstack([1,2])
means_dat_s = means_dat_s.astype(str)
std_dat = ref_dat.groupby(['cohort', 'age_class', 'gender']).agg({std : 'std' for std in stds})
std_dat = std_dat.apply(lambda x: round(x,2), axis=1)
std_dat_s = std_dat.stack().unstack([1,2])
std_dat_s= std_dat_s.astype(str)
std_dat_s = std_dat_s.apply(lambda x: '+/-'+x)
std_table = means_dat_s + std_dat_s


# In[129]:


std_table


# In[130]:


import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
#careful here, we're dealing with different volumes
metrics_to_plot=['micro_arousal_index', 'tst', 'sol_aasm','waso_aasm','n1_percentage','n2_percentage',
       'n3_percentage', 'rem_percentage', 'n2_latency_aasm','n3_latency_aasm','rem_latency_aasm',
       'awakenings', 'sleep_efficiency']

for metric in metrics_to_plot:
    sns.displot(x= all_recs[metric], hue = all_recs['cohort'], kde = True)
    plt.title(metric+" distribution for insomniacs vs. healthy people, night level")
    plt.show()


# In[131]:


import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
#same plots but for means and stds - careful here, we're dealing with different volumes

metrics_to_plot=['micro_arousal_index_mean', 'tst_mean',
       'sol_aasm_mean', 'waso_aasm_mean', 'n1_percentage_mean',
       'n2_percentage_mean', 'n3_percentage_mean', 'rem_percentage_mean',
       'n2_latency_aasm_mean', 'n3_latency_aasm_mean', 'rem_latency_aasm_mean',
       'rem_latency_mean', 'awakenings_mean', 'sleep_efficiency_mean',
       'micro_arousal_index_std', 'tst_std', 'sol_aasm_std', 'waso_aasm_std',
       'n1_percentage_std', 'n2_percentage_std', 'n3_percentage_std',
       'rem_percentage_std', 'n2_latency_aasm_std', 'n3_latency_aasm_std',
       'rem_latency_aasm_std', 'rem_latency_std', 'awakenings_std',
       'sleep_efficiency_std']

for metric in metrics_to_plot:
    sns.displot(x= ref_dat[metric], hue = ref_dat['cohort'], kde = True)
    plt.title(metric+" distribution for insomniacs vs. healthy people, mean per user level")
    plt.show()


# In[ ]:


#statistical tests


# In[201]:


sns.pairplot(data=ref_dat, hue="cohort")


# In[ ]:


#adding spectral power info


# In[161]:


all_recs['record'].nunique()


# In[151]:


sp_pow = pd.DataFrame(RecordSpectralPower.objects.filter(record_id__in=all_recs['record']).values('record_id', 'channel', 'sleep_stage', 'slow_wave_activity', 'delta', 'theta', 'alpha','sigma', 'timebins'))
wm = lambda x: np.average(x, weights=sp_pow.loc[x.index, "timebins"])
sp_pow=sp_pow.rename(columns={'record_id':'record'}).merge(all_recs[['record','record__user']], how ='left', on = 'record')
ss = {0:"wake", 1:"n1", 2:"n2", 3: "n3", 4:"rem"}
sp_pow['sleep_stage'] = sp_pow['sleep_stage'].map(lambda x: ss[x])


# In[158]:


sp_pow.record.nunique()


# In[193]:


#taking the weighted mean : here, we have a mean across all epochs of a given user for a given phase of sleep
sp_pow_agg = sp_pow.groupby(['record__user', 'channel', 'sleep_stage']).agg({'slow_wave_activity':wm ,
                                                            'delta':wm,
                                                            'theta':wm,
                                                            'alpha':wm,
                                                            'sigma':wm,
                                                       'timebins':'sum'})
sp_pow_agg.reset_index(inplace=True)


# In[194]:


sp_pow_agg['delta'] = sp_pow_agg['delta'].map(np.log)
sp_pow_agg['theta'] = sp_pow_agg['theta'].map(np.log)
sp_pow_agg['alpha'] = sp_pow_agg['alpha'].map(np.log)
sp_pow_agg['sigma'] = sp_pow_agg['sigma'].map(np.log)
sp_pow_agg['slow_wave_activity'] = sp_pow_agg['slow_wave_activity'].map(np.log)


# In[195]:


#select another channel if needed, but select one...
sp_pow_f7_f8 = sp_pow_agg[sp_pow_agg.channel=="eeg4"]
sp_pow_f7_f8 = sp_pow_f7_f8.drop(['channel', 'timebins'], axis = 1)


# In[196]:


sp_pow_f7_f8 = sp_pow_f7_f8.merge(ref_dat.rename(columns={'dreemer':'record__user'})[['record__user','age_class','age','age_at_start','gender','cohort']])


# In[144]:


means_sp = sp_pow_f7_f8.groupby(['cohort','age_class', 'gender', 'sleep_stage']).agg({'delta':'mean', 'theta':'mean', 'alpha':'mean', 'sigma':'mean'})


# In[145]:


std_sp = sp_pow_f7_f8.groupby(['cohort','age_class', 'gender', 'sleep_stage']).agg({'delta':'std', 'theta':'std', 'alpha':'std', 'sigma':'std'})


# In[147]:


means_sp = means_sp.apply(lambda x: round(x,2), axis=1)
means_sp_s = means_sp.stack().unstack([1,2])
means_sp_s = means_sp_s.astype(str)
std_sp = std_sp.apply(lambda x: round(x,2), axis=1)
std_sp_s = std_sp.stack().unstack([1,2])
std_sp_s= std_sp_s.astype(str)
std_sp_s = std_sp_s.apply(lambda x: '+/-'+x)
means_table_sp = means_sp_s + std_sp_s


# In[148]:


means_table_sp


# In[162]:


means_dat


# In[176]:


metrics_to_plot=['micro_arousal_index_mean', 'tst_mean',
       'sol_aasm_mean', 'waso_aasm_mean', 'n1_percentage_mean',
       'n2_percentage_mean', 'n3_percentage_mean', 'rem_percentage_mean',
       'n2_latency_aasm_mean', 'n3_latency_aasm_mean', 'rem_latency_aasm_mean',
       'rem_latency_mean', 'awakenings_mean', 'sleep_efficiency_mean',
       'micro_arousal_index_std', 'tst_std', 'sol_aasm_std', 'waso_aasm_std',
       'n1_percentage_std', 'n2_percentage_std', 'n3_percentage_std',
       'rem_percentage_std', 'n2_latency_aasm_std', 'n3_latency_aasm_std',
       'rem_latency_aasm_std', 'rem_latency_std', 'awakenings_std',
       'sleep_efficiency_std']

for metric in metrics_to_plot:
    sns.relplot(
    data=ref_dat, x="age_at_start", y=metric,
    col="cohort", hue="gender", style="gender",
    kind="scatter"
)
    plt.show()


# In[200]:


for metric in ['sigma','alpha','theta', 'delta']:
    sns.relplot(data=sp_pow_f7_f8, x="age_at_start", y=metric, col="sleep_stage", hue="cohort", style="gender", kind="scatter")
    plt.show()

