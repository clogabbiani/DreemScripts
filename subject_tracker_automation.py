
# coding: utf-8

# # Helena Automation

# ### Importing 

# In[427]:


import pandas as pd
import datetime as dt
from client.models import *
from django.db.models import *
import json
import dateutil.parser
import numpy as np
from datetime import datetime, timedelta
#import math
import uuid
from runscript_functions.runscript_reporter import runscript_reporter
import pygsheets
import os

# In[429]:



# Call Dreem API for Asceneuron client data 
pharma =  "Asceneuron"
protocol = "ASN51-101"

import dreemapi
from dreemapi import Client

c = Client("https://asn51-101-elt.dreem.com/v1/dreem/")


# In[430]:


def try_parse(x):
try:
    return dateutil.parser.parse(x)
except:
    return None

def invert_m_d(x):
try:
    list_x = x.split(sep='/')
    return list_x[1] + '/' + list_x[0] + '/' + list_x[2]
except:
    return None

def get_weekstart(date):
dt = datetime.strptime(date, '%Y-%m-%d')
start = dt - timedelta(days=dt.weekday())
return(start)

def my_isnan(x):
try:
    return math.isnan(x)
except TypeError:
    return False

def get_info(rec):
inf = c.record_detail(rec)
return (inf['device'], inf['reference']) 



########################################################################################################################


# GET EXCEL SHEET WHERE TO WRITE TO

#first, get current state of affairs
gc = pygsheets.authorize(service_file='creds-gsheet_writer.json')

#open the google spreadsheet
sh = gc.open('Copy of ASN51-101 Subject tracker')

# sheet of all users informations
wks = sh[2]
last_sheet = wks.get_as_df(start='A5', end='U252000',has_header=False, empty_value=None, include_tailing_empty=True)

# sheet of user - dosage date (day 0)
dosage_date_sheet = sh[6].get_as_df(start='A1', end='B25000',has_header=True, empty_value=None, include_tailing_empty=True)

#Get the already written Zendesk Ticket ids, with small modif to manage two absurd tickets, to rollback in a month or so
s1 = set(last_sheet.iloc[:,0])
s2 = list(last_sheet[0])  # list of users in the excel sheet


########################################################################################################################

def run():

report_obj=runscript_reporter(s_name='subject_tracker_automation',logging_table_model=AnalyticsRunscriptRefreshTime)

# REPORT PHARMA
# get list of report associated to Asceneuron client
l = []
count = 0
for rec in list(c.record_list()):
    try:
        l.append(c.get_latest_report(rec['id']).json())
    except:
        count+=1
        pass
if count >0 :
    print(f"{count} records couldn't be retrieved")

report = pd.DataFrame(l)
report = report[report.record.notnull()]

# add columns device and reference
columns = report['record'].apply(get_info)
device, reference = map(list,zip(*columns)) 
report['device'] = device
report['reference'] = reference

# get devices IDs and nicknames from our analytics db
device_df = pd.DataFrame(HeadbandDevice.objects.filter(login_uuid__in=report['device']).values('login_uuid','nickname')).rename(columns={'login_uuid':'device'})

# setting 'device' field as a string in order to perform the concatenation
device_df.device = device_df.device.map(str)
report.device = report.device.map(str)

# merge report with device nickname
report = report.merge(device_df, how = 'left', on = "device", validate="m:1")

# dreemer list of users related to Asceneuron client
dreemers = [{'user': d['dreemer'], 'email' : d['email']} for d in list(c.dreemer_list()) if d['email'] != None]
dreemers = pd.DataFrame([d for d in dreemers if protocol in d['email']])
dreemers['user'] = dreemers['user'].map(uuid.UUID)

# Get list of rec IDs associated to ASN users
recs_protocol = pd.DataFrame(PipelineRecord.objects.filter(user__in=dreemers['user']).values('id', 'pharma_id', 'user', 'start_time','stop_time', 'upload_time'))

# merging records ID with dreemer df by user value
recs_protocol = recs_protocol.merge(dreemers.rename(columns = {'dreemer':'user'}), how = 'left', on ='user')

# keep only reports with higher endpoints_version
keep = report.groupby('record').agg({'dreem_record_endpoints_version': 'max'}).reset_index()[['record', 'dreem_record_endpoints_version']]
keep = list(keep.apply(tuple, axis = 1))
report = report[report[['record', 'dreem_record_endpoints_version']].apply(tuple, axis = 1).isin(keep)]

# fill endpoints info into the all_recs dataframe
all_recs=pd.concat([report,report['endpoints'].apply(lambda x:pd.Series(x))],axis=1).drop('endpoints', axis = 1)

# clean all_recs
all_recs = all_recs[all_recs.tst>=60]
all_recs['record'] = all_recs['record'].map(uuid.UUID)
all_recs = all_recs[all_recs.record.isin(recs_protocol['id'])]

# merge all_recs with recs ID and format some columns
all_recs_usr = all_recs.merge(recs_protocol.rename(columns = {'id': 'record'}), how = 'left', on = 'record')
all_recs_usr["start_date_full"]=all_recs_usr.sleep_onset.apply(lambda x: try_parse(x))
all_recs_usr["start_date"]=all_recs_usr.start_date_full.dt.date
all_recs_usr['date_week'] = all_recs_usr['start_date'].dropna().apply(lambda x: x.isocalendar()[1])
all_recs_usr['date_week_start'] = all_recs_usr['start_date'].dropna().apply(lambda x: get_weekstart(str(x)))
all_recs_usr = all_recs_usr.drop('user', axis = 1)
all_recs_usr = all_recs_usr.sort_values(by=['email'])

# Remove Users with ASN51-101_000
be_removed = 'ASN51-101_000'
ind = [be_removed in all_recs_usr.email.iloc[ind] for ind in range(all_recs_usr.shape[0])]
all_recs_usr = all_recs_usr.drop(all_recs_usr[ind].index)



########################################################################################################################


# columns that are useful for the ecel file
columns = ['email','reference','nickname','proportion_off_head','proportion_scorable','record_duration','start_time','stop_time'] # AND DAY EMPTY

# dataframe with selected columns
asceneuron_df = all_recs_usr[columns]

# merge this information to all_recs_usr dataset
asceneuron_df['record_duration'] = asceneuron_df['record_duration'].map(lambda x: str(timedelta(seconds=x))[:-3].replace(':','h'))
asceneuron_df['proportion_scorable'] = round(asceneuron_df['proportion_scorable']*100, 2)
asceneuron_df['proportion_off_head'] = round(asceneuron_df['proportion_off_head']*100, 2)
asceneuron_df['date'] = 'today'
for i in range(asceneuron_df.shape[0]):
    if (int(asceneuron_df['start_time'].iloc[i].strftime("%H")) <= 10):
        asceneuron_df['date'].iloc[i] = (asceneuron_df['stop_time'].iloc[i]).strftime("%d/%m/%Y")
    else:
        asceneuron_df['date'].iloc[i] = (asceneuron_df['start_time'].iloc[i]).strftime("%d/%m/%Y")
asceneuron_df = asceneuron_df.rename(columns={"nickname": "device","email":"user"})
asceneuron_df['day'] = np.nan
asceneuron_df['comment'] = ''
asceneuron_df.drop(['stop_time'], axis=1)

# Create a list of df for each user witch will be uploaded on excel sheet
to_google = pd.DataFrame(asceneuron_df.groupby(['user','device','date']).agg({'proportion_scorable':list,'proportion_off_head':list,'record_duration':list,'day':'max','reference': list, 'comment':'first'})).reset_index()     # lambda x: ''.join(str(x))
email_list = sorted(list(set(to_google['user'])))


########################################################################################################################


# merge with whole dataframe
to_google = to_google.merge(dosage_date_sheet.rename(columns={'User':'user'}), how='left', on='user')

# compute day column as difference between date and dosage date
to_google['day'] = (to_google['date'].apply(invert_m_d).apply(try_parse) - to_google['Dosage Date'].apply(invert_m_d).apply(try_parse)).apply(_d)



########################################################################################################################


to_google_list = []

for index in range(len(email_list)):

    # df 1: with email and device nickname
    df_user_t = to_google[to_google.user == email_list[index]][['user', 'device']].iloc[:1].reset_index().drop(['index'], axis=1)
    df_user = pd.DataFrame(np.repeat(df_user_t.values, 7, axis=0))
    df_user.columns = df_user_t.columns

    # df 2: with dates, reference, score, off-head and duration (and day)
    temp = to_google[to_google.user == email_list[index]].sort_values(by='date')
    temp['night_date'] = temp['date'].apply(invert_m_d).apply(try_parse)
    temp = temp.sort_values(by='night_date')
    df_metrics = temp[['date','reference','proportion_off_head','proportion_scorable','record_duration','day','comment']].transpose()
    df_metrics = df_metrics.reset_index().drop(['index'], axis=1).iloc[:,~df_metrics.columns.duplicated()]

    # 1 manage reference number
    ref_list = [ref for ref in df_metrics.iloc[1]]
    ref_list_str = [' & '.join(list(map(str, ele))) for ele in ref_list]
    df_metrics.iloc[1] = 'Ref : ' + pd.Series(ref_list_str)

    # 2 manage Off Head values
    off_head_list = [off for off in df_metrics.iloc[2]]
    off_head_list_str = [' & '.join(list(map(str, ele))) for ele in off_head_list]
    df_metrics.iloc[2] = 'Off head : ' + pd.Series(off_head_list_str)

    # 3 manage Quality Index values
    quality_list = [ref for ref in df_metrics.iloc[3]]
    quality_list_str = [' & '.join(list(map(str, ele))) for ele in quality_list]
    df_metrics.iloc[3] = 'Record Quality Index : ' + pd.Series(quality_list_str)

    # 4 manage Duration values
    duration_list = [ref for ref in df_metrics.iloc[4]]
    duration_list_str = [' & '.join(list(map(str, ele))) for ele in duration_list]
    df_metrics.iloc[4] = 'Duration : ' + pd.Series(duration_list_str)

    # 5 manage day values
    day_list = [ref for ref in df_metrics.iloc[5]]
    df_metrics.iloc[5] = 'Day ' + pd.Series(day_list)

    df_metrics.columns.name = ''

    # concatenate the two df
    final_df = pd.concat([df_user,df_metrics], axis=1)
    to_google_list.append(final_df)



########################################################################################################################


# Write on Excel sheet 
for profile in range(len(to_google_list)):
    dfs = dict(tuple(to_google_list[profile].groupby(to_google_list[profile]['user'])))

    if((list(dfs.keys())[-1] in s2) == False):
        #Update the existing worksheet
        if profile == 0:
            wks.set_dataframe(dfs[list(dfs.keys())[0]], (5, 1), copy_head=False)
        else:
            wks.set_dataframe(dfs[list(dfs.keys())[0]], ((5+(7*profile)), 1), copy_head=False)        

        wks.adjust_column_width(start=0, end=len(dfs[list(dfs.keys())[0]].columns)+2, pixel_size=140)
        wks.adjust_row_height(0,len(s1)+len(dfs[list(dfs.keys())[0]])+2, pixel_size=40)
    else: 
        print('User ',list(dfs.keys())[-1], ' already present')


########################################################################################################################

report_obj.finished()  



