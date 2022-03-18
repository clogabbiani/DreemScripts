
# coding: utf-8

# In[1]:


import pandas as pd
import datetime as dt
import uuid
import numpy as np
import pygsheets
from dateutil.parser import parse
import math
from math import *
from client.models import *
from django.db.models import *
from django.db.models import Q
from runscript_functions.runscript_reporter import runscript_reporter
import datetime
from datetime import datetime

def try_parse(x):
    try:
        return dateutil.parser.parse(x)
    except:
        return None

def try_uuid(u):
    try:
        return uuid.UUID(u)
    except:
        return None

def my_isnan(x):
    try:
        return math.isnan(x)
    except TypeError:
        return False


# In[2]:


from _analytics_local_tokens import * 
tok = os.environ.get('ZENDESK_TOKEN')  #ZENDESK_TOKEN


# In[8]:


########################################################################################################################
#first, get current state of affairs
gc = pygsheets.authorize(service_file='creds-gsheet_writer.json')

#open the google spreadsheet
sh = gc.open('Complaint Dreem 2 MD')  #Copy of COPY - Complaint Dreem 2 MD
template = sh[0]

#Get the last worksheet
wks = sh[len(sh.worksheets())-1]

last_sheet = wks.get_as_df(start='A3', end='Y10000000',has_header=False, empty_value=None, include_tailing_empty=True)

#Get the already written Zendesk Ticket ids, with small modif to manage two absurd tickets, to rollback in a month or so
s1 = set(last_sheet.iloc[:,0])
s1 = {x for x in s1 if not my_isnan(x)}

times = wks.get_as_df(start = 'F3', end = 'F10000000',has_header=False, empty_value=None, include_tailing_empty=True).iloc[:,0]
df_times = pd.DataFrame(data=times.values, columns=['creation_date'])

date_list = [d for d in df_times['creation_date'].dropna()]
date_list = [dt.datetime.strptime(x,"%Y-%m-%d %H:%M:%S").timestamp() for x in date_list]
last_ticket_date = max(date_list)


# In[31]:


def run():
    
    report_obj=runscript_reporter(s_name='complaint_automation',logging_table_model=AnalyticsRunscriptRefreshTime)


    ########################################################################################################################

    #first, get current state of affairs
    gc = pygsheets.authorize(service_file='creds-gsheet_writer.json')

    #open the google spreadsheet
    sh = gc.open('Complaint Dreem 2 MD')  #Copy of COPY - Complaint Dreem 2 MD
    template = sh[0]

    #Get the last worksheet
    wks = sh[len(sh.worksheets())-1]

    last_sheet = wks.get_as_df(start='A3', end='Y10000000',has_header=False, empty_value=None, include_tailing_empty=True)

    #Get the already written Zendesk Ticket ids, with small modif to manage two absurd tickets, to rollback in a month or so
    s1 = set(last_sheet.iloc[:,0])
    s1 = {x for x in s1 if not my_isnan(x)}

    times = wks.get_as_df(start = 'F3', end = 'F10000000',has_header=False, empty_value=None, include_tailing_empty=True).iloc[:,0]
    df_times = pd.DataFrame(data=times.values, columns=['creation_date'])

    date_list = [d for d in df_times['creation_date'].dropna()]
    date_list = [dt.datetime.strptime(x,"%Y-%m-%d %H:%M:%S").timestamp() for x in date_list]
    last_ticket_date = max(date_list)


    ########################################################################################################################

    ticket_df = pd.DataFrame(ZendeskTicketV2
                             .objects
                             .filter(tags__contains='dreem_md', created_at__gte = last_ticket_date)
                             .values('ticket_id','requester_id__external_id', 'created_at', 'requester_id__name', 'requester_id__email', 'requester_id', 'subject', 'description', 'priority', 'assignee_id', 'tags')
                             .exclude(ticket_id__in = s1)).rename(columns = {'requester_id__external_id': 'dreemer','requester_id__name': 'name','requester_id__email': 'email'})

    ticket_df['dreemer'] = ticket_df['dreemer'].apply(lambda x : try_uuid(x))

    #HeadbandHardware name
    #Get the hb_id of the dreemers 
    hb_md = HeadbandDevice.objects.filter(hardware__name = 'V2+ Med').values_list('login_uuid',flat=True)
    dreem_md = pd.DataFrame(DreemerCohortsV2.objects.filter(dreemer__in = ticket_df.dreemer, dreemer__current_device__in=hb_md).values('dreemer', 'dreemer__current_device')).rename(columns = {'dreemer__current_device': 'device'})

    #Get the serial numbers of devices used in tickets for specific dreemers
    serials = pd.DataFrame(AllotmentAllotment.objects.filter(device__in = dreem_md.device, dreemer__in = ticket_df.dreemer)
         .values('device', 'dreemer', 'device__serial_number'))

    df_hb = dreem_md.merge(serials, on = ['device', 'dreemer'], how = 'left').rename(columns = {'device__serial_number': 'serial_number'})
    df_hb.dreemer = df_hb.dreemer.map(str)

    # merge everything into a single df
    ticket_df.dreemer = ticket_df.dreemer.map(str)
    df = ticket_df.merge(df_hb, how = 'left', on = 'dreemer')


    ########################################################################################################################

    # Filled from database
    # A : F
    to_google = df[['ticket_id', 'serial_number', 'created_at', 'name', 'email']].rename(columns = {'ticket_id':'ZenDesk Ticked ID', 
                'serial_number':'Device Involved: Serial Number', 'created_at': 'Aware date 3', 'name': 'Client/user 4','email':'Customer contact information 5'})
    to_google['Complaint Open Date'] = to_google['Aware date 3']

    # Fixed columns
    # G : H
    to_google['Country of Origin'] = 'US'
    to_google['Complaint Owner'] = 'Alessandro Stella'

    to_google.sort_values(by = 'Aware date 3', inplace=True)
    to_google.reset_index(drop = True, inplace=True)

    # bring every date back to datetime format
    to_google['Aware date 3'] = to_google['Aware date 3'].apply(datetime.fromtimestamp)
    to_google['Complaint Open Date'] = to_google['Complaint Open Date'].apply(datetime.fromtimestamp)

    last_ticket_date = datetime.fromtimestamp(last_ticket_date)


    ########################################################################################################################

    #Split the dataframe into different dataframes depending on month of date
    dfs = dict(tuple(to_google.groupby(to_google['Aware date 3'].dt.month)))

    #For the name of the worksheet, get the current year.
    year = str(dt.datetime.utcnow().year)

    if (len(list(dfs.keys())) == 1) & (last_ticket_date.month != list(dfs.keys())[0]):  #If the current worksheet is complete and just the next worksheet to be updated
        month = str(list(dfs.keys())[0])    #Current month is stored in the keys of dictionary of dataframes

        title = month + year            #Title of the worksheet is month+year. eg: 82020
        wks = pygsheets.Spreadsheet.add_worksheet(sh, title = title , src_tuple=(sh.id,template.id), index=None)
        wks.set_dataframe(dfs[list(dfs.keys())[0]], 'A3', copy_head=False)

        wks.adjust_column_width(start=0, end=len(dfs[list(dfs.keys())[0]].columns)+2, pixel_size=150)
        wks.adjust_row_height(0,len(dfs[list(dfs.keys())[0]])+2, pixel_size=80)


    elif len(list(dfs.keys())) == 1:    #If there is just the current worksheet to be updated
        #Update the existing worksheet
        wks.set_dataframe(dfs[list(dfs.keys())[0]], (len(s1)+2, 1), copy_head=False)

        wks.adjust_column_width(start=0, end=len(dfs[list(dfs.keys())[0]].columns)+2, pixel_size=150)
        wks.adjust_row_height(0,len(s1)+len(dfs[list(dfs.keys())[0]])+2, pixel_size=80)

    elif len(list(dfs.keys())) > 1: #If there are more than 1 month to be written in the spreadsheet, first update the current one, then create the next ones
        #Update the existing worksheet
        wks.set_dataframe(dfs[list(dfs.keys())[0]], (len(s1)+2, 1), copy_head=False)

        wks.adjust_column_width(start=0, end=len(dfs[list(dfs.keys())[0]].columns)+2, pixel_size=150)
        wks.adjust_row_height(0,len(s1)+len(dfs[list(dfs.keys())[0]])+2, pixel_size=80)

        for i in range(1,len(list(dfs.keys()))):
            month = str(list(dfs.keys())[i])    #Current month is stored in the keys of dictionary of dataframes

            title = month + year                #Title of the worksheet is month+year. eg: 82020
            wks = pygsheets.Spreadsheet.add_worksheet(sh, title = title , src_tuple=(sh.id,template.id), index=None)
            wks.set_dataframe(dfs[list(dfs.keys())[i]], 'A3', copy_head=False)

            wks.adjust_column_width(start=0, end=len(dfs[list(dfs.keys())[i]].columns)+2, pixel_size=150)
            wks.adjust_row_height(0,len(dfs[list(dfs.keys())[i]])+2, pixel_size=80)

            
    ########################################################################################################################
    
    report_obj.finished()  

