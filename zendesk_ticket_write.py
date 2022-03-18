
# coding: utf-8

# In[1]:


from client.models import *
from django.db.models import *
from runscript_functions.runscript_reporter import runscript_reporter
import os
import requests
import json
import pandas as pd
import datetime as dt
import numpy as np
import dateutil.parser
import datetime as dt
from datetime import timedelta
from datetime import datetime


# In[2]:


def format_ts(column):
    return column.map(lambda x: dt.datetime.strptime(x,"%Y-%m-%dT%H:%M:%SZ").timestamp())


# In[3]:


# In[6]: 
tok = os.environ.get('ZENDESK_TOKEN')  #ZENDESK_TOKEN
user = "alexandrechouraki@dreem.com"

# In[4]:


def format_data(url_):
    
    all_data = []
    
    while url_:
        
        # Do the HTTP get request
        response = requests.get(
            url_, 
            auth=(user + '/token', tok),
            verify = True
        )

        data=response.json()
    
        try:
            url_ = data['next_page']
            print("loading",url_)
        except KeyError:
            url_ = False # til no more pages to load

        all_data.append(data)
    
    return all_data


# In[4]:


from _analytics_local_tokens import * 
tok = os.environ.get('ZENDESK_TOKEN')  #ZENDESK_TOKEN


# In[5]:


def run():
    
    report_obj=runscript_reporter(s_name='zendesk_ticket_write',logging_table_model=AnalyticsRunscriptRefreshTime)
    
    ########################################################################################################################

    # get last user id into the zendesk table created in the db
    users_df = pd.DataFrame(ZendeskUserV2.objects.values('user_id','created_at')).sort_values(by='user_id',ascending=False)
    last_user_creation_date = users_df.iloc[0].created_at
    last_user_creation_date = dt.datetime.utcfromtimestamp(last_user_creation_date).strftime("%Y-%m-%d")

    # Set the request parameters
    url_tickets = 'https://rythmhelp.zendesk.com/api/v2/tickets'  
    base_url = 'https://rythmhelp.zendesk.com/api/v2/users/search.json?query=type:user created>={}'
    url_users = base_url.format(last_user_creation_date)
    #url_users = 'https://rythmhelp.zendesk.com/api/v2/users'
    user = 'alexandrechouraki@dreem.com'
    pwd = tok

    print('Loading tickets data')
    tickets_data = format_data(url_tickets)
    tickets_data = [dic['tickets'] for dic in tickets_data]
    tickets_data = [ticket for ticket_list in tickets_data for ticket in ticket_list]
    tickets_data = pd.DataFrame(tickets_data)
    print('Ended loading tickets data')

    print('Loading users data')
    users_data = format_data(url_users)
    users_data = [dic['users'] for dic in users_data]
    users_data = [user for user_list in users_data for user in user_list]
    users_data = pd.DataFrame(users_data)
    print('Ended loading users data')

    ########################################################################################################################

    ## TICKET MANAGING
    
    # Data processing: format date, rename columns, remove null values, remove columns, parse dates
    tickets_data = tickets_data.rename(columns={"id":'ticket_id'})

    tickets_data = tickets_data.where(pd.notnull(tickets_data),None)

    tickets_data = tickets_data.drop(columns=['url', 'recipient', 'forum_topic_id', 'problem_id', 'has_incidents', 'is_public',
                                              'due_at', 'satisfaction_rating', 'followup_ids', 'external_id', 'via', 'type', 
                                              'raw_subject', 'collaborator_ids', 'follower_ids', 'email_cc_ids', 'custom_fields', 
                                              'fields', 'allow_channelback', 'allow_attachments', 'sharing_agreement_ids', 'submitter_id'
                                              ])

    tickets_data[["updated_at", "created_at"]] = tickets_data[["updated_at", "created_at"]].apply(format_ts, axis=0)


    # fill NaN values with 0 so that it's possible to create an entry on the db table
    tickets_data['assignee_id'] = tickets_data['assignee_id'].fillna(0)
    tickets_data['organization_id'] = tickets_data['organization_id'].fillna(0)
    tickets_data['group_id'] = tickets_data['group_id'].fillna(0)

    #filtering out tickets that are not closed, we'll write them when they are
    tickets_data = tickets_data[tickets_data.status=="closed"]

    # Write on the ZendeskTicketV2 table all the ticket witch are not present yet 
    already_written = ZendeskTicketV2.objects.all().values_list('ticket_id', flat = True)
    to_load = tickets_data[~tickets_data.ticket_id.isin(already_written)]
    to_load = to_load.to_dict(orient='record')
    to_load = [ZendeskTicketV2(**x) for x in to_load]
    msg = ZendeskTicketV2.objects.bulk_create(to_load,batch_size=200)
    if msg:
        print(f"successfully wrote {str(len(to_load))} items")
    
    ########################################################################################################################

    ## USER MANAGING
    
    # Data processing: format date, rename columns, remove null values, remove columns, parse dates
    users_data = users_data.rename(columns={"id":'user_id'})

    users_data = users_data.where(pd.notnull(users_data),None)

    users_data = users_data.drop(columns=['time_zone', 'iana_time_zone', 'phone',
           'url', 'shared_phone_number', 'photo', 'locale_id',
           'locale', 'organization_id', 'role', 'verified', 'tags',
           'alias', 'active', 'shared', 'shared_agent', 'last_login_at',
           'two_factor_auth_enabled', 'signature', 'details', 'notes', 'role_type',
           'custom_role_id', 'moderator', 'ticket_restriction',
           'only_private_comments', 'restricted_agent', 'suspended',
           'default_group_id', 'report_csv', 'user_fields'])

    users_data[["updated_at", "created_at"]] = users_data[["updated_at", "created_at"]].apply(format_ts, axis=0)

    # write on the ZendeskTicketV2 table all the ticket witch are not present yet 
    already_written = list(user_df.user_id)
    to_load = users_data[~users_data.user_id.isin(already_written)]
    to_load = to_load.to_dict(orient='record')
    to_load = [ZendeskUserV2(**x) for x in to_load]
    msg = ZendeskUserV2.objects.bulk_create(to_load,batch_size=200)
    if msg:
        print(f"successfully wrote {str(len(to_load))} items")

    
    ########################################################################################################################

    report_obj.finished()  

