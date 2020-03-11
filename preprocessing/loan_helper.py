#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 18:41:52 2019

@author: Andrea Koltai
"""

import pandas as pd

def column_description(data_lc, description):
    
    '''
    From the imported dataset (data_lc) and the imported description this function
    returns a dataset with all the column names, dtypes, examples and detailed
    description.
    '''
    #selected two rows from the lendingclub dataframe called data_lc
    desc = pd.DataFrame(data_lc.iloc[[5,6],:].T.reset_index())
    desc.columns = ['col_name', 'exmp1', 'exmp2']
    
    #from the datatypes created a dataframe
    dtypes = pd.DataFrame(data_lc.dtypes.T.reset_index()).rename(columns={'index':'col_name', 0:'dtype'})
    
    #merged the selected rows with the dtypes
    desc = pd.merge(desc, dtypes, how='left', on='col_name')
    
    #merged the selected rows with the description
    desc = pd.merge(desc, description, left_on='col_name', right_on='LoanStatNew', how='left')
    desc.drop(columns='LoanStatNew', inplace=True)
    
    #created a column to see the number of nonnan values for continuous
    nan_counts = pd.DataFrame(data_lc.isna().sum()).reset_index().rename(columns={'index':'col_name', 0:'nan_counts'})
    desc = pd.merge(desc, nan_counts, how='left', on='col_name')
    
    return desc
    
    
    

def data_converting(dataset):
    '''
    This function contains every important data converting task must be done
    before the train test split.
    '''

    #emp_length convert to numeric and missing 
    emp_lens = {'10+ years': 10, '9 years': 9, '4 years': 4, '2 years': 2, '< 1 year': 0,
            '1 year': 1, '6 years': 6, '5 years': 5, '8 years': 8, '7 years': 7, '3 years':3}
    for emp_key, emp_val in emp_lens.items():
        dataset.loc[dataset.emp_length == emp_key, 'emp_length'] = emp_val
    dataset.emp_length = dataset.emp_length.astype(float)
    dataset.emp_length.fillna(0, inplace=True)
    
    #Converting grades to integers
    grades = {'A':7, 'B':6, 'C':5, 'D':4, 'E':3, 'F':2, 'G':1}
    dataset.loc[:,'grade'] = dataset.grade.apply(lambda x: grades[x])
    
    #convert dates to numeric (years)
    dataset.loc[:,'earliest_cr_line'] = dataset.earliest_cr_line.apply(lambda x: 2015 - int(x[-4:]))
    
    #revol_util convert to numeric, replace nan with median
    dataset.loc[:,'revol_util'] = dataset.revol_util.str.strip('%').astype(float)
    
    #create less category for purpose
    purposes = {'educational': 'other', 'vacation':'major_purchase', 'wedding':'major_purchase', 'renewable_energy':'home_improvement'}
    for purpose_key, purpose_val in purposes.items():
        dataset.loc[dataset.purpose==purpose_key, 'purpose'] = purpose_val
    
    #converting states to regions
    regions = {'W':  ['CA', 'OR', 'UT','WA', 'CO', 'NV', 'AK', 'MT', 'HI', 'WY', 'ID'], 
               'SW': ['AZ', 'TX', 'NM', 'OK'],
               'SE': ['GA', 'NC', 'VA', 'FL', 'KY', 'SC', 'LA', 'AL', 'WV', 'DC', 'AR', 'DE', 'MS', 'TN' ],
               'MW': ['IL', 'MO', 'MN', 'OH', 'WI', 'KS', 'MI', 'SD', 'IA', 'NE', 'IN', 'ND'],
               'NE': ['CT', 'NY', 'PA', 'NJ', 'RI','MA', 'MD', 'VT', 'NH', 'ME']}
    
    for region, states in regions.items():
        for state in states:
            dataset.loc[dataset.addr_state == state, 'region'] = region
    dataset.drop(columns = ['addr_state'], inplace=True)
    
    #fill nan values with zero
    dataset.mths_since_recent_bc_dlq.fillna(0, inplace=True)
    dataset.mths_since_recent_revol_delinq.fillna(0, inplace=True)
    dataset.emp_length.fillna(0, inplace=True)
    dataset.mo_sin_old_il_acct.fillna(0, inplace=True)
    
    #convert term to numeric
    dataset['term'] = dataset.term.apply(lambda x: 36 if x == ' 36 months' else 60)
    
    #convert grades to numeric
    grades = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7}
    dataset['grade'] = dataset.grade.apply(lambda x: grades[x])
    
    return dataset
    
    
    
def data_cleaning(dataset):
    '''
    This function contains every important feature engineering and cleaning task which
    should be done after train test splitting.
    '''
   
    
    #impute mean values to employment length variable
    dataset.loc[dataset.emp_length.isna(), 'emp_length'] = round(dataset.emp_length.mean())
    
    #truncated extreme values
    inc_trunc = dataset.annual_inc.quantile(q=0.99)
    dataset.loc[dataset.annual_inc>inc_trunc, 'annual_inc'] = dataset.annual_inc.quantile(q=0.99)
    
    
    #impute median values to missing revol_util
    dataset.loc[dataset.revol_util.isna(), 'revol_util'] = dataset.revol_util.median()
    
    #impute median values inplace the nan values
    dataset.loc[dataset.dti.isna(), 'dti'] = dataset.dti.median()
    
    
    return dataset


def get_correlated(corr, treshold = 0.8):
    '''
    Input: correlation matrix dataframe, the treshold above which to examine
    Output: dataframe with correlated columns and the correlation metrics
    '''
    corr_df = []
    for i in range(corr.shape[1]):
        for j in range(i+1, corr.shape[1]):
            if corr.iat[i,j]>treshold:
                corr_df.append([corr.index[i], corr.columns[j], corr.iat[i,j]])
    corr_df = pd.DataFrame(corr_df, columns = ['col_name1','col_name2', 'r_square'])
    return corr_df
    