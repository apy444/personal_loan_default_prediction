#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 18:41:52 2019

@author: flatironschool
"""



def data_converting(dataset):
    '''This file contains every important data converting task must be done
    before train test split.'''
    

    #emp_length convert to numeric and missing 
    emp_lens = {'10+ years': 10, '9 years': 9, '4 years': 4, '2 years': 2, '< 1 year': 0,
            '1 year': 1, '6 years': 6, '5 years': 5, '8 years': 8, '7 years': 7, '3 years':3}
    for emp_key, emp_val in emp_lens.items():
        dataset.loc[dataset.emp_length == emp_key, 'emp_length'] = emp_val
        
    
    #Converting grades to integers
    grades = {'A':7, 'B':6, 'C':5, 'D':4, 'E':3, 'F':2, 'G':1}
    dataset.loc[:,'grade'] = dataset.grade.apply(lambda x: grades[x])
    
    
    #convert date form to numeric (years)
    dataset.loc[:,'earliest_cr_line'] = dataset.earliest_cr_line.apply(lambda x: 2015 - int(x[-4:]))
    
    #revol_util convert to numeric, replace nan with median
    dataset.loc['revol_util'] = dataset.revol_util.str.strip('%').astype(float)
    
    #create less category for purpose
    purposes = {'educational': 'other', 'vacation':'major_purchase', 'wedding':'major_purchase', 'renewable_energy':'home_improvement'}
    for purpose_key, purpose_val in purposes.items():
        dataset.loc[dataset.purpose==purpose_key, 'purpose'] = purpose_val
    
    return dataset
    
    
    
def data_cleaning(dataset):
    '''This function contains every important feature engineering and cleaning task which
    should be done after train test splitting.'''
   
    
    #impute mean values to employment length variable
    dataset.loc[dataset.emp_length.isna(), 'emp_length'] = round(dataset.emp_length.mean())
    
    
    #truncated extreme values
    inc_trunc = dataset.annual_inc.quantile(q=0.995)
    dataset.loc[dataset.annual_inc>inc_trunc, 'annual_inc'] = inc_trunc
    
    
    #impute median values to missing revol_util
    dataset.loc[dataset.revol_util.isna(), 'revol_util'] = dataset.revol_util.median(skipna=True)
    
    
    return dataset

    
    