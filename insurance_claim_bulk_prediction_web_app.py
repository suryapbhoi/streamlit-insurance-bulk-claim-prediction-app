# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 04:16:53 2022

@author: surya_pc
"""
# NumPy for numerical computing
import numpy as np
import pandas as pd

# Pickle for reading model files
import pickle

# scikitlearn
import sklearn
sklearn.set_config(print_changed_only = False)

# Ignore sklearn's FutureWarning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

#Ignore Pandas SettingWithCopyWarning 
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

#streamlit
import streamlit as st

import datetime


# loading the saved model
clf = pickle.load(open('enet_model.sav', 'rb'))

# creating a function for Prediction
def fn_insurance_claim_prediction(input_data_file_path):
    
             
    # Load the test dataframe
    test_df = pd.read_csv(input_data_file_path, on_bad_lines='skip')
    
    # Separate the id from the test dataset
    id_df = test_df['id']
    
    # Choose only the feature-selection columns of the test dataset into X_df
    reqd_features = ['ps_calc_02', 'ps_ind_12_bin', 'ps_ind_07_bin', 'ps_ind_15', 'ps_car_08_cat', 'ps_car_02_cat', 'ps_car_13', 
                     'ps_car_07_cat', 'ps_car_04_cat', 'ps_car_14', 'ps_ind_02_cat', 'ps_ind_17_bin', 'ps_car_11', 'ps_ind_18_bin', 
                     'ps_car_12', 'ps_ind_04_cat', 'ps_reg_03', 'ps_reg_01', 'ps_car_15', 'ps_calc_01', 'ps_car_09_cat', 
                     'ps_ind_14', 'ps_ind_16_bin', 'ps_ind_05_cat', 'ps_ind_11_bin', 'ps_ind_01', 'ps_reg_02', 'ps_ind_08_bin', 
                     'ps_car_10_cat', 'ps_car_03_cat', 'ps_car_05_cat', 'ps_car_01_cat', 'ps_car_06_cat', 'ps_calc_03', 
                     'ps_ind_10_bin', 'ps_ind_06_bin', 'ps_ind_03', 'ps_ind_09_bin']
    
    required_test_df = test_df[reqd_features]
    
    
    prediction = (clf.predict_proba(required_test_df)[:,1] >= 0.035).astype(int)
    prediction_df = pd.DataFrame(prediction)
    
    result_df = pd.concat([test_df, prediction_df], axis = 1, ignore_index=True)
    result_df.columns = ['id','ps_ind_01','ps_ind_02_cat','ps_ind_03','ps_ind_04_cat','ps_ind_05_cat','ps_ind_06_bin','ps_ind_07_bin','ps_ind_08_bin','ps_ind_09_bin','ps_ind_10_bin','ps_ind_11_bin','ps_ind_12_bin','ps_ind_13_bin','ps_ind_14','ps_ind_15','ps_ind_16_bin','ps_ind_17_bin','ps_ind_18_bin',
                         'ps_reg_01','ps_reg_02','ps_reg_03','ps_car_01_cat','ps_car_02_cat','ps_car_03_cat','ps_car_04_cat','ps_car_05_cat','ps_car_06_cat','ps_car_07_cat','ps_car_08_cat','ps_car_09_cat','ps_car_10_cat','ps_car_11_cat','ps_car_11','ps_car_12','ps_car_13',
                         'ps_car_14','ps_car_15','ps_calc_01','ps_calc_02','ps_calc_03','ps_calc_04','ps_calc_05','ps_calc_06','ps_calc_07','ps_calc_08','ps_calc_09','ps_calc_10','ps_calc_11','ps_calc_12','ps_calc_13','ps_calc_14','ps_calc_15_bin','ps_calc_16_bin','ps_calc_17_bin',
                         'ps_calc_18_bin','ps_calc_19_bin','ps_calc_20_bin','target']
    
    # Save the results_df dataframe to a new file
    #result = result_df.to_csv('D:/_Project_31/_Phase 5 submissions/test_subset_prediction_'+datetime.datetime.now().strftime("%Y%m%d%H%M%S")+'.csv', index=None)
    
    
    # download the prediction dataframe as csv file
    @st.cache
    def convert_df_to_csv(df):
      # IMPORTANT: Cache the conversion to prevent computation on every rerun
      return df.to_csv(index=None).encode('utf-8')
    
    
    st.download_button(
      label="Download prediction file as CSV",
      data=convert_df_to_csv(result_df),
      file_name='insurance_claims_prediction_'+datetime.datetime.now().strftime("%Y%m%d%H%M%S")+'.csv',
      mime='text/csv',
    )
    
    # check whether the generated dataframe has exactly 59 columns
    if (len(result_df.columns) == 59):
        return "Success: Prediction File generated"
    else:
        return "Some error occured"

def main(): 
    
    # giving a title
    st.title('Bulk Insurace Claim Prediction Web App')
    
    
    # getting the input data from the user
    user_input = st.text_input('Enter the input CSV file URL:')
    
       
    # code for Prediction
    claim_prediciton = ''
    
    # creating a button for Prediction
    if st.button('Claim Prediction Result'):
        claim_prediciton = fn_insurance_claim_prediction(user_input)
        
    st.success(claim_prediciton)
    
    
if __name__ == '__main__':
    main()