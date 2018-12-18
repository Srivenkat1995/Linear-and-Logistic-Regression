import pandas as pd
import numpy as np 
import random

##################################################################################

same_pair_data = pd.read_csv('same_pairs.csv',index_col= False)
different_pair_data = pd.read_csv('diffn_pairs.csv',index_col = False)
human_observed_data = pd.read_csv('HumanObserved-Features-Data.csv',index_col= False)
random_data = []

human_feature_concatenation = open('Human_Feature_Concatenation.csv', 'w')
human_feature_subtraction = open('Human_Feature_Subtraction.csv','w')

for x in range(len(same_pair_data)):
  random_data.append(random.randint(0,len(different_pair_data)))
random_data.sort()
#print(random_data)

def append_feature_Details(features):
    
    for i in range(1,10):
        string = 'f' + str(i)
        human_feature_concatenation.write('%d,' %features[string])
    return    

def subtract_feature_Details(feature1, feature2):
    for i in range(1,10):
        value1 = feature1.iat[0,i+1] 
        value2 = feature2.iat[0,i+1]
        print(value1,value2)
        result = abs(value1 - value2)
        print(result)
        human_feature_subtraction.write(str(result))
        human_feature_subtraction.write(',')
    return

for index,rows in same_pair_data.iterrows():
    
    feature_A_details = human_observed_data[human_observed_data.img_id == rows['img_id_A']]
    feature_B_details = human_observed_data[human_observed_data.img_id == rows['img_id_B']]
   
    human_feature_concatenation.write(rows['img_id_A'] + ',' + rows['img_id_B']+ ',')
    human_feature_subtraction.write(rows['img_id_A'] + ',' + rows['img_id_B']+',')
   
    append_feature_Details(feature_A_details)
    append_feature_Details(feature_B_details)
   
    subtract_feature_Details(feature_A_details,feature_B_details)
   
    human_feature_concatenation.write(str(rows['target']))
   
    human_feature_subtraction.write(str(rows['target']))
   
    human_feature_concatenation.write('\n')
    human_feature_subtraction.write('\n')

for index,rows in different_pair_data.iterrows():
     
    if index in random_data: 
        feature_A_details = human_observed_data[human_observed_data.img_id == rows['img_id_A']]
        feature_B_details = human_observed_data[human_observed_data.img_id == rows['img_id_B']]

        human_feature_concatenation.write(rows['img_id_A'] + ',' + rows['img_id_B'] + ',')
        human_feature_subtraction.write(rows['img_id_A'] + ',' + rows['img_id_B']+',')


        append_feature_Details(feature_A_details)
        append_feature_Details(feature_B_details)

        subtract_feature_Details(feature_A_details,feature_B_details)

        human_feature_subtraction.write(str(rows['target']))

        human_feature_concatenation.write(str(rows['target']))

        human_feature_subtraction.write('\n')
        human_feature_concatenation.write('\n')
    else:
        continue

human_feature_concatenation.close()
human_feature_subtraction.close()

#######################################################################################################

same_gsv_data = pd.read_csv('same_gsc_pairs.csv',index_col= False)
different_pair_gsc_data = pd.read_csv('different_gsc_pairs.csv',index_col = False)
gsc_observed_data = pd.read_csv('GSC-Features.csv',index_col= False)

gsc_feature_concatenation = open('GSC_Feature_Concatenation.csv','w')
gsc_feature_subtraction = open('GSC_Feature_Subtraction.csv','w')

for x in range(2500):
  random_data.append(random.randint(0,len(same_gsv_data)))
random_data.sort()


def append_gsc_feature_Details(features):
    
    for i in range(1,512):
        string = 'f' + str(i)
        gsc_feature_concatenation.write('%d,' %features[string])
    return  

def subtract_feature_gsc_Details(feature1, feature2):
    for i in range(1,512):
        value1 = feature1.iat[0,i+1] 
        value2 = feature2.iat[0,i+1]
        print(value1,value2)
        result = abs(value1 - value2)
        print(result)
        gsc_feature_subtraction.write(str(result))
        gsc_feature_subtraction.write(',')
    return

for index,rows in same_gsv_data.iterrows():
    
    if index in random_data:
        feature_A_details = gsc_observed_data[gsc_observed_data.img_id == rows['img_id_A']]
        feature_B_details = gsc_observed_data[gsc_observed_data.img_id == rows['img_id_B']]
   
        gsc_feature_concatenation.write(rows['img_id_A'] + ',' + rows['img_id_B']+ ',')
        gsc_feature_subtraction.write(rows['img_id_A'] + ',' + rows['img_id_B']+',')
   
        append_gsc_feature_Details(feature_A_details)
        append_gsc_feature_Details(feature_B_details)
   
        subtract_feature_gsc_Details(feature_A_details,feature_B_details)
   
        gsc_feature_concatenation.write(str(rows['target']))
   
        gsc_feature_subtraction.write(str(rows['target']))
   
        gsc_feature_concatenation.write('\n')
        gsc_feature_subtraction.write('\n')    

for index,rows in different_pair_gsc_data.iterrows():
     
    if index in random_data: 
        feature_A_details = gsc_observed_data[gsc_observed_data.img_id == rows['img_id_A']]
        feature_B_details = gsc_observed_data[gsc_observed_data.img_id == rows['img_id_B']]

        gsc_feature_concatenation.write(rows['img_id_A'] + ',' + rows['img_id_B'] + ',')
        gsc_feature_subtraction.write(rows['img_id_A'] + ',' + rows['img_id_B']+',')


        append_gsc_feature_Details(feature_A_details)
        append_gsc_feature_Details(feature_B_details)

        subtract_feature_gsc_Details(feature_A_details,feature_B_details)

        gsc_feature_subtraction.write(str(rows['target']))

        gsc_feature_concatenation.write(str(rows['target']))

        gsc_feature_subtraction.write('\n')
        gsc_feature_concatenation.write('\n')
    else:
        continue
