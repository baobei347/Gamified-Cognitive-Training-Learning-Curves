import pandas as pd
import torch


def extractdata_basic(df):

    subjectcount = len(df.anon_id.unique())
    df['sub'] = pd.factorize(df.anon_id)[0]
    sub = df['sub']
    y = df['score_pctile']
    j = df['nth_play_specific'] - 1

    sub = torch.tensor(sub.values).long()
    y = torch.tensor(y.values).float()
    j = torch.tensor(j.values).float()

    return y, j, sub, subjectcount


def extractdata_gfactor(df):

    subjectcount = len(df.anon_id.unique())
    df['sub'] = pd.factorize(df.anon_id)[0]
    sub = df['sub']
    y = df['score_pctile']
    j = df['nth_play_specific']-1
    k = df['nth_play_other_total']

    sub = torch.tensor(sub.values).long()
    y = torch.tensor(y.values).float()
    j = torch.tensor(j.values).float()
    k = torch.tensor(k.values).float()
    
    return y, j, k, sub, subjectcount


def extractdata_faculty(df):

    subjectcount = len(df.anon_id.unique())
    df['sub'] = pd.factorize(df.anon_id)[0]
    sub = df['sub']
    y = df['score_pctile']
    j = df['nth_play_specific']-1
    k1 = df['area_Attention']
    k2 = df['area_Flexibility']
    k3 = df['area_Reasoning']
    k4 = df['area_Memory']
    k5 = df['area_Language']
    k6 = df['area_Math']

    sub = torch.tensor(sub.values).long()
    y = torch.tensor(y.values).float()
    j = torch.tensor(j.values).float()
    k1 = torch.tensor(k1.values).float()
    k2 = torch.tensor(k2.values).float()
    k3 = torch.tensor(k3.values).float()
    k4 = torch.tensor(k4.values).float()
    k5 = torch.tensor(k5.values).float()
    k6 = torch.tensor(k6.values).float()
    
    return y, j, k1, k2, k3, k4, k5, k6, sub, subjectcount


def extractdata_faculty_fined(df):

    subjectcount = len(df.anon_id.unique())
    df['sub'] = pd.factorize(df.anon_id)[0]
    sub = df['sub']
    y = df['score_pctile']
    j = df['nth_play_specific']-1
    k1 = df['attribute_Divided_Attention']
    k2 = df['attribute_Face_Name_Recall']
    k3 = df['attribute_Field_of_View']
    k4 = df['attribute_Information_Processing']
    k5 = df['attribute_Logical_Reasoning']
    k6 = df['attribute_Numerical_Calculation']
    k7 = df['attribute_Planning']
    k8 = df['attribute_Response_Inhibition']
    k9 = df['attribute_Selective_Attention']
    k10 = df['attribute_Spatial_Orientation']
    k11 = df['attribute_Spatial_Reasoning']
    k12 = df['attribute_Spatial_Recall']
    k13 = df['attribute_Task_Switching']
    k14 = df['attribute_Verbal_Fluency']
    k15 = df['attribute_Visualization']
    k16 = df['attribute_Vocabulary_Proficiency']
    k17 = df['attribute_Working_Memory']

    sub = torch.tensor(sub.values).long()
    y = torch.tensor(y.values).float()
    j = torch.tensor(j.values).float()
    k1 = torch.tensor(k1.values).float()
    k2 = torch.tensor(k2.values).float()
    k3 = torch.tensor(k3.values).float()
    k4 = torch.tensor(k4.values).float()
    k5 = torch.tensor(k5.values).float()
    k6 = torch.tensor(k6.values).float()
    k7 = torch.tensor(k7.values).float()
    k8 = torch.tensor(k8.values).float()
    k9 = torch.tensor(k9.values).float()
    k10 = torch.tensor(k10.values).float()
    k11 = torch.tensor(k11.values).float()
    k12 = torch.tensor(k12.values).float()
    k13 = torch.tensor(k13.values).float()
    k14 = torch.tensor(k14.values).float()
    k15 = torch.tensor(k15.values).float()
    k16 = torch.tensor(k16.values).float()
    k17 = torch.tensor(k17.values).float()

    return y, j, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, sub, subjectcount


def extractdata_element(df):

    subjectcount = len(df.anon_id.unique())
    df['sub'] = pd.factorize(df.anon_id)[0]
    sub = df['sub']
    y = df['score_pctile']
    j = df['nth_play_specific']-1
    k1 = df['Memory']
    k2 = df['Memory_Updating']
    k3 = df['Multiple_Object_Monitoring']
    k4 = df['Selective_Attention']
    k5 = df['Divided_Attention']
    k6 = df['Task_Switching']
    k7 = df['Vocabulary_Knowledge']
    k8 = df['Word_Generation']
    k9 = df['Planning']
    k10 = df['Calculation']
    k11 = df['Quantitative_Reasoning']
    k12 = df['Spatial_Reasoning']
    k13 = df['Logical_Reasoning']
    k14 = df['Response_Inhibition']

    sub = torch.tensor(sub.values).long()
    y = torch.tensor(y.values).float()
    j = torch.tensor(j.values).float()
    k1 = torch.tensor(k1.values).float()
    k2 = torch.tensor(k2.values).float()
    k3 = torch.tensor(k3.values).float()
    k4 = torch.tensor(k4.values).float()
    k5 = torch.tensor(k5.values).float()
    k6 = torch.tensor(k6.values).float()
    k7 = torch.tensor(k7.values).float()
    k8 = torch.tensor(k8.values).float()
    k9 = torch.tensor(k9.values).float()
    k10 = torch.tensor(k10.values).float()
    k11 = torch.tensor(k11.values).float()
    k12 = torch.tensor(k12.values).float()
    k13 = torch.tensor(k13.values).float()
    k14 = torch.tensor(k14.values).float()
    
    return y, j, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, sub, subjectcount


def extractdata_element_stimulus(df):

    subjectcount = len(df.anon_id.unique())
    df['sub'] = pd.factorize(df.anon_id)[0]
    sub = df['sub']
    y = df['score_pctile']
    j = df['nth_play_specific']-1
    k1 = df['Memory']
    k2 = df['Memory_Updating']
    k3 = df['Multiple_Object_Monitoring']
    k4 = df['Selective_Attention']
    k5 = df['Divided_Attention']
    k6 = df['Task_Switching']
    k7 = df['Vocabulary_Knowledge']
    k8 = df['Word_Generation']
    k9 = df['Planning']
    k10 = df['Calculation']
    k11 = df['Quantitative_Reasoning']
    k12 = df['Spatial_Reasoning']
    k13 = df['Logical_Reasoning']
    k14 = df['Response_Inhibition']
    k15 = df['Objects']
    k16 = df['Locations']
    k17 = df['Single_Letters']
    k18 = df['Words']
    k19 = df['Single_Digits']
    k20 = df['Numbers']
    k21 = df['Cursor_Keys']
    k22 = df['Keyboard_Entry']
    k23 = df['Mouse_Pointer']
    k24 = df['Response_Time_Pressure']

    sub = torch.tensor(sub.values).long()
    y = torch.tensor(y.values).float()
    j = torch.tensor(j.values).float()
    k1 = torch.tensor(k1.values).float()
    k2 = torch.tensor(k2.values).float()
    k3 = torch.tensor(k3.values).float()
    k4 = torch.tensor(k4.values).float()
    k5 = torch.tensor(k5.values).float()
    k6 = torch.tensor(k6.values).float()
    k7 = torch.tensor(k7.values).float()
    k8 = torch.tensor(k8.values).float()
    k9 = torch.tensor(k9.values).float()
    k10 = torch.tensor(k10.values).float()
    k11 = torch.tensor(k11.values).float()
    k12 = torch.tensor(k12.values).float()
    k13 = torch.tensor(k13.values).float()
    k14 = torch.tensor(k14.values).float()
    k15 = torch.tensor(k15.values).float()
    k16 = torch.tensor(k16.values).float()
    k17 = torch.tensor(k17.values).float()
    k18 = torch.tensor(k18.values).float()
    k19 = torch.tensor(k19.values).float()
    k20 = torch.tensor(k20.values).float()
    k21 = torch.tensor(k21.values).float()
    k22 = torch.tensor(k22.values).float()
    k23 = torch.tensor(k23.values).float()
    k24 = torch.tensor(k24.values).float()
    
    return y, j, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, sub, subjectcount



