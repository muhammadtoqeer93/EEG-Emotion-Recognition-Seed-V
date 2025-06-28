#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!/usr/bin/env python

import numpy as np
import mne
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def get_Sessiondata_frame(session,start_second,end_second,label,selected_channels,high_ban,low_band):
    df_person=pd.DataFrame()
    df_session=pd.DataFrame()
    df_class_ = pd.DataFrame()
    for filename in session:
        load_file=filename+".cnt"
        eeg_raw = mne.io.read_raw_cnt(load_file,preload=True)
        eeg_raw=eeg_raw.pick_channels(selected_channels)
        raw_filtered = eeg_raw.copy().filter(high_ban, low_band)
        ica.fit(raw_filtered)
        filtered_data=raw_filtered.get_data()
        mne.filter.resample(filtered_data)
        sum_all=(sum(filtered_data[0])+sum(filtered_data[1])+sum(filtered_data[2])+sum(filtered_data[3])+sum(filtered_data[4]))
        for index in range(len(label)):
            class_ = filtered_data[:, start_second[index]*1000 : end_second[index]*1000]
            for i in range(len(class_)):
                for j in range(len(class_[i])):
                    class_[i][j]=class_[i][j]-sum_all
            class_ = class_.transpose()
            df_class_=pd.DataFrame(class_)
            df_class_['label']=label[index]
            df_person=pd.concat([df_person, df_class_],ignore_index=True)
        df_session=pd.concat([df_session, df_person], ignore_index=True)
    return df_session


session1=['1_1_20180804','2_1_20180416','3_1_20180414','4_1_20180414','5_1_20180719','6_1_20180713','7_1_20180411','8_1_20180717','9_1_20180724','10_1_20180507','11_1_20180510','12_1_20180515','13_1_20180720','14_1_20180420','15_1_20180724','16_1_20180805']

session2=['1_2_20180810','2_2_20180419','3_2_20180419','4_2_20180417','5_2_20180728','6_2_20180731','7_2_20180418','8_2_20180802','9_2_20180804','10_2_20180524','11_2_20180508','12_2_20180508','13_2_20180806','14_2_20180423','15_2_20180807','16_2_20180815']

session3=['1_3_20180808','2_3_20180425','3_3_20180424','4_3_20180501','5_3_20180723','6_3_20180802','7_3_20180422','8_3_20180726','9_3_20180728','10_3_20180626','11_3_20180522','12_3_20180517','13_3_20180725','14_3_20180427','15_3_20180730','16_3_20180813']


start_second1= [30, 132, 287, 555, 773, 982, 1271, 1628, 1730, 2025, 2227, 2435, 2667, 2932, 3204]
end_second1= [102, 228, 524, 742, 920, 1240, 1568, 1697, 1994, 2166, 2401, 2607, 2901, 3172, 3359]

start_second2= [30, 299, 548, 646, 836, 1000, 1091, 1392, 1657, 1809, 1966, 2186, 2333, 2490, 2741]
end_second2= [267, 488, 614, 773, 967, 1059, 1331, 1622, 1777, 1908, 2153, 2302, 2428, 2709, 2817]

start_second3= [30, 353, 478, 674, 825, 908, 1200, 1346, 1451, 1711, 2055, 2307, 2457, 2726, 2888]
end_second3= [321, 418, 643, 764, 877, 1147, 1284, 1418, 1679, 1996, 2275, 2425, 2664, 2857, 3066]


label1=[4,1,3,2,0,4,1,3,2,0,4,1,3,2,0]
label2=[2,1,3,0,4,4,0,3,2,1,3,4,1,2,0]
label3=[2,1,3,0,4,4,0,3,2,1,3,4,1,2,0]

selected_channels=['AF3', 'AF4', 'T7', 'T8', 'PZ']


high_ban = 1.0
low_band  = 499


ica = mne.preprocessing.ICA(n_components=5, random_state=4, max_iter=1000)

session1=get_Sessiondata_frame(session1,start_second1,end_second1,label1,selected_channels,high_ban,low_band)
session1.to_csv('session1.csv', index=False)

session2=get_Sessiondata_frame(session2,start_second2,end_second2,label2,selected_channels,high_ban,low_band)
session2.to_csv('session2.csv', index=False)

session3=get_Sessiondata_frame(session3,start_second3,end_second3,label3,selected_channels,high_ban,low_band)
session3.to_csv('session3.csv', index=False)

