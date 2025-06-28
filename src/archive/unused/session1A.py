#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python

import numpy as np
import mne
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def get_Sessiondata_frame(session,start_second,end_second,label,selected_channels,high_ban,low_band):
    df_session=pd.DataFrame()
    df_person=pd.DataFrame()
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


session1=['9_1_20180724','10_1_20180507','11_1_20180510','12_1_20180515','13_1_20180720','14_1_20180420','15_1_20180724','16_1_20180805']


start_second1= [30, 132, 287, 555, 773, 982, 1271, 1628, 1730, 2025, 2227, 2435, 2667, 2932, 3204]
end_second1= [102, 228, 524, 742, 920, 1240, 1568, 1697, 1994, 2166, 2401, 2607, 2901, 3172, 3359]


label1=[4,1,3,2,0,4,1,3,2,0,4,1,3,2,0]

selected_channels=['AF3', 'AF4', 'T7', 'T8', 'PZ']


high_ban = 1.0
low_band  = 499


ica = mne.preprocessing.ICA(n_components=5, random_state=4, max_iter=1000)

session1=get_Sessiondata_frame(session1,start_second1,end_second1,label1,selected_channels,high_ban,low_band)
session1.to_csv('session1.csv', index=False)


# In[ ]:




