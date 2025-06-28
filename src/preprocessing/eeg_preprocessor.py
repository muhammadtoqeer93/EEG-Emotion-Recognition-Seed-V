# File: src/preprocessing/eeg_preprocessor.py

import numpy as np
import pandas as pd
import mne
from sklearn.preprocessing import MinMaxScaler

ica = mne.preprocessing.ICA(n_components=5, random_state=4, max_iter=1000)

def get_Sessiondata_frame(session, start_second, end_second, label, selected_channels, high_band, low_band):
    df_session = pd.DataFrame()

    for filename in session:
        load_file = filename + ".cnt"
        eeg_raw = mne.io.read_raw_cnt(load_file, preload=True)
        eeg_raw = eeg_raw.pick_channels(selected_channels)
        raw_filtered = eeg_raw.copy().filter(high_band, low_band)
        ica.fit(raw_filtered)
        filtered_data = raw_filtered.get_data()
        mne.filter.resample(filtered_data)

        sum_all = sum(filtered_data[0]) + sum(filtered_data[1]) + sum(filtered_data[2]) + sum(filtered_data[3]) + sum(filtered_data[4])

        for index in range(len(label)):
            class_ = filtered_data[:, start_second[index]*1000 : end_second[index]*1000]
            class_ = class_ - sum_all  # Element-wise subtraction
            class_ = class_.transpose()

            df_class_ = pd.DataFrame(class_)
            df_class_['label'] = label[index]
            df_session = pd.concat([df_session, df_class_], ignore_index=True)

    return df_session
