# File: scripts/generate_session1.py

import sys
import os
sys.path.append(os.path.abspath(".."))

from src.preprocessing.eeg_preprocessor import get_Sessiondata_frame

session1 = [
    '9_1_20180724','10_1_20180507','11_1_20180510','12_1_20180515',
    '13_1_20180720','14_1_20180420','15_1_20180724','16_1_20180805'
]

start_second1 = [30, 132, 287, 555, 773, 982, 1271, 1628, 1730, 2025, 2227, 2435, 2667, 2932, 3204]
end_second1 = [102, 228, 524, 742, 920, 1240, 1568, 1697, 1994, 2166, 2401, 2607, 2901, 3172, 3359]
label1 = [4, 1, 3, 2, 0, 4, 1, 3, 2, 0, 4, 1, 3, 2, 0]

selected_channels = ['AF3', 'AF4', 'T7', 'T8', 'PZ']
high_band = 1.0
low_band = 499

df = get_Sessiondata_frame(session1, start_second1, end_second1, label1, selected_channels, high_band, low_band)
df.to_csv('outputs/session1.csv', index=False)

print("session1.csv created successfully.")
