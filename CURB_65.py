import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

raw_data = pd.read_csv("Datathon2019/Datathon_dataset/final.csv")

curb65_list = [0 for i in range(len(raw_data))]
print(raw_data)
for i in range(len(raw_data)):
    if raw_data['bun'].iloc[i] >= 20:
        curb65_list[i] += 1
    if raw_data['resp_rate'].iloc[i] >= 30:
        curb65_list[i] += 1
    if raw_data['sys_bp'].iloc[i] < 90:
        curb65_list[i] += 1
    if raw_data['age'].iloc[i] >= 65:
        curb65_list[i] += 1
    if raw_data['total'].iloc[i] < 15:
        curb65_list[i] += 1
    
raw_data['curb65_score'] = curb65_list
raw_data.to_csv('Datathon2019/Datathon_dataset/finalWithCurb65.csv')
print(raw_data)