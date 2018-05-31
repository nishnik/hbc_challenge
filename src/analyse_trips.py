import pandas as pd
training = pd.read_csv('../train_trip_info.csv')
data = training.as_matrix()

start_loc_to_final_loc = {}
final_loc_from_start_loc = {}
for i in range(9): # trips are from 0 to 9
    start_loc_to_final_loc[i] = []
    final_loc_from_start_loc[i] = []

for row in data:
    start_id = row[2]
    stop_id = row[3]
    start_loc_to_final_loc[start_id].append(stop_id)
    final_loc_from_start_loc[stop_id].append(start_id)

import numpy as np

for i in range(9): # trips are from 0 to 9
    print ("\nNode name: ", i)
    print ("\t People go to: ", np.unique(start_loc_to_final_loc[i]))
    print ("\t People come from: ", np.unique(final_loc_from_start_loc[i]))

