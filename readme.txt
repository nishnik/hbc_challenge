Human Behavior Challenge (HBC2018) for understanding human behavior

## Data files

A zip file contains all the training and test trajectories, and a ground truth label file for the training set.

When you unzip the dataset, you find

- ./test/***.csv
- ./train/***.csv
- ./goal_info.csv
- ./train_trip_info.csv
- ./test_trip_info.csv

where *** is the trajectory number (000, 001, ..., 262 for train, 000, 001, ..., 233 for test).


## File format

### train and test trajectories

A single CSV file (000.csv, 001.csv, ...) contains a trajectory, and each line represents the information of a location of a smartphone. In addition to longitude and latitude, some other information is provided;

- lat (float): latitude
- lon (float): longitude
- tripID (int): ID of this trajectory. Note that values of this field is the same in the same trajectory csv file.
- elapsedTime_sec (float): elapsed time (in second) from the beginning of this trajectory
The first line has the header (strings). Fields are separated by a single comma.

Here is an example:

```
lat,lon,tripID,elapsedTime_sec
35.157885,136.926141,1,0.0
35.15775299999999,136.92623033333334,1,8.082
35.15777023076923,136.92619484615386,1,10.118
35.15771669230769,136.92623269230768,1,12.017
35.15771669230769,136.92623269230768,1,14.019
```

- Different trajectories have different number of locations.
- Time intervals between successive two locations are about few seconds when the smartphone catches BLE beacon signals well, otherwise interval may vary up to tens of seconds.
- Trajectories in the training and test sets are in the same format.
- Ground truth information for training trajectories are given in a separate file.

Note that first and last few seconds (uniform in 5 to 60 seconds) are omitted. The task is hence to infer the future and past of test trajectories and predict the goal and start.

### start and goal locations: goal_info.csv

A single txt file of locations of the pre-defined goals.

- LocID (int): goal ID digit (from 0 to 8) to be predicted for test trajectory files.
- lat (float): latitude
- lon (float): longitude

The first line has the header (strings). Fields are separated by a single comma.

### trajectoy info: train_trip_info.csv and test_trip_info.csv

Each txt file describes trips (trajectories).

- tripID (int): ID of a trajectory.
- duration (float): duration (in minute) of this trajectory.
- startLocID (int): goal ID digit (from 0 to 8) from which this trajectory starts.
- endLocID (int): goal ID digit (from 0 to 8) to which this trajectory reaches. This filed is removed from test_trip_info.csv.
- age (int): age of the user of the trajectory. There are missing values.
- gender (char): male (m), female (f). There are missing values.

The first line has the header (strings). Fields are separated by a single comma.

Here is an example of train_trip_info.csv:

```
tripID,duration,startLocID,destLocID,age,gender
1,7.696116666666667,3,4,26,m
5,13.35055,5,1,22,m
10,2.3670333333333335,4,6,19,f
11,3.53645,2,0,16,f
13,3.07525,4,6,21,m
15,4.80535,5,1,26,m
```

Note that test_trip_info.csv has only three fields: tripID, age, and gender. Here is an example:

```
tripID,age,gender
2,50,f
3,20,m
6,7,m
9,23,m
```


## Disclaimer

Before starting data collection, written agreements were obtained by all the participants for the use of scientific purpose and for this competition.

## License of the dataset

The dataset was collected for scientific purpose. If you use the dataset for any scientific purposes except this competition, please refer the following paper:

- Shinsuke Kajioka, Takuto Sakuma, Kazuya Nishi, Yuta Umezu, Masayuki Karasuyama, Toru Tamaki, Takuya Maekawa, Takahiro Uchiya, Hiroshi Matsuo, Ichiro Takeuchi, Comparative Pattern Mining of Human Trajectory Data, IPSJ SIG-UBI Technical Reports, Vol. 2018-UBI-57, No. 39, ISSN 2188-8698, pp. 1-6, Feb 2018. http://id.nii.ac.jp/1001/00185912/

Please contact the corresponding researcher, Prof. Ichiro Takeuchi, if you would like to use the dataset for any other purposes, or access un-preprocessed original raw data.


## Acknowledgements

This competition and project are supported in part by JSPS KAKENHI Grant Numbers JP16H06535, JP16K21735, JP16H06540, JP16H06539, JP16H06538.

