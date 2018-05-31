
# coding: utf-8

# visualization script for HBC2018


import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path

import os


# In[ ]:


goal_df = pd.read_csv('goal_info.csv', index_col=False)
train_trip_df = pd.read_csv('train_trip_info.csv', index_col=False)
test_trip_df = pd.read_csv('test_trip_info.csv', index_col=False)

img = plt.imread('map.png')

# In[ ]:


csv_files = sorted(Path(os.path.join('.','train')).glob('*.csv'))
pngdir = os.path.join('.','train','png')
os.makedirs(pngdir, exist_ok=True)


for f in csv_files:

    df = pd.read_csv(f)
    tripID = int(df.iloc[0].tripID)
    startLocID = train_trip_df.loc[train_trip_df.tripID == tripID].startLocID.as_matrix()[0]
    destLocID = train_trip_df.loc[train_trip_df.tripID == tripID].destLocID.as_matrix()[0]
    age = train_trip_df.loc[train_trip_df.tripID == tripID].age.as_matrix()[0]
    gender = train_trip_df.loc[train_trip_df.tripID == tripID].gender.as_matrix()[0]

    fig, ax = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    df['elapsedTime'] = df['elapsedTime_sec']


    ax[0].plot(df['elapsedTime'], df['lat'], lw=1, color='k', label='')
    ax[0].plot(df['elapsedTime'], df['lat'], lw=0, markersize=3, marker='.', color='k', label='')
    ax[0].axhline(y=goal_df.at[startLocID, 'lat'], color='b', linewidth=1, label='startLocID {}'.format(startLocID))
    ax[0].axhline(y=goal_df.at[destLocID, 'lat'], color='r', linewidth=1, label='destLocID {}'.format(destLocID))

    ax[1].plot(df['elapsedTime'], df['lon'], lw=1, color='k')
    ax[1].plot(df['elapsedTime'], df['lon'], lw=0, markersize=3, marker='.', color='k')
    ax[1].axhline(y=goal_df.at[startLocID, 'lon'], color='b', linewidth=1)
    ax[1].axhline(y=goal_df.at[destLocID, 'lon'], color='r', linewidth=1)

    ax[0].set_title('tripID {0}: age {1:.0f} gender {2}'.format(tripID, age, gender))
    ax[-1].set_xlabel('elapsed time [sec]')

    ax[0].set_ylabel('latitude')
    ax[1].set_ylabel('longitude')

    ax[0].set_ylim( 35.155,  35.160)
    ax[1].set_ylim(136.923, 136.928)

    for a in ax:
        a.set_xlim(0, 1200) # 1200sec = 60min
        a.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:.3f}'))

    ax[0].legend()
    plt.savefig(os.path.join(pngdir, f.name + '-plot.pdf'))
#    plt.show()
    plt.close()



    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca()
    
    ax.imshow(img, alpha=0.5, origin='lower', extent=[136.923, 136.928, 35.160, 35.155])

    df.plot(x='lon', y='lat', ax=ax, legend=False, style='.-', lw=1, alpha=0.4, c='k', label='')
    ax.scatter(df.at[0, 'lon'], df.at[0, 'lat'], s=300, c='b', alpha=0.3, edgecolors='k', label='first point')
    ax.scatter(df.at[len(df)-1, 'lon'], df.at[len(df)-1, 'lat'], s=300, c='r', alpha=0.3, edgecolors='k', label='last point')

    for j in range(len(goal_df)):
        ax.scatter(goal_df.at[j, 'lon'], goal_df.at[j, 'lat'], s=100, c='g', alpha=0.3, edgecolors='k')
        
        ax.annotate(j, xy=(goal_df.at[j, 'lon'], goal_df.at[j, 'lat']),
                    size=8, color='k',
                    horizontalalignment='center', verticalalignment='center')
        
        
    ax.scatter(goal_df.at[startLocID, 'lon'], goal_df.at[startLocID, 'lat'], s=100, c='b', label='startLocID')
    ax.scatter(goal_df.at[destLocID, 'lon'], goal_df.at[destLocID, 'lat'], s=100, c='r', label='destLocID')

    ax.set_ylim([ 35.155,  35.160])
    ax.set_xlim([136.923, 136.928])
    ax.set_title('tripID {0}: age {1:.0f} gender {2}'.format(tripID, age, gender))

    ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:.3f}'))
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:.3f}'))

    plt.legend()
    plt.savefig(os.path.join(pngdir, f.name + '-map.pdf'))
#    plt.show()
    plt.close()


#     break


# In[ ]:



csv_files = sorted(Path(os.path.join('.','test')).glob('*.csv'))
pngdir = os.path.join('.','test','png')
os.makedirs(pngdir, exist_ok=True)


for f in csv_files:

    df = pd.read_csv(f)
    tripID = int(df.iloc[0].tripID)
    age = test_trip_df.loc[test_trip_df.tripID == tripID].age.as_matrix()[0]
    gender = test_trip_df.loc[test_trip_df.tripID == tripID].gender.as_matrix()[0]
#    startLocID = train_trip_df.loc[train_trip_df.tripID == tripID].startLocID.as_matrix()[0]
#    destLocID = train_trip_df.loc[train_trip_df.tripID == tripID].destLocID.as_matrix()[0]

    fig, ax = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    df['elapsedTime'] = df['elapsedTime_sec']


    ax[0].plot(df['elapsedTime'], df['lat'], lw=1, color='k', label='')
    ax[0].plot(df['elapsedTime'], df['lat'], lw=0, markersize=3, marker='.', color='k', label='')
#    ax[0].axhline(y=goal_df.at[startLocID, 'lat'], color='b', linewidth=1, label='startLocID {}'.format(startLocID))
#    ax[0].axhline(y=goal_df.at[destLocID, 'lat'], color='r', linewidth=1, label='destLocID {}'.format(destLocID))

    ax[1].plot(df['elapsedTime'], df['lon'], lw=1, color='k')
    ax[1].plot(df['elapsedTime'], df['lon'], lw=0, markersize=3, marker='.', color='k')
#    ax[1].axhline(y=goal_df.at[startLocID, 'lon'], color='b', linewidth=1)
#    ax[1].axhline(y=goal_df.at[destLocID, 'lon'], color='r', linewidth=1)

    ax[0].set_title('tripID {0}: age {1:.0f} gender {2}'.format(tripID, age, gender))
    ax[-1].set_xlabel('elapsed time [sec]')

    ax[0].set_ylabel('latitude')
    ax[1].set_ylabel('longitude')

    ax[0].set_ylim( 35.155,  35.160)
    ax[1].set_ylim(136.923, 136.928)

    for a in ax:
        a.set_xlim(0, 1200) # 1200sec = 60min
        a.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:.3f}'))

#    ax[0].legend()
    plt.savefig(os.path.join(pngdir, f.name + '-plot.pdf'))
#    plt.show()
    plt.close()



    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca()
    
    ax.imshow(img, alpha=0.5, origin='lower', extent=[136.923, 136.928, 35.160, 35.155])

    df.plot(x='lon', y='lat', ax=ax, legend=False, style='.-', lw=1, alpha=0.4, c='k', label='')
    ax.scatter(df.at[0, 'lon'], df.at[0, 'lat'], s=300, c='b', alpha=0.3, edgecolors='k', label='first point')
    ax.scatter(df.at[len(df)-1, 'lon'], df.at[len(df)-1, 'lat'], s=300, c='r', alpha=0.3, edgecolors='k', label='last point')

    for j in range(len(goal_df)):
        ax.scatter(goal_df.at[j, 'lon'], goal_df.at[j, 'lat'], s=100, c='g', alpha=0.3, edgecolors='k')
        
        ax.annotate(j, xy=(goal_df.at[j, 'lon'], goal_df.at[j, 'lat']),
                    size=8, color='k',
                    horizontalalignment='center', verticalalignment='center')
        
        
#    ax.scatter(goal_df.at[startLocID, 'lon'], goal_df.at[startLocID, 'lat'], s=100, c='b', label='startLocID')
#    ax.scatter(goal_df.at[destLocID, 'lon'], goal_df.at[destLocID, 'lat'], s=100, c='r', label='destLocID')

    ax.set_ylim([ 35.155,  35.160])
    ax.set_xlim([136.923, 136.928])
    ax.set_title('tripID {0}: age {1:.0f} gender {2}'.format(tripID, age, gender))

    ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:.3f}'))
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:.3f}'))

    plt.legend()
    plt.savefig(os.path.join(pngdir, f.name + '-map.pdf'))
#    plt.show()
    plt.close()


#     break

