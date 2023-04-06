import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Initialize empty variables for 1a and 1b CSV filenames
_acsv = ''
_bcsv = ''

# Get directory to search from command line argument
directory = sys.argv[1]
vis = sys.argv[2]

# Recursively search through directory for CSV files
for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith('.csv') and ('_1a' in file or '_1b' in file or '_2a' in file or '_2b' in file):
            if '_2a' in file:
                _acsv = os.path.join(root, file)
            if '_2b' in file:
                _bcsv = os.path.join(root, file)
            if '_1a' in file:
                _acsv = os.path.join(root, file)
            elif '_1b' in file:
                _bcsv = os.path.join(root, file)

#print('a and b posner csv s found')
#print('a CSV file:', _acsv)
#print('b CSV file:', _bcsv)

# Load the CSV file into a pandas DataFrame
pos_a_df = pd.read_csv(_acsv)
pos_b_df = pd.read_csv(_bcsv)

# select columns and clean
cols = ["block_name", "valid_cue", "key_resp.rt", "key_resp.keys", "cue_dir", "stim_pos"]

a_df = pos_a_df[cols]
b_df = pos_b_df[cols]

a_df = a_df[(a_df['block_name'] == 'trials')]
b_df = b_df[(b_df['block_name'] == 'trials')]

a_df['cue_dir'] = a_df['cue_dir'].replace({1: 'left', 2: 'right', 3: 'neutral'})
b_df['cue_dir'] = b_df['cue_dir'].replace({1: 'left', 2: 'right', 3: 'neutral'})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,6))

a_df['stim_pos'].astype(str)


def boxplot_val(my_df, ax, titley, ylab):
    # convert to reaction time column to dtype to lists
    my_df['key_resp.rt'] = my_df['key_resp.rt'].apply(lambda x: [x])
    # remove all double taps
    my_df = my_df[~my_df['key_resp.rt'].astype(str).str.contains(',', na=False)]
    my_df = my_df[~my_df['key_resp.rt'].astype(str).str.contains('nan', na=False)]

    # Split the DataFrame into two based on the 'valid_cue' column
    valid_df = my_df[my_df['valid_cue']==True]
    invalid_df = my_df[my_df['valid_cue']==False]

    # Extract the 'key_resp.rt' column and convert it to a list for plotting
    valid_data = valid_df['key_resp.rt'].apply(lambda x: float(x[0].strip('[]'))).tolist()
    print("Valid mean: ", np.mean(valid_data))
    invalid_data = invalid_df['key_resp.rt'].apply(lambda x: float(x[0].strip('[]'))).tolist()
    print("Invalid mean: ", np.mean(invalid_data))
    
    # Plot the boxplots side by side
    ax.boxplot([valid_data, invalid_data], widths=0.5, sym='o', notch=False)
    for i, data in enumerate([valid_data, invalid_data]):
        offset = i + 1
        y = data
        x = np.random.normal(offset, 0.04, size=len(y))
        ax.scatter(x, y, alpha=0.5, edgecolors='none')

    # Add labels and titles
    ax.set_ylim(0.223, 0.4)
    ax.set_xticklabels(['Valid', 'Invalid'])
    if ylab:
        ax.set_ylabel('Response Time')
    ax.set_title(titley)
    print("....")

def boxplot_stim_pos(my_df, ax, titley, ylab):
    my_df['key_resp.rt'] = my_df['key_resp.rt'].apply(lambda x: [x])
    # remove all double taps
    my_df = my_df[~my_df['key_resp.rt'].astype(str).str.contains(',', na=False)]
    my_df = my_df[~my_df['key_resp.rt'].astype(str).str.contains('nan', na=False)]
    my_df = my_df[my_df['valid_cue']==True]
    
    # Split the DataFrame into two based on the 'valid_cue' column
    left_df = my_df[my_df['stim_pos']=='(-5, -1)']
    right_df = my_df[my_df['stim_pos']=='(5, -1)']

    # Extract the 'key_resp.rt' column and convert it to a list for plotting
    left_data = left_df['key_resp.rt'].apply(lambda x: float(x[0].strip('[]'))).tolist()
    print("Left mean: ", np.mean(left_data))
    right_data = right_df['key_resp.rt'].apply(lambda x: float(x[0].strip('[]'))).tolist()
    print("Right mean: ", np.mean(right_data))
    
    ax.set_ylim(0.2, 0.4)
    # Plot the boxplots side by side
    ax.boxplot([left_data, right_data], widths=0.5, sym='o', notch=False)
    for i, data in enumerate([left_data, right_data]):
        offset = i + 1
        y = data
        x = np.random.normal(offset, 0.04, size=len(y))
        ax.scatter(x, y, alpha=0.5, edgecolors='none')

    # Add labels and titles
    ax.set_xticklabels(['left', 'right'])
    if ylab:
        ax.set_ylabel('Response Time')
    ax.set_title(titley)
    print("....")


if vis=='0':
    boxplot_val(a_df, ax1, 'Pre neurofeedback posner', True)
    boxplot_val(b_df, ax2, 'Post neurofeedback posner', 0)

if vis=='1':
    boxplot_stim_pos(a_df, ax1, 'Pre neurofeedback posner', True)
    boxplot_stim_pos(b_df, ax2, 'Post neurofeedback posner', 0)

plt.show(block=True)
