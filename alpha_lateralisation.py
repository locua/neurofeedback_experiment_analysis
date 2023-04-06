import h5py
import numpy as np
import scipy.signal as signal
from matplotlib import pyplot as plt

import sys

filename = sys.argv[1]

def print_structure(name, obj):
    print(name, '->', type(obj))

with h5py.File(filename, 'r') as f:
    f.visititems(print_structure)

with h5py.File(filename, 'r') as f:
    raw_data = f['protocol9/signals_stats/AAI']

print('raw_data: ')
print(raw_data)

sys.exit(0)

def filter_alpha_band(data, fs):
    low_freq = 8
    high_freq = 12
    nyquist = 0.5 * fs
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    return signal.lfilter(b, a, data)

def filter_alpha_band(data, fs):
    low_freq = 8
    high_freq = 12
    nyquist = 0.5 * fs
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    return signal.lfilter(b, a, data)

alpha_filtered_data = np.apply_along_axis(filter_alpha_band, 0, raw_data, fs)

# Define the indices of left and right channels
left_channels = [0, 2, 4]  # Replace with the actual indices of left homologous channels
right_channels = [1, 3, 5]  # Replace with the actual indices of right homologous channels

# Calculate power and lateralization index
power_left = np.sum(alpha_filtered_data[left_channels] ** 2, axis=0)
power_right = np.sum(alpha_filtered_data[right_channels] ** 2, axis=0)
LI = (power_right - power_left) / (power_right + power_left)

# plot the lateralisation
plt.figure(figsize=(10, 5))
plt.plot(LI)
plt.xlabel('Time (samples)')
plt.ylabel('Lateralization Index')
plt.title('Alpha Wave Lateralization Index')
plt.show()

