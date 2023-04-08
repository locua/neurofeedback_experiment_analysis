import xml.etree.ElementTree as ET
import h5py
import pandas as pd
import numpy as np

def _get_channels_and_fs(xml_str_or_file):
    root = ET.fromstring(xml_str_or_file)
    if root.find('desc').find('channels') is not None:
        channels = [k.find('label').text for k in root.find('desc').find('channels').findall('channel')]
    else:
        channels = [k.find('name').text for k in root.find('desc').findall('channel')]
    fs = int(root.find('nominal_srate').text)
    return channels, fs

def _get_signals_list(xml_str):
    root = ET.fromstring(xml_str)
    derived = [s.find('sSignalName').text for s in root.find('vSignals').findall('DerivedSignal')]
    composite = []
    if root.find('vSignals').findall('CompositeSignal')[0].find('sSignalName') is not None:
        composite = [s.find('sSignalName').text for s in root.find('vSignals').findall('CompositeSignal')]
    return derived + composite


def _get_info(f):
    if 'channels' in f:
        channels = [ch.decode("utf-8")  for ch in f['channels'][:]]
        fs = int(f['fs'][()])
    else:
        channels, fs = _get_channels_and_fs(f['stream_info.xml'][0])
    signals = _get_signals_list(f['settings.xml'][0])
    n_protocols = len([k for k in f.keys() if ('protocol' in k and k != 'protocol0')])
    block_names = [f['protocol{}'.format(j+1)].attrs['name'] for j in range(n_protocols)]
    return fs, channels, block_names, signals


def load_data(file_path):
    with h5py.File(file_path) as f:
        # load meta info
        fs, channels, p_names, signals = _get_info(f)

        # load raw data
        data = [f['protocol{}/raw_data'.format(k + 1)][:] for k in range(len(p_names))]
        df = pd.DataFrame(np.concatenate(data), columns=channels)

        # load signals data
        signals_data = [f['protocol{}/signals_data'.format(k + 1)][:] for k in range(len(p_names))]
        df_signals = pd.DataFrame(np.concatenate(signals_data), columns=['signal_'+s for s in signals])
        df = pd.concat([df, df_signals], axis=1)

        # load timestamps
        if 'timestamp' in df:
            timestamp_data = [f['protocol{}/timestamp_data'.format(k + 1)][:] for k in range(len(p_names))]
            df['timestamps'] = np.concatenate(timestamp_data)

        # events data
        events_data = [f['protocol{}/mark_data'.format(k + 1)][:] for k in range(len(p_names))]
        df['events'] = np.concatenate(events_data)

        # reward data
        if 'protocol1/reward_data' in f:
            reward_data = [f['protocol{}/reward_data'.format(k + 1)][:] for k in range(len(p_names))]
            df['reward'] = np.concatenate(reward_data)

        # participant response data
        if 'protocol1/choice_data' in f:
            choice_data = [f['protocol{}/choice_data'.format(k + 1)][:] for k in range(len(p_names))]
            df['choice'] = np.concatenate(choice_data)

        if 'protocol1/answer_data' in f:
            answer_data = [f['protocol{}/answer_data'.format(k + 1)][:] for k in range(len(p_names))]
            df['answer'] = np.concatenate(answer_data)

        # Probe data
        if 'protocol1/probe_data' in f:
            probe_data = [f['protocol{}/probe_data'.format(k + 1)][:] for k in range(len(p_names))]
            df['probe'] = np.concatenate(probe_data)

        # Chunk data
        if 'protocol1/chunk_data' in f:
            chunk_data = [f['protocol{}/chunk_data'.format(k + 1)][:] for k in range(len(p_names))]
            df['chunk_n'] = np.concatenate(chunk_data)

        # Posner Cue data
        if 'protocol1/cue_data' in f:
            cue_data = [f['protocol{}/cue_data'.format(k + 1)][:] for k in range(len(p_names))]
            df['cue'] = np.concatenate(cue_data)

        # Posner stim data
        if 'protocol1/posner_stim_data' in f:
            posner_stim_data = [f['protocol{}/posner_stim_data'.format(k + 1)][:] for k in range(len(p_names))]
            df['posner_stim'] = np.concatenate(posner_stim_data)

        # Posner stim time
        if 'protocol1/posner_stim_time' in f:
            posner_stim_time = [f['protocol{}/posner_stim_time'.format(k + 1)][:] for k in range(len(p_names))]
            df['posner_time'] = np.concatenate(posner_stim_time)

        # response data
        if 'protocol1/response_data' in f:
            response_data = [f['protocol{}/response_data'.format(k + 1)][:] for k in range(len(p_names))]
            df['response_data'] = np.concatenate(response_data)

        # set block names and numbers
        df['block_name'] = np.concatenate([[p]*len(d) for p, d in zip(p_names, data)])
        df['block_number'] = np.concatenate([[j + 1]*len(d) for j, d in enumerate(data)])
    return df, fs, channels, p_names

import matplotlib.pyplot as plt

def plot_basic(dat):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dat)
    ax.axhline(color='black')
    plt.show()

def plot_raw_all(dat):
    fig, ax = plt.subplots(figsize=(12, 6))
    cols = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7','F8', 'T7', 'T8', 'P7', 'P8', 'FZ', 'CZ', 'PZ', 'OZ', 'FC1', 'FC2',
       'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'TP9', 'TP10', 'POZ', 'ECG',
       'F1', 'F2', 'C1', 'C2', 'P1', 'P2', 'AF3', 'AF4', 'FC3', 'FC4', 'CP3',
       'CP4', 'PO3', 'PO4', 'F5', 'F6', 'C5', 'C6', 'P5', 'P6', 'AF7', 'AF8',
       'FT7', 'FT8', 'TP7', 'TP8', 'PO7', 'PO8', 'FT9', 'FT10', 'FPZ', 'CPZ']
    for col in cols:
        ax.plot(df[col])
    plt.show()

if __name__ == '__main__':
    import sys
    file_path = sys.argv[1]
    df, fs, channels, p_names = load_data(file_path)
    print(df.groupby('block_number')['block_name'].first())
    print(df.head())
    print(df.columns)
    print(df.signal_left.describe())

    #plot_basic(df.signal_left)
    #plot_basic(df.signal_AAI)
    plot_raw_all(df)
