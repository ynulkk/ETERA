import os
import math
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datasets import utils as dataset_utils
from scipy import signal
from scipy.io import loadmat
from collections import defaultdict
from scipy.signal import butter, lfilter, welch
from scipy.fftpack import fft, ifft

class EEGPreprocessConfig(object):
	def __init__(self):
		""" EEG input: [T, B, H, W] = [8, 10, 64, 64] """
		self.image_frames = 4
		self.image_channels = 10

		self.num_classes = 3

		self.num_trials = 15
		self.frequency = 200
		self.num_channels = 62
		self.window = 2.0
		self.frame_window = 0.5

		self.eeg_raw_data_path = "./eeg_raw_data"
		self.eeg_datasets_path = "./eeg_datasets"
		self.dataset = "SEED_IV"

def eeg_preprocess_config() -> EEGPreprocessConfig:
	parser = argparse.ArgumentParser(description='Initialize SEED dataset preprocessing configuration')
	parser.add_argument('--dataset', dest="dataset", required=False, type=str, default="SEED_IV",
						choices=["SEED", "SEED_IV"], help='Dataset to preprocess.')

	""" EEG input: [T, B, H, W] = [8, 10, 64, 64] """
	parser.add_argument('--image_frames', dest="image_frames", required=False, type=int, default=4, help='The time frames of the EEG.')
	parser.add_argument('--image_channels', dest="image_channels", required=False, type=int, default=6, help='The number of frequency bands for the EEG.')

	parser.add_argument('--num_classes', dest="num_classes", required=False, type=int, default=3,
						help='The number of categories in the EEG dataset , SEED: 3; SEED-IV: 4; DEAP: 9.')

	parser.add_argument('--num_trials', dest="num_trials", required=False, type=int, default=15,
						help='The number of experiments included in each experiment in the EEG dataset, SEED: 15; SEED-IV: 24.')
	parser.add_argument('--frequency', dest="frequency", required=False, type=int, default=200,
						help='Sampling rate of EEG dataset, SEED: 200Hz; SEED-IV: 200Hz; DEAP: 512Hz, downsampling to 128Hz.')
	parser.add_argument('--num_channels', dest="num_channels", required=False, type=int, default=62, help='Number of electrodes in EEG.')
	parser.add_argument('--window', dest="window", required=False, type=float, default=2.0,
						help='Window length for non overlapping segmentation of EEG, in seconds.')
	parser.add_argument('--frame_window', dest="frame_window", required=False, type=float, default=0.5,
						help='Divide each time window into non overlapping frames twice, in seconds.')

	parser.add_argument('--eeg_raw_data_path', dest="eeg_raw_data_path", required=False, type=str,
						default="./eeg_raw_data",
						help='Relative path (absolute path) of EEG dataset requiring preprocessing.')
	parser.add_argument('--eeg_datasets_path', dest="eeg_datasets_path", required=False, type=str,
						default="./eeg_datasets",
						help='Relative path (absolute path) of preprocessed EEG dataset.')

	eeg_preprocess_config = EEGPreprocessConfig()
	args = parser.parse_args()
	eeg_preprocess_config.dataset = args.dataset
	eeg_preprocess_config.image_frames = args.image_frames
	eeg_preprocess_config.image_channels = args.image_channels

	eeg_preprocess_config.num_classes = args.num_classes

	eeg_preprocess_config.num_trials = args.num_trials
	eeg_preprocess_config.frequency = args.frequency
	eeg_preprocess_config.num_channels = args.num_channels
	eeg_preprocess_config.window = args.window
	eeg_preprocess_config.frame_window = args.frame_window

	eeg_preprocess_config.eeg_raw_data_path = args.eeg_raw_data_path
	eeg_preprocess_config.eeg_datasets_path = args.eeg_datasets_path

	return eeg_preprocess_config

def butter_bandpass_filter(trial_signal, lowcut, highcut, fs=200, order=3):
	nyq = 0.5 * fs
	b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
	y = lfilter(b, a, trial_signal)
	return y

def compute_DE(eeg_signal) -> float:
	variance = np.var(eeg_signal, ddof=1)
	return math.log(2 * math.pi * math.e * variance) / 2

def compute_PSD(eeg_signal, freq_idx=0, frequency=1024) -> float:
	freq_start, freq_end = [1, 4, 8, 14, 31, 51], [4, 8, 14, 31, 51, 75]
	n_channels, n_samples = eeg_signal.shape[0], eeg_signal.shape[1]


	eeg_signal_psd = np.zeros((n_channels,))
	Hwindow = np.array([0.5 - 0.5 * np.cos(2 * np.pi * n / (n_samples + 1)) for n in range(1, n_samples + 1)])

	nfft = 2 ** math.ceil(math.log2(n_samples))
	freqs = np.fft.fftfreq(nfft, d=1 / nfft)  # Calculate the frequency value on the frequency axis
	for i in range(n_channels):
		channel_data = eeg_signal[i, :]
		freq_indices = np.where((freqs >= freq_start[freq_idx]) & (freqs <= freq_end[freq_idx]))[0]

		assert len(freq_indices) > 0, 'len(freqs_indices) <= 0'

		eeg_Hdata = channel_data * Hwindow
		eeg_signal_fft = fft(eeg_Hdata, nfft)

		power_spectrum = np.abs(eeg_signal_fft)
		eeg_signal_psd[i] = np.sum(power_spectrum[freq_indices] ** 2) / (freq_end[freq_idx] - freq_start[freq_idx])
	return eeg_signal_psd

def compute_PSD_welch(eeg_signal, freq_idx=0, frequency=200) -> float:
	freq_start, freq_end = [1, 4, 8, 14, 31, 51], [4, 8, 14, 31, 51, 75]
	n_channels, n_samples = eeg_signal.shape[0], eeg_signal.shape[1]	# "hamming"

	eeg_signal_psd = np.zeros((n_channels,))
	nperseg = 1  # Choose one tenth of the signal length as the window size
	nfft = 2 ** math.ceil(math.log2(n_samples))
	for i in range(n_channels):
		channel_data = eeg_signal[i, :]
		freqs, data_psd = welch(x=channel_data, fs=frequency, nfft=nfft, detrend=None, window='hann',
								nperseg=nperseg, scaling='density')

		freq_indices = np.where((freqs >= freq_start[freq_idx]) & (freqs <= freq_end[freq_idx]))[0]
		eeg_signal_psd[i] = np.sum(data_psd[freq_indices])
	return eeg_signal_psd


def extract_spectral_de_psd(eeg_signal, frequency=200, image_frames=8, image_channels=5, window=2.0, frame_window=0.5, filter_order=3):
	num_channels, num_sample = eeg_signal.shape[0], eeg_signal.shape[1]
	window_length, frame_window_length = int(window * frequency), int(frame_window * frequency)
	start = int(5 * frequency)
	delta_1D = np.zeros(shape=[0], dtype=float)
	theta_1D = np.zeros(shape=[0], dtype=float)
	alpha_1D = np.zeros(shape=[0], dtype=float)
	beta_1D = np.zeros(shape=[0], dtype=float)
	gamma_1D = np.zeros(shape=[0], dtype=float)
	gamma2_1D = np.zeros(shape=[0], dtype=float)

	for channel in range(num_channels):
		trial_signal = eeg_signal[channel]
		clear_signal = butter_bandpass_filter(trial_signal, 0.5, 75, frequency, order=filter_order)

		delta = butter_bandpass_filter(clear_signal, 1, 4, frequency, order=filter_order)
		theta = butter_bandpass_filter(clear_signal, 4, 8, frequency, order=filter_order)
		alpha = butter_bandpass_filter(clear_signal, 8, 14, frequency, order=filter_order)
		beta = butter_bandpass_filter(clear_signal, 14, 31, frequency, order=filter_order)
		gamma = butter_bandpass_filter(clear_signal, 31, 51, frequency, order=filter_order)
		gamma2 = butter_bandpass_filter(clear_signal, 51, 75, frequency, order=filter_order)


		delta_1D = np.append(delta_1D, delta)
		theta_1D = np.append(theta_1D, theta)
		alpha_1D = np.append(alpha_1D, alpha)
		beta_1D = np.append(beta_1D, beta)
		gamma_1D = np.append(gamma_1D, gamma)
		gamma2_1D = np.append(gamma2_1D, gamma2)

	data_2D = np.stack(
		[
			delta_1D,
			theta_1D,
			alpha_1D,
			beta_1D,
			gamma_1D,
			gamma2_1D
		])
	data_2D = data_2D.reshape([image_channels, num_channels, -1])

	mean_values = np.mean(data_2D[:, :, 0 : start], axis=2, keepdims=True)
	data_2D = data_2D[:, :, start : start + ((num_sample - start) // (window_length)) * window_length]
	data_2D = data_2D - mean_values

	data_3D = data_2D.reshape([image_channels, num_channels, -1, image_frames, frame_window_length])

	data_3D = np.transpose(data_3D,
						   axes=[2, 3, 0, 1, 4])  # [-1, image_frames, image_channels, num_channels, window_length]

	""" Calculate PSD features """
	data_psd = np.empty([data_3D.shape[0], data_3D.shape[1], data_3D.shape[2], data_3D.shape[3]])
	for i in range(data_3D.shape[0]):
		for j in range(data_3D.shape[1]):
			for k in range(data_3D.shape[2]):
				data_psd[i][j][k] = compute_PSD_welch(data_3D[i][j][k], k, frequency)
				#data_psd[i][j][k] = compute_PSD(data_3D[i][j][k], k)
	psd_3D = np.array(data_psd)

	""" Calculate DE features """
	data_3D = data_3D.reshape([-1, frame_window_length])
	de_1D = np.array([compute_DE(signal) for signal in data_3D])
	de_1D = de_1D.reshape([-1, num_channels])
	de_3D = de_1D.reshape([-1, image_frames, image_channels, num_channels])

	""" Connect DE and PSD features in the frequency dimension"""
	de_psd_3D = np.empty([de_3D.shape[0], de_3D.shape[1], de_3D.shape[2] * 2, de_3D.shape[3]])
	for i in range(de_3D.shape[0]):
		for j in range(de_3D.shape[1]):
			for k in range(de_3D.shape[2]):
				de_psd_3D[i][j][k * 2] = de_3D[i][j][k]
				de_psd_3D[i][j][k * 2 + 1] = psd_3D[i][j][k]
	return de_psd_3D  # [-1, image_frames, image_channels, num_channels]

def process(eeg_file_name, subject_name: str, session_ID: int, session_labels, image_frames: int, image_channels: int, num_trials: int, num_channels: int, window=2.0, frame_window=0.5, filter_order=3,):
	data = loadmat(eeg_file_name)
	print("File: ", eeg_file_name)

	feats_3D = np.empty([0, image_frames, 2 * image_channels, num_channels])
	labels = np.empty([0])

	for trial in range(num_trials):
		trial_signal = data[subject_name + '_eeg' + str(trial + 1)]
		de_3D = extract_spectral_de_psd(trial_signal, image_frames=image_frames, image_channels=image_channels, window=window, frame_window=frame_window,filter_order=filter_order,)

		feats_3D = np.vstack([feats_3D, de_3D])
		labels = np.append(labels, np.array([session_labels[session_ID - 1][trial]] * de_3D.shape[0]))

	print("EEG Features shape:", feats_3D.shape)
	return feats_3D, labels

def SEED_run(eeg_raw_data_path: Path, eeg_datasets_path: str, image_frames: int, image_channels: int, num_trials: int, num_channels: int, window=2.0, frame_window=0.5, filter_order=3,):
	subject_short_names = {
		'1': 'djc',
		'2': 'jl',
		'3': 'jj',
		'4': 'lqj',
		'5': 'ly',
		'6': 'mhw',
		'7': 'phl',
		'8': 'sxy',
		'9': 'wk',
		'10': 'ww',
		'11': 'wsf',
		'12': 'wyw',
		'13': 'xyl',
		'14': 'ys',
		'15': 'zjy'
	}

	session_count = defaultdict(int)
	session_labels = [
		[2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0],
		[2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0],
		[2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0]
	]
	for eeg_file_name in map(Path, dataset_utils.list_files_sorted(eeg_raw_data_path, pattern='*_*.mat*')):
		subject_id = eeg_file_name.name.split('_')[0]  # extracts the subject id from filename
		session_count[subject_id] += 1

		# spec_feats_4D = [num_trials, -1, image_frames, image_channels, num_channels]
		spec_feats_3D, labels = process(eeg_file_name=eeg_file_name,
										subject_name=subject_short_names[subject_id],
										session_ID=session_count[subject_id],
										session_labels=session_labels,
										image_frames=image_frames,
										image_channels=image_channels,
										num_trials=num_trials,
										num_channels=num_channels,
										window=window,
										frame_window=frame_window,
										filter_order=filter_order,)
		if eeg_datasets_path == None:
			eeg_datasets_path = "E:/Datasets/SEED_depsd_datasets_1S_62_new"
		session_path = os.path.join(eeg_datasets_path,
									f'subject_{subject_id}/session_{session_count[subject_id]}').replace('\\', '/')
		if not os.path.exists(session_path):
			os.makedirs(session_path)

		for idx, spec in enumerate(spec_feats_3D):
			file_path = os.path.join(session_path,
									 f'subject_{subject_id}_session_{session_count[subject_id]}_{idx + 1}_{int(labels[idx])}.npy').replace(
				'\\', '/')
			with open(file_path, 'wb') as f:
				np.save(f, spec)

		print(f"Saved preprocess data for subject_{subject_id}_session_{session_count[subject_id]}.")

def SEED_IV_run(eeg_raw_data_path: Path, eeg_datasets_path: str, image_frames: int, image_channels: int, num_trials: int, num_channels: int,  window=2.0, frame_window=0.5):
	subject_short_names = {
		'1': 'cz',
		'2': 'ha',
		'3': 'hql',
		'4': 'ldy',
		'5': 'ly',
		'6': 'mhw',
		'7': 'mz',
		'8': 'qyt',
		'9': 'rx',
		'10': 'tyc',
		'11': 'whh',
		'12': 'wll',
		'13': 'wq',
		'14': 'zjd',
		'15': 'zjy'
	}

	session_labels = [
		[1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
		[2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
		[1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0]
	]
	for session in range(0, 3):
		eeg_session_path = Path(os.path.join(eeg_raw_data_path, f'{session + 1}').replace('\\', '/'))
		for eeg_file_name in map(Path, dataset_utils.list_files_sorted(eeg_session_path, pattern='*_*.mat*')):
			subject_id = eeg_file_name.name.split('_')[0]  # extracts the subject id from filename

			# spec_feats_3D = [num_trials, -1, image_frames, image_channels, num_channels]
			spec_feats_3D, labels = process(eeg_file_name=eeg_file_name,
											subject_name=subject_short_names[subject_id],
											session_ID=session + 1,
											session_labels=session_labels,
											image_frames=image_frames,
											image_channels=image_channels,
											num_trials=num_trials,
											num_channels=num_channels,
											window=window,
											frame_window=frame_window,
											)
			if eeg_datasets_path == None:
				eeg_datasets_path = "E:/Datasets/SEED_IV_depsd_datasets_1S_62"
			session_path = os.path.join(eeg_datasets_path,
										f'subject_{subject_id}/session_{session + 1}').replace('\\', '/')
			if not os.path.exists(session_path):
				os.makedirs(session_path)

			for idx, spec in enumerate(spec_feats_3D):
				file_path = os.path.join(session_path,
										 f'subject_{subject_id}_session_{session + 1}_{idx + 1}_{int(labels[idx])}.npy').replace('\\', '/')
				with open(file_path, 'wb') as f:
					np.save(f, spec)

			print(f"Saved preprocess data for subject_{subject_id}_session_{session + 1}.")

if __name__ == '__main__':
	eeg_config = eeg_preprocess_config()

	if eeg_config.dataset == "SEED_IV":
		SEED_IV_run(
			eeg_raw_data_path=eeg_config.eeg_raw_data_path,
			eeg_datasets_path=eeg_config.eeg_datasets_path,
			image_frames=eeg_config.image_frames,
			image_channels=eeg_config.image_channels,
			num_trials=eeg_config.num_trials,
			num_channels=eeg_config.num_channels,
			window=eeg_config.window,
			frame_window=eeg_config.frame_window,
		)
	else:
		SEED_run(
			eeg_raw_data_path=Path(eeg_config.eeg_raw_data_path),
			eeg_datasets_path=eeg_config.eeg_datasets_path,
			image_frames=eeg_config.image_frames,
			image_channels=eeg_config.image_channels,
			num_trials=eeg_config.num_trials,
			num_channels=eeg_config.num_channels,
			window=eeg_config.window,
			frame_window=eeg_config.frame_window,
			filter_order=5,
		)


