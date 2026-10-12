import os
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

class DeepfakeAudioDataset(Dataset):
    def __init__(self, data_dir, file_list, labels, sample_rate=16000, duration=3.0):
        self.data_dir = data_dir
        self.file_list = file_list
        self.labels = labels
        self.sample_rate = sample_rate
        self.max_length = int(sample_rate * duration)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        
        audio, sr = librosa.load(file_path, sr=self.sample_rate)
        
        if len(audio) < self.max_length:
            audio = np.pad(audio, (0, self.max_length - len(audio)))
        else:
            audio = audio[:self.max_length]
            
        waveform_tensor = torch.tensor(audio, dtype=torch.float32)
        
        # Sahi spelling: melspectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        spec_tensor = torch.tensor(mel_spec_db, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return {
            'waveform': waveform_tensor,
            'spectrogram': spec_tensor,
            'label': label
        }