import torch
import torch.nn as nn
from transformers import Wav2Vec2Model
class BankingDeepfakeDetector(nn.Module):
    def __init__(self, num_classes=2):
        super(BankingDeepfakeDetector, self).__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        for param in self.wav2vec.parameters():
            param.requires_grad = False
        self.wav2vec_fc = nn.Linear(768, 256)
        self.spec_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.spec_fc = nn.Linear(128, 256)
        self.classifier = nn.Sequential(
            nn.Linear(256 + 256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, waveforms, spectrograms):
        outputs = self.wav2vec(waveforms)
        wav_features = outputs.last_hidden_state.mean(dim=1)
        wav_features = torch.relu(self.wav2vec_fc(wav_features))
        spec_features = self.spec_cnn(spectrograms)
        spec_features = spec_features.view(spec_features.size(0), -1)
        spec_features = torch.relu(self.spec_fc(spec_features))
        combined = torch.cat((wav_features, spec_features), dim=1)
        logits = self.classifier(combined)
        return logits