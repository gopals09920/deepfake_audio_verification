import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
import kagglehub
from dataset import DeepfakeAudioDataset
from model import BankingDeepfakeDetector

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Kagglehub se dataset path fetch karein
    print("Locating dataset via kagglehub...")
    data_dir = kagglehub.dataset_download("birdy654/deep-voice-deepfake-voice-recognition")
    print(f"Dataset path: {data_dir}")

    train_files = []
    train_labels = []

    # Dataset folder ke andar ki saari audio files aur folders scan karein
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(('.wav', '.mp3', '.flac')):
                rel_path = os.path.relpath(os.path.join(root, file), data_dir)
                train_files.append(rel_path)
                
                # Agar file ya folder ke naam mein 'fake' hai toh label 1, warna 0 (real)
                if 'fake' in file.lower() or 'fake' in root.lower():
                    train_labels.append(1)
                else:
                    train_labels.append(0)

    print(f"Total audio files found: {len(train_files)}")

    if len(train_files) == 0:
        print("Koi audio file nahi mili!")
        return

    batch_size = 2
    epochs = 3
    learning_rate = 2e-4
    num_classes = 2

    train_dataset = DeepfakeAudioDataset(data_dir=data_dir, file_list=train_files, labels=train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = BankingDeepfakeDetector(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    print("Starting Training Loop with Real Dataset...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            waveforms = batch['waveform'].to(device)
            spectrograms = batch['spectrogram'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(waveforms, spectrograms)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/banking_deepfake_detector.pth")
    print("Training complete! Model saved to models/banking_deepfake_detector.pth")

if __name__ == "__main__":
    train_model()