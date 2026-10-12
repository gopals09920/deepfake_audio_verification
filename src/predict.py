import os
import torch
import librosa
import numpy as np
from model import BankingDeepfakeDetector

def predict_audio(file_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model initialize karein aur trained weights load karein
    model = BankingDeepfakeDetector(num_classes=2).to(device)
    model_path = "models/banking_deepfake_detector.pth"
    
    if not os.path.exists(model_path):
        print(f"Error: Trained model weights nahi mile {model_path} par!")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Audio load aur preprocess karein (jaise dataset mein kiya tha)
    sample_rate = 16000
    duration = 3.0
    max_length = int(sample_rate * duration)

    audio, sr = librosa.load(file_path, sr=sample_rate)
    
    if len(audio) < max_length:
        audio = np.pad(audio, (0, max_length - len(audio)))
    else:
        audio = audio[:max_length]
        
    waveform_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)
    
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    spec_tensor = torch.tensor(mel_spec_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        outputs = model(waveform_tensor, spec_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

    class_names = ["Real Audio", "Deepfake Audio"]
    result = class_names[predicted_class.item()]
    conf_score = confidence.item() * 100

    print(f"\n--- Prediction Result ---")
    print(f"File: {file_path}")
    print(f"Prediction: {result}")
    print(f"Confidence: {conf_score:.2f}%")
    print("-------------------------\n")

if __name__ == "__main__":
    # Test karne ke liye hamari hi dummy ya koi audio file ka path dein
    test_file = "./dataset/sample_real_1.wav"
    predict_audio(test_file)