import os
import sys
import time
import numpy as np
import torch
import librosa
import sounddevice as sd

# Ensure VRAM allocation environment variables are set
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Append source directory to path to resolve local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import MODELS_DIR, SAMPLE_RATE, AUDIO_DURATION, N_MELS, N_FFT, HOP_LENGTH, DEVICE, CLASSES
from cnn_model import build_cnn_model

def ensure_length(y, target_len):
    """
    Pads with zeros if shorter than target_len, or truncates if longer.
    """
    if len(y) < target_len:
        return np.pad(y, (0, target_len - len(y)), mode="constant")
    elif len(y) > target_len:
        return y[:target_len]
    return y

def load_selected_model(model_type):
    """
    Loads and returns the DeepNoiseCNN model for the specified weight type.
    model_type can be 'augmented' or 'clean'.
    """
    model_path = os.path.join(MODELS_DIR, f"best_cnn_{model_type}.pth")
    if not os.path.exists(model_path):
        # Fallback to general best_cnn.pth if specific suffix doesn't exist
        fallback_path = os.path.join(MODELS_DIR, "best_cnn.pth")
        if os.path.exists(fallback_path):
            print(f"[*] Suffix-specific weights not found. Using fallback weights: {fallback_path}")
            model_path = fallback_path
        else:
            print(f"[ERROR] Model weights file not found at {model_path}")
            print("Please run train.py first to train the models.")
            sys.exit(1)
            
    print(f"[*] Loading DeepNoiseCNN weights from: {model_path}")
    model = build_cnn_model(num_classes=len(CLASSES))
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

def record_audio(duration, samplerate):
    """
    Records mono audio from the microphone for the specified duration and samplerate.
    Shows a progress bar to the user during recording.
    """
    target_samples = int(duration * samplerate)
    print("\nGet ready to make a sound!")
    for i in range(3, 0, -1):
        print(f"Recording starts in {i}...")
        time.sleep(1)
        
    print("\n🎙️  === RECORDING STARTED ===")
    
    # Start asynchronous recording
    recording = sd.rec(target_samples, samplerate=samplerate, channels=1, dtype='float32')
    
    # Display a simple text progress bar
    num_steps = 20
    step_duration = duration / num_steps
    for step in range(num_steps):
        percent = int((step + 1) / num_steps * 100)
        bar = "█" * (step + 1) + "░" * (num_steps - step - 1)
        sys.stdout.write(f"\rRecording: |{bar}| {percent}% ({((step+1)*step_duration):.1f}s / {duration:.1f}s)")
        sys.stdout.flush()
        time.sleep(step_duration)
        
    sd.wait()  # Ensure recording is fully complete
    print("\n🛑  === RECORDING STOPPED ===\n")
    
    # Flatten array to 1D
    y = recording.flatten()
    return y

def classify_waveform(y, model):
    """
    Preprocesses the raw audio waveform and uses the model to predict the class.
    Returns the predicted class index and softmax probabilities.
    """
    # 1. Normalize amplitude
    max_val = np.max(np.abs(y))
    if max_val > 0.0:
        y = y / max_val
        
    # 2. Standardize duration
    target_len = int(SAMPLE_RATE * AUDIO_DURATION)
    y = ensure_length(y, target_len)
    
    # 3. Extract Log-Mel Spectrogram features
    mel_spec = librosa.feature.melspectrogram(
        y=y, 
        sr=SAMPLE_RATE, 
        n_fft=N_FFT, 
        hop_length=HOP_LENGTH, 
        n_mels=N_MELS
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=1.0)
    
    # 4. Format shape to match network inputs: [Batch=1, Height=128, Width=216, Channels=1]
    input_tensor = torch.tensor(mel_spec_db, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    input_tensor = input_tensor.to(DEVICE)
    
    # 5. Model prediction pass
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=-1).cpu().numpy()[0]
        
    pred_idx = np.argmax(probs)
    return pred_idx, probs

def display_classification_results(pred_idx, probs):
    """
    Displays the predictions and class probability distribution in a nice CLI panel.
    """
    pred_class = CLASSES[pred_idx]
    pred_prob = probs[pred_idx]
    
    print("="*60)
    print(f"                   LIVE CLASSIFICATION RESULT")
    print("="*60)
    print(f" Predicted Class: 🌟 {pred_class.upper()} 🌟 (Confidence: {pred_prob*100:.2f}%)")
    print("-"*60)
    print(" Class Probabilities:")
    
    # Sort classes by probability to display best guesses first
    sorted_indices = np.argsort(probs)[::-1]
    for idx in sorted_indices:
        class_name = CLASSES[idx]
        prob = probs[idx]
        # Text-based horizontal bar chart
        bar_len = int(prob * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  {class_name.capitalize():<10} | {bar} | {prob*100:6.2f}%")
        
    print("="*60)

def main():
    print("="*60)
    print("           DEEPNOISE: LIVE DEMONSTRATION TOOL")
    print("="*60)
    print(f" Device selected: {DEVICE.upper()}")
    print(f" Classification Target: {len(CLASSES)} Animal Classes")
    print(f" Target Classes: {', '.join(CLASSES)}")
    
    # Query default microphone name
    try:
        default_input = sd.default.device[0]
        if default_input >= 0:
            device_info = sd.query_devices(default_input, 'input')
            print(f" 🎙️  Active Microphone: {device_info['name']}")
        else:
            print(" 🎙️  Active Microphone: Default System Device (No valid index)")
    except Exception:
        print(" 🎙️  Active Microphone: Default System Device")
        
    # Interactive Model Selection
    print("\nSelect the model version to load:")
    print("  [1] Augmented Model (Folds 1-3, validated on Fold 4)")
    print("  [2] Clean Model (Folds 1-3, validated on Fold 4 - No Augmentation)")
    print("  [3] Full Production Model (Folds 1-5, with 10% clean validation split - Recommended)")
    
    choice = input("\nEnter choice (1, 2 or 3, default is 3): ").strip()
    if choice == "1":
        model_type = "augmented"
    elif choice == "2":
        model_type = "clean"
    else:
        model_type = "full"
        
    # Load model
    model = load_selected_model(model_type)
    
    print("\nSystem ready! Press Enter to record 5.0 seconds of audio.")
    try:
        while True:
            input("\n👉 Press Enter to start recording (or Ctrl+C to quit)...")
            
            # Record audio
            try:
                waveform = record_audio(AUDIO_DURATION, SAMPLE_RATE)
            except Exception as e:
                print(f"\n[ERROR] Audio recording failed: {e}")
                print("Make sure a microphone is connected and configured.")
                continue
                
            # Check saturation / silence
            peak = np.max(np.abs(waveform))
            if peak >= 0.98:
                print("\n" + "!"*60)
                print("⚠️  WARNING: Microphone is saturating (clipping)!")
                print("   The recorded audio signal is too loud. This can distort the sound")
                print("   and confuse the model. Try moving further away or lowering your")
                print("   microphone volume/gain in system settings.")
                print("!"*60)
            elif peak < 0.01:
                print("\n" + "!"*60)
                print("⚠️  WARNING: Recording is extremely quiet!")
                print(f"   Peak amplitude was only {peak:.5f}. The model might only be hearing")
                print("   background room noise (which often triggers frog/cat predictions).")
                print("   Try making the sound louder or closer to the microphone.")
                print("!"*60)
            else:
                print(f"\n[*] Audio level checks passed (Peak amplitude: {peak:.3f})")
                
            # Optional playback to verify sound quality
            play_choice = input("\n🔊 Play back the recording to check quality? (y/N): ").strip().lower()
            if play_choice == 'y':
                print("Playing recorded audio...")
                sd.play(waveform, SAMPLE_RATE)
                sd.wait()
                print("Playback finished.")
                
            # Classify
            print("\nProcessing audio and running model inference...")
            pred_idx, probs = classify_waveform(waveform, model)
            
            # Show output
            display_classification_results(pred_idx, probs)
            
    except KeyboardInterrupt:
        print("\n\nExiting Live Demonstration. Goodbye!")

if __name__ == "__main__":
    main()
