# DeepNoise Project: Dataset Definition

## Dataset Strategy
**Chosen Option:** We selected the option to **Use a public dataset** (specifically, the DCASE 2020 Challenge Task 2 Development Dataset, which contains a curated subset of the MIMII industrial machine recordings).

This curated dataset contains real-world recordings of four common industrial machine types (fans, pumps, valves, and slide rails) operating in both healthy (normal) and anomalous (faulty) states, mixed with real factory background noises at different volume levels.

## Dataset Details

- **Source of the data:** 
  * **Repository:** Zenodo (DOI: 10.5281/zenodo.3678171)
  * **Verified URL:** [https://zenodo.org/record/3678171](https://zenodo.org/record/3678171)
  * **Zip Files to Download (Total ~4.6 GB):**
    1. `dev_data_fan.zip` (1.4 GB)
    2. `dev_data_pump.zip` (1.0 GB)
    3. `dev_data_valve.zip` (1.1 GB)
    4. `dev_data_slider.zip` (1.1 GB)
- **Number of classes:** 8 distinct classes
- **The 8 Acoustic Classes:**
  1. **Fan Normal:** Smooth, continuous hum of a healthy industrial cooling fan.
  2. **Fan Anomaly:** Squeaking or rattling indicating bearing friction or physical damage.
  3. **Pump Normal:** Healthy vibration and continuous flow sounds.
  4. **Pump Anomaly:** Cavitation, clogging, or grinding pump sounds.
  5. **Valve Normal:** Clean, repetitive opening and closing clicking sounds.
  6. **Valve Anomaly:** Gas/liquid leakage hiss, or irregular clicking due to contamination.
  7. **Slider Normal:** Regular, smooth sliding rails moving back and forth.
  8. **Slider Anomaly:** Scraping, scratching, or uneven sliding due to rails wear or misalignment.
- **Expected number of audio samples per class:** Over 1,000 samples for normal classes, and 200–400 for anomalous classes. 
- **Approximate duration of each audio sample:** Standardized to **4 seconds** per sample. (MIMII clips are 10 seconds long; we segment them into 4-second windows to increase sample size and reduce VRAM usage).
- **File format:** `.wav` (Waveform Audio File Format, 16-bit, mono) to preserve audio fidelity.
- **Sampling rate:** Resampled from 16,000 Hz to a standardized **22,050 Hz (22.05 kHz)**.
- **Dataset Balance:** **Unbalanced**. Normal operation samples are significantly more abundant than anomaly samples. We will use data augmentation (e.g. noise injection, time-shifting) and weighted loss during training.

## Possible Limitations of the Dataset
1. **Unbalanced Anomaly Data:** The model might struggle to recognize anomalies if not compensated by weighted loss functions or augmentation.
2. **Factory Background Noise:** The recordings are mixed with real factory background noise at different SNRs (6 dB, 0 dB, -6 dB). This makes the task realistic but highly challenging for the network to extract clean signal signatures.
3. **Domain Shift:** Different machines of the same type might sound slightly different depending on their manufacturer or installation location.

