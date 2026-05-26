# DeepNoise Project: Audio Representation Strategy

## Overview
Raw audio is a 1D sequence of amplitude values over time. While architectures like 1D-CNNs or RNNs can process this, modern acoustic classification heavily relies on 2D Convolutional Neural Networks (CNNs). To use 2D-CNNs, we must transform the 1D audio signal into a 2D image-like representation. 

Below is a comparison of three common representations.

---

## 1. Raw Waveform
The unmodified digital signal representing air pressure changes over time.

* **What it captures:** Instantaneous amplitude at each sampling point.
* **Preserves time and frequency?:** Time is explicitly preserved. Frequency is implicitly present but difficult for simple networks to extract without deep feature learning.
* **Suitable for a 2D-CNN?:** No. It requires 1D-CNNs (like WaveNet architectures).
* **Advantages:** No loss of information; zero preprocessing time.
* **Disadvantages:** Highly sensitive to phase shifts and noise; very large input dimension, making it computationally expensive and hard to learn structural patterns.
* **Decision:** **Will NOT use.** It is inefficient for standard 2D image classification architectures.

---

## 2. MFCCs (Mel-Frequency Cepstral Coefficients)
A highly compressed representation that extracts the "envelope" of the power spectrum, mimicking human hearing characteristics.

* **What it captures:** Timbre and spectral envelope shape.
* **Preserves time and frequency?:** It preserves time and a highly compressed form of frequency (decorrelated cepstral coefficients).
* **Suitable for a 2D-CNN?:** Yes, it forms a 2D matrix (time vs. coefficients).
* **Advantages:** Extremely low dimensionality (typically 13 to 40 coefficients per time frame); excellent for human speech recognition because it removes pitch and background harmonics.
* **Disadvantages:** Lossy compression. It discards fine-grained spectral details that are crucial for distinguishing non-speech mechanical sounds. The lack of spatial relation between adjacent coefficients makes 2D convolutions less intuitive.
* **Decision:** **Will NOT use.** While useful, it discards too much spectral information necessary for detecting mechanical friction or impacts.

---

## 3. Mel-spectrogram
A visual representation of the spectrum of frequencies of a signal as it varies with time, mapped to the Mel scale (which is logarithmic and mimics human ear frequency resolution).

* **What it captures:** Energy of different frequency bands over time.
* **Preserves time and frequency?:** Yes. The X-axis represents time, the Y-axis represents frequency (Mel-scaled), and the color intensity represents amplitude/energy.
* **Suitable for a 2D-CNN?:** **Yes, highly suitable.** It functions exactly like a 1-channel (grayscale) image.
* **Advantages:** Retains critical spectral details; maps low frequencies with high resolution and high frequencies with lower resolution, which matches how critical mechanical faults often present; highly compatible with transfer learning on image-based CNN architectures.
* **Disadvantages:** Higher dimensionality than MFCCs, requiring slightly more memory.
* **Decision:** **WILL USE as our primary representation.**

---

## Final Selection and Justification
The project will use the **Mel-spectrogram** as the primary audio representation. 

**Justification:** The Mel-spectrogram converts our 1D acoustic signal into a rich 2D time-frequency image. Industrial anomalies, such as bearing faults (which appear as high-frequency energy spikes) or imbalances (which appear as low-frequency periodic pulses), are visually distinct in a spectrogram. By using this representation, we can leverage powerful, well-researched 2D Convolutional Neural Networks (CNNs) originally designed for computer vision to detect these acoustic patterns efficiently and accurately.

**Parameter Baseline and Flexibility:**
Our initial feature extraction will use a standard configuration: `n_fft = 2048` (window size), `hop_length = 512` (stride), and `n_mels = 128` (frequency bands), producing a `128 x 173` input grid. Because of the fundamental time-frequency trade-off (Gabor's Limit), these parameters are not locked in. If early testing shows the model struggles to detect rapid metallic impacts (which require higher time resolution/smaller hop and window sizes) or constant motor hums (which benefit from higher frequency resolution), we will experimentally tune these parameters during the development phase.

