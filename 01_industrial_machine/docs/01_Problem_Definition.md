# DeepNoise Project: Problem Definition & Scenario

## Chosen Scenario
**Chosen Option:** Predictive maintenance using machine sounds

## Context of the Problem
In modern manufacturing and industrial facilities, machinery such as motors, pumps, and fans operate continuously. The acoustic profile of these machines is highly indicative of their mechanical health. By monitoring these sounds, we can detect anomalies before they lead to catastrophic mechanical failures. 

## Why the Classification Task is Relevant
Unplanned downtime in industrial settings costs millions of dollars annually. Traditional maintenance is either reactive (fixing after breaking) or preventive (fixing based on a schedule, which can be unnecessary). Acoustic predictive maintenance allows for condition-based monitoring, detecting early signs of wear (like friction or impacts) safely and non-invasively, saving money and preventing accidents.

## Acoustic Classes (Categories)
The system will classify short audio segments into one of the following **eight classes**, representing healthy and faulty states of four key industrial machines:

1. **Fan Normal:** Smooth, continuous hum of a healthy industrial cooling fan.
2. **Fan Anomaly:** Squeaking or rattling indicating bearing friction or physical damage.
3. **Pump Normal:** Healthy vibration and continuous flow sounds.
4. **Pump Anomaly:** Cavitation, clogging, or grinding pump sounds.
5. **Valve Normal:** Clean, repetitive opening and closing clicking sounds.
6. **Valve Anomaly:** Gas/liquid leakage hiss, or irregular clicking due to contamination.
7. **Slider Normal:** Regular, smooth sliding rails moving back and forth.
8. **Slider Anomaly:** Scraping, scratching, or uneven sliding due to rails wear or misalignment.

## Possible Challenges
- **High Background Noise:** Industrial environments are extremely noisy. The target acoustic signal in MIMII is mixed with real background noise at low Signal-to-Noise Ratios (SNR, up to -6 dB), which can mask subtle faults.
- **Class Similarity:** Certain mechanical faults might produce similar frequency components (e.g. grinding in pumps vs scraping in sliders), making them hard to distinguish.
- **Data Imbalance:** While 'Normal Operation' data is abundant, recording actual 'Fault' data is rare and difficult to obtain in real-world settings, leading to fewer anomaly samples per class.

