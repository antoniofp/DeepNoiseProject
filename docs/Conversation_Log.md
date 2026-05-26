# DeepNoise Project - Conversation & Activity Log

This log documents the design process, technical decisions, and discussions between us and the AI tool, which was a combination of Gemini CLI and Antigravity CLI. Both are Google tools and use Gemini models as their main agent; this main agent was instructed to act as a Lead AI Engineer.

---

## Phase 1: Context & Initial Design

### Scenario Context Sourcing
- **Usuario:** "este es el contexto del proyectoContext Modern industrial, urban, and technological environments produce large amounts of acoustic information..."
- **IA (Gemini):** Created the initial `Context.md` establishing the core goal of acoustic event classification with at least four classes.

### Project Development Mandate
- **Usuario:** "ahora, el pryecto esta pensado en desarrollar 100% con IA, considera que lo vas a hacer todo tu, esta solo es la primera parte vas a desarrollar el pipeline completo de que tenemos que hacer definir el alcance etc.Objective Design a complete technical plan..."
- **IA (Gemini):** Created the master `Technical_Plan.md` outlining specific targets for signal transformation, baseline modeling, CNN architecture, and autonomous dataset downloading.

In the next queries, the AI was given the instructions from Canvas, and the output for that project phase was reviewed. We also checked for alignment to the project's requirements and objectives.

### Scenario Selection (Predictive Maintenance)
- **Usuario:** "Define the scenario your team will address. You may choose one of the following options or propose a different one: [...] , registra todo lo que estas haciendo, al final quiero un registro de toda la conversacion en otro md"
- **IA (Gemini):** Created `01_Problem_Definition.md` choosing the predictive maintenance scenario (machine operating sounds) and initialized this activity log.

### Audio Representation Strategy
- **Usuario:** "ahora sigue estoAudio Representation Strategy Raw audio is a one-dimensional signal, but convolutional neural networks are often applied to two-dimensional structures..."
- **IA (Gemini):** Created `03_Audio_Representation.md` comparing Raw Waveform, MFCCs, and Mel-spectrograms, and justified selecting the Mel-spectrogram for 2D convolutions.

### Machine Learning Pipeline Design
- **Usuario:** "Proposed Machine Learning PipeLine Design the complete pipeline of your system..."
- **IA (Gemini):** Created `04_ML_Pipeline.md` defining the 9 pipeline stages (acquisition, preprocessing, segmentation, extraction, train/val/test split, baseline model, CNN training, evaluation, and error analysis).

### Baseline Model Selection
- **Usuario:** "Baseline Model Before training a CNN, your team must propose a simple baseline model..."
- **IA (Gemini):** Created `05_Baseline_Model.md` proposing a Random Forest classifier trained on 1D globally averaged Mel-spectrogram features to establish a minimum performance threshold.

### Convolutional Neural Network Architecture
- **Usuario:** "Proposed Neural Network Propose a convolutional neural network architecture for the classification task..."
- **IA (Gemini):** Created `06_CNN_Architecture.md` defining the lightweight custom 2D CNN layers, compile parameters (Adam, Categorical Crossentropy), metrics (Recall, Macro F1), and overfitting risks.

### Validation Framework
- **Usuario:** "Validation Strategy Explain how your team will evaluate the system in the implementation stage..."
- **IA (Gemini):** Created `07_Validation_Strategy.md` detailing the 70/15/15 stratified split and explaining why Accuracy alone is highly misleading for unbalanced datasets.

### AI Mandate Instructions
- **Usuario:** "Predictive maintenance using machine sounds... osea quiero que pongas explicitamente cual escogiste de estos, además con los markdowns que tienes tienes ya el contexto del proyecto, pero tienes todo lo que queremos osea otra vez que te abra vas a entender que tienes que registar los log, etc."
- **IA (Gemini):** Created `GEMINI.md` to persist the lead developer role, logging guidelines, and roadmap context for subsequent sessions.

---

## Phase 2: Technical Architecture Refinement (Used voice-to-text tool to aid with more easily writing prompts that are longer and more detailed)

### Scenario Validation
- **Usuario:** "Okay, but before doing that, I want to go into the more fundamental aspects of this plan. My main concern is to select an appropriate environment where we will be identifying these sounds. Right now, I think we are set in an industrial environment and I just want to know how easy it is to train a network to identify these sounds based on the availability of datasets. [...] It's important that we have made a good decision before locking in the plan and start developing."
- **IA (Gemini):** Compared the industrial scenario to alternative domains (urban security, home safety, animals/instruments). Proved that the MIMII dataset provides a high-quality, pre-labeled baseline for predictive maintenance, making training highly feasible.

### Role of Convolutional Layers
- **Usuario:** "Ok, this scenario... It seems nice... Now, before continuing, I want you to teach me some stuff. I know what a fully connected layer is. I think that is a multilayer perceptron. [...] But in a convolutional neural network, there are many other, like, stages. I don't know what they really are. [...]. I'd like you to help me understand why... what each type of layer does."
- **IA (Gemini):** Explained the core concept of 2D convolutional layers, how filters/kernels slide over the image to detect localized shapes (edges, textures), and how multiple filters run in parallel.

### Filtering and Spectrogram Generation Order
- **Usuario:** "The fact that the 2D grid is smaller is just a side effect of using filters, right? You cannot convolute outside the pixel grid. So, it necessarily gets a bit smaller. Right? [...] And are these filters made, like, in parallel or in series, one after another [...] And, to my knowledge, a filter doesn't create a spectrogram. To create a spectrogram you have to perform Fourier analysis. So you've got me a bit confused here with all the stuff you've said."
- **IA (Gemini):** Clarified that the spectrogram is created via Fourier transform *before* the CNN. Explained that filters run in parallel, and size reduction is a boundary side-effect that can be bypassed using zero-padding.

### Fourier Parameters definition
- **Usuario:** "Okay, so we must also agree on the parameters of the Fourier analysis and the parameters that convert this Fourier analysis into a spectrogram, right?"
- **IA (Gemini):** Outlined the key parameters of the Mel-spectrogram conversion: window length (`n_fft`), step size (`hop_length`), frequency bins (`n_mels`), and logarithmic scaling (`power_to_db`).

### Gabor's Limit Trade-off
- **Usuario:** "So hop length divides along the time and number of melbands divides along the frequency. Knowing this, we should modify whatever documents we must do to say that this input image of the spectrogram is the starting one but can be modified if after testing we found out that it is not optimal. And yes, I would like the trade-off in between changing these parameters [...]"
- **IA (Gemini):** Explained the time-frequency resolution trade-off (Gabor's Limit) where narrow windows capture precise timing (metallic clicks) but blur pitches, while wide windows blur timing but capture precise pitch (steady hums).

### ReLU & MaxPooling Functions
- **Usuario:** "Ok, let's move on to the next layer"
- **IA (Gemini):** Outlined the purpose of ReLU activation (replacing negative values with zero to represent non-linear systems) and MaxPooling (downsampling for dimension reduction and translation invariance/timing robustness).

### Preservation of 2D Dimensions
- **Usuario:** "Ok, this makes sense. So, we take the raw audio data, we apply FFT with the Melbands structure in mind, then We apply our filters or the convolution layer, but... we still have a 2x2 matrix, a perceptron, which I guess it's the part where... ReLU activation function comes in. Doesn't it need, like, just a vector? [...] I mean, to my knowledge, what comes out of the ReLU part is a bunch of numbers in a 1D vector, so I don't really get at what time max pooling comes in. Please clear this out."
- **IA (Gemini):** Clarified that both ReLU and MaxPooling operate directly on 2D images, and the feature maps are only flattened into a 1D vector at the very end of all convolutional/pooling blocks to enter the dense perceptron layers.

### Source of Negative Activation Values
- **Usuario:** "Oh, okay, so a relu is just... it inputs... the input is the value of the pixel and the output is whatever the relu gives out. I found this confusing because if relu is just a linear function with everything behind zero turned to zero and pixels don't have... ...negative values. What's the point of this relu layer? [...]"
- **IA (Gemini):** Explained that raw input pixels are positive, but convolutional filters contain learned negative weights, which produce negative activation values during convolution, making ReLU necessary.

### GPU Iteration Speed
- **Usuario:** "Yeah, we should add this. Thanks for remembering. Also, since I am running a dedicated GPU, it wouldn't be too much of a time-consuming task to change a bit the architecture of the network, in case... ...the results are not satisfying enough and we re-trained it, right?"
- **IA (Gemini):** Confirmed that training on an RTX 3050 Ti would take under 3 minutes, permitting fast hyperparameter tuning and model retraining.

### Reverting Reflection Section
- **Usuario:** "Okay, please revert back to Section 10 to what it was. You are not supposed to edit that without me explicitly saying you to do so. [...] My prof will get angry if it detects use of AI in those words. I will ask you for information maybe, but not to write it all."
- **IA (Gemini):** Reverted Section 10 of the final report back to the user's original text, agreeing that the reflection must be in the student's own voice.

### Mel-Spectrogram vs. Normal Spectrogram
- **Usuario:** "What is the difference between a male spectrogram and a normal spectrogram? Because the way you describe them, they seem like exactly the same, minus maybe their normalization."
- **IA (Gemini):** Clarified that it is the "Mel-spectrogram" and explained that it warps the linear Hz scale logarithmically to mimic human pitch perception, prioritizing lower frequency detail.

### Dataset Sourcing Verification
- **Usuario:** "Ok, this makes sense. Now, let's review our dataset sources and structure. In the instructions, as you mentioned, and as you can see, we must use a dataset with at least four classes. And also, the links that you provide to the datasets, maybe you hallucinated them. [...] So please check that the datasets will be useful for us that contain everything we need and that they are easily accessible from our references to the datasets. Take your time for this."
- **IA (Gemini):** Verified the Zenodo URLs for the DCASE 2020 Task 2 Development Dataset as a curated, compact subset (fans and pumps) to avoid massive Zenodo download issues.

### Class Expansion to 8 Classes
- **Usuario:** "Yeah, and they don't have to be four classes. They must be at least four so we can add much more many classes. And also the... The file size is not really a concern unless it's something like more than 20 gigabytes. [...] I would say it's okay that you modify the dataset definition. and the corresponding part in the final report."
- **IA (Gemini):** Expanded the project design to 8 classes by incorporating valves and slide rails from the DCASE dataset, and updated the corresponding configuration files.

### DCASE Directory Restructuring
- **Usuario:** "Thanks. I checked the link that you placed in the references and I can find the page to the dataset and one thing that confused me is that... it is already divided in training and testing. Like, shouldn't we do this? I don't get it."
- **IA (Gemini):** Explained that the DCASE competition is unsupervised, whereas our project is a supervised classification task, requiring our scripts to merge and re-split the files.

### Preprocessing and Restructuring in Pipeline
- **Usuario:** "Is this already in the final report? Or in whatever file this preprocessing script should be mentioned?"
- **IA (Gemini):** Confirmed it wasn't, and added the folder-restructuring steps to the ML Pipeline, preprocessing, and report documents.

### Final Log & Document Polish
- **Usuario:** "i added some small comments to the ai log, can you make sure they format ok? and i'd like you to check part 10, do not rephrase it or chage my words, i just want you to check the spelling and make sure it is formatted correctly, add this query to the log as well. after this, the final report and log should be ready to be converted to pdfs and submitted."
- **IA (Gemini):** Formatted and corrected the spelling of the user's log comments, performed spelling checks (capitalizing "I") on Section 10 of the final report, and appended this final interaction to the log.

