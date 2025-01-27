# ISP'25: Speech Synthesis and Voice Cloning

> Code for the simple text-to-speech model used for demonstration and practice purposes at the "Speech Synthesis and Voice Cloning" course during ISP'25 at Skoltech.

## Demos

The model components and training example are provided in the following demonstration notebooks:

- Inference using the pre-trained models: [inference.ipynb](notebooks/inference.ipynb)
- Fine-tuning the pre-trained model on custom data: [training.ipynb](notebooks/training.ipynb)

## Model

The model takes inspiration from [FastPitch](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/FastPitch) 
and [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS) and introduces a few modifications and simplifications. Its modules are:  

- Transformer-based `TextEncoder` with [ALiBi embeddings](https://github.com/ofirpress/attention_with_linear_biases)
- `Aligner` between text and mel spectrograms with CUDA-supported Monotonic Alignment Search
- Flow Matching and Transformer-based `TemporalAdaptor` for modeling the distribution of token duration, pitch, and energy
- Transformer-based `MelDecoder` with ALiBi embeddings

## Dataset

The dataset for training the models should have the following structure:

```
DATASET_ROOT
  wavs
    audio_1.wav
    audio_2.wav
    ...
    audio_N.wav
  meta.csv
```

The metadata file should have the following structure:

```
wavs/audio_1.wav|This is the sample text.
wavs/audio_2.wav|The second audio св+язяно с +этим т+екстом.
...
wavs/audio_N.wav|нижний текст.
```

In other words, the metadata files should contain the "|"-separated paths to audios (relative to the dataset root) and matched texts.
