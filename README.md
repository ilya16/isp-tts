# isp-tts: Simple Educational TTS model

> A minimal, educational text-to-speech (TTS) system developed for the  
> [**Speech Synthesis and Voice Cloning** course](https://github.com/ilya16/speech-synthesis-course)
> during the **[Independent Study Period 2025](https://student.skoltech.ru/isp) (ISP'25)** at **Skoltech**.

## Demos

The model components and training example are provided in the following demonstration notebooks:

- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ilya16/isp-tts/blob/main/notebooks/inference.ipynb) 
  [**inference.ipynb**](notebooks/inference.ipynb): demo with the TTS inference using the pre-trained models
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ilya16/isp-tts/blob/main/notebooks/training.ipynb)
  [**training.ipynb**](notebooks/training.ipynb): code for fine-tuning the pre-trained model on custom data

## Model

The model architecture takes inspiration from [FastPitch](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/FastPitch) 
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


## License

Prepared for academic and non-commercial use.  
Inspired by open-source projects and educational resources in speech synthesis research.