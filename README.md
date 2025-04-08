# Singularity Images with AI Tools for MetaCentrum

This repository contains a build recipe for constructing a Singularity image based on the NVIDIA TensorFlow container (`nvcr.io/nvidia/tensorflow:24.12-tf2-py3`) with a comprehensive set of AI tools for research and teaching.

The image is tailored for use on the Czech national academic infrastructure **MetaCentrum** and was built to support scientific research in artificial intelligence and machine learning, particularly in **speech technologies**, **natural language processing**, and **deep learning**.

## Author

**Jan Švec**  
Department of Cybernetics  
Faculty of Applied Sciences  
University of West Bohemia, Pilsen, Czech Republic  
Email: <honzas@kky.zcu.cz>

## Purpose

The main goals of this image are:
- Support academic research in AI, especially multilingual and multimodal processing.
- Provide a reproducible, consistent environment for teaching AI-related topics to students.
- Facilitate high-performance GPU-enabled computing via MetaCentrum with optimized builds of CUDA-dependent libraries.

## Included Tools and Libraries

The image includes:

### System and Development Tools
- Build tools: `g++`, `make`, `cmake`, `autoconf`, `libtool`, `git`, `svn`
- Audio & multimedia: `sox`, `ffmpeg`, `libsndfile-dev`, `espeak-ng`, `portaudio19-dev`
- System utilities: `tmux`, `vim`, `ncdu`, `htop`, `mc`

### Python Libraries and Frameworks
- Deep learning: `TensorFlow 2.17.0+nv24.12`, `PyTorch 2.6.0`, `JAX`, `Flax`, `Keras`
- NLP: `Transformers`, `SentencePiece`, `Sentence-Transformers`, `Evaluate`, `SeqEval`, `Datasets`, `Hugging Face PEFT`, `TRLLib`, `Whisper`
- ASR: `SpeechBrain`, `PyCTCDecode`, `JiWER`, `Pyannote.audio`, `KenLM`, `phonemizer`, `pyreaper`
- ML libraries: `Scikit-learn`, `XGBoost`, `mlxtend`, `UMAP`, `Lightning`, `TorchMetrics`, `TorchInfo`
- Audio processing: `Librosa`, `Pydub`, `PyWorld`, `resampy`, `ffmpeg-normalize`, `mutagen`, `Encodec`
- Scientific computing: `NumPy`, `SciPy`, `Cython`, `Pandas`, `Matplotlib`, `Seaborn`, `Biopython`, `ML-DTypes`
- Utilities: `ujson`, `more-itertools`, `shortuuid`, `anyascii`, `aiohttp`, `flask`, `packaging`, `Pebble`
- Experimentation: `Wandb`, `TensorBoard`, `Papermill`, `Notebook`, `JupyterLab`, `ipywidgets`

### CUDA and Performance
- GPU build of `llama.cpp` with `gguf-py`
- `bitsandbytes`, `flash-attn`, `xlstm`, `liger-kernel`

### Other Notable Tools
- `ollama` support (for language model serving)
- `gruut` phonemization support for multiple languages
- Jupyter ecosystem for interactive research and teaching

### Cleanups
- Unnecessary preinstalled packages (like cudf) are removed to reduce image size.
- JupyterLab in latest version.
- Temporary files are cleaned up.

## Usage

To build the image:

```bash
export SINGULARITY_TMPDIR=...
export SINGULARITY_CACHEDIR=...
singularity build -f tensorflow24.12-r8.simg tensorflow24.12-r8.recipe
```

This results to in ~13GB image tensorflow24.12-r8.simg .
