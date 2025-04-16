# Singularity Images with AI Tools for MetaCentrum

This repository contains a build recipe for constructing a Singularity image based on the NVIDIA TensorFlow container (`nvcr.io/nvidia/tensorflow:24.12-tf2-py3`) with a comprehensive set of AI tools for research and teaching.

The image is tailored for use on the Czech national academic infrastructure **MetaCentrum** and was built to support scientific research in artificial intelligence and machine learning, particularly in **speech technologies**, **natural language processing**, and **deep learning**.

## Docker Compatibility & Use as AI/ML Foundation

Although this project is Singularity-centric, the underlying setup is fully compatible with **Docker-based workflows**. The software stack, base image, and build logic can be easily transposed into a Dockerfile, making this repository a **perfect starting point for Dockerization** of:

- AI/ML experimentation environments
- Model training pipelines (including MLPs and transformers)
- Reproducible research containers
- GPU-accelerated inference and deployment setups

This compatibility makes it highly adaptable for both academic HPC environments (Singularity) and cloud-native or on-prem setups (Docker).

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

## Example PBS Job Submission Script (`qsub`)

This script runs a Jupyter Lab session from the Singularity image on MetaCentrum GPU nodes:

```bash
#!/bin/bash
#PBS -l select=1:ncpus=2:mem=10gb:scratch_local=24gb:ngpus=1
#PBS -l walltime=24:00:00
#PBS -j oe

echo ${PBS_O_LOGNAME:?This script must be run under PBS scheduling system, execute: qsub $0}

# define variables
SING_IMAGE="singularity/tensorflow24.12-r8.simg"
HOMEDIR=/home/$USER # substitute username and path to to your real username and path
HOSTNAME=`hostname -f`
JUPYTER_PORT="8888"
IMAGE_BASE=`basename $SING_IMAGE`
export PYTHONUSERBASE=$HOMEDIR/.local-${IMAGE_BASE}
export JUPYTER_CONFIG_DIR=$HOMEDIR/.local-${IMAGE_BASE}
export JUPYTER_DATA_DIR=$HOMEDIR/.local-${IMAGE_BASE}
export JUPYTER_RUNTIME_DIR=$SCRATCHDIR/jupyter-runtime

mkdir -p ${PYTHONUSERBASE}/lib/python3.12/site-packages

#find nearest free port to listen
isfree=$(netstat -taln | grep $JUPYTER_PORT)
while [[ -n "$isfree" ]]; do
    JUPYTER_PORT=$[JUPYTER_PORT+1]
    isfree=$(netstat -taln | grep $JUPYTER_PORT)
done


# test if scratch directory is set
# if scratch directory is not set, issue error message and exit
test -n "$SCRATCHDIR" || { echo >&2 "Variable SCRATCHDIR is not set!"; exit 1; }
trap '/software/meta-utils/public/clean_scratch' EXIT TERM

#set SINGULARITY variables for runtime data
export SINGULARITY_CACHEDIR=$HOMEDIR
export SINGULARITY_LOCALCACHEDIR=$SCRATCHDIR
export SINGULARITYENV_PYTHONUSERBASE=$PYTHONUSERBASE
export SINGULARITYENV_TF_GPU_ALLOCATOR=cuda_malloc_async
export SINGULARITYENV_TMPDIR=$SCRATCHDIR
export SINGULARITYENV_PREPEND_PATH=$PYTHONUSERBASE/bin:$PATH
export SINGULARITYENV_JUPYTER_CONFIG_DIR=$JUPYTER_CONFIG_DIR
export SINGULARITYENV_JUPYTER_DATA_DIR=$JUPYTER_DATA_DIR
export SINGULARITYENV_JUPYTER_RUNTIME_DIR=$JUPYTER_RUNTIME_DIR

LOCAL_SING_IMAGE=$SCRATCHDIR/$IMAGE_BASE
echo "Copying $SING_IMAGE to local scratch (started $(date))"
cp $SING_IMAGE $LOCAL_SING_IMAGE
echo "Copying $SING_IMAGE to local scratch (done $(date))"

TOKEN=$(uuidgen)
echo "$(date +%Y%m%d-%H%M%S) $PBS_JOBID running on $HOSTNAME, URL http://$HOSTNAME:$JUPYTER_PORT/lab?token=$TOKEN " >> $HOMEDIR/JupyterLab_jobs.txt
export SINGULARITYENV_JUPYTER_TOKEN=$TOKEN

# Define bind paths (edit as needed)
BIND=""
if [[ -d /auto ]]; then
  BIND="$BIND --bind /auto"
fi
if [[ -d $SCRATCHDIR ]]; then
  BIND="$BIND --bind $SCRATCHDIR"
fi
if [[ -d /storage ]]; then
  BIND="$BIND --bind /storage"
fi
if [[ -n /etc/krb5.conf ]]; then
  BIND="$BIND --bind /etc/krb5.conf"
fi
if [[ -n /etc/ssh/ssh_config ]]; then
  BIND="$BIND --bind /etc/ssh/ssh_config"
fi

echo "$PBS_JOBID singularity started on $HOSTNAME"
singularity exec --nv -H $HOMEDIR \
                 $BIND \
                 $LOCAL_SING_IMAGE jupyter-lab --port $JUPYTER_PORT --ip 0.0.0.0 --no-browser --notebook-dir=$HOMEDIR
echo "$PBS_JOBID singularity terminated on $HOSTNAME"
```
