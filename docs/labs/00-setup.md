# Lab 0 — Environment Setup

**Time:** 15 minutes · **Hardware:** any laptop (GPU optional)

## Create the environment

```bash
mkdir -p ~/nca-genl-labs && cd ~/nca-genl-labs
python3 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
```

## Install the core packages

```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
pip install spacy nltk
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers datasets sentence-transformers accelerate
pip install faiss-cpu
pip install evaluate rouge-score sacrebleu
pip install peft
```

!!! note "GPU users"
    Install the CUDA build of PyTorch instead — pick the right command for your CUDA
    version from [pytorch.org](https://pytorch.org/get-started/locally/). Everything in
    these labs also runs on CPU, just more slowly.

Download the spaCy model and NLTK data:

```bash
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

## Verify

Save as `check_env.py` and run it:

```python
import sys
import numpy as np
import torch
import spacy
import transformers

print(f"python        {sys.version.split()[0]}")
print(f"numpy         {np.__version__}")
print(f"torch         {torch.__version__}")
print(f"transformers  {transformers.__version__}")
print(f"spacy         {spacy.__version__}")

print(f"\nCUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device: {torch.cuda.get_device_name(0)}")
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  memory: {total:.1f} GB")
else:
    print("  running on CPU — fine for every lab in this course")

nlp = spacy.load("en_core_web_sm")
doc = nlp("NVIDIA announced new GPUs in Santa Clara.")
print("\nspaCy entities:", [(e.text, e.label_) for e in doc.ents])
```

Expected output ends with something like:

```text
spaCy entities: [('NVIDIA', 'ORG'), ('Santa Clara', 'GPE')]
```

## Optional: check your GPU from the shell

```bash
nvidia-smi
```

Read the output as exam practice — it shows driver and CUDA version, GPU name, **memory
used / total** (the number that decides what models fit), utilisation percentage, and the
processes holding memory.

## Optional: a local LLM for Labs 4 and 6

The labs work with small Hugging Face models on CPU. If you want a more capable local
model for the generation step, install [Ollama](https://ollama.com) and pull a small one:

```bash
ollama pull llama3.2:3b
```

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `OSError: Can't find model 'en_core_web_sm'` | Re-run `python -m spacy download en_core_web_sm` inside the activated venv |
| Very slow model downloads | Set `HF_HOME` to a fast disk; models cache under `~/.cache/huggingface` |
| `faiss` import error on Apple Silicon | `pip install faiss-cpu --no-cache-dir`, or use `sklearn.neighbors.NearestNeighbors` instead |
| CUDA out of memory | Reduce batch size; use a smaller model; add `torch_dtype=torch.float16` |

Next: **[Lab 1 — spaCy & NumPy](01-spacy-numpy.md)**.
