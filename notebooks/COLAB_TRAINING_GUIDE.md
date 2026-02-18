# Step-by-step guide: Train on Google Colab (dumb-proof)

This guide assumes you want to train the **VQ-VAE + PixelSNAIL** anomaly detection model on **Google Colab** using the **DCASE 2020 Task 2** dev dataset. Follow the steps in order.

---

## Part 1 — Project index (what this repo does)

### Layout

```
semcom_asd_vqar/
├── configs/
│   ├── default.yaml    # Local/default paths and epochs
│   └── colab.yaml      # Colab: longer epochs, paths for Drive
├── notebooks/
│   ├── colab_train.ipynb   # Run this on Colab
│   └── COLAB_TRAINING_GUIDE.md  # This file
├── src/
│   ├── main.py             # Entry: train or eval from YAML + overrides
│   ├── data/
│   │   └── preprocessing.py   # DCASE2020Task2Dataset, mel spectrograms, stats
│   ├── engine/
│   │   ├── train.py        # Phase 1 (VQ-VAE) + Phase 2 (PixelSNAIL prior)
│   │   └── test.py         # Eval: ROC AUC from NLL and MSE
│   ├── models/
│   │   ├── autoencoders.py # MobileNetV2_8x_VQVAE, etc.
│   │   ├── pixelsnail.py   # Prior for code indices
│   │   ├── encoder.py, decoder.py, quantizer.py
│   └── utils/
│       ├── config.py       # load_config, deep_merge
│       ├── logger.py       # Logging helpers
│       └── audio.py        # load_wav, LogMelSpectrogramExtractor, collect_audio_files
├── requirements.txt
└── README.md
```

### Config (YAML)

- **mode**: `train` or `eval`
- **checkpoints**: `dir` (root directory; default `checkpoints`). Models go to `{dir}/models/`, stats to `{dir}/stats/`.
- **data**: `root_dir` (dataset path), `appliance` (e.g. `fan`), `test_size`, `batch_size`, `random_state`, `max_samples_stats`
- **device**: `cuda` or `cpu`
- **phase1**: VQ-VAE — `num_epochs`, `lr`, `checkpoint` (save path, typically `{checkpoints.dir}/models/...`)
- **phase2**: PixelSNAIL prior — `num_epochs`, `lr`, `checkpoint`
- **eval**: `vqvae_checkpoint`, `prior_checkpoint` (for evaluation)
- **logging**: `log_dir`, `name`

**Checkpoint layout** (best practice):
```
checkpoints/
├── models/          # VQ-VAE and PixelSNAIL prior .pth files
│   ├── mobilenetv2_8x_vqvae.pth
│   └── pixelsnail_prior.pth
└── stats/           # Precomputed train mean/std per appliance
    └── {appliance}_train_stats.pt
```

The notebook uses `configs/colab.yaml` and overrides `data.root_dir`, `checkpoints.dir`, checkpoint paths, and `logging.log_dir` from its config cell.

### Data format (DCASE 2020 Task 2 dev)

- **root_dir** must contain: `{root_dir}/{appliance}/train/` and (for eval) `{root_dir}/{appliance}/test/`.
- **Train**: only normal sounds; `.wav` files directly under `.../fan/train/`.
- **Test**: files named `normal_*` or `anomaly_*`; label 0 = normal, 1 = anomaly.

Example:

```
/path/to/dcase2020-task2-dev-dataset/
└── fan/
    ├── train/
    │   ├── normal_id_00_00000000.wav
    │   └── ...
    └── test/
        ├── normal_id_00_00000010.wav
        ├── anomaly_id_00_00000020.wav
        └── ...
```

### Training flow

1. **Phase 1 — VQ-VAE**: Train `MobileNetV2_8x_VQVAE` on normal log-mel spectrograms (reconstruction + VQ loss). Saves best model to `phase1.checkpoint`.
2. **Phase 2 — PixelSNAIL**: Freeze VQ-VAE; train PixelSNAIL prior on VQ code indices (NLL). Saves best prior to `phase2.checkpoint`.
3. **Eval** (optional): Load both checkpoints, compute anomaly scores (MSE and NLL) on test set, report ROC AUC.

Entry point: `run(config_path=..., overrides=..., mode="train"|"eval", log_dir=...)` from `src.main`.

### Logging

- **Where:** `main.run()` creates a logger that writes to **both** the notebook (stdout) and a file: `{log_dir}/{name}_{mode}.log` (e.g. `./logs/main_train.log`). On Colab with the clone on Drive, that file is under the repo folder on Drive.
- **When:** The file handler flushes after every log line so lines appear on disk (and in Drive) immediately. If you see no file or no new lines, the process may still be in the **dataset stats** phase (see troubleshooting below).
- **What:** Config and device at start; then “Computing dataset mean/std…” or “Loaded train stats from …”; then “Data loading”; then per-epoch train/val losses (Phase 1 VQ-VAE, then Phase 2 prior).

### Precomputed train stats (optional)

To avoid the slow stats phase when data is on Drive, you can **precompute mean/std locally** and put the file in the checkpoints directory so Colab loads it:

1. **On your laptop** (with the dataset on fast disk), run:
   ```bash
   python scripts/compute_train_stats.py --root_dir /path/to/dcase2020-task2-dev-dataset --appliance fan --checkpoint_dir checkpoints/stats
   ```
   This creates `checkpoints/stats/fan_train_stats.pt`. Optionally use `--max_samples 2000` for a faster run.

2. **Upload** `checkpoints/stats/fan_train_stats.pt` to your repo’s `checkpoints/stats/` folder on Drive (or copy it into the cloned project’s checkpoints dir on Colab).

3. When you run training, the code will **load** stats from that file and skip the computation phase. The log will say “Loaded train stats from …”.

---

## Part 2 — Prerequisites (before you start)

1. **Google account** and **Google Colab** (colab.research.google.com).
2. **GPU runtime** — Training on CPU is very slow. Before running: **Runtime → Change runtime type → Hardware accelerator: GPU** (e.g. T4). Then run your cells; the log should show `Device: cuda`.
3. **DCASE 2020 Task 2 dev dataset**  
   - Download from: [DCASE 2020 Task 2](https://dcase.community/challenge2020/task-unsupervised-detection-of-anomalous-sounds).  
   - Upload the unpacked folder to **Google Drive** (e.g. `My Drive/datasets/dcase2020-task2-dev-dataset`).  
   - You will point the notebook to this path (e.g. `/content/drive/MyDrive/datasets/dcase2020-task2-dev-dataset`).
4. **This repo** will be cloned from GitHub into your Drive in Step 2 below (no need to upload the zip yourself).

---

## Part 3 — Step-by-step (run cells in this order)

Open **Google Colab** (colab.research.google.com), then:

### Step 1 — New notebook and mount Drive

1. **File → New notebook** (or upload `notebooks/colab_train.ipynb` from this repo if you already have it).
2. If you are **not** using the pre-made `colab_train.ipynb`, create cells as below. If you **are** using `colab_train.ipynb`, just run the cells in order.
3. **First code cell:** mount Google Drive so the VM can see your dataset and the cloned repo:

```python
from google.colab import drive
drive.mount("/content/drive")
```

4. When the prompt appears, click the link, allow access, copy the code back into the cell if asked. Run the cell. You should see: `Mounted at /content/drive`.

---

### Step 2 — Clone the project into Drive

Run this in the **next** cell (so the Colab VM has the full project, including `src/`):

```python
import os

REPO_DIR = "/content/drive/MyDrive/semcom_asd_vqar"
REPO_URL = "https://github.com/raidantimosquitos/semcom_asd_vqar.git"

if os.path.isdir(REPO_DIR):
    !cd "{REPO_DIR}" && git pull
else:
    os.makedirs(os.path.dirname(REPO_DIR), exist_ok=True)
    !git clone {REPO_URL} {REPO_DIR}
```

- If the repo is **private**, replace `REPO_URL` with an HTTPS URL that includes a personal access token, or set up SSH on Colab and use the SSH clone URL.
- You can change `REPO_DIR` to another path under `/content/drive/MyDrive/` if you prefer.

---

### Step 3 — Set project root and install dependencies

Run this **exactly once** before any training or eval. It sets the project root and installs PyYAML (Colab usually has torch/sklearn already):

```python
import sys
import os

try:
    PROJECT_ROOT = REPO_DIR   # from the clone cell above
except NameError:
    PROJECT_ROOT = "/content/drive/MyDrive/semcom_asd_vqar"

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

!pip install -q PyYAML
```

- **Important:** If you skip this cell or run the “Run training” cell before this one, you will get `ModuleNotFoundError: No module named 'src'` or `No module named 'src.data'`. Always run Step 1 → 2 → 3 before training.

---

### Step 4 — Set dataset path and checkpoint/log dirs

Set your dataset path and where to save checkpoints and logs. These overrides are merged into `configs/colab.yaml` when you call `run()`.

```python
# Where the DCASE 2020 Task 2 dev dataset is (on Drive after mount)
DATA_ROOT = "/content/drive/MyDrive/datasets/dcase2020-task2-dev-dataset"

# Where to save models and logs (inside the cloned project on Drive)
CHECKPOINT_DIR = "./checkpoints"
MODELS_DIR = f"{CHECKPOINT_DIR}/models"
LOG_DIR = "./logs"

OVERRIDES = {
    "data": {"root_dir": DATA_ROOT},
    "phase1": {
        "num_epochs": 50,
        "checkpoint": f"{MODELS_DIR}/mobilenetv2_8x_vqvae.pth",
    },
    "phase2": {
        "num_epochs": 80,
        "checkpoint": f"{MODELS_DIR}/pixelsnail_prior.pth",
    },
    "eval": {
        "vqvae_checkpoint": f"{MODELS_DIR}/mobilenetv2_8x_vqvae.pth",
        "prior_checkpoint": f"{MODELS_DIR}/pixelsnail_prior.pth",
    },
    "logging": {"log_dir": LOG_DIR},
}

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
```

- **Change `DATA_ROOT`** to the path where you actually put the dataset (e.g. under `/content/drive/MyDrive/...`).
- Optional: change `phase1.num_epochs` / `phase2.num_epochs` in `OVERRIDES` to train shorter or longer.

---

### Step 5 — Run training

Only run this **after** Steps 1–4. This runs Phase 1 (VQ-VAE) then Phase 2 (PixelSNAIL); checkpoints and logs go to the paths you set in Step 4.

```python
from src.main import run

run(
    config_path=os.path.join(PROJECT_ROOT, "configs", "colab.yaml"),
    overrides=OVERRIDES,
    mode="train",
    log_dir=LOG_DIR,
)
```

- Training can take a long time (50 + 80 epochs by default). Colab may disconnect; checkpoints are on Drive if you used the paths above.
- To run **evaluation** later (ROC AUC on test set), use the same `PROJECT_ROOT`, `OVERRIDES`, `LOG_DIR`, and call:

```python
run(
    config_path=os.path.join(PROJECT_ROOT, "configs", "colab.yaml"),
    overrides=OVERRIDES,
    mode="eval",
    log_dir=LOG_DIR,
)
```

---

## Part 4 — Checklist (dumb-proof)

- [ ] Drive mounted (`/content/drive`).
- [ ] Clone cell run; repo is at `REPO_DIR` (e.g. `/content/drive/MyDrive/semcom_asd_vqar`).
- [ ] Project/deps cell run; `PROJECT_ROOT` set, `os.chdir(PROJECT_ROOT)` done, `pip install -q PyYAML` run.
- [ ] Config cell run; `DATA_ROOT` points to your DCASE dataset folder; `CHECKPOINT_DIR` and `LOG_DIR` set.
- [ ] Run training cell once; no need to re-run Steps 1–3 unless you restart the runtime.

---

## Part 5 — Troubleshooting

| Problem | What to do |
|--------|------------|
| `ModuleNotFoundError: No module named 'src'` or `No module named 'src.data'` | Run **Step 3** (project root + `sys.path` + `chdir`) and then run the training cell again. Do not run the training cell before Step 3. |
| `FileNotFoundError` or “No files” for dataset | Check that `DATA_ROOT` is the folder that contains `fan/train/` (and `fan/test/` for eval). After mount, the path must be like `/content/drive/MyDrive/.../dcase2020-task2-dev-dataset`. |
| No log file in `logs/` or no training progress for a long time | (1) Log file is written under the cloned project’s `logs/` (e.g. `.../semcom_asd_vqar/logs/main_train.log`). The handler flushes after each line so it should appear on Drive; refresh the folder. (2) The first phase is **computing dataset mean/std** over all train files (or `max_samples_stats` if set). With data on Drive this can take 30+ min if `max_samples_stats` is null. Use `max_samples_stats: 500` in config (Colab config already does this) so stats finish in a few minutes, then you’ll see “Data loading” and epoch logs. |
| Log shows `Device: cpu` but config has `cuda` | The runtime has no GPU. Use **Runtime → Change runtime type → Hardware accelerator: GPU**, then re-run from Step 1 (or at least the run cell). |
| Colab disconnected / session lost | Re-run Steps 1 → 2 → 3 → 4, then run training again. If checkpoints were saved to Drive (e.g. under the cloned repo), Phase 1/2 will resume from the saved files. |
| Private repo clone fails | Use an HTTPS URL with a personal access token, or configure SSH keys in Colab and use the SSH clone URL. |
| Out of GPU memory | In the config overrides, reduce `data.batch_size` (e.g. from 32 to 16 or 8). |

---

## Quick reference — Cell order in `colab_train.ipynb`

1. **Mount Drive** → `drive.mount("/content/drive")`
2. **Clone repo** → clone into `/content/drive/MyDrive/semcom_asd_vqar`
3. **Project + deps** → `PROJECT_ROOT`, `sys.path`, `chdir`, `pip install -q PyYAML`
4. **Config** → `DATA_ROOT`, `OVERRIDES`, `makedirs` for checkpoints/logs
5. **Run** → `from src.main import run` and `run(..., mode="train", ...)`

Run 1 → 2 → 3 → 4 → 5 in order every time you open the notebook or after a runtime restart.
