# Notebooks

## Colab training (`colab_train.ipynb`)

Use this notebook to run **training on Google Colab** with longer epochs.

**→ Step-by-step (dumb-proof) guide:** [**COLAB_TRAINING_GUIDE.md**](COLAB_TRAINING_GUIDE.md) — project index, prerequisites, exact cell order, checklist, troubleshooting.

**Short version:** Run cells **1 → 2 → 3 → 4 → 5** in order:

1. **Mount Drive** — `drive.mount("/content/drive")`
2. **Clone repo** — clone into `/content/drive/MyDrive/semcom_asd_vqar`
3. **Project & deps** — set `PROJECT_ROOT`, `sys.path`, `chdir`, `pip install -q PyYAML`
4. **Config** — set `DATA_ROOT` to your DCASE dataset path; checkpoints/logs under the cloned project
5. **Run** — `run(..., mode="train")`

To run **evaluation** after training: same setup, then `run(..., mode="eval", log_dir=LOG_DIR)`.
