# Notebooks

## Colab training (`colab_train.ipynb`)

Use this notebook to run **training on Google Colab** with longer epochs.

1. **Upload to Colab**
   - Zip the project (or clone from Git in a Colab cell) and upload to Colab.
   - Upload the [DCASE2020 Task 2 dev dataset](https://dcase.community/challenge2020/task-unsupervised-detection-of-anomalous-sounds) to Colab or to Google Drive.

2. **Open the notebook**
   - Upload `notebooks/colab_train.ipynb` to Colab (File → Upload notebook), or open it from Drive if you store the project there.

3. **Set paths**
   - In cell 2: set `PROJECT_ROOT` to the project root (e.g. `/content/dcase-2022-vq-vae-ar`).
   - In cell 3: set `DATA_ROOT` to the dataset root; optionally set `CHECKPOINT_DIR` and `LOG_DIR` to a Drive path so checkpoints and logs persist.

4. **Run all**
   - Run the cells in order. Training uses `configs/colab.yaml` and the overrides (e.g. 50 + 80 epochs). You can change `OVERRIDES["phase1"]["num_epochs"]` and `OVERRIDES["phase2"]["num_epochs"]` for different run lengths.

To run **evaluation** in Colab, use the same setup and call:
```python
run(config_path="configs/colab.yaml", overrides=OVERRIDES, mode="eval", log_dir=LOG_DIR)
```
