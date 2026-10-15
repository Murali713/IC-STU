# IC-STU: Identifiability-Certified Spatio-Temporal Unfolding for EEG Artifact Removal

PyTorch implementation of IC-STU for single-channel EOG artifact removal on the
[EEGdenoiseNet](https://github.com/ncclabsustech/EEGdenoiseNet) benchmark.

The network unfolds a group-sparse proximal gradient solver (7 stages, shared brain and artifact
dictionaries), and at inference returns a per-segment certificate `kappa_hat` with a reliability flag.

## Run in Google Colab

Upload `IC_STU_EEG_Colab.ipynb` to Colab and choose **Runtime > Run all**. The notebook downloads the
data, trains for 100 epochs (about 10-15 min on CPU) and prints train/validation values per epoch.

## Run locally

```bash
pip install -r requirements.txt
git clone --depth 1 https://github.com/ncclabsustech/EEGdenoiseNet.git
mkdir data && cp EEGdenoiseNet/data/*_all_epochs.npy data/
python ic_stu.py --data_dir data --artifact eog --epochs 100 --out_dir runs/eog
```

Outputs: `history.csv` (per-epoch train/validation values), `history.png`, `best.pt`, `results.json`.

## Files

| File | Purpose |
|---|---|
| `ic_stu.py` | Model, loss, metrics, training and evaluation |
| `IC_STU_EEG_Colab.ipynb` | Same code as a Colab notebook |
| `diag_sparsity.py` | Refit sparsity check on a trained checkpoint |
| `results/` | Output of one 100-epoch run (seed 0) |

## Results of the included run (EOG, EEGdenoiseNet test set, seed 0)

| Metric | Value |
|---|---|
| Parameters | 524,311 |
| RRMSEt, stage output | 0.561 |
| RRMSEt, after support refit | 0.797 |
| Mean certificate kappa_hat | 0.492 |
| Segments flagged (kappa_hat < 0.4) | 19.9% |
| Mean RRMSEt, flagged / unflagged | 0.890 / 0.774 |

These numbers do **not** reproduce the values in the manuscript (RRMSEt 0.341). The refit with the
manuscript's sparsity levels (28 brain atoms, 6 artifact atoms) is worse than the unrefitted stage output.
Loss scaling, dictionary initialization (DCT for brain, Haar for artifact) and sparsity levels are
implementation choices that the manuscript does not fully specify and that were not tuned here.

## Notes on implementation choices

- Reconstruction terms of the loss are per-element means; the dictionary terms are divided by K.
- Certificate is computed on the selected supports enlarged by a margin of 2 atoms per block.
- Training uses the stage outputs; support selection, refit and certificate run only at inference.
- Only the single-channel EEGdenoiseNet setting is implemented.

## Data

EEGdenoiseNet: Zhang et al., "EEGdenoiseNet: a benchmark dataset for deep learning solutions of EEG
denoising", Journal of Neural Engineering, 18(5), 056057, 2021. The data files are not included in
this repository.
