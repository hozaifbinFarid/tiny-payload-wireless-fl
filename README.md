# Tiny-Payload Wireless Federated Learning

Experiment code for the paper **"Finite payload bias and delivery correction in wireless federated learning"** (Hozaif Bin Farid, 2026; submitted to *Telecommunication Systems*).
Code archive (all versions): [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23057925.svg)](https://doi.org/10.5281/zenodo.23057925)  
Code v1.0.1: <https://doi.org/10.5281/zenodo.23065574>  
Result archives and checkpoints: <https://doi.org/10.5281/zenodo.23184027>

## What this is

Random sparsification reduces the communication cost of federated learning, but it also makes the length of each client's update depend on the sampled gradient. Under a fixed transmission deadline, longer updates are less likely to arrive, so successful delivery becomes informative about the update itself and simply dropping late updates biases the aggregate. This repository contains a single, resumable Google Colab notebook that simulates orthogonal wireless uplinks with such random-length payloads and compares six aggregation rules: uncorrected dropped updates, an oracle channel correction at the mean count, a past-delivery EMA heuristic, an oracle length-conditioned inverse-probability-weighted (IPW) correction, a probe-estimated IPW correction, and a fixed-size importance-sampling baseline. It runs three studies (UCI HAR with subject clients, UCI HAR with Dirichlet clients, and PAMAP2 with subject clients) with 12 seeds each. The notebook also produces the bias diagnostics, learning curves, matched-cost comparisons, held-out-subject test metrics, paired comparisons, and figures reported in the paper. It makes no claim about a deployed 6G system, differential privacy, or a faithful reproduction of JCDO or CA-Fed.

## Repository contents

| File | Purpose |
|---|---|
| `Tiny_Payload_Wireless_Federated_Learning.ipynb` | The complete experiment: data preparation, simulation, checkpointing, analysis, figures |
| `CITATION.cff` | Citation metadata |
| `LICENSE` | Code license |

## How to run a study

1. Open the notebook in Google Colab (**File → Upload notebook**, or open it from this repository).
2. Choose **Runtime → Change runtime type → T4 GPU**.
3. In the configuration cell, set `STUDY_PRESET` to one of the values below. Either edit the default in this line, or set the environment variable in an earlier cell (`os.environ['TINYFL_STUDY'] = 'har_subjects'`):

   ```python
   STUDY_PRESET = os.environ.get('TINYFL_STUDY', 'pamap2_subjects')
   ```

4. Choose **Runtime → Run all** and authorize the Google Drive mount when prompted.
5. Run one study at a time. Each study has its own output folder and its own frozen run plan.

| `STUDY_PRESET` | Dataset | Client allocation | Clients per round | Validation subjects |
|---|---|---|---|---|
| `har_subjects` | UCI HAR | One training subject per client | 10 | 3 |
| `har_dirichlet` | UCI HAR | Classwise Dirichlet split, α = 0.5 | 10 | 3 |
| `pamap2_subjects` | PAMAP2 | One training subject per client | 5 | 1 |

Every study uses 12 seeds, expected payload counts k ∈ {32, 128, 512}, and 600 rounds. Each study plans **264 fits**: 216 main runs (12 seeds × 3 payload sizes × 6 methods), 24 MLP robustness runs, and 24 correlated-channel (Markov) robustness runs.

**Smoke test.** To rehearse the whole pipeline in a few minutes, set `os.environ['TINYFL_PROFILE'] = 'smoke'` before the configuration cell. Smoke results are not paper results.

**Interruptions.** Runs are checkpointed. If Colab disconnects, reconnect and choose Run all again: completed runs are skipped and partial runs resume from their last checkpoint.

**Outputs** are written to Google Drive under `MyDrive/TinyPayloadFL_<STUDY_PRESET>/` (for `pamap2_subjects` the folder is `MyDrive/TinyPayloadFL_pamap2/`). Each study creates a `study_<id>/` folder containing the frozen run plan, data and split manifests, run checkpoints, tables, figures (PNG and PDF), and `REPORT.md`. A results ZIP (`tiny_payload_fl_results_<id>.zip`) is written at the top of the output folder.

## Environment used for the reported results

| Item | Value |
|---|---|
| Platform | Google Colab |
| GPU | NVIDIA Tesla T4 |
| Python | 3.13.15 |
| PyTorch | 2.11.0+cu128 |
| NumPy | 2.1.3 |
| SciPy | 1.16.3 |

The notebook installs missing NumPy/SciPy/pandas/matplotlib automatically and uses the PyTorch build that Colab provides. It also runs on CPU with a NumPy backend, but this is much slower. Deterministic algorithms are enabled and TF32 is disabled. Bitwise reproducibility across different GPU models or library versions is not guaranteed; each checkpoint stores hashes of the code and prepared data so that every run can be traced to the version that produced it.

## Datasets

Both datasets are downloaded automatically by the notebook from the UCI Machine Learning Repository. Please consult the dataset pages for their terms of use and cite the original papers.

- **UCI HAR** (Human Activity Recognition Using Smartphones): <https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones>. Anguita, Ghio, Oneto, Parra, Reyes-Ortiz, "A Public Domain Dataset for Human Activity Recognition Using Smartphones," ESANN 2013.
- **PAMAP2** (Physical Activity Monitoring): <https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring>. Reiss and Stricker, "Introducing a New Benchmarked Dataset for Activity Monitoring," ISWC 2012.

Subject splits are subject-holdout. For PAMAP2 there is no official train/test division; this work uses subjects 1-5 and 7 for training, subject 6 for validation, and subjects 8 and 9 for testing. The validation subject(s) in each study are chosen by a seeded rule (`split_seed = 20250919`) and are recorded in the split manifest.

## Approximate run time

| Study | GPU time reserved by the notebook's ledger (Colab T4) |
|---|---|
| `pamap2_subjects` | 0.4 GPU-hours |
| `har_subjects` | 0.5 GPU-hours |
| `har_dirichlet` | 1 GPU-hours |

Values are `reserved_seconds / 3600` from each study's `gpu_budget.json`. The ledger reserves time in conservative blocks, so these figures slightly overstate actual compute. For PAMAP2, the runtime pilot projected about 0.36 GPU-hours of round compute for the 264 fits, before data preparation, diagnostics and checkpoint I/O. The notebook stops scheduling new computation at 9 allocated GPU-hours. Colab's own usage counter remains the authority on your allowance; disconnect the runtime when the notebook finishes.

## Known limitations

- The PAMAP2 features include the IMU orientation channels, which the dataset documentation marks as invalid.
- PAMAP2 windows are formed after rows from non-target activities are removed, so a window can span a removed segment; each window takes its label from its center sample.
- Simulated channels are orthogonal uplinks with block fading; setup, control and downlink are assumed reliable. Wire bytes are logical attempted datagrams, not PHY symbols or energy.
- The 12 seeds capture training randomness only, not variation across subjects or datasets.

## Citation

If you use this code, please cite the paper and the archived code release:

```
Farid, H. B. Finite payload bias and delivery correction in wireless
federated learning. Submitted to Telecommunication Systems, 2026.

Farid, H. B. Finite payload bias and delivery correction in wireless
federated learning: experiment code (v1.0.1). Zenodo, 2026.
https://doi.org/10.5281/zenodo.23065574

Farid, H. B. Finite payload bias and delivery correction in wireless
federated learning: result archives and checkpoints. Zenodo, 2026.
https://doi.org/10.5281/zenodo.23184027
```

See also `CITATION.cff`.

## License

Code released under the MIT license (see `LICENSE`). The datasets are subject to their own terms.
