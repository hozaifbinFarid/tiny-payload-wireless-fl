# Analysis assets

This directory accompanies **Finite payload bias and delivery correction in wireless federated learning**, submitted to *Telecommunication Systems*. It contains the verification, statistical-analysis and figure-generation scripts and the verified numerical tables. The training notebook is in the repository root.

The numerical exports come from the three result archives (Zenodo, DOI 10.5281/zenodo.23184027). All 7,404 archive files were parsed or read, all 2,394 checkpoint checksums matched, and re-evaluation of the 792 saved final models reproduced their predictions, confusion matrices and test metrics. Training was not rerun. These checks establish consistency of the processed snapshots and saved results; they do not independently validate raw PAMAP2 preprocessing.

## Contents and units

| File in `audit/` | Contents |
| --- | --- |
| `verified_final_test.csv` | 792 seed-level final test records, including the main and focused robustness runs |
| `verified_history.csv` | 19,800 recorded validation observations; the initial observation is included |
| `verified_diagnostics.csv` | 135 diagnostic rows: 81 Monte Carlo estimates and 54 analytically assigned oracle zeros |
| `verified_matched_cost.csv` | 1,944 method/budget observations selected from the recorded validation histories |
| `test_paired_recomputed.csv` | 51 endpoint contrasts: Holm adjustment over 45 main tests and a separate family of six robustness tests |
| `matched_paired_recomputed.csv` | 135 paired validation contrasts with Holm adjustment over all 135 |
| `verified_calibration.csv` | 108 archived probe-calibration summaries |
| `verified_per_subject.csv` | Recomputed test accuracy for each held-out subject and saved final model |
| `verified_confusion_matrices.json` | Every recomputed final-model confusion matrix |
| `*_clients.json` | Client sample counts and class composition |
| `run_inventory.csv` | Run identifiers, specifications and checkpoint identities |
| `study_details.json` | Saved configurations, dataset manifests, runtime versions and accounting records |
| `audit_status.json` | File-type counts, checkpoint-check count and verification errors (empty) |
| `input_member_manifest.csv` | Archive/member names, byte counts and SHA-256 checksums for all 7,404 files |

CSV accuracy, balanced-accuracy and paired-difference values are fractions unless the column explicitly indicates another unit. Multiply accuracy by 100 for percent and a paired difference by 100 for percentage points. Confidence limits use the same units as their estimate. Wire costs are bytes; the documents and figures use decimal MB (1 MB = 1,000,000 bytes). Simulated time, GPU reservation time and measured round-computation time are separate quantities. Diagnostic bias is a norm relative to the reference gradient norm. See the supplement for exact estimands and uncertainty definitions.

`HAR_subject` identifies the original subject-client HAR study; `HAR_Dirichlet` identifies the classwise Dirichlet redistribution; `PAMAP2_subject` identifies the subject-client PAMAP2 study. An `iid` channel label describes independently redrawn slow channel states and does not establish IID client data.

## Regenerate the nine figures

The included exports suffice; original result archives are not needed for this step. In a Python environment with the packages listed in `requirements-analysis.txt`, run from this directory:

```sh
python make_figures.py
```

This writes EPS vector files with embedded fonts and 600 dpi PNGs into `../Figures/`. It does not update the images embedded in Word automatically. Figure data, labels, uncertainty and axis units follow the accompanying documents.

| Figure | Content |
| --- | --- |
| Fig1 | Mean-length and marginal-probability diagnostic bias |
| Fig2 | Paired final test accuracy differences from fixed-size correction |
| Fig3 | Validation trajectories against full attempted wire cost |
| Fig4 | Complete communication-cost components at k = 128 |
| Fig5 | Largest periodically logged delivery inverse weights |
| FigS1 | Training-client class proportions |
| FigS2 | All matched simulated-time validation contrasts |
| FigS3 | Held-out subject accuracy at k = 128 |
| FigS4 | PAMAP2 class confusion matrices at k = 128 |

## Repeat the saved-model audit

Download the three result ZIPs from Zenodo (DOI 10.5281/zenodo.23184027); they are not duplicated in this repository. Put `TinyPayloadFL2.zip`, `TinyPayloadFL_noniid.zip` and `TinyPayloadFL_pamap2.zip` in `analysis/inputs/`, or pass their containing directory to the first command:

```sh
python prepare_archives.py inputs
python audit_sources.py
python summarize_calibration.py
python make_figures.py
```

The first command validates ZIP CRCs and every file against the member checksum manifest (`audit/input_member_manifest.csv`), then extracts into `analysis/extracted/<archive-stem>/`. The second reads all extracted files, verifies archived identities and checkpoints, re-evaluates the saved final models on the prepared feature arrays, and rewrites the result exports and paired comparisons in `audit/`. The calibration command exports existing probe records; it does not refit their models. Expect floating-point loss differences at numerical precision across environments. The archived original training environment is recorded separately in `study_details.json`.

The training notebook (`Tiny_Payload_Wireless_Federated_Learning.ipynb`, repository root) runs the studies; these scripts are a verification and figure-reproduction layer and do not rerun training. The HAR Dirichlet and PAMAP2 studies were executed with code fingerprint `ae2bb80e…`; the HAR subjects study was executed with an earlier version of the same notebook (fingerprint `650e5800…`), before the Dirichlet and PAMAP2 extensions were added (see `audit/study_details.json`). The PAMAP2 orientation-channel issue and the window-boundary handling remain limitations; see the manuscript.

All code in this directory was written during manuscript preparation. `requirements-analysis.txt` records the environment used for this analysis, not the original training environment. No new learning experiments or raw-data preprocessing are claimed.
