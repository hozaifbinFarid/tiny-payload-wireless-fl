"""Recreate the calibration export from the saved run records, without refitting."""
from pathlib import Path
import json
import pandas as pd
ROOT = Path(__file__).resolve().parent
labels = {'TinyPayloadFL2':'HAR_subject', 'TinyPayloadFL_noniid':'HAR_Dirichlet', 'TinyPayloadFL_pamap2':'PAMAP2_subject'}
rows = []
for study in sorted((ROOT/'extracted').glob('*/study_*')):
    for path in sorted(study.glob('runs/*/probe_calibration.json')):
        spec = json.loads((path.parent/'run_spec.json').read_text())['spec']
        record = json.loads(path.read_text())
        rows.append({'study':labels[study.parent.name], 'k':spec['k'], 'seed':spec['seed'],
                     'heldout_brier':record['heldout_brier'], 'probe_uplink_bytes':record['uplink_bytes'],
                     'probe_sim_seconds':record['sim_seconds']})
if len(rows) != 108:
    raise RuntimeError(f'Expected 108 calibration records; found {len(rows)}. Prepare all three archives first.')
pd.DataFrame(rows).to_csv(ROOT/'audit'/'verified_calibration.csv', index=False)
print('Exported 108 recorded calibration summaries; no fitting or training was run.')
