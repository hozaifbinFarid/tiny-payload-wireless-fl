"""Validate and extract the three original result ZIPs for independent re-evaluation."""
from pathlib import Path, PurePosixPath
import argparse, csv, hashlib, zipfile
ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('input_directory', nargs='?', type=Path, default=ROOT/'inputs')
args = parser.parse_args()
expected = {}
with (ROOT/'audit'/'input_member_manifest.csv').open(newline='') as f:
    for row in csv.DictReader(f): expected[(row['archive'],row['member'])] = row
names = ['TinyPayloadFL2.zip','TinyPayloadFL_noniid.zip','TinyPayloadFL_pamap2.zip']
checked = 0
for name in names:
    source = args.input_directory/name
    destination = ROOT/'extracted'/source.stem
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source) as archive:
        bad = archive.testzip()
        if bad is not None: raise RuntimeError(f'ZIP CRC failure: {name}: {bad}')
        files = [entry for entry in archive.infolist() if not entry.is_dir()]
        actual = {(name,entry.filename) for entry in files}
        wanted = {key for key in expected if key[0] == name}
        if actual != wanted: raise RuntimeError(f'Archive member list differs: {name}')
        for entry in files:
            relative = PurePosixPath(entry.filename)
            if relative.is_absolute() or '..' in relative.parts:
                raise RuntimeError(f'Unsafe path in {name}: {entry.filename}')
            data = archive.read(entry)
            row = expected[(name,entry.filename)]
            if len(data) != int(row['bytes']) or hashlib.sha256(data).hexdigest() != row['sha256']:
                raise RuntimeError(f'Checksum mismatch: {name}: {entry.filename}')
            target = destination.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            checked += 1
    print(f'{name}: validated and extracted {len(files)} files')
print(f'Validated {checked} members against the original audit manifest.')
