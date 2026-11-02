"""Reproduce the frozen forecasts and paper metrics without a simulator.

Run from the release root: python scripts/reproduce_frozen.py
The original predictor's pure predict() function is loaded unchanged from the
frozen source snapshot. No training or physical-output lookup occurs in it.
"""
from pathlib import Path
import ast
import csv
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT/'RELEASE_MANIFEST.json').exists() and (ROOT/'release/pwm-tmsp/RELEASE_MANIFEST.json').exists():
    ROOT = ROOT/'release/pwm-tmsp'


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    manifest = json.loads((ROOT/'RELEASE_MANIFEST.json').read_text())
    for name, item in manifest['files'].items():
        p = ROOT/name
        if not p.is_file() or sha(p) != item['sha256']:
            raise ValueError('Release integrity mismatch: '+name)
    source = ROOT/'models/frozen/source/benchmarks/retimed_prediction_v56.py'
    module = ast.parse(source.read_text(encoding='utf-8'))
    function = next(x for x in module.body if isinstance(x, ast.FunctionDef) and x.name == 'predict')
    scope = {'np': np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), scope)
    card = json.loads((ROOT/'models/frozen/network_cards.json').read_text())
    with np.load(ROOT/'data/inputs/frozen_inputs.npz') as f:
        widths = f['widths_ns']
    committed = json.loads((ROOT/'results/predictions/committed_predictions.json').read_text())
    for layer in ['L0','L1','L2_static_candidate']:
        actual = scope['predict'](widths, card, layer)
        expected = committed[layer]
        if len(actual) != len(expected): raise ValueError('Forecast length differs: '+layer)
        maximum = 0.0
        for a, b in zip(actual, expected):
            if a['node'] != b['node'] or a['sample'] != b['sample']:
                raise ValueError('Forecast association differs')
            maximum = max(maximum, *(abs(a[k]-b[k]) for k in ['rise_ns','fall_ns','width_ns','available_ns']))
        if maximum > 1e-6: raise ValueError('Forecast exceeds one-femtosecond tolerance: '+layer)
        print(f'{layer}: {len(actual)} forecasts; maximum difference {maximum:.9g} ns')
    with (ROOT/'data/measurements/network1000_outputs.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 6000: raise ValueError('Expected 6000 scored word outputs')
    if len({(r['node'],r['sample_id']) for r in rows}) != 6000:
        raise ValueError('Duplicate measurement identifiers')
    metrics = {}
    expected = json.loads((ROOT/'results/summary/network1000_summary.json').read_text())
    for layer in ['L0','L1','L2_static_candidate']:
        width = np.array([float(r[layer+'_width_ns'])-float(r['measured_width_ns']) for r in rows])
        edge = np.array([float(r[layer+'_'+k+'_ns'])-float(r['measured_'+k+'_ns']) for r in rows for k in ['rise','fall']])
        final = np.array([float(r[layer+'_width_ns'])-float(r['measured_width_ns']) for r in rows if r['node']=='s4c2'])
        m=dict(width_RMS_ns=float(np.sqrt(np.mean(width**2))),
               edge_P99_percent_FS=float(np.quantile(np.abs(edge),0.99)/2),
               final_width_RMS_ns=float(np.sqrt(np.mean(final**2))))
        metrics[layer]=m
        for k,v in m.items():
            ref=manifest['expected_metrics'][layer][k]
            if not np.isclose(v,ref,rtol=1e-9,atol=1e-9):
                raise ValueError(f'Metric mismatch: {layer}/{k}: {v} != {ref}')
    for name,limit in [('width_error_ns',2.0),('anchor_error_ns',0.2)]:
        value=max(abs(float(r[name])) for r in rows)
        if value>limit:raise ValueError('Physical error criterion exceeded')
    reduction=100*(1-metrics['L2_static_candidate']['width_RMS_ns']/metrics['L1']['width_RMS_ns'])
    report=dict(status='pass',release_files_verified=len(manifest['files']),
                input_count=len(widths),measured_outputs=len(rows),metrics=metrics,
                L2_vs_L1_RMS_reduction_percent=reduction,
                scope='forecast reproduction and independent arithmetic from released event measurements; no new SPICE run')
    out=ROOT/'verification';out.mkdir(exist_ok=True)
    (out/'reproduction.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__ == '__main__': main()
