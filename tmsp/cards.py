"""Exact-node parameter adapter. Does not fit or interpolate coefficients."""
import csv
import hashlib
from pathlib import Path
from .models import Primitive, DEFAULT

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'design/baseline/data/fits/reference_params.csv'
METRICS = ROOT / 'design/baseline/data/metrics.csv'
# Installed wheels carry identical generated tables, independent of workspace layout.
if not SOURCE.exists():SOURCE=Path(__file__).resolve().parent/'data/reference_params.csv'
if not METRICS.exists():METRICS=Path(__file__).resolve().parent/'data/metrics.csv'


def _matches(row, corner, temp, vdd):
    return row['corner'] == corner and abs(float(row['temp']) - temp) < 1e-9 and abs(float(row['vdd']) - vdd) < 1e-9


def node_card(kind, layer='L1', corner='tt', temp=27., vdd=1.8, *, replay=False):
    temp, vdd = float(temp), float(vdd)
    if layer == 'L0':
        raise ValueError('ideal architectural models require explicit design gains, not fitted cards')
    with SOURCE.open(newline='', encoding='utf-8') as f:
        rows = [r for r in csv.DictReader(f) if r['primitive'] == kind and _matches(r, corner, temp, vdd)]
    if not rows:
        raise ValueError('node not measured')
    # Some derived parameters have a stale L2 label in CSV (DD-02 v1.1
    # explicitly promotes MM tie correction and TR leakage to L1).
    params = {r['param']: float(r['value']) for r in rows if r['value'] and r['param'] in set(DEFAULT) | {'K_v', 'K_t', 'vdd0', 'tau_leak'}}
    if not all(__import__('math').isfinite(v) for v in params.values()):
        raise ValueError('unidentified parameter in node')
    if kind == 'DLY':
        params['delay_cycles'] = 0  # characterised eight-inverter chain; not a z^-1 memory
    if kind == 'MM':
        params['mode'] = 'MAX'  # only mm_max was characterised
    adjustments = {}
    if kind == 'TR':
        params['trigger_mode'] = 'clk'
        # The static fit already contains the native testbench hold droop.
        # Applying the separately measured leakage again double-counts it.
        # Dynamic hold association needs a separate reference-hold extraction.
        adjustments['lambda_leak_identified_not_enabled'] = params.pop('lambda_leak', None)
    if kind == 'TVC':
        # v0 was fitted to v_hold1, measured 20 ns after pulse end.
        adjustments['v0_at_20ns'] = params['v0']
        params['v0'] -= params.get('droop_rate', 0.) * 20.
        adjustments['v0_referenced_to_hold0'] = params['v0']
    ranges = {'TR': (20, 160), 'DLY': (20, 160), 'ADD': (20, 80),
              'MM': (20, 120), 'TVC': (20, 160), 'INT': (20, 120)}
    lo, hi = ranges.get(kind, (1e-6, 200))
    if kind == 'DLY':
        with METRICS.open(newline='', encoding='utf-8') as f:
            observed = [float(r['width_in_ns']) for r in csv.DictReader(f)
                        if r['kind'] == 'dly' and _matches(r, corner, temp, vdd) and r['status'] == 'VALID']
        # SPICE PULSE width excludes ramps; threshold-crossing width includes
        # half of each ramp. The API envelope must use measured event widths.
        lo, hi = min(observed), max(observed)
    voltage_range = (-10., 10.)
    if kind in {'TR', 'VTC'}:
        fit = next(r for r in rows if r['model'] == 'L1' and r['param'] in {'alpha0', 'K_v'})
        xmin, xmax = float(fit['x_min']), float(fit['x_max'])
        if kind == 'TR':
            lo, hi = xmin, min(xmax, 160 if replay else (100 if layer == 'L1' else 140))
        else:
            voltage_range = (xmin, min(xmax, .8 if replay or layer == 'L2' else .6))
    exact_pairs = None
    if kind in {'ADD', 'MM'}:
        with METRICS.open(newline='', encoding='utf-8') as f:
            points = [r for r in csv.DictReader(f) if r['kind'] == kind.lower() and _matches(r, corner, temp, vdd) and r['status'] == 'VALID']
        exact_pairs = sorted(set((round(float(r['in_wa']) * 1e6), round(float(r['in_wb']) * 1e6)) for r in points))
    card = dict(kind=kind, layer=layer, params=params,
                model_id=f'{kind}.sky130.{corner}.{temp:g}.{vdd:g}.{layer}',
                input_width=(round(lo * 1e6), round(hi * 1e6)),
                output_width=(1, 200_000_000), voltage_range=voltage_range,
                anchor=(temp, vdd), environment=((temp, temp), (vdd, vdd)),
                stochastic=False, exact_pairs=exact_pairs,
                enable_secondary=layer == 'L2',
                active_mechanisms=['static_cubic'] if layer == 'L2' else [])
    # Missing sigma_ind is metadata, never silently assigned a physical zero.
    provenance = dict(source=str(SOURCE.relative_to(ROOT)),
                      source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                      sigma_ind=None, sigma_ind_status='unidentified',
                      execution_mode='deterministic', parameter_scope='exact_node',
                      replay_full_fit_envelope=replay,
                      evidence='transistor schematic; ideal passives',
                      timing_validated=False, dynamics_validated=False,
                      stochastic_validated=False)
    provenance['reference_adjustments'] = adjustments
    return card, provenance


def instantiate_node(*args, **kwargs):
    card, _ = node_card(*args, **kwargs)
    return Primitive(**card)
