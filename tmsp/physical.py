"""Explicit adapter for the baseline's triggered capture/replay circuits.

No regression is performed. Programmed PULSE widths are analytically
translated to threshold-to-threshold widths; trigger coordinates are shifted
by their half-rise time. ADD/MM have an explicit external replay clock here,
unlike their default DD-02 placement. No default specification is changed.
"""
import copy
import math
import numpy as np
from .cards import node_card
from .core import Clock, Pulse, Result, Status, is_time, quantize
from .models import Primitive


def triggered_card(kind, layer='L1', corner='tt', temp=27., vdd=1.8):
    if kind not in {'TR', 'ADD', 'MM'}:
        raise ValueError('triggered capture adapter supports TR/ADD/MM only')
    cfg, prov = node_card(kind, layer, corner, temp, vdd, replay=True)
    p = cfg['params']
    # Baseline inputs have rise=fall=.2 ns. PULSE's programmed high duration
    # excludes ramps; measured width includes half the rise and fall.
    width_shift_fs = 200_000.
    raw_latency = p['latency']
    p['latency'] = raw_latency - .1
    if p['latency'] < 0:
        raise ValueError('trigger crossing reference incompatible with latency')
    cfg['input_width'] = tuple(round(w+width_shift_fs) for w in cfg['input_width'])
    if cfg['exact_pairs']:
        cfg['exact_pairs'] = [tuple(round(w+width_shift_fs) for w in pair) for pair in cfg['exact_pairs']]
    transforms = dict(programmed_to_measured_width_shift_fs=width_shift_fs,
                      programmed_to_trigger_crossing_ns=.1,
                      latency_programmed_ns=raw_latency, latency_crossing_ns=p['latency'])
    if kind == 'TR':
        alpha, beta = p['alpha0'], p['beta0']
        leakage = prov['reference_adjustments']['lambda_leak_identified_not_enabled']
        if leakage is None or not math.isfinite(leakage):
            raise ValueError('TR hold sensitivity unidentified')
        # run_one('tr') sets rise=220-programmed_width: falling threshold
        # crossing is 220.3 ns for every width. Trigger crossing is 300.1 ns.
        reference_c = 79.8 + p['latency']
        p['alpha0'] = alpha
        p['beta0'] = beta - alpha*width_shift_fs - leakage*reference_c
        if layer == 'L2':
            base = np.poly1d([p['q_w3']/(200e6)**3, p['q_w2']/(200e6)**2,
                              alpha+p['alpha_delta'], beta+p['beta_delta']])
            rebased = base(np.poly1d([1., -width_shift_fs]))
            rebased += np.poly1d([-leakage*reference_c])
            p['alpha_delta'] = float(rebased[1]-p['alpha0'])
            p['beta_delta'] = float(rebased[0]-p['beta0'])
            p['q_w2'] = float(rebased[2]*(200e6)**2)
            p['q_w3'] = float(rebased[3]*(200e6)**3)
        p['lambda_leak'] = leakage
        transforms.update(reference_hold_formula='H_ns=79.8+latency_crossing_ns (end-aligned baseline)',
                          hold_sensitivity_fs_per_ns=leakage)
    elif kind == 'ADD':
        p['bias0'] -= (p['g_a']+p['g_b'])*width_shift_fs
    else:
        p['output_offset'] -= p['output_gain']*width_shift_fs
    cfg['model_id'] += '.triggered_capture.v1'
    prov.update(reference_transforms=transforms,
                placement='external_triggered_capture', fit_performed=False,
                timing_validated=False, hold_reference_association='prediction_under_test')
    return cfg, prov


class TriggeredPrimitive:
    """Preserve numerical transform/state, explicitly re-time ADD/MM outputs."""
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        self.model = Primitive(**config)
        self.reset()

    def reset(self, t0=0):
        self.model.reset(t0)
        self.last_output = -1

    def process(self, inputs, *, clock, ctrl=None, env=None):
        if not isinstance(clock, Clock) or not is_time(clock.t):
            return self.model._fail(Status.UNDEF, None, 'invalid replay clock')
        raw = [x.value if isinstance(x, Result) else x for x in inputs.values()]
        if not raw or any(not isinstance(x, Pulse) for x in raw):
            return self.model._fail(Status.MISS, clock.sample, 'complete input pulses required')
        if any(x.sample != clock.sample for x in raw):
            return self.model._fail(Status.MISS, clock.sample, 'trigger sample label mismatch')
        if any(x.fall.t > clock.t for x in raw):
            return self.model._fail(Status.DOMAIN, clock.sample, 'replay trigger precedes input completion')
        result = self.model.process(inputs, clock=clock, ctrl=ctrl, env=env)
        if result.value is None or self.model.kind == 'TR':
            return result
        rise = clock.t + quantize(self.model.p['latency']*1e6)
        fall = rise + result.value.width_fs
        if not is_time(fall):
            return self.model._fail(Status.DOMAIN, clock.sample, 'output timestamp overflow')
        if rise < self.last_output:
            return self.model._fail(Status.UNDEF, clock.sample, 'triggered output overlaps prior sample')
        self.last_output = fall
        pulse = Pulse.at(rise, result.value.width_fs, clock.sample, self.model.model_id,
                         result.value.rise.slew)
        obs = {**result.obs, 't_valid': fall, 'placement': 'external_triggered_capture'}
        return Result(result.status, pulse, result.flags, obs, result.diagnostics)
