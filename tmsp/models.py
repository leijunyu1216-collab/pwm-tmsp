"""Complete-sample primitive evaluator, not a streaming circuit scheduler.

DD-02 equations use fs for widths, ns for latency parameters and V/s on
events. Only the evidenced static L2 mechanism of TR/VTC is enabled here.
"""
import copy
import hashlib
import json
import math
from .core import (Clock, Edge, Pulse, Voltage, Result, Status, NUMERIC,
                   FS, is_time, primary, quantize)

CLASSES = {'TR', 'DLY', 'ADD', 'MM', 'VTC', 'TVC', 'INT'}
CTRL = {'TR': {'gain_code', 'offset_code'}, 'ADD': {'gain_code_a', 'gain_code_b', 'offset_code'},
        'VTC': {'gain_code', 'offset_code'}, 'TVC': {'gain_code', 'offset_code'},
        'INT': {'gain_code', 'leak_code', 'offset_code'}, 'DLY': set(), 'MM': set()}
DEFAULT = dict(alpha0=1., alpha_trim=0., beta0=0., beta_q=0., latency=0.,
               gain_lsb=0., offset_lsb=0., sigma_j_common=0., sigma_j_diff=0.,
               sigma_n=0., sigma_v=0., sigma_ap=0., lambda_leak=0.,
               slew_out=None, delay_cycles=1, delay_offset=0., skew=0.,
               skew_a=0., skew_b=0., g_a=1., g_b=1., bias0=0.,
               mode='MIN', tie_policy='A', decision_offset=0., output_gain=1.,
               output_offset=0., tie_width_error=0., trigger_mode='input_fall',
               w0=0., v0=0., beta_q_v=0., offset_lsb_v=0., aperture=0.,
               droop_rate=0., a0=.99, b0=1., leak_lsb=0., charge_q=0.,
               alpha_delta=0., beta_delta=0., q_w2=0., q_w3=0.,
               gain_delta=0., offset_delta=0., kappa_v=0., kappa_v3=0.,
               supply_sens=0., temp_sens=0., supply_sens_v=0., temp_sens_v=0.)


class Primitive:
    def __init__(self, kind, layer='L0', params=None, *, model_id=None, seed=20261022,
                 t_ref_fs=FS, full_scale_fs=None, input_width=(1, FS),
                 output_width=(1, FS), voltage_range=(-10., 10.),
                 warmup_cycles=0, clip_policy='error', d_min=0, d_max=None,
                 enable_secondary=True, active_mechanisms=None, stochastic=False,
                 anchor=(27., 1.8), environment=None, exact_pairs=None):
        if kind not in CLASSES or layer not in {'L0', 'L1', 'L2'}:
            raise ValueError('unknown primitive/layer')
        self.kind, self.layer = kind, layer
        self.p = {**DEFAULT, **(params or {})}
        unknown = set(params or {}) - set(DEFAULT) - {'K_v', 'K_t', 'vdd0', 'tau_leak'}
        if unknown:
            raise ValueError(f'unsupported parameters: {sorted(unknown)}')
        self.model_id = model_id or f'{kind}.reference'
        self.seed, self.t_ref = seed, t_ref_fs
        self.fs = t_ref_fs if full_scale_fs is None else full_scale_fs
        self.input_width, self.output_width = tuple(input_width), tuple(output_width)
        self.voltage_range = tuple(voltage_range)
        self.warmup_cycles, self.clip_policy = warmup_cycles, clip_policy
        self.d_min, self.d_max = d_min, d_max
        self.secondary = layer == 'L2' and enable_secondary
        self.mechanisms = tuple(active_mechanisms or (['static_cubic'] if self.secondary else []))
        self.stochastic, self.anchor = stochastic, tuple(anchor)
        self.environment = environment
        self.exact_pairs = exact_pairs
        if not is_time(t_ref_fs) or t_ref_fs == 0 or self.fs != self.t_ref:
            raise ValueError('reference period and FS must agree')
        if type(warmup_cycles) is not int or warmup_cycles < 0:
            raise ValueError('invalid warmup')
        if clip_policy not in {'error', 'sat'} or not is_time(d_min):
            raise ValueError('invalid clipping/d_min')
        if d_max is not None and (not is_time(d_max) or d_max < d_min):
            raise ValueError('invalid d_max')
        for lo, hi in (self.input_width, self.output_width):
            if not is_time(lo) or not is_time(hi) or not 0 < lo <= hi:
                raise ValueError('invalid width envelope')
        if not all(math.isfinite(v) for v in self.voltage_range) or self.voltage_range[0] >= self.voltage_range[1]:
            raise ValueError('invalid voltage range')
        if self.secondary and (kind not in {'TR', 'VTC'} or self.mechanisms != ('static_cubic',)):
            raise ValueError('only evidenced TR/VTC static cubic L2 is implemented')
        if len(self.mechanisms) > 2 or layer != 'L2' and self.mechanisms:
            raise ValueError('invalid mechanism declaration')
        for key, value in self.p.items():
            if value is not None and isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError(f'nonfinite parameter {key}')
            if key.startswith('sigma_') and value < 0:
                raise ValueError('negative noise strength')
        if self.p['latency'] < 0 or self.p['delay_offset'] < 0:
            raise ValueError('negative delay')
        if type(self.p['delay_cycles']) is not int or self.p['delay_cycles'] < 0:
            raise ValueError('invalid cycle delay')
        if self.p['slew_out'] is not None and self.p['slew_out'] <= 0:
            raise ValueError('invalid output slew')
        if self.p['mode'] not in {'MIN', 'MAX'} or self.p['tie_policy'] not in {'A', 'B', 'META'}:
            raise ValueError('invalid selection policy')
        if self.p['trigger_mode'] not in {'input_fall', 'clk'}:
            raise ValueError('invalid trigger mode')
        if kind == 'VTC' and 'K_v' not in self.p or kind == 'TVC' and 'K_t' not in self.p:
            raise ValueError('conversion gain must be supplied')
        if not 0 <= self.p['a0'] < 1:
            raise ValueError('a0 outside stable range')
        if 'tau_leak' in self.p:
            if self.p['tau_leak'] <= 0 or not math.isclose(self.p['a0'], math.exp(-self.t_ref / 1e6 / self.p['tau_leak']), rel_tol=1e-8):
                raise ValueError('tau_leak inconsistent with a0')
        self.checksum = hashlib.sha256(json.dumps(self._config(), sort_keys=True).encode()).hexdigest()
        self.reset()

    def _config(self):
        return dict(kind=self.kind, layer=self.layer, params=self.p, seed=self.seed,
                    model_id=self.model_id, t_ref=self.t_ref, fs=self.fs,
                    input_width=self.input_width, output_width=self.output_width,
                    voltage_range=self.voltage_range, warmup=self.warmup_cycles,
                    clip=self.clip_policy, d_min=self.d_min, d_max=self.d_max,
                    secondary=self.secondary, mechanisms=self.mechanisms,
                    stochastic=self.stochastic, anchor=self.anchor,
                    environment=self.environment, exact_pairs=self.exact_pairs)

    def reset(self, t0=0, initial_state=0.):
        if not is_time(t0) or not math.isfinite(initial_state):
            raise ValueError('invalid reset')
        self.state = dict(y=float(initial_state), count=0, last_sample=-1,
                          input_ends={}, last_output=-1, last_clock=None,
                          t0=t0, ctrl={})

    def snapshot(self):
        return dict(checksum=self.checksum, state=copy.deepcopy(self.state))

    def restore(self, snapshot):
        if snapshot['checksum'] != self.checksum:
            raise ValueError('snapshot belongs to another configuration')
        self.state = copy.deepcopy(snapshot['state'])

    def _normal(self, sample, error):
        # Independent keyed streams; adding noise terms cannot shift prior draws.
        key = f'{self.seed}|{self.model_id}|{sample}|{error}'.encode()
        digest = hashlib.sha256(key).digest()
        u = (int.from_bytes(digest[:8], 'big') + .5) / 2**64
        v = (int.from_bytes(digest[8:16], 'big') + .5) / 2**64
        u = min(max(u, 2**-53), 1 - 2**-53)
        return math.sqrt(-2 * math.log(u)) * math.cos(2 * math.pi * v)

    def _noise(self, key, sample):
        if self.layer == 'L0' or not self.stochastic or self.p[key] == 0:
            return 0.
        return self.p[key] * self._normal(sample, key)

    def _fail(self, status, sample, detail, flags=()):
        diagnostic = dict(status=status.value, flags=list(flags), port=self.kind,
                          sample=sample, edge=None, t=None, error_id='runtime',
                          detail=detail, repro={'seed': self.seed, 'model_id': self.model_id})
        return Result(status, flags=tuple(flags), diagnostics=(diagnostic,))

    def process(self, inputs, *, clock=None, ctrl=None, env=None):
        """Evaluate one complete sample; `inputs` uses DD-02 port names.

        Stateful calls require strictly increasing labels. Result inputs carry
        status/flags between models. No missing port is filled with a value.
        """
        names = {'TR': ['tin'], 'DLY': ['tin'], 'ADD': ['a', 'b'],
                 'MM': ['a', 'b'], 'VTC': ['vin'], 'TVC': ['tin'], 'INT': ['din']}[self.kind]
        vals, flags = {}, []
        failures = []
        for name in names:
            obj = inputs.get(name)
            if obj is None:
                failures.append(Status.MISS)
                continue
            if isinstance(obj, Result):
                flags += [obj.status, *obj.flags]
                if obj.status not in NUMERIC:
                    failures.append(obj.status)
                    continue
                obj = obj.value
            vals[name] = obj
        if failures:
            return self._fail(primary([*failures, *flags]), None, 'missing/invalid upstream input', flags)
        if set(inputs) - set(names):
            return self._fail(Status.EXTRA, None, 'unexpected port')
        samples = []
        for name, obj in vals.items():
            if isinstance(obj, Pulse) and name != 'vin':
                status = obj.validate()
                samples.append(obj.sample)
                if status != Status.VALID:
                    return self._fail(status, obj.sample, 'invalid pulse')
                if not self.input_width[0] <= obj.width_fs <= self.input_width[1]:
                    return self._fail(Status.DOMAIN, obj.sample, 'input width outside envelope')
                if obj.rise.t < self.state['input_ends'].get(name, self.state['t0']):
                    return self._fail(Status.UNDEF, obj.sample, 'input pulse overlap or before reset')
            elif isinstance(obj, Voltage) and name == 'vin':
                samples.append(obj.sample)
                if not is_time(obj.t) or type(obj.sample) is not int or obj.sample < 0 or not math.isfinite(obj.v):
                    return self._fail(Status.UNDEF, obj.sample, 'invalid voltage sample')
                if obj.status not in NUMERIC:
                    return self._fail(obj.status, obj.sample, 'invalid voltage status')
                flags.append(obj.status)
                if not self.voltage_range[0] <= obj.v <= self.voltage_range[1]:
                    return self._fail(Status.DOMAIN, obj.sample, 'input voltage outside envelope')
                if obj.t < self.state['input_ends'].get(name, self.state['t0']):
                    return self._fail(Status.UNDEF, obj.sample, 'voltage time moves backward')
            else:
                return self._fail(Status.UNDEF, None, 'incorrect port domain')
        sample = samples[0]
        if any(k != sample for k in samples):
            return self._fail(Status.MISS, sample, 'input sample labels do not match')
        if sample <= self.state['last_sample']:
            return self._fail(Status.UNDEF, sample, 'sample label repeated or backward')
        need_clock = self.kind in {'VTC', 'TVC', 'INT'} or self.kind == 'TR' and self.p['trigger_mode'] == 'clk'
        if need_clock and clock is None:
            return self._fail(Status.MISS, sample, 'required trigger/sample clock absent')
        if clock is not None:
            if not isinstance(clock, Clock) or not is_time(clock.t) or type(clock.sample) is not int:
                return self._fail(Status.UNDEF, sample, 'invalid clock')
            if clock.sample != sample:
                return self._fail(Status.MISS, sample, 'clock label mismatch')
            if clock.t < self.state['t0']:
                return self._fail(Status.UNDEF, sample, 'clock before reset')
        env = env or self.anchor
        if len(env) != 2 or not all(math.isfinite(v) for v in env):
            return self._fail(Status.DOMAIN, sample, 'invalid temperature/supply')
        if self.environment and not all(lo <= v <= hi for v, (lo, hi) in zip(env, self.environment)):
            return self._fail(Status.DOMAIN, sample, 'environment outside node envelope')
        if self.exact_pairs is not None:
            pair = tuple(vals[n].width_fs for n in names)
            if pair not in self.exact_pairs:
                return self._fail(Status.DOMAIN, sample, 'uncharacterised two-input pair')
        controls = dict(self.state['ctrl'])
        obs_status = Status.VALID
        for key, code in (ctrl or {}).items():
            if key not in CTRL[self.kind] or type(code) is not int:
                raise ValueError('invalid control field or type')
            controls[key] = min(127, max(-128, code))
            if controls[key] != code:
                obs_status = Status.SAT
        # Commit labels once a structurally valid transaction is accepted.
        self.state['last_sample'] = sample
        self.state['count'] += 1
        self.state['ctrl'] = controls
        for name, obj in vals.items():
            self.state['input_ends'][name] = obj.fall.t if isinstance(obj, Pulse) else obj.t
        if self.state['count'] <= self.warmup_cycles:
            return self._fail(Status.RESET, sample, 'warmup')
        try:
            return self._evaluate(vals, sample, clock, controls, env, flags, obs_status)
        except (ValueError, OverflowError) as exc:
            return self._fail(Status.DOMAIN, sample, str(exc), flags)

    def _evaluate(self, v, k, clock, c, env, flags, obs_status):
        p, linear = self.p, self.layer != 'L0'
        get = lambda name: p[name] if linear else 0.
        code = lambda name: c.get(name, 0) if linear else 0
        nc = self._noise('sigma_j_common', k)
        nw = self._noise('sigma_j_diff', k)
        dt, dv = env[0] - self.anchor[0], env[1] - self.anchor[1]
        drift = get('supply_sens') * dv + get('temp_sens') * dt
        delay = p['latency'] * 1e6
        slew = None if p['slew_out'] is None else p['slew_out'] * 1e9
        selected, new_y = None, None
        if self.kind == 'TR':
            x = v['tin']
            ready = max(x.fall.t, clock.t if clock else x.fall.t)
            hold = (ready + delay - x.fall.t) / 1e6
            alpha = p['alpha0'] * (1 + p['alpha_trim'] + p['gain_lsb'] * code('gain_code')) if linear else 1.
            width = alpha * x.width_fs + get('beta0') + get('beta_q') + get('offset_lsb') * code('offset_code')
            width += get('lambda_leak') * hold + self._noise('sigma_n', k) + drift + nw
            if self.secondary:
                u = x.width_fs / self.t_ref
                width += p['alpha_delta'] * x.width_fs + p['beta_delta'] + p['q_w2'] * u**2 + p['q_w3'] * u**3
            rise = (ready, delay + nc)
        elif self.kind == 'DLY':
            x = v['tin']
            ready = x.rise.t
            base_delay = p['delay_cycles'] * self.t_ref + p['delay_offset'] * 1e6
            width = (p['alpha0'] * (1 + p['alpha_trim']) if linear else 1) * x.width_fs
            width += get('beta0') + get('beta_q') + get('lambda_leak') * base_delay / 1e6 + drift + nw
            rise = (ready, base_delay + get('skew') * 1e6 + nc)
            slew = x.rise.slew
        elif self.kind == 'ADD':
            a, b = v['a'], v['b']
            ready = max(a.fall.t, b.fall.t)
            width = p['g_a'] * (1 + get('gain_lsb') * code('gain_code_a')) * a.width_fs
            width += p['g_b'] * (1 + get('gain_lsb') * code('gain_code_b')) * b.width_fs + p['bias0']
            width += get('beta0') + get('beta_q') + get('offset_lsb') * code('offset_code') + drift + nw
            rise = (ready, max(a.fall.t - ready + get('skew_a') * 1e6,
                               b.fall.t - ready + get('skew_b') * 1e6) + delay + nc)
        elif self.kind == 'MM':
            a, b = v['a'], v['b']
            qa, qb = a.width_fs + get('decision_offset') / 2, b.width_fs - get('decision_offset') / 2
            if qa == qb and p['tie_policy'] == 'META':
                return self._fail(Status.META, k, 'exact equal comparison keys', flags)
            selected = ('a' if p['tie_policy'] == 'A' else 'b') if qa == qb else (
                'a' if (qa < qb) == (p['mode'] == 'MIN') else 'b')
            x = v[selected]
            ready = x.rise.t
            width = (p['output_gain'] if linear else 1) * x.width_fs + get('output_offset') + drift + nw
            if a.width_fs == b.width_fs:
                width += get('tie_width_error')
            rise = (ready, delay + nc)
            slew = x.rise.slew
        elif self.kind == 'VTC':
            x = v['vin']
            if x.t > clock.t:
                raise ValueError('voltage sampled before valid time')
            ready = clock.t
            width = p['K_v'] * (1 + get('gain_lsb') * code('gain_code')) * x.v + p['w0']
            width += get('beta_q') + get('offset_lsb') * code('offset_code') + nw
            width += p['K_v'] * self._noise('sigma_v', k) + drift
            if self.secondary:
                width += p['gain_delta'] * x.v + p['offset_delta'] + p['kappa_v'] * x.v**2 + p['kappa_v3'] * x.v**3
            rise = (ready, delay + nc)
        elif self.kind == 'TVC':
            x = v['tin']
            ready = clock.t
            hold = (clock.t - x.fall.t) / 1e6
            if hold < 0:
                raise ValueError('negative hold time')
            voltage = p['K_t'] * (1 + get('gain_lsb') * code('gain_code')) * x.width_fs + p['v0']
            voltage += get('beta_q_v') + get('offset_lsb_v') * code('offset_code') + get('droop_rate') * hold
            voltage += get('supply_sens_v') * dv + get('temp_sens_v') * dt + self._noise('sigma_v', k)
            voltage, clipped = self._clip(voltage, self.voltage_range)
            if clipped:
                flags.append(Status.SAT)
            valid = ready + quantize(delay + get('aperture') * 1e6 + self._noise('sigma_ap', k))
            if not is_time(valid):
                raise ValueError('voltage timestamp overflow')
            self._timing(ready, valid, valid, flags)
            self.state['y'] = voltage
            status = primary(flags)
            return Result(status, Voltage(valid, voltage, k, status), tuple(sorted(set(flags) - {status, Status.VALID})),
                          dict(status=obs_status.value, sample=k, t_valid=valid, value=voltage))
        else:  # INT
            x = v['din']
            ready = clock.t
            if ready < x.fall.t:
                raise ValueError('integrator clock precedes input completion')
            last = self.state['last_clock']
            period = self.t_ref if last is None else ready - last
            if period <= 0:
                raise ValueError('nonpositive integrator sample period')
            a = p['a0'] * (1 + get('leak_lsb') * code('leak_code'))
            if not 0 <= a < 1:
                raise ValueError('controlled integrator unstable')
            a = a ** (period / self.t_ref)
            b = p['b0'] * (1 + get('gain_lsb') * code('gain_code'))
            new_y = a * self.state['y'] + b * x.width_fs + get('beta0') + get('beta_q')
            new_y += get('offset_lsb') * code('offset_code') + get('charge_q') * x.width_fs + self._noise('sigma_n', k)
            new_y, clipped = self._clip(new_y, self.output_width)
            if clipped:
                flags.append(Status.SAT)
            width = new_y + nw
            rise = (ready, delay + nc)
        width, clipped = self._clip(width, self.output_width)
        if clipped:
            flags.append(Status.SAT)
        r, w = rise[0] + quantize(rise[1]), quantize(width)
        # Quantize width once, then preserve the integer difference exactly.
        f = r + w
        if not is_time(f):
            raise ValueError('falling timestamp overflow')
        self._timing(ready, r, f, flags)
        if r < self.state['last_output']:
            return self._fail(Status.UNDEF, k, 'output overlap/backward time', flags)
        if new_y is not None:
            self.state['y'] = new_y
            self.state['last_clock'] = ready
        self.state['last_output'] = f
        status = primary(flags)
        output = Pulse.at(r, w, k, self.model_id, slew)
        return Result(status, output, tuple(sorted(set(flags) - {status, Status.VALID})),
                      dict(status=obs_status.value, sample=k, t_valid=f,
                           width_fs=w, selected=selected, controls=c.copy()))

    def _clip(self, value, bounds):
        if not math.isfinite(value):
            raise ValueError('nonfinite numerical output')
        if bounds[0] <= value <= bounds[1]:
            return value, False
        if self.clip_policy == 'error':
            raise ValueError('output outside envelope')
        return min(bounds[1], max(bounds[0], value)), True

    def _timing(self, ready, first, last, flags):
        if first < max(self.state['t0'], ready + self.d_min):
            raise ValueError('output violates minimum delay/causality')
        if self.d_max is not None and last > ready + self.d_max:
            flags.append(Status.LATE)
