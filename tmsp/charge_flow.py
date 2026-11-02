"""Experimental scalar flow-coordinate backend (volts, nanoseconds).

Event scheduling/quantization belongs to the existing integer-fs interface.
This backend assumes matched autonomous fields and refuses extrapolation.
"""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class FlowCoordinate:
    voltage_V: tuple
    exposure_ns: tuple

    def __post_init__(self):
        v=np.asarray(self.voltage_V,dtype=float);t=np.asarray(self.exposure_ns,dtype=float)
        if v.ndim!=1 or len(v)<2 or v.shape!=t.shape or not np.isfinite(v).all() or not np.isfinite(t).all():
            raise ValueError('finite paired one-dimensional samples required')
        if not np.all(np.diff(v)>0) or not np.all(np.diff(t)>0) or t[0]!=0:
            raise ValueError('strictly monotone flow with zero initial exposure required')

    def voltage(self,exposure_ns):
        if not math.isfinite(exposure_ns) or not 0<=exposure_ns<=self.exposure_ns[-1]:
            raise ValueError('stored state outside reachable flow domain')
        return float(np.interp(exposure_ns,self.exposure_ns,self.voltage_V))

    def exposure(self,voltage_V):
        if not math.isfinite(voltage_V) or not self.voltage_V[0]<=voltage_V<=self.voltage_V[-1]:
            raise ValueError('voltage outside characterized flow domain')
        return float(np.interp(voltage_V,self.voltage_V,self.exposure_ns))

    def synthesize(self,widths_ns,rates,replay_rate,guard_ns=0):
        widths=np.asarray(widths_ns,dtype=float);a=np.asarray(rates,dtype=float)
        if widths.ndim!=1 or widths.shape!=a.shape or not np.isfinite(widths).all() or not np.isfinite(a).all():
            raise ValueError('finite paired widths and field multipliers required')
        if np.any(widths<0) or np.any(a<0) or not math.isfinite(replay_rate) or replay_rate<=0 or not math.isfinite(guard_ns) or guard_ns<0:
            raise ValueError('positive replay and nonnegative capture/guard required')
        total=float(a@widths)
        if total+guard_ns>self.exposure_ns[-1]:
            raise ValueError('architecture exceeds stored-state headroom')
        return dict(stored_voltage_V=self.voltage(total),width_ns=total/replay_rate,
                    headroom_ns=self.exposure_ns[-1]-total-guard_ns)


def residual_bound(widths_ns,capture_residual_rates,replay_rate,replay_residual_rate,boundary_ns,ideal_width_ns):
    w=np.asarray(widths_ns,dtype=float);r=np.asarray(capture_residual_rates,dtype=float)
    if w.ndim!=1 or w.shape!=r.shape or not np.isfinite(w).all() or not np.isfinite(r).all():
        raise ValueError('finite paired widths and residual rates required')
    scalars=(replay_rate,replay_residual_rate,boundary_ns,ideal_width_ns)
    if not all(math.isfinite(x) for x in scalars) or np.any(w<0) or np.any(r<0) or replay_residual_rate<0 or replay_rate<=replay_residual_rate or boundary_ns<0 or ideal_width_ns<0:
        raise ValueError('nonnegative bounds and positive replay margin required')
    return float((r@w+replay_residual_rate*ideal_width_ns+boundary_ns)/(replay_rate-replay_residual_rate))
