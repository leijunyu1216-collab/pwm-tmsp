"""Experimental causal exposure-state boundary generator.

An online behavioral controller; analog sensing, DAC circuitry and their energy
are characterized independently. Public sample times are integer femtoseconds.
"""
from dataclasses import dataclass
import math
import numpy as np

@dataclass(frozen=True)
class BoundaryCard:
    bias_V:tuple
    relative_rate:tuple
    coordinate_ns:tuple
    voltage_V:tuple
    phase_ns:float
    sample_fs:int=250_000
    bits:int=16
    fullscale_V:float=1.8
    def __post_init__(self):
        for name in ['bias_V','relative_rate','coordinate_ns','voltage_V']:
            object.__setattr__(self,name,tuple(getattr(self,name)))
        for x,y in [(self.bias_V,self.relative_rate),(self.coordinate_ns,self.voltage_V)]:
            if len(x)!=len(y) or len(x)<2 or not all(math.isfinite(v) for v in (*x,*y)) or not all(a<b for a,b in zip(x,x[1:])):
                raise ValueError('finite monotone lookup domain required')
        if type(self.sample_fs)is not int or self.sample_fs<=0 or type(self.bits)is not int or not 1<=self.bits<=32 or not math.isfinite(self.phase_ns) or not math.isfinite(self.fullscale_V) or self.fullscale_V<=0:
            raise ValueError('invalid sampling or encoding card')
        if not all(v>0 for v in self.relative_rate) or not all(a<b for a,b in zip(self.voltage_V,self.voltage_V[1:])):
            raise ValueError('positive rates and increasing encoding required')

class ExposureBoundary:
    def __init__(self,card,start_fs=0):
        if type(start_fs)is not int or start_fs<0:raise ValueError('invalid start')
        self.card=card;self.next_fs=start_fs;self.exposure_ns=0.;self.previous_rate=None
    def sample(self,time_fs,bias_V):
        c=self.card
        if type(time_fs)is not int or time_fs!=self.next_fs:raise ValueError('sample clock mismatch')
        if not math.isfinite(bias_V) or not c.bias_V[0]<=bias_V<=c.bias_V[-1]:raise ValueError('bias outside characterized domain')
        # The command uses the previous exposure state. Today's observation
        # affects only subsequent commands: one sample of computation latency.
        coordinate=c.phase_ns+self.exposure_ns
        if not c.coordinate_ns[0]<=coordinate<=c.coordinate_ns[-1]:raise ValueError('boundary outside characterized domain')
        voltage=float(np.interp(coordinate,c.coordinate_ns,c.voltage_V))
        if not 0<=voltage<=c.fullscale_V:raise ValueError('DAC range exceeded')
        code=round(voltage/c.fullscale_V*(2**c.bits-1))
        if self.previous_rate is not None:self.exposure_ns+=c.sample_fs/1e6*(self.previous_rate-1.)
        self.previous_rate=float(np.interp(bias_V,c.bias_V,c.relative_rate));self.next_fs+=c.sample_fs
        return code*c.fullscale_V/(2**c.bits-1)
    def snapshot(self):return dict(next_fs=self.next_fs,exposure_ns=self.exposure_ns,previous_rate=self.previous_rate)
    def restore(self,state):
        if set(state)!={'next_fs','exposure_ns','previous_rate'}:raise ValueError('invalid state')
        if type(state['next_fs'])is not int or state['next_fs']<0 or not math.isfinite(state['exposure_ns']):raise ValueError('invalid state')
        if state['previous_rate'] is not None and (not math.isfinite(state['previous_rate']) or state['previous_rate']<=0):raise ValueError('invalid rate state')
        self.next_fs=state['next_fs'];self.exposure_ns=state['exposure_ns'];self.previous_rate=state['previous_rate']
