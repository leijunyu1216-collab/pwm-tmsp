"""Causal width-domain reference for unit-delay plus registered half-addition."""
import copy
import numpy as np
from .core import Pulse,Result,Status,FS


class RegisteredLowpass:
    """Input k and stored k-1 generate output frame k+1; integer-fs interface."""
    def __init__(self,layer='L0',card=None):
        if layer not in ['L0','L1','L2']:raise ValueError('unknown layer')
        if layer!='L0' and card is None:raise ValueError('measured component card required')
        self.layer=layer;self.card=copy.deepcopy(card);self.reset()

    def reset(self):self.state=dict(next_sample=0,previous_stored_fs=None)
    def snapshot(self):return copy.deepcopy(self.state)
    def restore(self,state):self.state=copy.deepcopy(state)

    def process(self,pulse):
        if not isinstance(pulse,Pulse) or pulse.validate()!=Status.VALID or pulse.sample!=self.state['next_sample']:
            return Result(Status.UNDEF)
        x=pulse.width_fs/1e6
        if self.layer!='L0':
            lo,hi=self.card['input_width_ns']
            if not lo<=x<=hi:return Result(Status.DOMAIN)
            d=float(np.polyval(self.card['delay_poly_ns'][self.layer],x))
        else:d=x
        previous=self.state['previous_stored_fs']
        if previous is None:
            self.state=dict(next_sample=pulse.sample+1,previous_stored_fs=round(d*1e6))
            return Result(Status.MISS,obs=dict(input_sample=pulse.sample,output_frame=pulse.sample+1))
        if self.layer=='L0':
            width_fs=round((pulse.width_fs+previous)/2)
            offset_fs=0
        else:
            ga,gb,bias=self.card['half_adder_affine_ns']
            width_fs=round((ga*x+gb*previous/1e6+bias)*1e6)
            offset_fs=round(self.card['output_rise_offset_ns']*1e6)
        if width_fs<=0:return Result(Status.DOMAIN)
        self.state=dict(next_sample=pulse.sample+1,previous_stored_fs=round(d*1e6))
        rise=(pulse.sample+1)*FS+40_100_000+offset_fs
        return Result(Status.VALID,Pulse.at(rise,width_fs,pulse.sample+1,'lowpass'),
            obs=dict(input_sample=pulse.sample,stored_input_sample=pulse.sample-1,output_frame=pulse.sample+1))
