"""Integer-only two-point digital pulse-width predistortion controller.

The actuator edits programmed input width. It is not an unmeasured analog
trim register. Control words latch only after both observations arrive.
"""
from dataclasses import dataclass


def div_even(n,d):
    if d<=0:raise ValueError('positive divisor required')
    sign=-1 if n<0 else 1;q,r=divmod(abs(n),d)
    return sign*(q+(2*r>d or 2*r==d and q%2==1))


@dataclass
class WidthCalibrator:
    gain_q14:int=16384
    offset_word:int=0
    offset_lsb_fs:int=1000
    revision:int=0

    def command(self,target_fs):
        if type(target_fs) is not int or target_fs<0:raise ValueError('integer target required')
        raw=div_even(self.gain_q14*target_fs,16384)+self.offset_word*self.offset_lsb_fs
        command=div_even(raw,1000)*1000
        if not 20_000_000<=command<=100_000_000:
            raise ValueError('input actuator outside validated20-100ns')
        return command

    def update(self,targets,commands,observations):
        if any(len(v)!=2 or any(type(x) is not int for x in v) for v in [targets,commands,observations]):
            raise ValueError('two integer observations and commands required')
        t0,t1=targets;u0,u1=commands;y0,y1=observations
        dy=y1-y0;du=u1-u0
        if t1<=t0 or dy<=0 or du<=0:raise ValueError('non-monotone calibration observations')
        g=div_even(du*16384,dy)
        b_fs=u0-div_even((y0-t0)*du,dy)-div_even(g*t0,16384)
        b=div_even(b_fs,self.offset_lsb_fs)
        if not 4096<=g<=32767 or not -131072<=b<=131071:
            raise ValueError('control saturation')
        changed=(g,b)!=(self.gain_q14,self.offset_word)
        self.gain_q14=g;self.offset_word=b;self.revision+=1
        return changed
