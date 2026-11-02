"""Validated-scope INT voltage state and fixed-period dual-store TR adapter."""
import copy
import math
from .core import Pulse,Clock,Voltage,Result,Status,FS,is_time
from .physical import TriggeredPrimitive


def integrate_voltage(v,ns,g0,g1,tau,on):
    if not on:return v*math.exp(-ns/tau)
    rate=g1-1/tau
    return v*math.exp(rate*ns)+g0*(math.expm1(rate*ns)/rate if abs(rate)>1e-15 else ns)


class DynamicINT:
    def __init__(self,card,initial_v=0.,t0_fs=0):
        self.card=copy.deepcopy(card)
        if len(card['active_mechanisms'])>2:raise ValueError('mechanism budget')
        self.reset(initial_v,t0_fs)

    def reset(self,initial_v=0.,t0_fs=0):
        lo,hi=self.card['envelope']['state_v']
        if not lo<=initial_v<=hi or not is_time(t0_fs):raise ValueError('initial state outside card')
        self.state=dict(voltage=initial_v,t=t0_fs,last_sample=-1)

    def process(self,pulse,observe_fs):
        if not isinstance(pulse,Pulse) or pulse.validate()!=Status.VALID:
            return Result(Status.UNDEF)
        if not is_time(observe_fs) or observe_fs<pulse.fall.t:
            return Result(Status.DOMAIN)
        if pulse.sample<=self.state['last_sample'] or pulse.rise.t<self.state['t']:
            return Result(Status.UNDEF)
        w=(pulse.width_fs-200_000)/1e6
        lo,hi=self.card['envelope']['programmed_width_ns']
        if not lo<=w<=hi:return Result(Status.DOMAIN)
        p=self.card['parameters'];v=self.state['voltage']
        advance=lambda vv,ns,on:integrate_voltage(vv,ns,p['g0_v_ns'],p['g1_per_ns'],p['tau_ns'],on)
        v=advance(v,(pulse.rise.t-self.state['t'])/1e6,False)
        v+=p['q_on_v'][0]+p['q_on_v'][1]*v
        v=advance(v,pulse.width_fs/1e6,True)
        v+=p['q_off_v'][0]+p['q_off_v'][1]*v
        v=advance(v,(observe_fs-pulse.fall.t)/1e6,False)
        lo,hi=self.card['envelope']['state_v']
        if not lo<=v<=hi:return Result(Status.DOMAIN)
        self.state=dict(voltage=v,t=observe_fs,last_sample=pulse.sample)
        return Result(Status.VALID,Voltage(observe_fs,v,pulse.sample),obs=dict(state_v=v))

    def snapshot(self):return copy.deepcopy(self.state)
    def restore(self,state):self.state=copy.deepcopy(state)


class TwoBankTR:
    """Fixed-period phase scheduler. Sample labels remain global and unchanged.

    This object schedules capture/replay transactions; it is not a generic
    event simulator. Reset availability, replay deadline and merged-output
    non-overlap are checked on every cycle.
    """
    def __init__(self,config,period_fs=FS,trigger_phase_fs=160_100_000,warmup=100,t0_fs=0):
        if period_fs!=FS:raise ValueError('card only characterized at5MS/s')
        if type(warmup) is not int or warmup<0 or not is_time(t0_fs):
            raise ValueError('invalid warmup or origin')
        if not is_time(trigger_phase_fs) or trigger_phase_fs>=period_fs:
            raise ValueError('invalid phase')
        self.config=copy.deepcopy(config);self.period=period_fs
        self.phase=trigger_phase_fs;self.warmup=warmup;self.t0=t0_fs
        self.banks=[TriggeredPrimitive({**config,'warmup_cycles':0,
            'model_id':config['model_id']+f'.bank{b}'}) for b in [0,1]]
        self.reset()

    def reset(self):
        for m in self.banks:m.reset(self.t0)
        self.state=dict(cycle=0,bank_release=[self.t0,self.t0],merged_release=self.t0)

    def process(self,pulse):
        k=self.state['cycle']
        if not isinstance(pulse,Pulse) or pulse.validate()!=Status.VALID or pulse.sample!=k:
            return Result(Status.UNDEF)
        base=self.t0+k*self.period;bank=k%2;clock=base+self.phase
        if pulse.rise.t<base or pulse.fall.t>clock:return Result(Status.DOMAIN)
        if self.state['bank_release'][bank]>base:return Result(Status.LATE)
        saved=self.banks[bank].model.snapshot();last=self.banks[bank].last_output
        result=self.banks[bank].process({'tin':pulse},clock=Clock(clock,k))
        if result.value is None or result.value.rise.t<self.state['merged_release']:
            self.banks[bank].model.restore(saved);self.banks[bank].last_output=last
            return result if result.value is None else Result(Status.UNDEF)
        self.state['bank_release'][bank]=result.value.fall.t
        self.state['merged_release']=result.value.fall.t
        self.state['cycle']+=1
        if k<self.warmup:return Result(Status.RESET,obs=dict(sample=k,bank=bank,clock_fs=clock))
        return Result(result.status,result.value,result.flags,
            {**result.obs,'bank':bank,'clock_fs':clock,'cycle':k},result.diagnostics)

    def snapshot(self):
        return dict(state=copy.deepcopy(self.state),banks=[m.model.snapshot() for m in self.banks],
                    last_outputs=[m.last_output for m in self.banks])

    def restore(self,snapshot):
        self.state=copy.deepcopy(snapshot['state'])
        for m,state,last in zip(self.banks,snapshot['banks'],snapshot['last_outputs']):
            m.model.restore(state);m.last_output=last
