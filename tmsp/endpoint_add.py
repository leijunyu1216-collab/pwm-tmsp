"""Ideal half-weight backend for the experimental common-endpoint realization.

Explicit clock phase is mandatory; a physical card is qualified separately.
All public timestamps are integer femtoseconds. No implicit one-cycle latency.
"""
from dataclasses import dataclass,asdict
from .core import Clock,Pulse,Result,Status,is_time


@dataclass(frozen=True)
class EndpointADD:
    fall_phase_fs:int
    input_phase_fs:int=30_100_000
    width_min_fs:int=4_000_000
    width_max_fs:int=196_000_000

    def evaluate(self,a,b,clock):
        if not isinstance(clock,Clock) or not is_time(clock.t) or type(clock.sample) is not int or clock.sample<0:
            return Result(Status.UNDEF,flags=('INVALID_CLOCK',))
        if any(not isinstance(p,Pulse) or p.validate()!=Status.VALID for p in (a,b)):
            return Result(Status.UNDEF,flags=('INVALID_INPUT',))
        if any(type(v) is not int or v<0 for v in (self.fall_phase_fs,self.input_phase_fs,self.width_min_fs,self.width_max_fs)) or self.width_max_fs<self.width_min_fs:
            return Result(Status.UNDEF,flags=('INVALID_CARD',))
        if a.sample!=b.sample or a.sample!=clock.sample or a.rise.t!=b.rise.t or a.rise.t!=clock.t+self.input_phase_fs:
            return Result(Status.DOMAIN,flags=('INPUT_PHASE_OR_SAMPLE',))
        if not all(self.width_min_fs<=p.width_fs<=self.width_max_fs for p in (a,b)):
            return Result(Status.DOMAIN,flags=('WIDTH_ENVELOPE',))
        total=a.width_fs+b.width_fs
        width=total//2
        if total%2 and width%2:width+=1  # exact nearest-even half-fs rounding
        fall=clock.t+self.fall_phase_fs;rise=fall-width
        if not is_time(fall) or not is_time(rise):return Result(Status.UNDEF,flags=('TIMESTAMP_RANGE',))
        if rise<max(a.fall.t,b.fall.t):return Result(Status.LATE,flags=('OUTPUT_PRECEDES_CAPTURE_END',))
        return Result(Status.VALID,Pulse.at(rise,width,clock.sample,'endpoint_add'),
                      obs={'anchor':'fall','availability_fs':fall,'capture_end_fs':max(a.fall.t,b.fall.t)})


class EndpointADDComponent:
    """CycleGraph adapter; complete widths are released at the falling event."""
    def __init__(self,model,observer=None):
        self.model=model;self.observer=observer

    def contract(self):
        return {'kind':'experimental_endpoint_ADD','model':asdict(self.model),'observer':self.observer}

    def step(self,ctx,inputs):
        result=self.model.evaluate(inputs['a'],inputs['b'],Clock(ctx.tick_fs,ctx.cycle))
        if self.observer and result.status==Status.VALID:
            p=result.value
            ctx.emit(self.observer,{'sample':p.sample,'width_fs':p.width_fs,'status':str(result.status),
                                   'rise_fs':p.rise.t,'fall_fs':p.fall.t,'anchor':'fall'},p.fall.t)
        return {'result':result}

    def snapshot(self):return {}

    def restore(self,state):
        if state!={}:raise ValueError('stateless component snapshot required')
