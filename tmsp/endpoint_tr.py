"""L0 common-fall register contract; physical qualifications remain external.

This backend preserves width and declares both edge phase and availability.
It does not infer L1/L2 parameters or accept misaligned physical parents.
"""
from dataclasses import dataclass,asdict
from .core import Clock,Pulse,Result,Status,is_time

@dataclass(frozen=True)
class EndpointTR:
    input_fall_phase_fs:int
    output_fall_phase_fs:int
    width_min_fs:int=4_000_000
    width_max_fs:int=196_000_000

    def evaluate(self,pulse,clock):
        if not isinstance(clock,Clock) or not is_time(clock.t) or type(clock.sample) is not int or clock.sample<0:
            return Result(Status.UNDEF,flags=('INVALID_CLOCK',))
        if not isinstance(pulse,Pulse) or pulse.validate()!=Status.VALID:
            return Result(Status.UNDEF,flags=('INVALID_INPUT',))
        if any(type(v) is not int or v<0 for v in (self.input_fall_phase_fs,self.output_fall_phase_fs,self.width_min_fs,self.width_max_fs)) or not 0<self.width_min_fs<=self.width_max_fs:
            return Result(Status.UNDEF,flags=('INVALID_CARD',))
        if pulse.sample!=clock.sample or pulse.fall.t!=clock.t+self.input_fall_phase_fs:
            return Result(Status.DOMAIN,flags=('INPUT_PHASE_OR_SAMPLE',))
        if not self.width_min_fs<=pulse.width_fs<=self.width_max_fs:
            return Result(Status.DOMAIN,flags=('WIDTH_ENVELOPE',))
        fall=clock.t+self.output_fall_phase_fs;rise=fall-pulse.width_fs
        if not is_time(fall) or not is_time(rise):return Result(Status.UNDEF,flags=('TIMESTAMP_RANGE',))
        if rise<pulse.fall.t:return Result(Status.LATE,flags=('OUTPUT_PRECEDES_CAPTURE_END',))
        return Result(Status.VALID,Pulse.at(rise,pulse.width_fs,clock.sample,'endpoint_tr'),
            obs={'anchor':'fall','availability_fs':fall,'capture_end_fs':pulse.fall.t,'layer':'L0'})

class EndpointTRComponent:
    def __init__(self,model,observer=None):self.model=model;self.observer=observer
    def contract(self):return {'kind':'experimental_endpoint_TR','model':asdict(self.model),'observer':self.observer}
    def step(self,ctx,inputs):
        result=self.model.evaluate(inputs['input'],Clock(ctx.tick_fs,ctx.cycle))
        if self.observer and result.status==Status.VALID:
            p=result.value
            ctx.emit(self.observer,{'sample':p.sample,'width_fs':p.width_fs,'status':str(result.status),
                'rise_fs':p.rise.t,'fall_fs':p.fall.t,'anchor':'fall'},p.fall.t)
        return {'result':result}
    def snapshot(self):return {}
    def restore(self,state):
        if state!={}:raise ValueError('stateless component snapshot required')
