"""Context-qualified L1 transfer. Unvalidated cards require explicit opt-in."""
import copy
import math
from .core import Clock,Pulse,Result,Status,is_time


class MeasuredTransfer:
    """Frozen offline coefficients;no oracle access or online fitting."""
    def __init__(self,card,allow_development=False):
        self.card=copy.deepcopy(card)
        self.allow_development=allow_development

    def evaluate(self,pulse,clock,context_sha256):
        c=self.card
        if not c.get('evaluator_supported',True):
            return Result(Status.DOMAIN,flags=('UNSUPPORTED_INTERFACE',))
        if c['release_state']!='validated' and not self.allow_development:
            return Result(Status.DOMAIN,flags=('CARD_NOT_INDEPENDENTLY_VALIDATED',))
        if context_sha256!=c['circuit_context_sha256']:
            return Result(Status.DOMAIN,flags=('UNIDENTIFIED_CIRCUIT_CONTEXT',))
        if not isinstance(clock,Clock) or not is_time(clock.t) or not isinstance(pulse,Pulse) or pulse.validate()!=Status.VALID:
            return Result(Status.UNDEF,flags=('INVALID_INPUT_OR_CLOCK',))
        if pulse.sample!=clock.sample:
            return Result(Status.DOMAIN,flags=('SAMPLE_ASSOCIATION',))
        if clock.t!=clock.sample*c['period_fs']:
            return Result(Status.DOMAIN,flags=('UNIDENTIFIED_CLOCK_SCHEDULE',))
        phase=clock.t+c['input_fall_phase_fs']
        if abs(pulse.fall.t-phase)>c['input_phase_tolerance_fs']:
            return Result(Status.DOMAIN,flags=('UNIDENTIFIED_INPUT_PHASE',))
        w=pulse.width_fs/1e6;lo,hi=c['input_width_ns']
        if not lo<=w<=hi:
            return Result(Status.DOMAIN,flags=('INPUT_WIDTH_DOMAIN',))
        gain,bias=c['width_affine_gain_offset_ns'];slope,offset=c['fall_error_affine_ns']
        width=gain*w+bias;error=slope*w+offset
        if not all(math.isfinite(x) for x in [gain,bias,slope,offset,width,error]) or width<=0:
            return Result(Status.DOMAIN,flags=('INVALID_CARD_OR_PREDICTION',))
        width_fs=round(width*1e6)
        fall=clock.t+c['output_fall_phase_fs']+round(error*1e6);rise=fall-width_fs
        if not is_time(rise) or not is_time(fall):
            return Result(Status.UNDEF,flags=('TIMESTAMP_RANGE',))
        if rise<pulse.fall.t:
            return Result(Status.LATE,flags=('OUTPUT_PRECEDES_CAPTURE_END',))
        return Result(Status.VALID,Pulse.at(rise,width_fs,clock.sample,'measured_transfer'),
            obs=dict(layer='L1',availability_fs=fall,release_state=c['release_state'],
                     source='frozen offline measured card',context_sha256=context_sha256))
