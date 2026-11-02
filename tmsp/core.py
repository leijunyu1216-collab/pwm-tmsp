"""Integer-femtosecond event representation; no global RNG or implicit units."""
from dataclasses import dataclass, field
from enum import StrEnum
import math

I64_MAX = 2**63 - 1
FS = 200_000_000


class Status(StrEnum):
    VALID = 'VALID'
    LATE = 'LATE'
    SAT = 'SAT'
    UNDEF = 'UNDEF'
    EXTRA = 'EXTRA'
    MISS = 'MISS'
    META = 'META'
    DOMAIN = 'DOMAIN'
    RESET = 'RESET'


ORDER = list(Status)
NUMERIC = {Status.VALID, Status.SAT, Status.LATE}


def primary(statuses):
    return max(statuses, key=ORDER.index, default=Status.VALID)


def is_time(t):
    return type(t) is int and 0 <= t <= I64_MAX


def quantize(t):
    if not math.isfinite(t):
        raise ValueError('non-finite time')
    value = round(t)
    if not is_time(value):
        raise ValueError('time outside nonnegative int64')
    return value


@dataclass(frozen=True)
class Edge:
    t: int
    pol: str
    chan: str
    sample: int
    slew: float | None = None  # V/s


@dataclass(frozen=True)
class Clock:
    t: int
    sample: int


@dataclass(frozen=True)
class Pulse:
    rise: Edge
    fall: Edge

    @property
    def width_fs(self):
        return self.fall.t - self.rise.t

    @property
    def sample(self):
        return self.rise.sample

    def validate(self):
        r, f = self.rise, self.fall
        if not all(is_time(x.t) and type(x.sample) is int and x.sample >= 0
                   and isinstance(x.chan, str) and bool(x.chan)
                   and (x.slew is None or math.isfinite(x.slew) and x.slew > 0)
                   for x in (r, f)):
            return Status.UNDEF
        if (r.pol, f.pol) != ('R', 'F') or r.chan != f.chan or r.sample != f.sample:
            return Status.UNDEF
        return Status.VALID if f.t > r.t else Status.UNDEF

    @classmethod
    def at(cls, t, width, sample=0, chan='in', slew=None):
        return cls(Edge(t, 'R', chan, sample, slew),
                   Edge(t + width, 'F', chan, sample, slew))


@dataclass(frozen=True)
class Voltage:
    t: int
    v: float
    sample: int
    status: Status = Status.VALID


@dataclass(frozen=True)
class Result:
    status: Status
    value: Pulse | Voltage | None = None
    flags: tuple = ()
    obs: dict = field(default_factory=dict)
    diagnostics: tuple = ()


def encode(x, profile='U', t_ref=FS, u_z=.5, w_min=1, w_max=FS):
    """Map a value to one/two widths. Invalid encoding raises ValueError."""
    if not math.isfinite(x) or not 0 < t_ref <= I64_MAX:
        raise ValueError('invalid value or reference')
    if profile == 'U':
        us = (x,)
    elif profile == 'B':
        us = (u_z + x / 2,)
    elif profile == 'D' and u_z >= .5:
        us = (u_z + x / 2, u_z - x / 2)
    else:
        raise ValueError('invalid profile/zero point')
    if profile in ('B', 'D') and abs(x) > 1 or profile == 'U' and not 0 <= x <= 1:
        raise ValueError('value outside encoding range')
    ws = tuple(quantize(u * t_ref) for u in us)
    if not all(w_min <= w <= w_max for w in ws):
        raise ValueError('encoded pulse outside width envelope')
    return ws if profile == 'D' else ws[0]


def decode(width, profile='U', t_ref=FS, u_z=.5):
    if profile == 'U':
        return width / t_ref
    if profile == 'B':
        return 2 * (width / t_ref - u_z)
    if profile == 'D':
        return (width[0] - width[1]) / t_ref
    raise ValueError('unknown profile')
