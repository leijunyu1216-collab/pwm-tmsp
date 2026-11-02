"""Finite integer-fs clock tables;physical timing,slopes and NP stay fixed."""
from decimal import Decimal
import re

PATTERN=re.compile(r'(?m)^(V\S+) (\S+_drive) 0 PULSE\(0 1.8 ([\d.]+)n ([\d.]+)n ([\d.]+)n ([\d.]+)n ([\d.]+)n (\d+)\)$')


def fs(value):
    exact=Decimal(value)*1000000
    if exact!=exact.to_integral_value():raise ValueError('clock time is not integer fs')
    return int(exact)


def points(groups):
    name,node,*params=groups;delay,rise,fall,width,period=map(fs,params[:5]);count=int(params[5])
    if min(delay,rise,fall,width,period,count)<=0 or rise+width+fall>=period:
        raise ValueError('invalid finite pulse clock')
    result=[(0,0)]
    for k in range(count):
        t=delay+k*period
        result.extend([(t,0),(t+rise,1.8),(t+rise+width,1.8),(t+rise+width+fall,0)])
    return result


def render(value):
    q,r=divmod(value,1000000)
    return f'{q}.{r:06d}n'


def rewrite(text):
    inventory=[]
    def one(m):
        values=points(m.groups());inventory.append(dict(source=m[1],node=m[2],pulses=int(m[8]),points=len(values),
            delay_fs=fs(m[3]),rise_fs=fs(m[4]),fall_fs=fs(m[5]),width_fs=fs(m[6]),period_fs=fs(m[7])))
        return f'{m[1]} {m[2]} 0 PWL(0 0\n'+'\n'.join(f'+ {render(t)} {v:g}' for t,v in values[1:])+' )'
    result,n=PATTERN.subn(one,text)
    if n!=87:raise ValueError(f'clock inventory changed:{n}')
    return result,inventory
