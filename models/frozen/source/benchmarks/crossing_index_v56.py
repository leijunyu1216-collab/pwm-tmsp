"""One-pass finite/time validation and exact0.9V crossing index for native raw."""
import numpy as np
from benchmarks.streaming_cli_probe_v47 import raw_read
from benchmarks.retimed_run_v56 import normalized


def build(path,block_rows=2048,live=False):
    if live:
        from benchmarks.audit_cost_diagnosis_v56 import partial_read
        mapped,names,_=partial_read(path)
    else:mapped,names=raw_read(path)
    nrows,ncols=mapped.shape;offset=mapped.offset
    selected=[i for i,n in enumerate(names) if n.startswith('v(')];keys=[normalized(names[i]) for i in selected]
    rising={k:[] for k in keys};falling={k:[] for k in keys};times=np.empty(nrows);first=0;previous=None
    with path.open('rb') as f:
        f.seek(offset)
        while first<nrows:
            count=min(block_rows,nrows-first);a=np.fromfile(f,dtype='<f8',count=count*ncols).reshape(count,ncols)
            if not np.isfinite(a).all():raise ValueError('nonfinite physical observations')
            t=a[:,names.index('time')]*1e9;times[first:first+count]=t
            if previous is not None:a=np.vstack((previous,a));t=np.r_[times[first-1],t]
            if not np.all(np.diff(t)>0):raise ValueError('time ordering failure')
            v=a[:,selected]
            for positive,dest in [(True,rising),(False,falling)]:
                condition=((v[:-1]<.9)&(v[1:]>=.9)) if positive else ((v[:-1]>.9)&(v[1:]<=.9))
                row,col=np.nonzero(condition)
                when=t[row]+(.9-v[row,col])*(t[row+1]-t[row])/(v[row+1,col]-v[row,col])
                for j,key in enumerate(keys):
                    events=when[col==j]
                    if len(events):dest[key].extend(events.tolist())
            previous=a[-1:].copy();first+=count
    return times,{k:(np.asarray(rising[k]),np.asarray(falling[k])) for k in keys},names
