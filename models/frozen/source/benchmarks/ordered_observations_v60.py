"""Audited native row order at equal times;never remove or perturb a sample.

The old strictly-increasing reader remains immutable.  This numerical version
uses (native SI time, original row ordinal). Equal-time intervals have zero
measure for quadrature. Large voltage jumps or digital decisions at the same
time are rejected rather than silently serialized into accepted PWM events.
"""
import json
from pathlib import Path
import numpy as np
from benchmarks.audit_cost_diagnosis_v56 import partial_read
from benchmarks.streaming_cli_probe_v47 import raw_read
from benchmarks.retimed_run_v56 import normalized

POLICY=dict(version='native-ordered-microsteps-v60',time_unit='native seconds',
    equal_time_order='original native row ordinal',original_rows_removed=0,
    timestamp_jitter_added=False,voltage_cluster_bound_V=.001,
    voltage_bound_source='existing numerical state comparison limit,not a new physical limit',
    digital_threshold_V=.9,equal_time_digital_crossings_allowed=0,
    backward_time_allowed=False,nonfinite_observations_allowed=False,
    zero_time_quadrature_weight=0.,scientific_gates_relaxed=False)


def build(path,block_rows=2048,live=False,*,audit_path=None):
    path=Path(path)
    if block_rows<2:raise ValueError('block_rows must be at least2')
    if live:mapped,names,tail=partial_read(path)
    else:mapped,names=raw_read(path);tail=0
    nrows,ncols=mapped.shape;offset=mapped.offset;del mapped
    if nrows<2:raise ValueError('insufficient complete physical rows')
    ti=names.index('time');selected=[i for i,n in enumerate(names) if n.startswith('v(')]
    keys=[normalized(names[i]) for i in selected]
    rising={k:[] for k in keys};falling={k:[] for k in keys};times=np.empty(nrows)
    first=0;previous=None;cluster_time=None;cluster_min=cluster_max=None
    pairs=[];max_spans=np.zeros(len(selected));column_delta=np.zeros(ncols)
    with path.open('rb') as f:
        f.seek(offset)
        while first<nrows:
            count=min(block_rows,nrows-first)
            a=np.fromfile(f,dtype='<f8',count=count*ncols).reshape(count,ncols)
            if not np.isfinite(a).all():raise ValueError('nonfinite physical observations')
            times[first:first+count]=a[:,ti]*1e9
            if previous is not None:a=np.vstack((previous,a))
            si=a[:,ti];t=si*1e9;dt=np.diff(si);row_base=first-(previous is not None)
            if np.any(dt<0):raise ValueError('native time reversal')
            if np.any((dt>0)&(np.diff(t)<=0)):raise ValueError('time-unit conversion collapses distinct native times')
            for i in np.flatnonzero(dt==0):
                v0=a[i,selected];v1=a[i+1,selected]
                crosses=((v0<.9)&(v1>=.9))|((v0>.9)&(v1<=.9))
                if crosses.any():raise ValueError('equal-time digital threshold crossing')
                if cluster_time is None or si[i]!=cluster_time:
                    cluster_time=si[i];cluster_min=v0.copy();cluster_max=v0.copy()
                cluster_min=np.minimum(cluster_min,v1);cluster_max=np.maximum(cluster_max,v1)
                span=cluster_max-cluster_min
                if np.any(span>POLICY['voltage_cluster_bound_V']):raise ValueError('equal-time voltage cluster exceeds existing1mV numerical bound')
                max_spans=np.maximum(max_spans,span)
                column_delta=np.maximum(column_delta,np.abs(a[i+1]-a[i]))
                pairs.append(dict(left_row=int(row_base+i),right_row=int(row_base+i+1),
                    native_time_s=float(si[i]),time_ns=float(t[i]),max_voltage_span_V=float(max(span,default=0.))))
            v=a[:,selected]
            for positive,dest in [(True,rising),(False,falling)]:
                condition=((v[:-1]<.9)&(v[1:]>=.9)) if positive else ((v[:-1]>.9)&(v[1:]<=.9))
                row,col=np.nonzero(condition)
                when=t[row]+(.9-v[row,col])*(t[row+1]-t[row])/(v[row+1,col]-v[row,col])
                for j,key in enumerate(keys):
                    events=when[col==j]
                    if len(events):dest[key].extend(events.tolist())
            previous=a[-1:].copy();first+=count
    audit=dict(policy=POLICY,raw_path=str(path.resolve()),complete_rows=nrows,columns=ncols,
        ignored_incomplete_live_bytes=tail,all_complete_observations_finite=True,
        first_ns=float(times[0]),last_ns=float(times[-1]),equal_time_pairs=len(pairs),
        equal_time_pairs_inventory=pairs,equal_time_digital_crossings=0,
        max_equal_time_voltage_cluster_span_V=float(max(max_spans,default=0.)),
        voltage_cluster_spans_V=dict(zip((names[i] for i in selected),map(float,max_spans))),
        all_column_equal_time_max_adjacent_changes=dict(zip(names,map(float,column_delta))),
        all_original_complete_rows_retained=True,original_rows_removed=0,gate_pass=True)
    if audit_path is not None:
        dest=Path(audit_path);dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('x',encoding='utf-8') as f:json.dump(audit,f,indent=2)
    return times,{k:(np.asarray(rising[k]),np.asarray(falling[k])) for k in keys},names
