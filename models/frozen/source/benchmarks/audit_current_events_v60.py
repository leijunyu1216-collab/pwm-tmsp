"""Audit complete current-threshold event lists,not only selected gap endpoints."""
import json,time
import numpy as np
from benchmarks.physical_common import ROOT,edges
from benchmarks.audit_cost_diagnosis_v56 import partial_read,stream_sha
from benchmarks.live_spice import now,write
from benchmarks.campaign_resources_v57 import trim_readonly_cache
from benchmarks.retimed_run_v56 import normalized

OUT=ROOT/'artifacts/ordered_observations_v60'
RAW=ROOT/'artifacts/ground_reduction_v59/regression/physical/transient.raw'


def query(t,signals,row):
    F=row['capture_reference_phase_ns'];b=row['bank'];prefix=f"x{row['node']}.xbank{b}"
    data=abs(signals[f'@mflat_{prefix}.xca.xsrc[id]'])+abs(signals[f'@mflat_{prefix}.xcb.xsrc[id]'])
    ref=abs(signals[f'@mflat_{prefix}.xcr.xsrc[id]'])
    reset=abs(signals[f'@mflat_{prefix}.xresetd[id]'])+abs(signals[f'@mflat_{prefix}.xresetr[id]'])
    result=[];level=row['local_on_current_A']
    for g in row['current_gaps']:
        frac=g['fraction']
        for name,off,on in [('precharge_to_data',row['precharge_control_fall_ns'],row['data_control_rise_ns']),
                            ('capture_to_replay',row['data_control_fall_ns'],row['replay_control_rise_ns'])]:
            if off is None or on is None or level<=0:continue
            m=(t>=min(off,on)-1)&(t<=max(off,on)+1.5)
            result.append((name,frac,float(frac*level),edges(t[m],data[m],frac*level)))
    if row['reuse_observed'] and row['replay_control_rise_ns'] is not None:
        start=row['replay_control_rise_ns'];end=row['replay_control_fall_ns']+row['reuse_control_margin_ns']
        refmask=(t>=start+.35)&(t<=start+.8)
        reflevel=float(np.median(ref[refmask])) if refmask.any() else 0.
        lev=max(level,reflevel);m=(t>=end-.5)&(t<=end+1);peak=float(max(reset[m]))
        mm=(t>=start)&(t<=end+1)
        for g in row['reuse_current_gaps']:
            frac=g['fraction']
            result.append(('reuse_source',frac,float(frac*lev),edges(t[mm],np.maximum(data,ref)[mm],frac*lev)))
            result.append(('reuse_reset',frac,float(frac*peak),edges(t[mm],reset[mm],frac*peak)))
    return result


def main():
    started=time.perf_counter();a,names,_=partial_read(RAW);t=np.asarray(a[:,names.index('time')])*1e9
    audit=json.loads((OUT/'exposed_214_replay/native_time_audit.json').read_text())
    variants={k:json.loads((OUT/f'{p}/system_live.json').read_text()) for k,p in [
        ('native','exposed_214_replay'),('first','diagnostic_first'),('last','diagnostic_last')]}
    records={k:{(r['node'],r['sample']):r for r in v['timings']} for k,v in variants.items()}
    removed={k:np.asarray(sorted(set(p[side] for p in audit['equal_time_pairs_inventory'])),dtype=int)
        for k,side in [('first','right_row'),('last','left_row')]}
    columns={normalized(n):i for i,n in enumerate(names)};comparisons=[];queries=0;edge_total=0
    for row in variants['native']['timings']:
        if not row['phase_evaluation_complete']:continue
        F=row['capture_reference_phase_ns'];lo=np.searchsorted(t,F-300,'left');hi=np.searchsorted(t,F+600,'right')
        prefix=f"x{row['node']}.xbank{row['bank']}"
        keys=[f'@mflat_{prefix}.{name}[id]' for name in ['xca.xsrc','xcb.xsrc','xcr.xsrc','xresetd','xresetr']]
        local=np.asarray(a[lo:hi][:,[columns[n] for n in keys]])
        u=t[lo:hi];signals={k:local[:,j] for j,k in enumerate(keys)}
        original=query(u,signals,row);queries+=len(original);edge_total+=sum(len(e) for *_,es in original for e in es)
        for kind in ['first','last']:
            drop=removed[kind];local_drop=drop[(drop>=lo)&(drop<hi)]-lo
            if not len(local_drop):
                # No changed sample in the entire phase window: byte-identical
                # query inputs and the previously compared scalar thresholds.
                other=records[kind][(row['node'],row['sample'])]
                if other['local_on_current_A']!=row['local_on_current_A']:raise ValueError('unexpected threshold change outside equal-time window')
                continue
            keep=np.ones(len(u),dtype=bool);keep[local_drop]=False
            other=query(u[keep],{k:v[keep] for k,v in signals.items()},records[kind][(row['node'],row['sample'])])
            if len(original)!=len(other):raise ValueError('current query inventory differs')
            for x,y in zip(original,other):
                if x[:2]!=y[:2]:raise ValueError('current query label differs')
                same=all(len(p)==len(q) for p,q in zip(x[3],y[3]))
                delta=max((float(max(abs(p-q))) for p,q in zip(x[3],y[3]) if len(p) and len(p)==len(q)),default=0.)
                comparisons.append(dict(node=row['node'],sample=row['sample'],projection=kind,query=x[0],fraction=x[1],
                    threshold_difference_A=abs(x[2]-y[2]),event_counts_identical=same,max_edge_difference_ns=delta))
        trim_readonly_cache()
    passed=all(x['event_counts_identical'] and x['max_edge_difference_ns']<=.01 for x in comparisons)
    result=dict(utc=now(),gate_pass=passed,wall_s=time.perf_counter()-started,
        complete_phases=sum(r['phase_evaluation_complete'] for r in variants['native']['timings']),
        all_current_queries=queries,all_current_rise_and_fall_events=edge_total,
        explicitly_affected_comparisons=len(comparisons),unaffected_windows_byte_identical=True,
        max_edge_difference_ns=max((x['max_edge_difference_ns'] for x in comparisons),default=0.),
        current_event_count_failures=[x for x in comparisons if not x['event_counts_identical']],
        comparisons=comparisons,raw_sha256=stream_sha(RAW),native_rows_removed=0,
        numerical_edge_bound_ns=.01,new_independent_seed_opened=False)
    write(OUT/'current_event_qualification.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='comparisons'},indent=2))
    if not passed:raise RuntimeError('complete current-event audit failed')


if __name__=='__main__':main()
