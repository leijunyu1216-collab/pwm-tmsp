"""Exposed-data audit of every voltage event, state and electrical boundary.

First/last projections exist only as adversarial diagnostic views. The actual
evaluator keeps every native row. No projected waveform is written or used as
physical acceptance evidence, nor is this old-data audit a new holdout.
"""
import json,re,time
from pathlib import Path
import numpy as np
from benchmarks.physical_common import ROOT,edges
from benchmarks.live_spice import now,write
from benchmarks.audit_cost_diagnosis_v56 import partial_read,stream_sha
from benchmarks.retimed_run_v56 import normalized,OUT as OLD
from benchmarks.campaign_observations_v60 import module,evaluate
from benchmarks.ordered_observations_v60 import build,POLICY
from benchmarks.campaign_resources_v57 import trim_readonly_cache

OUT=ROOT/'artifacts/ordered_observations_v60'
PARENT=ROOT/'artifacts/ground_reduction_v59'


class Column:
    def __init__(self,a,ids,col):self.a,self.ids,self.col=a,ids,col
    def __getitem__(self,s):return self.a[self.ids[s],self.col]
    def __array__(self,dtype=None,copy=None):return np.asarray(self[:],dtype=dtype)


class View:
    def __init__(self,a,ids):self.a,self.ids=a,ids
    def __getitem__(self,key):
        rows,col=key
        if rows!=slice(None):raise ValueError('only full-column lazy views supported')
        return Column(self.a,self.ids,col)


def projected_index(a,names,ids):
    selected=[i for i,n in enumerate(names) if n.startswith('v(')];keys=[normalized(names[i]) for i in selected]
    result={k:[[],[]] for k in keys};ti=names.index('time');t=np.asarray(a[ids,ti])*1e9;previous=None
    if not np.all(np.diff(t)>0):raise ValueError('diagnostic projection not strict')
    for first in range(0,len(ids),2048):
        b=np.asarray(a[ids[first:first+2048]])
        if previous is not None:b=np.vstack((previous,b))
        u=b[:,ti]*1e9
        for j,k in zip(selected,keys):
            for polarity,values in enumerate(edges(u,b[:,j])):result[k][polarity].extend(values.tolist())
        previous=b[-1:].copy();trim_readonly_cache()
    return t,{k:tuple(np.asarray(v) for v in values) for k,values in result.items()}


def compare_scientific(a,b):
    differences=[];numeric={};ignored={'analysis_source_sha256','capture_analysis_sha256','raw_sha256','utc','evaluation_wall_s'}
    def visit(x,y,key):
        leaf=key.rsplit('/',1)[-1]
        if leaf in ignored:return
        if isinstance(x,dict):
            if x.keys()!=y.keys():differences.append(key+':keys');return
            for k in x:visit(x[k],y[k],key+'/'+k)
        elif isinstance(x,list):
            if len(x)!=len(y):differences.append(key+':length');return
            for i,(u,v) in enumerate(zip(x,y)):visit(u,v,key+'/'+str(i))
        elif isinstance(x,bool) or x is None or isinstance(x,str):
            if x!=y:differences.append(key)
        elif isinstance(x,(int,float)):
            if not np.isfinite(x) or not np.isfinite(y):
                if x!=y:differences.append(key)
            else:numeric[key]=abs(float(x)-float(y))
        elif x!=y:differences.append(key)
    visit(a,b,'')
    edge=max((v for k,v in numeric.items() if k.endswith('_ns')),default=0.)
    state=max((v for k,v in numeric.items() if k.endswith('_V') or '/useful_read_aux_voltage_span_V/' in k),default=0.)
    return dict(discrete_differences=differences,maximum_ns_field_difference=edge,maximum_voltage_field_difference_V=state,
        maximum_numeric_field_difference=max(numeric.values(),default=0.),
        nonzero_numeric_differences={k:v for k,v in numeric.items() if v},
        gate_pass=not differences and edge<=.01 and state<=.001)


def clock_inventory(body):
    clocks=[]
    pattern=r'(?m)^(V\S+) (\S+_drive) 0 PWL\((.*?)\s\)\s*$'
    lines=body.splitlines()
    for m in re.finditer(pattern,body,re.S):
        name,drive,text=m.groups();tokens=text.replace('\n+',' ').split()
        if len(tokens)%2:raise ValueError('clock point pairs')
        knots=np.asarray([float(v[:-1])*1e-9 if v.endswith('n') else float(v) for v in tokens[::2]])
        levels=np.asarray(list(map(float,tokens[1::2])))
        resistors=[s.split() for s in lines if s.startswith('rflat_') and drive in s.split()[1:3]]
        if len(resistors)!=1 or float(resistors[0][3])!=100:raise ValueError('clock source boundary')
        r=resistors[0];receiver=r[2] if r[1]==drive else r[1]
        clocks.append((name,receiver,knots,levels))
    if len(clocks)!=87:raise ValueError('expected87 physical finiteR clocks')
    return clocks


def boundary_energy(a,names,ids,body):
    clocks=clock_inventory(body);ti=names.index('time');supply=0.;ports=np.zeros((len(clocks),3));previous=None
    for first in range(0,len(ids),4096):
        b=np.asarray(a[ids[first:first+4096]])
        if previous is not None:b=np.vstack((previous,b))
        t=b[:,ti];supply+=float(np.trapezoid(-1.8*b[:,names.index('i(vdd)')],t))
        for j,(_,node,u,v) in enumerate(clocks):
            source=np.interp(t,u,v);received=b[:,names.index('v('+node+')')]
            p=source*(source-received)/100
            ports[j]+=np.array([np.trapezoid(p,t),np.trapezoid(np.maximum(p,0),t),np.trapezoid((source-received)**2/100,t)])
        previous=b[-1:].copy();trim_readonly_cache()
    return dict(supply_energy_J=supply,boundary_net_energy_J=float(supply+ports[:,0].sum()),
        clock_ports={name:dict(net_energy_J=float(p[0]),positive_draw_J=float(p[1]),resistor_heat_J=float(p[2])) for (name,*_),p in zip(clocks,ports)},
        actual_clock_generator_included=False,full_system_PPA_qualified=False)


def run():
    OUT.mkdir(exist_ok=True);start=time.perf_counter()
    write(OUT/'observation_policy.json',dict(utc=now(),policy=POLICY,tier='exposed numerical development',new_seed_opened=False,
        unchanged_limits=dict(edge_ns=.01,state_V=.001,energy_relative=.001),
        null='ordered microsteps change events or scientific gate outcomes',alternative='all native rows preserve the declared observations within original numerical limits'))
    # Full existing short data must reproduce its old evaluator exactly.
    c=json.loads((OLD/'tail_development/contract.json').read_text());c['warmup']=0
    from benchmarks.retimed_prediction_v56 import predict
    card=json.loads((OLD/'periodic_short27_prediction_cards.json').read_text())
    pred={k:predict(c['input_widths_ns'],card,k) for k in ['L0','L1','L2_static_candidate']}
    short=evaluate(PARENT/'runs/ground_short27/transient.raw',OUT/'short_replay',c,pred,20,live=False)
    expected=json.loads((PARENT/'short_assessment/system_live.json').read_text())
    actual=json.loads((OUT/'short_replay/system_live.json').read_text());same=compare_scientific(actual,expected)
    if not same['gate_pass'] or same['maximum_numeric_field_difference']!=0:raise RuntimeError('strict short evaluator changed scientific fields')
    write(OUT/'short_equivalence.json',same)
    # The exposed stopped trace is a live complete-row prefix,not terminal64310ns.
    stage=PARENT/'regression';raw=stage/'physical/transient.raw'
    c=json.loads((stage/'evaluation_contract.json').read_text());pred=json.loads((stage/'predictions.json').read_text())
    long=evaluate(raw,OUT/'exposed_214_replay',c,pred,214,live=True)
    audit=json.loads((OUT/'exposed_214_replay/native_time_audit.json').read_text());a,names,tail=partial_read(raw)
    reference=json.loads((OUT/'exposed_214_replay/system_live.json').read_text())
    t,index,_=build(raw,live=True);ids=np.arange(len(a));body=(stage/'generated_electrical.cir').read_text()
    energies={'native_order':boundary_energy(a,names,ids,body)};comparisons={};events={};projections={}
    for kind,side in [('first','right_row'),('last','left_row')]:
        remove=np.asarray([row[side] for row in audit['equal_time_pairs_inventory']],dtype=int)
        keep=np.ones(len(a),dtype=bool);keep[remove]=False;chosen=ids[keep]
        ut,ui=projected_index(a,names,chosen);event_rows=[]
        for name in index:
            count=all(len(x)==len(y) for x,y in zip(index[name],ui[name]))
            delta=max((float(np.max(np.abs(x-y))) for x,y in zip(index[name],ui[name]) if len(x) and len(x)==len(y)),default=0.)
            event_rows.append(dict(node=name,event_counts_identical=count,max_edge_difference_ns=delta))
        events[kind]=event_rows
        m=module();m.partial_read=lambda p:(View(a,chosen),names,tail);m.crossing_build=lambda p,**kw:(ut,ui,names)
        result=m.evaluate(raw,OUT/('diagnostic_'+kind),c,pred,214,live=True)
        comparison=compare_scientific(reference,json.loads((OUT/('diagnostic_'+kind)/'system_live.json').read_text()))
        comparisons[kind]=comparison;projections[kind]=result['gate_pass']
        energies[kind]=boundary_energy(a,names,chosen,body)
    voltage_max=audit['max_equal_time_voltage_cluster_span_V']
    energy_deltas={}
    for kind in ['first','last']:
        pairs=[('supply',energies['native_order']['supply_energy_J'],energies[kind]['supply_energy_J']),
               ('boundary',energies['native_order']['boundary_net_energy_J'],energies[kind]['boundary_net_energy_J'])]
        for port in energies['native_order']['clock_ports']:
            for key,value in energies['native_order']['clock_ports'][port].items():pairs.append((port+'/'+key,value,energies[kind]['clock_ports'][port][key]))
        energy_deltas[kind]={name:abs(x-y)/max(abs(x),abs(y),np.finfo(float).tiny) for name,x,y in pairs}
    all_events=all(r['event_counts_identical'] and r['max_edge_difference_ns']<=.01 for rows in events.values() for r in rows)
    energy_max=max(x for d in energy_deltas.values() for x in d.values())
    passed=bool(short['gate_pass'] and long['gate_pass'] and all_events and voltage_max<=.001 and energy_max<=.001
        and all(x['gate_pass'] for x in comparisons.values()) and all(projections.values()))
    result=dict(utc=now(),gate_pass=passed,wall_s=time.perf_counter()-start,strict_short_fields_exact=True,
        all455_observations_validated=True,all_voltage_event_comparisons=events,full_phase_and_aux_comparisons=comparisons,
        max_equal_time_voltage_span_V=voltage_max,energies=energies,relative_energy_differences=energy_deltas,
        maximum_boundary_or_port_energy_relative_difference=energy_max,exposed214_scoped_gate=long['gate_pass'],
        native_rows_retained=len(a),native_rows_removed=0,diagnostic_projections_only=True,
        audit_raw_sha256=stream_sha(raw),original_failed_freeze_accepted=False,terminal64310ns_completed=False,
        model_refit=False,new_independent_seed_opened=False,global_D1_D6_acceptance=False)
    write(OUT/'physical_qualification.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['energies','all_voltage_event_comparisons','full_phase_and_aux_comparisons','relative_energy_differences']},indent=2))
    if not passed:raise RuntimeError('ordered-observation exposed qualification failed')


if __name__=='__main__':run()
