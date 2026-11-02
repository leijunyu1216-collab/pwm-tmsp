"""Readonly complete-row snapshots and scoped continuous-campaign evaluation."""
import json,time
from pathlib import Path
import numpy as np
from benchmarks.audit_cost_diagnosis_v56 import partial_read
from benchmarks.retimed_run_v56 import normalized
from benchmarks.physical_common import ROOT
from benchmarks.live_spice import write,now
from benchmarks.crossing_index_v56 import build as crossing_build
from benchmarks.bounded_phase_scorer_v56 import modules


def contract(widths,physical,warm,targets,tail):
    from benchmarks.retimed_run_v56 import OUT
    c=json.loads((OUT/'tail_development/contract.json').read_text());values=list(map(float,widths))
    for name,w in c['words'].items():
        count=physical['words'][name]['count'];w['count']=count
        w['expected']=([sorted(values[k:k+5])[2] for k in range(len(values)-4)] if name=='s4c2' else values[:count])
    c.update(input_widths_ns=values,expected_stop_ns=physical['stop_ns'],warmup=warm,target_count=targets,tail=tail,
        expected_initialized_outputs=sum(w['count'] for w in c['words'].values()),
        score_main_counts={n:w['count']-tail for n,w in c['words'].items()},
        scope='frozen six-word joint system campaign;all checkpoints one continuous trajectory',
        independent_seed_opened=True,new_independent_seed_opened=True)
    return c


def evaluate(raw,folder,c,predictions,limit,*,live=True,cut_ns=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    a,native,_=partial_read(raw) if live else (*__import__('benchmarks.streaming_cli_probe_v47',fromlist=['raw_read']).raw_read(raw),0)
    t,index,_=crossing_build(raw,live=live);names=[normalized(n) for n in native]
    if cut_ns is not None:
        end=np.searchsorted(t,cut_ns,side='right');t=t[:end];a=a[:end]
        index={n:tuple(x[x<=t[-1]] for x in events) for n,events in index.items()}
    z={n:a[:,i] for i,n in enumerate(names)}
    base,aux=modules();base.PRECOMPUTED_CROSSINGS=index;aux.PRECOMPUTED_CROSSINGS=index
    from benchmarks.campaign_resources_v57 import trim_readonly_cache
    base.MEMORY_TRIM=trim_readonly_cache
    inp=folder/'inputs/system';stage=folder/'stage/system';inp.mkdir(parents=True);stage.mkdir(parents=True)
    write(inp/'contract.json',c);write(stage/'summary.json',dict(status='running' if live else 'complete',simulation_valid=not live))
    base.OUT=folder;aux.OUT=folder;base.read=lambda _: (t,z)
    start=time.perf_counter();r=aux.assess('system',live=True)
    warm=c['warmup'];chosen=[];phase=[];auxiliary=[]
    def begin(n):return warm+(4 if n=='front' else 4-int(n[7:]) if n.startswith('history') else 0)
    def selected(row):return begin(row['node'])<=row['sample']<begin(row['node'])+limit
    chosen=[row for row in r['rows'] if selected(row)]
    phase=[row for row in r['timings'] if selected(row)]
    auxiliary=[row for row in r['auxiliary_switch_timings'] if selected(row)]
    warmrows=[row for row in r['rows'] if row['node']=='s4c2' and row['sample']<warm]
    all_covered_ports=all(row['port_within_bounds'] for row in r['rows'])
    all_completed_phases=all(row['phase_gate_pass'] for row in r['timings'] if row['phase_evaluation_complete'])
    all_completed_aux=all(row['gate_pass'] for row in r['auxiliary_switch_timings'] if row['phase_complete'])
    local_domains=all(row['input_width_ns'] is not None and 2<=row['input_width_ns']<=198 for row in chosen)
    input_rows=[];ir,iff=index['tin']
    for k,width in enumerate(c['input_widths_ns']):
        F=40.1+200*k
        if F+width+.01>t[-1]:break
        rr=ir[np.searchsorted(ir,F-1):np.searchsorted(ir,F+199)];ff=iff[np.searchsorted(iff,F-1):np.searchsorted(iff,F+199)]
        ok=len(rr)==len(ff)==1 and ff[0]>rr[0] and abs(rr[0]-F)<=.002 and abs(ff[0]-rr[0]-width)<=.002
        input_rows.append(dict(sample=k,valid=bool(ok)))
    metrics={}
    for layer,forecast in predictions.items():
        lookup={(p['node'],p['sample']):p for p in forecast};width_err=[];edge_err=[];final_width_err=[]
        for row in chosen:
            name,k=row['node'],row['sample'];p=lookup[(name,k)];F=c['words'][name]['phase']+200*k+200.1
            rr,ff=index[name];rr=rr[np.searchsorted(rr,F-199):np.searchsorted(rr,F+1)];ff=ff[np.searchsorted(ff,F-199):np.searchsorted(ff,F+1)]
            if len(rr)!=1 or len(ff)!=1:continue
            e=float((ff[0]-rr[0])-p['width_ns']);width_err.append(e)
            edge_err.extend([float(rr[0]-p['rise_ns']),float(ff[0]-p['fall_ns'])])
            if name=='s4c2':final_width_err.append(e)
        metrics[layer]=dict(width_RMS_ns=float(np.sqrt(np.mean(np.square(width_err)))) if width_err else None,
            edge_P99_percent_FS=float(np.percentile(np.abs(edge_err),99)/2) if edge_err else None,
            final_width_RMS_ns=float(np.sqrt(np.mean(np.square(final_width_err)))) if final_width_err else None,
            compared_word_outputs=len(width_err),compared_edges=len(edge_err))
    physical_pass=bool(len(chosen)==len(c['words'])*limit and all_covered_ports and all_completed_phases and all_completed_aux
        and len(warmrows)==warm and all(row['valid'] for row in warmrows)
        and len(phase)==len(c['words'])*limit and all(row['phase_gate_pass'] and row['reuse_observed'] for row in phase)
        and len(auxiliary)==5*limit and all(row['gate_pass'] for row in auxiliary)
        and local_domains and input_rows and all(row['valid'] for row in input_rows))
    model_pass=bool(metrics['L2_static_candidate']['compared_word_outputs']==len(c['words'])*limit
        and metrics['L2_static_candidate']['edge_P99_percent_FS'] is not None and metrics['L2_static_candidate']['edge_P99_percent_FS']<=1.)
    if not live:physical_pass=bool(physical_pass and r['gate_pass'])
    result=dict(utc=now(),checkpoint_targets=limit,actual_snapshot_last_ns=float(t[-1]),
        simulation_terminal=not live,scoped_physical_gate_pass=physical_pass,scoped_prediction_gate_pass=model_pass,
        gate_pass=physical_pass and model_pass,scheduled_warmup_outputs=warm,valid_warmup_outputs=sum(row['valid'] for row in warmrows),
        selected_word_outputs=len(chosen),selected_phase_transactions=len(phase),selected_aux_transactions=len(auxiliary),
        checked_complete_inputs=len(input_rows),all_complete_inputs_match=bool(input_rows and all(row['valid'] for row in input_rows)),
        local_card_domains_hold=local_domains,all_covered_ports_pass=all_covered_ports,
        all_completed_phase_gates_pass=all_completed_phases,all_completed_aux_gates_pass=all_completed_aux,
        failed_ports=[row for row in r['rows'] if not row['port_within_bounds']],
        failed_selected_phases=[row for row in phase if not row['phase_gate_pass'] or not row['reuse_observed']],
        failed_selected_aux=[row for row in auxiliary if not row['gate_pass']],metrics=metrics,
        final_measured_widths_ns=[row['output_width_ns'] for row in chosen if row['node']=='s4c2'],
        full_event_counts_exact=r['event_count_exact'],full_development_style_gate=r['gate_pass'],
        evaluation_wall_s=time.perf_counter()-start,global_D1_D6_acceptance=False)
    write(folder/'checkpoint.json',result);return result


def replay():
    from benchmarks.retimed_run_v56 import OUT
    from benchmarks.retimed_prediction_v56 import predict
    c=json.loads((OUT/'tail_development/contract.json').read_text());c['warmup']=0
    card=json.loads((OUT/'grid_short27_prediction_cards.json').read_text())
    forecasts={layer:predict(c['input_widths_ns'],card,layer) for layer in ['L0','L1','L2_static_candidate']}
    path=ROOT/'artifacts/campaign_qualification_v57/replay20'
    result=evaluate(OUT/'runs/grid_short27/transient.raw',path,c,forecasts,20,live=False)
    write(path/'scope.json',dict(development_replay_only=True,new_independent_seed_opened=False))
    print(json.dumps({k:v for k,v in result.items() if not k.startswith('failed')},indent=2))
    if not result['gate_pass']:raise RuntimeError('checkpoint replay failed')


if __name__=='__main__':replay()
