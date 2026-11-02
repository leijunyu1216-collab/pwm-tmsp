"""Generate bounded-window current scoring; compare every result to the original."""
import json,types,time
from pathlib import Path
import numpy as np
from benchmarks.physical_common import ROOT
from benchmarks.live_spice import write,now
from benchmarks.retimed_run_v56 import OUT,normalized
from benchmarks.streaming_cli_probe_v47 import raw_read
from benchmarks.audit_cost_diagnosis_v56 import stream_sha

FOLDER=OUT/'bounded_scorer'
HELPER='''
class _WindowSignals:
    def __init__(self,source,span):self.source,self.span=source,span
    def __getitem__(self,key):return self.source[key][self.span]

def _phase_window(t,z,F):
    if callable(globals().get('MEMORY_TRIM')):MEMORY_TRIM()
    span=slice(np.searchsorted(t,F-300,side='left'),np.searchsorted(t,F+600,side='right'))
    return t[span],_WindowSignals(z,span)

'''


def once(s,old,new):
    if s.count(old)!=1:raise ValueError('source pattern drift:'+old)
    return s.replace(old,new)


def modules():
    FOLDER.mkdir(exist_ok=True)
    src=ROOT/'benchmarks/assess_wide_v34.py';s=src.read_text()
    s=s.replace('def assess(label,live=False):',HELPER+'def assess(label,live=False):')
    s=once(s,'t,z=read(label);cache={};','t,z=read(label);full_t,full_z=t,z;cache=dict(globals().get("PRECOMPUTED_CROSSINGS",{}));')
    s=once(s,"cache[name]=edges(t,z[name])","cache[name]=edges(full_t,full_z[name])")
    s=once(s,"if live and t[-1]<outF+1:continue","if live and full_t[-1]<outF+1:continue\n            t,z=_phase_window(full_t,full_z,F)")
    s=once(s,"    summary=json.loads((folder/'summary.json').read_text())","    t,z=full_t,full_z\n    summary=json.loads((folder/'summary.json').read_text())")
    path=FOLDER/'bounded_wide.py';path.write_text(s,encoding='utf-8')
    base=types.ModuleType('bounded_wide_v56');base.__file__=str(path);exec(compile(s,str(path),'exec'),base.__dict__)
    src2=ROOT/'benchmarks/assess_capture_network_v36.py';s2=src2.read_text()
    s2=once(s2,'from benchmarks import assess_wide_v34 as base','base=OPTIMIZED_BASE')
    s2=once(s2,"base.OUT=OUT;result=base.assess(label,live);t,z=base.read(label)","base.OUT=OUT;result=base.assess(label,live);t,z=base.read(label);full_t,full_z=t,z")
    s2=once(s2,"cache[name]=edges(t,z[name])","cache[name]=edges(full_t,full_z[name])")
    s2=once(s2,'rows=[];cache={}','rows=[];cache=dict(globals().get("PRECOMPUTED_CROSSINGS",{}))')
    s2=once(s2,"b=phase['bank'];F=phase['capture_reference_phase_ns'];prefix=f'x{name}.xbank{b}'","b=phase['bank'];F=phase['capture_reference_phase_ns'];prefix=f'x{name}.xbank{b}'\n        t,z=base._phase_window(full_t,full_z,F)")
    path2=FOLDER/'bounded_capture.py';path2.write_text(s2,encoding='utf-8')
    aux=types.ModuleType('bounded_capture_v56');aux.__file__=str(path2);aux.OPTIMIZED_BASE=base;exec(compile(s2,str(path2),'exec'),aux.__dict__)
    write(FOLDER/'transformation.json',dict(utc=now(),original_sources_sha256={str(src.relative_to(ROOT)):stream_sha(src),str(src2.relative_to(ROOT)):stream_sha(src2)},
        generated_sources_sha256={path.name:stream_sha(path),path2.name:stream_sha(path2)},
        changes='bounded F-300toF+600ns views for per-transaction current calculations;global edge caches unchanged',
        thresholds_or_metrics_changed=False,golden_widths_used_to_select_windows=False,window_depends_only_on_frozen_schedule=True))
    return base,aux


def main(label='tail_short27'):
    from benchmarks import assess_wide_v34 as original_base
    from benchmarks import assess_capture_network_v36 as original_aux
    from benchmarks.crossing_index_v56 import build
    base,aux=modules();folder=OUT/'runs'/label;a,native=raw_read(folder/'transient.raw');names=[normalized(n) for n in native]
    indexing_start=time.perf_counter();indexed_t,crossings,_=build(folder/'transient.raw');index_wall=time.perf_counter()-indexing_start
    base.PRECOMPUTED_CROSSINGS=crossings;aux.PRECOMPUTED_CROSSINGS=crossings
    t=a[:,names.index('time')]*1e9;z={n:a[:,i] for i,n in enumerate(names)}
    c=json.loads((OUT/'tail_development/contract.json').read_text());summ=json.loads((folder/'summary.json').read_text())
    results=[];runtimes=[]
    for number,(b,s) in enumerate([(original_base,original_aux),(base,aux)]):
        target=FOLDER/('original' if number==0 else 'bounded');inp=target/'inputs'/label;stage=target/'stage'/label
        inp.mkdir(parents=True,exist_ok=True);stage.mkdir(parents=True,exist_ok=True)
        write(inp/'contract.json',c);write(stage/'summary.json',dict(status=summ['status'],simulation_valid=summ['terminal_complete']))
        old_out,old_read,old_s=b.OUT,b.read,s.OUT
        try:
            b.OUT=target;b.read=lambda _: (t,z);s.OUT=target;start=time.perf_counter();result=s.assess(label,live=True);runtimes.append(time.perf_counter()-start)
        finally:b.OUT,b.read,s.OUT=old_out,old_read,old_s
        results.append({k:v for k,v in result.items() if k not in ['analysis_source_sha256','capture_analysis_sha256']})
    equal=results[0]==results[1]
    result=dict(utc=now(),label=label,all_scientific_fields_exactly_equal=equal,
        compared_port_transactions=len(results[0]['rows']),compared_phase_transactions=len(results[0]['timings']),
        compared_auxiliary_transactions=len(results[0]['auxiliary_switch_timings']),original_wall_s=runtimes[0],bounded_wall_s=runtimes[1],one_pass_index_wall_s=index_wall,
        primary_gate_passes=[r['gate_pass'] for r in results],scope='evaluator equivalence on exposed complete development;not a system speed benchmark',
        full_long_storage_and_input_generator_qualified=False,new_independent_seed_opened=False)
    write(FOLDER/'equivalence.json',result);print(json.dumps(result,indent=2))
    if not equal:raise AssertionError('bounded current scoring changed a scientific field')


if __name__=='__main__':main()
