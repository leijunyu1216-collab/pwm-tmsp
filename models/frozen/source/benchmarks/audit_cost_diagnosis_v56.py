"""Readonly audit of exposed profiles, including incomplete binary prefixes."""
import hashlib,json,re
from pathlib import Path
import numpy as np
from benchmarks.cost_diagnosis_v56 import OUT,BASE,LABEL,analyze,reference
from benchmarks.live_spice import now,write


def stream_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*2**20),b''):h.update(block)
    return h.hexdigest()


def partial_read(path):
    with Path(path).open('rb') as f:
        head=f.read(100000);marker=b'Binary:\n';offset=head.index(marker)+len(marker)
    text=head[:offset].decode('ascii');n=int(re.search(r'No. Variables:\s*(\d+)',text)[1])
    names=re.findall(r'(?m)^\s*\d+\s+(\S+)\s+\S+\s*$',text)
    if len(names)!=n:raise ValueError('schema mismatch')
    rows,remainder=divmod(Path(path).stat().st_size-offset,n*8)
    return np.memmap(path,dtype='<f8',mode='r',offset=offset,shape=(rows,n)),names,remainder


def main():
    # The first profiling attempt completed SPICE but did not request raw output
    # correctly. Preserve the error; use its log for timing only.
    p=OUT/'klu8_100ns';text=(p/'ngspice.log').read_text(errors='replace')
    if not (p/'summary.json').exists():
        stats={k.strip():float(v) for k,v in re.findall(r'(?m)^([^\n=]+)=\s*([0-9.eE+-]+)\s*$',text)}
        write(p/'summary.json',dict(status='MISSING_RAW',statistics=stats,terminal_complete=False,
            scope='timing log only;no waveform comparison;excluded from matched speed ratio',new_independent_seed_opened=False))
    candidate=OUT/'active_final_dense2310/transient.raw'
    if candidate.exists():
        a,names,tail=partial_read(candidate);stop=float(a[-1,names.index('time')]*1e9)
        result=analyze(a,names,stop)
        write(candidate.parent/'partial_reference_audit.json',dict(utc=now(),scope='completed prefix comparison only;no terminal or cost acceptance',
            complete_rows=len(a),ignored_trailing_bytes=tail,last_ns=stop,**result))
    ref,rn=reference();t=ref[:,rn.index('time')]*1e9
    from benchmarks.physical_common import edges
    rises,falls=edges(t,ref[:,rn.index('s4c2')])
    rows=[]
    for p in sorted(OUT.glob('*/summary.json')):
        r=json.loads(p.read_text());rows.append(dict(label=p.parent.name,status=r.get('status'),runtime_s=r.get('runtime_s'),
            last_ns=r.get('last_ns'),terminal_complete=r.get('terminal_complete'),reference_agreement=r.get('observed_reference_agreement'),
            max_edge_difference_ns=r.get('max_edge_difference_ns'),final_median_pairs=r.get('final_median_output_pairs'),
            statistics=r.get('statistics',{}),summary_sha256=stream_sha(p)))
    write(OUT/'evaluation_summary.json',dict(utc=now(),parent='v54 executor cost failure;v55 prerequisites',
        evaluation_summary='exposed numerical/executor diagnosis;new independent campaign still unopened',
        first_final_rise_ns=float(rises[0]),first_final_fall_ns=float(falls[0]),old_1510ns_prefix_cannot_include_final_pair=bool(falls[0]>1510),
        claim_update='no full cost qualification unless terminal,active-output agreement,all declared observations and budgets all pass',
        baseline_relation='original source and reference retained',failure_mode='execution cost and evaluation-window mismatch',
        controls=rows,new_independent_seed_opened=False,formal_freeze_executed=False,
        full_system_layout_area_complete=False,full_system_routing_complete=False,full_system_three_noise_complete=False,
        next_action='select only a reference-qualified numerical route;complete same-implementation cost and short development before freezing'))
    print(json.dumps({'first_final_fall_ns':float(falls[0]),'controls':[{k:v for k,v in r.items() if k not in ['statistics','summary_sha256']} for r in rows]},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
