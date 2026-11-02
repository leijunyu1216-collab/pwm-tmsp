"""Physical cost/short-development controls for the new retimed implementation."""
import argparse,json,os,re,subprocess,time
import numpy as np
import psutil
from benchmarks.physical_common import ROOT,edges
from benchmarks.live_spice import now,write
from benchmarks.streaming_cli_probe_v47 import EXE,raw_read
from benchmarks.network_streaming_pilot_v52 import raw_progress
from benchmarks.audit_cost_diagnosis_v56 import stream_sha

OUT=ROOT/'artifacts/retimed_lattice_v56'


def normalized(n):
    if n.startswith('v('):return n[2:-1]
    if n.startswith('i(@'):return n[2:-1]
    if n.startswith('i('):return n[2:-1]+'#branch'
    return n


def score_ports(a,names,c,stop,precomputed=None,times=None):
    t=a[:,names.index('time')]*1e9 if times is None else times;rows=[]
    for name,w in c['words'].items():
        rr,ff=precomputed[name] if precomputed is not None else edges(t,a[:,names.index('v('+name+')')])
        for k,expected in enumerate(w['expected']):
            F=w['phase']+200*k+200.1
            if F+1>stop:continue
            r=rr[np.searchsorted(rr,F-199,'left'):np.searchsorted(rr,F+1,'left')]
            f=ff[np.searchsorted(ff,F-199,'left'):np.searchsorted(ff,F+1,'left')]
            ok=len(r)==len(f)==1 and f[0]>r[0]
            width=float(f[0]-r[0]-expected) if ok else None;anchor=float(f[0]-F) if ok else None
            rows.append(dict(node=name,sample=k,expected_ns=expected,valid=bool(ok),width_error_ns=width,anchor_error_ns=anchor,
                within_bounds=bool(ok and abs(width)<=2 and abs(anchor)<=.2)))
    med=[r for r in rows if r['node']=='s4c2'];good=[r for r in rows if r['valid']]
    return dict(rows=rows,evaluated_outputs=len(rows),complete_outputs=len(good),final_median_output_pairs=sum(r['valid'] for r in med),
        max_abs_width_ns=max((abs(r['width_error_ns']) for r in good),default=None),
        max_abs_anchor_ns=max((abs(r['anchor_error_ns']) for r in good),default=None),
        covered_port_gate=bool(rows and all(r['within_bounds'] for r in rows)),
        median_endpoint4_covered=any(r['valid'] and r['expected_ns']==4 for r in med),
        median_endpoint196_covered=any(r['valid'] and r['expected_ns']==196 for r in med))


def run(label,stop,budget,step=.125,unpruned=False,tail=False,phase_grid=False,periodic=False):
    folder=OUT/'runs'/label;folder.mkdir(parents=True,exist_ok=False)
    source=OUT/('periodic_grid/case.cir' if periodic else 'phase_grid/case.cir' if phase_grid else 'tail_development/case.cir' if tail else 'inputs/development24/case.cir' if unpruned else 'pruned/case.cir')
    original=source.read_text();electrical=original.split('.tran')[0]
    if (phase_grid or periodic) and unpruned:
        from benchmarks.retimed_lattice_v56 import BASE
        old_path=(OUT/'pruned/geometry_models.cir').as_posix()
        full_path=(BASE/'flat/distributed_challenge24/geometry_models.cir').as_posix()
        if electrical.count(old_path)!=1:raise ValueError('geometry include inventory changed')
        electrical=electrical.replace(old_path,full_path)
    from benchmarks.retimed_contract_v56 import effective
    c=json.loads((OUT/'tail_development/contract.json').read_text()) if tail else effective();cols=c['output_columns']
    raw=folder/'transient.raw';log=folder/'ngspice.log';net=folder/'case.cir'
    net.write_text(electrical+f'\n.save {" ".join(cols)}\n.tran {step}n {stop}n 0 {step}n\n.control\nset klu\nset num_threads=8\necho PROFILE_BEFORE_RUN\nrun {raw.resolve().as_posix()}\necho PROFILE_AFTER_RUN\nrusage all\nquit\n.endc\n.end\n',encoding='ascii')
    cmd=[str(EXE),'-b','-n','-r',str(raw.resolve()),'-o',str(log.resolve()),str(net.resolve())]
    write(folder/'contract.json',dict(utc=now(),tier='new implementation exposed development',netlist_sha256=stream_sha(net),source_sha256=stream_sha(source),
        stop_ns=stop,budget_s=budget,maxstep_ns=step,unpruned=unpruned,tail_development=tail,phase_grid=phase_grid,periodic_grid=periodic,columns=len(cols),command=cmd,
        new_independent_seed_opened=False,formal_freeze_executed=False,system_speed_benchmark=False))
    env=os.environ.copy();env['PATH']=str(EXE.parent)+os.pathsep+'D:/MinGW-w64/mingw64/bin'+os.pathsep+env['PATH'];env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    start=time.perf_counter();p=subprocess.Popen(cmd,cwd=folder,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
    tracked=psutil.Process(p.pid);peak=0;status='complete';last=-5;phase={}
    with (folder/'monitor.jsonl').open('w',encoding='utf-8',buffering=1) as monitor:
        while p.poll() is None:
            elapsed=time.perf_counter()-start
            try:peak=max(peak,tracked.memory_info().rss)
            except psutil.Error:pass
            if elapsed-last>=2:
                text=log.read_text(errors='replace') if log.exists() else ''
                for marker in ['PROFILE_BEFORE_RUN','Initial Transient Solution','PROFILE_AFTER_RUN']:
                    if marker in text and marker not in phase:phase[marker]=elapsed
                row=dict(utc=now(),elapsed_s=elapsed,pid=p.pid,peak_RSS_bytes=peak,phase_first_observed_s=phase.copy(),**raw_progress(raw));write(folder/'progress.json',row);monitor.write(json.dumps(row)+'\n');last=elapsed
            if elapsed>=budget or peak>2*2**30:status='TIMEOUT' if elapsed>=budget else 'RESOURCE_ABORT';p.kill();p.wait();break
            time.sleep(.05)
    text=log.read_text(errors='replace') if log.exists() else ''
    result=dict(status=status,returncode=p.returncode,runtime_s=time.perf_counter()-start,peak_RSS_bytes=peak,phase_first_observed_s=phase,
        terminal_complete=False,new_independent_seed_opened=False,formal_freeze_executed=False,system_speed_benchmark=False,full_dense_phase_contract_qualified=False,**raw_progress(raw))
    result['statistics']={k.strip():float(v) for k,v in re.findall(r'(?m)^([^\n=]+)=\s*([0-9.eE+-]+)\s*$',text)}
    if status=='complete' and p.returncode==0 and raw.exists() and 'run simulation(s) aborted' not in text.lower():
        from benchmarks.crossing_index_v56 import build
        a,names=raw_read(raw);t,index,_=build(raw)
        if abs(t[-1]-stop)<1e-6:
            result.update(terminal_complete=True,raw_sha256=stream_sha(raw),**score_ports(a,names,c,stop,index,t))
            # Expose a disk-backed, zero-copy schema view for the unchanged
            # internal current/phase evaluator. The header offset is explicit.
            write(folder/'stream_schema.json',dict(names=[normalized(n) for n in names],columns=len(names),offset_bytes=a.offset,dtype='<f8',points=len(a)))
    write(folder/'summary.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','statistics']},indent=2),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('label');p.add_argument('--stop',type=float,default=1510);p.add_argument('--budget',type=float,default=360);p.add_argument('--step',type=float,default=.125);p.add_argument('--unpruned',action='store_true');p.add_argument('--tail',action='store_true');p.add_argument('--phase-grid',action='store_true');p.add_argument('--periodic',action='store_true');a=p.parse_args();run(a.label,a.stop,a.budget,a.step,a.unpruned,a.tail,a.phase_grid,a.periodic)
