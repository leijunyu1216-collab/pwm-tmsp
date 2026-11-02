"""Bounded, exposed electrical-reference cost diagnostics. No held-out generator."""
import argparse,json,os,re,subprocess,time
from pathlib import Path
import numpy as np
import psutil
from benchmarks.physical_common import ROOT,sha
from benchmarks.live_spice import now,write
from benchmarks.streaming_cli_probe_v47 import EXE,raw_read
from benchmarks.network_streaming_pilot_v52 import BASE,LABEL,raw_progress,edges

OUT=ROOT/'artifacts/cost_diagnosis_v56'


def reference():
    folder=BASE/'stage'/LABEL
    schema=json.loads((folder/'stream_schema.json').read_text())
    a=np.memmap(folder/'transient_stream.f64',dtype='<f8',mode='r').reshape(-1,schema['columns'])
    return a,schema['names']


def analyze(a,names,stop):
    ref,rn=reference();t=a[:,names.index('time')]*1e9;rt=ref[:,rn.index('time')]*1e9
    observations=[]
    for name in names:
        if not name.startswith('v('):continue
        node=name[2:-1]
        if node not in rn:continue
        first=edges(t,a[:,names.index(name)],stop);second=edges(rt,ref[:,rn.index(node)],stop)
        same=all(len(x)==len(y) for x,y in zip(first,second))
        delta=max([float(np.max(abs(x-y))) for x,y in zip(first,second) if len(x) and len(x)==len(y)],default=0.)
        observations.append(dict(node=node,rises=len(first[0]),falls=len(first[1]),same_counts=same,max_edge_difference_ns=delta))
    states=[n for n in names if n.startswith('v(') and (n.endswith(('.data)','.reference)','.fixed)')) or n in ['v(pbias)','v(nbias)'])]
    m=t<stop-.25
    state_errors={n:float(np.max(abs(a[m,names.index(n)]-np.interp(t[m],rt,ref[:,rn.index(n[2:-1])])))) for n in states}
    en=float(np.trapezoid(-1.8*a[m,names.index('i(vdd)')],t[m])*1e3)
    gold=float(np.trapezoid(-1.8*np.interp(t[m],rt,ref[:,rn.index('vdd#branch')]),t[m])*1e3)
    result=dict(ports=observations,state_errors_V=state_errors,supply_energy_pJ=en,reference_supply_energy_pJ=gold,
        energy_relative_difference=abs(en-gold)/abs(gold),max_edge_difference_ns=max(x['max_edge_difference_ns'] for x in observations),
        final_median_output_pairs=next((x['falls'] for x in observations if x['node']=='s4c2'),0))
    result['observed_reference_agreement']=all(x['same_counts'] for x in observations) and result['max_edge_difference_ns']<=.01 and max(state_errors.values(),default=float('inf'))<=.001 and result['energy_relative_difference']<=.001
    return result


def run(label,stop,budget,threads=1,solver='klu',extra='',source=None,maxstep=.125,dense=False):
    folder=OUT/label;folder.mkdir(parents=True,exist_ok=False)
    source=Path(source) if source else BASE/'flat'/LABEL/'case.cir'
    electrical=source.read_text().split('.tran')[0]
    if solver=='sparse':electrical=electrical.replace('.options klu','')
    contract=json.loads((BASE/'inputs'/LABEL/'contract.json').read_text());ports=['tin',*contract['words']]
    states=['pbias','nbias','xs4c2.fixed','xs4c2.xbank0.data','xs4c2.xbank0.reference']
    columns=[f'v({n})' for n in ports+states]+['i(vdd)']
    if dense:columns=re.search(r'(?m)^wrdata trace.txt (.+)$',source.read_text())[1].split()
    net=folder/'case.cir';raw=folder/'transient.raw';log=folder/'ngspice.log'
    control=('set klu\n' if solver=='klu' else 'unset klu\n')+f'set num_threads={threads}\necho PROFILE_BEFORE_RUN\nrun {raw.resolve().as_posix()}\necho PROFILE_AFTER_RUN\nrusage all\nquit\n'
    net.write_text(electrical+extra+f'\n.save {" ".join(columns)}\n.tran 0.125n {stop}n 0 {maxstep}n\n.control\n'+control+'.endc\n.end\n',encoding='ascii')
    cmd=[str(EXE),'-b','-n','-r',str(raw.resolve()),'-o',str(log.resolve()),str(net.resolve())]
    write(folder/'contract.json',dict(utc=now(),tier='exposed cost diagnosis',source=str(source),source_sha256=sha(source),deck_sha256=sha(net),
        stop_ns=stop,budget_s=budget,threads=threads,solver=solver,extra_options=extra,maxstep_ns=maxstep,command=cmd,
        new_independent_seed_opened=False,system_speed_benchmark=False,observations='all original requested vectors' if dense else 'reduced diagnostic ports;not full phase contract'))
    env=os.environ.copy();env['PATH']=str(EXE.parent)+os.pathsep+'D:/MinGW-w64/mingw64/bin'+os.pathsep+env['PATH']
    env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    start=time.perf_counter();p=subprocess.Popen(cmd,cwd=folder,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
    tracked=psutil.Process(p.pid);peak=0;status='complete';last=-5;cpu={};phase_times={}
    with (folder/'monitor.jsonl').open('w',encoding='utf-8',buffering=1) as out:
        while p.poll() is None:
            elapsed=time.perf_counter()-start
            try:peak=max(peak,tracked.memory_info().rss);cpu=tracked.cpu_times()._asdict()
            except psutil.Error:pass
            if elapsed-last>=2:
                text=log.read_text(errors='replace') if log.exists() else ''
                for marker in ['PROFILE_BEFORE_RUN','Initial Transient Solution','PROFILE_AFTER_RUN']:
                    if marker in text and marker not in phase_times:phase_times[marker]=elapsed
                row=dict(utc=now(),elapsed_s=elapsed,pid=p.pid,peak_RSS_bytes=peak,cpu_seconds=cpu,phase_first_observed_s=phase_times.copy(),**raw_progress(raw))
                write(folder/'progress.json',row);out.write(json.dumps(row)+'\n');last=elapsed
            if elapsed>=budget or peak>2*2**30:
                status='TIMEOUT' if elapsed>=budget else 'RESOURCE_ABORT';p.kill();p.wait();break
            time.sleep(.05)
    text=log.read_text(errors='replace') if log.exists() else ''
    result=dict(utc=now(),status=status,returncode=p.returncode,runtime_s=time.perf_counter()-start,peak_RSS_bytes=peak,
        phase_first_observed_s=phase_times,cpu_seconds=cpu,terminal_complete=False,**raw_progress(raw),
        new_independent_seed_opened=False,system_speed_benchmark=False,full_dense_phase_contract_qualified=False)
    result['statistics']={k.strip():float(v) for k,v in re.findall(r'(?m)^([^\n=]+)=\s*([0-9.eE+-]+)\s*$',text)}
    if status=='complete' and p.returncode==0 and not raw.exists():
        result['status']='MISSING_RAW';result['evaluation_failure']='simulation log alone does not establish waveform agreement'
    if status=='complete' and p.returncode==0 and raw.exists() and 'run simulation(s) aborted' not in text.lower():
        a,names=raw_read(raw);t=a[:,names.index('time')]*1e9
        if np.isfinite(a).all() and np.all(np.diff(t)>0) and abs(t[-1]-stop)<1e-6:
            result.update(terminal_complete=True,raw_sha256=sha(raw),**analyze(a,names,stop))
    write(folder/'summary.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['ports','state_errors_V']},ensure_ascii=False),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('label');p.add_argument('--stop',type=float,default=100);p.add_argument('--budget',type=float,default=75)
    p.add_argument('--threads',type=int,default=1);p.add_argument('--solver',choices=['klu','sparse'],default='klu');p.add_argument('--extra',default='');p.add_argument('--source');p.add_argument('--maxstep',type=float,default=.125);p.add_argument('--dense',action='store_true')
    a=p.parse_args();run(a.label,a.stop,a.budget,a.threads,a.solver,a.extra,a.source,a.maxstep,a.dense)
