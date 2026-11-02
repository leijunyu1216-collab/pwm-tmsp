"""Bounded one-solve binary-output pilot on the actual frozen 22-word network.

Only executor and requested terminal time change. This is exposed development,
not a fresh holdout, long-sequence acceptance or a model speed benchmark.
"""
import json, os, re, subprocess, time
from pathlib import Path
import numpy as np
import psutil
from benchmarks.physical_common import ROOT, sha
from benchmarks.live_spice import now, write
from benchmarks.streaming_cli_probe_v47 import EXE, raw_read

OUT = ROOT/'artifacts/network_streaming_v52'
BASE = ROOT/'artifacts/observation_recovery_v45/retry1'
LABEL = 'shared_challenge24'


def raw_progress(path):
    if not path.exists(): return {}
    with path.open('rb') as f:
        head=f.read(min(path.stat().st_size,100000)); end=head.find(b'Binary:\n')
        if end < 0: return dict(raw_bytes=path.stat().st_size)
        offset=end+len(b'Binary:\n'); text=head[:offset].decode('ascii')
        columns=int(re.search(r'No. Variables:\s*(\d+)',text)[1])
        rows=max(0,(path.stat().st_size-offset)//(columns*8))
        if rows:
            f.seek(offset+(rows-1)*columns*8); last=float(np.frombuffer(f.read(8),dtype='<f8')[0])*1e9
        else:last=None
    return dict(raw_bytes=path.stat().st_size,raw_columns=columns,raw_rows_written=rows,last_ns=last)


def edges(t, values, horizon):
    mask=t < horizon-.25; t=t[mask]; values=values[mask]
    result=[]
    for positive in [True,False]:
        ids=np.flatnonzero(((values[:-1]<.9)&(values[1:]>=.9)) if positive else ((values[:-1]>=.9)&(values[1:]<.9)))
        result.append(t[ids]+(.9-values[ids])*(t[ids+1]-t[ids])/(values[ids+1]-values[ids]))
    return result


def main():
    assert not OUT.exists(); OUT.mkdir()
    original=BASE/'flat'/LABEL/'case.cir'; source=original.read_text()
    columns=re.search(r'(?m)^wrdata trace.txt (.+)$',source)[1].split()
    stop=1510.; budget=360.; deck=OUT/'case.cir'; raw=OUT/'transient.raw'; log=OUT/'ngspice.log'
    electrical=source.split('.tran')[0]
    deck.write_text(electrical+f'.save {" ".join(columns)}\n.tran 0.125n {stop}n 0 0.125n\n'
        '.control\nset klu\nset num_threads=8\n.endc\n.end\n',encoding='ascii')
    command=[str(EXE),'-b','-n','-r',str(raw.resolve()),'-o',str(log.resolve()),str(deck.resolve())]
    contract=dict(utc=now(),question='Does one-solve streaming preserve the real-network prefix at bounded memory?',
        tier='auxiliary/development executor qualification',source=str(original),source_sha256=sha(original),
        netlist_sha256=sha(deck),exe_sha256=sha(EXE),columns=len(columns),physical_words=22,
        physical_change=False,stop_ns=stop,timeout_s=budget,max_RSS_bytes=2*2**30,
        settings='same Gear/tolerances/KLU/maxstep and8device threads;CLI replaces DLL callback',
        edge_difference_bound_ns=.01, voltage_difference_bound_V=.001,
        new_independent_seed_opened=False,complete_network_long_sequence_qualification=False,
        system_speed_benchmark=False,command=command)
    write(OUT/'contract.json',contract)
    env=os.environ.copy();env['PATH']=str(EXE.parent)+os.pathsep+'D:/MinGW-w64/mingw64/bin'+os.pathsep+env['PATH']
    env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    begin=time.perf_counter(); peak=0; status='complete'
    process=subprocess.Popen(command,cwd=OUT,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
                             creationflags=subprocess.CREATE_NO_WINDOW)
    tracked=psutil.Process(process.pid); last_update=-10
    with (OUT/'monitor.jsonl').open('a',encoding='utf-8',buffering=1) as monitor:
        while process.poll() is None:
            elapsed=time.perf_counter()-begin
            try:peak=max(peak,tracked.memory_info().rss)
            except psutil.Error:pass
            if elapsed-last_update>=5:
                state=dict(utc=now(),pid=process.pid,elapsed_s=elapsed,peak_RSS_bytes=peak,**raw_progress(raw))
                write(OUT/'progress.json',state);monitor.write(json.dumps(state)+'\n');last_update=elapsed
            if elapsed>budget or peak>contract['max_RSS_bytes']:
                status='TIMEOUT' if elapsed>budget else 'RESOURCE_ABORT';process.kill();process.wait();break
            time.sleep(.1)
    duration=time.perf_counter()-begin; console=log.read_text(errors='replace') if log.exists() else ''
    result=dict(utc=now(),status=status,returncode=process.returncode,runtime_s=duration,peak_RSS_bytes=peak,
        initialization_report_seen='Initial Transient Solution' in console,
        klu_selected='Using KLU as Direct Linear Solver' in console,
        abort_message_seen='run simulation(s) aborted' in console.lower(),**raw_progress(raw),
        physical_network_tested=True,qualification_pass=False,system_speed_benchmark=False,
        new_independent_seed_opened=False,log_sha256=sha(log) if log.exists() else None)
    write(OUT/'summary.json',result)
    if status!='complete' or process.returncode!=0 or result['abort_message_seen']:
        print(json.dumps(result,indent=2));return
    data,names=raw_read(raw);ti=names.index('time');t=data[:,ti]*1e9
    assert np.isfinite(data).all() and np.all(np.diff(t)>0) and abs(t[-1]-stop)<1e-6
    schema=json.loads((BASE/'stage'/LABEL/'stream_schema.json').read_text()); n=schema['columns']
    reference=np.memmap(BASE/'stage'/LABEL/'transient_stream.f64',dtype='<f8',mode='r').reshape(-1,n)
    oldnames=schema['names']; rt=reference[:,oldnames.index('time')]*1e9
    input_contract=json.loads((BASE/'inputs'/LABEL/'contract.json').read_text())
    observations=[]; maxedge=0.; event_count=0
    for node in ['tin',*input_contract['words']]:
        a=edges(t,data[:,names.index('v('+node+')')],stop)
        b=edges(rt,reference[:,oldnames.index(node)],stop)
        same=all(len(x)==len(y) for x,y in zip(a,b)); deltas=[float(np.max(abs(x-y))) for x,y in zip(a,b) if len(x) and len(x)==len(y)]
        delta=max(deltas,default=0.);maxedge=max(maxedge,delta);event_count+=sum(len(x) for x in a)
        observations.append(dict(node=node,rises=len(a[0]),falls=len(a[1]),reference_rises=len(b[0]),
                                 reference_falls=len(b[1]),same_counts=same,max_edge_difference_ns=delta))
    # Compare stored/bias/reference state, with time interpolation only for executor comparison.
    state_names=[x for x in oldnames if x.endswith(('.data','.reference','.fixed')) or x in ['pbias','nbias']]
    state_deltas={}
    selected=t<stop-.25
    for name in state_names:
        candidate='v('+name+')'
        if candidate not in names:continue
        error=data[selected,names.index(candidate)]-np.interp(t[selected],rt,reference[:,oldnames.index(name)])
        state_deltas[name]=float(np.max(abs(error)))
    end=stop-.25; times=np.r_[t[t<end],end]
    power=-1.8*np.interp(times,t,data[:,names.index('i(vdd)')])
    reference_power=-1.8*np.interp(times,rt,reference[:,oldnames.index('vdd#branch')])
    energy=float(np.trapezoid(power,times)*1e3);refenergy=float(np.trapezoid(reference_power,times)*1e3)
    result.update(raw_completed=True,actual_stop_ns=float(t[-1]),raw_sha256=sha(raw),
        compared_edges=event_count,port_observations=observations,max_edge_difference_ns=maxedge,
        state_max_difference_V=max(state_deltas.values(),default=0),state_differences=state_deltas,
        state_comparison_uses_interpolation=True,supply_net_energy_pJ=energy,reference_supply_net_energy_pJ=refenergy,
        supply_relative_energy_difference=abs(energy-refenergy)/abs(refenergy),
        scope='actual full topology1510ns prefix only;prefix executor agreement,not6510ns qualification or independent result')
    result['qualification_pass']=bool(result['klu_selected'] and all(x['same_counts'] for x in observations)
        and maxedge<=.01 and result['state_max_difference_V']<=.001 and result['supply_relative_energy_difference']<=1e-3)
    write(OUT/'summary.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
