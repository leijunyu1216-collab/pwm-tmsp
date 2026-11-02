"""SPICE execution with persistent logs and observable process status."""
import json
import os
import re
import subprocess
import time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import psutil
from benchmarks.physical_common import NG,sha


def now():return datetime.now(timezone.utc).isoformat()
def write(path,data):Path(path).write_text(json.dumps(data,indent=2),encoding='utf-8')


def run_live(directory,deck,columns,timeout=360,max_rss_bytes=None):
    directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=True)
    if (directory/'run.json').exists():raise RuntimeError('immutable run directory already contains metadata:'+str(directory))
    path=directory/'case.cir';path.write_text(deck,encoding='ascii')
    command=[str(NG),'-b',str(path)]
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    start=time.perf_counter();started=now();history=directory/'monitor.jsonl';log=directory/'ngspice.log'
    with log.open('wb',buffering=0) as output,history.open('w',encoding='utf-8',buffering=1) as monitor:
        process=subprocess.Popen(command,cwd=directory,stdout=output,stderr=subprocess.STDOUT,env=env)
        observed=psutil.Process(process.pid);last=-5
        while process.poll() is None:
            elapsed=time.perf_counter()-start
            if elapsed-last>=5:
                tail=log.read_bytes()[-16000:].decode('utf-8',errors='replace')
                refs=re.findall(r'Reference value\s*:\s*([\d.eE+-]+)',tail)
                state=dict(utc=now(),elapsed_s=elapsed,pid=process.pid,log_bytes=log.stat().st_size,
                    last_reported_time_s=float(refs[-1]) if refs else None,
                    observed_phase='transient_progress' if refs else 'process_running_phase_unobserved')
                try:state.update(cpu_seconds=sum(observed.cpu_times()[:2]),rss_bytes=observed.memory_info().rss)
                except psutil.Error:pass
                write(directory/'status.json',state);monitor.write(json.dumps(state)+'\n');last=elapsed
                stop_reason=('CANCEL_REQUESTED' if (directory/'cancel.request').exists() else
                    'MEMORY_BUDGET' if max_rss_bytes and state.get('rss_bytes',0)>max_rss_bytes else None)
                if stop_reason:
                    process.kill();process.wait()
                    meta=dict(command=command,execution_status='RESOURCE_ABORT',stop_reason=stop_reason,
                        runtime_s=time.perf_counter()-start,netlist_sha256=sha(path),columns=columns,started_utc=started,
                        last_state=state,log_sha256=sha(log),max_rss_bytes=max_rss_bytes)
                    write(directory/'run.json',meta);raise RuntimeError('live transient stopped by resource/cancel contract:'+str(directory))
            if elapsed>=timeout:
                process.kill();process.wait()
                meta=dict(command=command,execution_status='TIMEOUT',timeout_limit_s=timeout,runtime_s=time.perf_counter()-start,
                    netlist_sha256=sha(path),columns=columns,started_utc=started,log_sha256=sha(log),
                    environment_overrides={k:env[k] for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']})
                write(directory/'run.json',meta);raise RuntimeError('live transient timed out;logs/status retained:'+str(directory))
            time.sleep(.2)
    text=log.read_text(encoding='utf-8',errors='replace')
    meta=dict(command=command,returncode=process.returncode,runtime_s=time.perf_counter()-start,started_utc=started,
        netlist_sha256=sha(path),columns=columns,log_sha256=sha(log),environment_overrides={k:env[k] for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']})
    write(directory/'run.json',meta)
    if process.returncode:raise RuntimeError('live SPICE failed:'+str(directory))
    if not (directory/'trace.txt').exists():raise RuntimeError('SPICE returned without trace:'+str(directory))
    a=np.loadtxt(directory/'trace.txt',skiprows=1)
    assert a.ndim==2 and a.shape[1]==len(columns)+1
    expected=float(re.search(r'\.tran\s+\S+\s+([\d.eE+-]+)n',deck)[1])
    meta.update(trace_sha256=sha(directory/'trace.txt'),expected_stop_ns=expected,actual_stop_ns=float(a[-1,0]*1e9))
    complete=np.isfinite(a).all() and meta['actual_stop_ns']>=expected-1e-5 and 'run simulation(s) aborted' not in text
    meta['execution_status']='complete' if complete else 'TRUNCATED_TRANSIENT';write(directory/'run.json',meta)
    if not complete:raise RuntimeError('live transient incomplete:'+str(directory))
    return a,meta
