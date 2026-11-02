"""Observable ngspice shared-API phases; not a system speed benchmark."""
import argparse
import ctypes as c
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
import numpy as np
import psutil
from benchmarks.physical_common import ROOT,NG,sha,footer
from benchmarks.live_spice import now,write

DLL=NG.parent/'ngspice.dll'
OUT=ROOT/'artifacts/spice_stage_v8'


def read_status_cache(path):
    """A transient Windows sharing violation must not abort a simulation."""
    try:
        value=json.loads(Path(path).read_text())
        return value if isinstance(value,dict) else None
    except (OSError,json.JSONDecodeError):
        return None


def write_status_cache(path,value):
    """Best-effort UI/monitor cache; append-only logs remain authoritative."""
    path=Path(path);temporary=path.with_suffix('.tmp')
    try:
        write(temporary,value);os.replace(temporary,path)
        return True
    except OSError:
        return False


class Values(c.Structure):
    _fields_=[('name',c.c_char_p),('creal',c.c_double),('cimag',c.c_double),('is_scale',c.c_bool),('is_complex',c.c_bool)]
class ValuesAll(c.Structure):
    _fields_=[('veccount',c.c_int),('vecindex',c.c_int),('vecsa',c.POINTER(c.POINTER(Values)))]
class Info(c.Structure):
    _fields_=[('number',c.c_int),('vecname',c.c_char_p),('is_real',c.c_bool),('pdvec',c.c_void_p),('pdvecscale',c.c_void_p)]
class InfoAll(c.Structure):
    _fields_=[('name',c.c_char_p),('title',c.c_char_p),('date',c.c_char_p),('type',c.c_char_p),('veccount',c.c_int),('vecs',c.POINTER(c.POINTER(Info)))]


def worker(config_path):
    config=json.loads(Path(config_path).read_text());folder=Path(config['folder']);os.chdir(folder)
    start=time.perf_counter();mutex=threading.Lock();state=dict(phase='worker_start',elapsed_s=0)
    log=(folder/'callbacks.jsonl').open('w',encoding='utf-8',buffering=1)
    def event(kind,**values):
        with mutex:
            row=dict(utc=now(),elapsed_s=time.perf_counter()-start,event=kind,**values)
            log.write(json.dumps(row)+'\n');state.update(row)
            write_status_cache(folder/'worker_state.json',state)
    event('WORKER_START',phase='dll_load')
    handles=[os.add_dll_directory(str(NG.parent)),os.add_dll_directory(str(NG.parent.parent.parent))]
    lib=c.CDLL(str(DLL));time_index=[None];last_point=[-1.];first=[True]
    stream=(folder/'transient_stream.f64').open('wb') if config.get('stream_vectors') else None
    Char=c.CFUNCTYPE(c.c_int,c.c_char_p,c.c_int,c.c_void_p)
    Exit=c.CFUNCTYPE(c.c_int,c.c_int,c.c_bool,c.c_bool,c.c_int,c.c_void_p)
    Data=c.CFUNCTYPE(c.c_int,c.POINTER(ValuesAll),c.c_int,c.c_int,c.c_void_p)
    Init=c.CFUNCTYPE(c.c_int,c.POINTER(InfoAll),c.c_int,c.c_void_p)
    BG=c.CFUNCTYPE(c.c_int,c.c_bool,c.c_int,c.c_void_p)
    @Char
    def text_callback(message,index,user):
        text=message.decode('utf-8',errors='replace')
        phase=state['phase']
        if 'Initial Transient Solution' in text:phase='initial_transient_solution_report'
        event('OUTPUT',phase=phase,text=text);return 0
    @Char
    def status_callback(message,index,user):
        event('SIMULATOR_STATUS',text=message.decode('utf-8',errors='replace'));return 0
    @Exit
    def exit_callback(status,immediate,quit,index,user):event('CONTROLLED_EXIT',status=status,quit=quit);return 0
    @Init
    def init_callback(pointer,index,user):
        info=pointer.contents
        names=[info.vecs[i].contents.vecname.decode() for i in range(info.veccount)]
        time_index[0]=next((i for i,n in enumerate(names) if n.lower()=='time'),None)
        if stream:write(folder/'stream_schema.json',dict(names=names,dtype='<f8',columns=len(names),scope='raw callback points; incomplete runs remain incomplete'))
        event('INIT_VECTOR_CALLBACK',phase='plot_vectors_initialized',plot_type=info.type.decode(),vector_count=info.veccount,time_index=time_index[0]);return 0
    @Data
    def data_callback(pointer,count,index,user):
        values=pointer.contents
        if stream:
            stream.write(np.array([values.vecsa[i].contents.creal for i in range(values.veccount)],dtype='<f8').tobytes())
        if time_index[0] is not None:
            t=values.vecsa[time_index[0]].contents.creal
            elapsed=time.perf_counter()-start
            if first[0] or elapsed-last_point[0]>=1:
                if stream:stream.flush()
                event('FIRST_TRANSIENT_DATA' if first[0] else 'TRANSIENT_PROGRESS',phase='transient_data_observed',time_s=t,point_index=values.vecindex)
                first[0]=False;last_point[0]=elapsed
        return 0
    @BG
    def bg_callback(running,index,user):event('BG_THREAD_CALLBACK',running=running);return 0
    callbacks=[text_callback,status_callback,exit_callback,data_callback,init_callback,bg_callback]
    lib.ngSpice_Init.argtypes=[Char,Char,Exit,Data,Init,BG,c.c_void_p];lib.ngSpice_Init.restype=c.c_int
    lib.ngSpice_Command.argtypes=[c.c_char_p];lib.ngSpice_Command.restype=c.c_int
    lib.ngSpice_Init(*callbacks,None);event('INIT_RETURN',phase='source_pending')
    def command(command,label):
        event(label+'_ENTER',phase=label.lower(),command=command)
        result=lib.ngSpice_Command(command.encode('ascii'))
        event(label+'_RETURN',returncode=result)
        return result
    # ngspice's shared command parser preserves quotes in a source filename.
    # The workspace paths contain no spaces; fail explicitly if that changes.
    if any(ch.isspace() for ch in config['deck']):
        raise ValueError('source path requires ngspice parser-safe staging')
    if command('source '+config['deck'],'SOURCE'):
        raise RuntimeError('ngspice source failed; see callback log')
    for cmd in config['commands']:
        if cmd.strip().lower()=='quit':continue
        command(cmd,'RUN' if cmd.strip().lower()=='run' else 'CONTROL')
    command('rusage','RUSAGE')
    if stream:stream.close()
    event('WORKER_DONE',phase='complete')


def probe(source,label,timeout=360,retain_all=True,max_rss_bytes=None,stream_vectors=False):
    folder=OUT/label;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'contract.json').exists():raise RuntimeError('immutable probe exists:'+str(folder))
    source=Path(source).resolve();original=source.read_text()
    control=re.search(r'(?ms)^\.control\s*\n(.*?)^\.endc',original)
    assert control
    commands=[line.strip() for line in control[1].splitlines() if line.strip()]
    wr=next(line for line in commands if line.startswith('wrdata '));columns=wr.split()[2:]
    # Preserve the original default vector retention, including currents used
    # by post-run expressions, as well as the physical electrical statements.
    saved='.save all\n' if retain_all else '.save '+' '.join(columns)+'\n'
    deck=original[:control.start()]+saved+original[control.end():]
    path=folder/'source.cir';path.write_text(deck,encoding='ascii')
    cfg=dict(folder=str(folder.resolve()),deck=path.resolve().as_posix(),commands=commands,stream_vectors=stream_vectors)
    config=folder/'worker.json';write(config,cfg)
    write(folder/'contract.json',dict(source=str(source),source_sha256=sha(source),instrumented_deck_sha256=sha(path),dll_sha256=sha(DLL),
        header_sha256=sha(NG.parent.parent/'include/ngspice/sharedspice.h'),timeout_s=timeout,
        changes='control commands invoked individually via shared API;save all original default vectors;electrical statements unchanged',
        scope='stage diagnosis;DLL is separate from CLI timing/qualification',retain_all=retain_all,max_rss_bytes=max_rss_bytes,stream_vectors=stream_vectors,worker_source_sha256=sha(__file__)))
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    command=[sys.executable,'-m','benchmarks.spice_stage_probe','--worker',str(config.resolve())]
    begin=time.perf_counter();final='complete'
    with (folder/'worker_console.log').open('wb') as log,(folder/'process.jsonl').open('w',encoding='utf-8',buffering=1) as monitor:
        process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env)
        tracked=psutil.Process(process.pid);last=-5
        while process.poll() is None:
            elapsed=time.perf_counter()-begin
            if elapsed-last>=5:
                row=dict(utc=now(),elapsed_s=elapsed,pid=process.pid)
                try:row.update(cpu_seconds=sum(tracked.cpu_times()[:2]),rss_bytes=tracked.memory_info().rss)
                except psutil.Error:pass
                cached=read_status_cache(folder/'worker_state.json')
                if cached is not None:row['worker_state']=cached
                else:row['worker_state_cache']='temporarily unavailable; callback/raw logs retained'
                monitor.write(json.dumps(row)+'\n');write_status_cache(folder/'status.json',row);last=elapsed
                if (max_rss_bytes and row.get('rss_bytes',0)>max_rss_bytes) or (folder/'cancel.request').exists():
                    process.kill();process.wait();final='RESOURCE_ABORT';break
            if elapsed>=timeout:process.kill();process.wait();final='TIMEOUT';break
            time.sleep(.2)
    if process.returncode and final not in ['TIMEOUT','RESOURCE_ABORT']:final='WORKER_ERROR'
    rows=[json.loads(line) for line in (folder/'callbacks.jsonl').read_text().splitlines()] if (folder/'callbacks.jsonl').exists() else []
    result=dict(label=label,status=final,returncode=process.returncode,runtime_s=time.perf_counter()-begin,
        source_enter=next((r['elapsed_s'] for r in rows if r['event']=='SOURCE_ENTER'),None),
        source_return=next((r['elapsed_s'] for r in rows if r['event']=='SOURCE_RETURN'),None),
        run_enter=next((r['elapsed_s'] for r in rows if r['event']=='RUN_ENTER'),None),
        first_transient_data=next((r['elapsed_s'] for r in rows if r['event']=='FIRST_TRANSIENT_DATA'),None),
        last_observed_state=rows[-1] if rows else None,callback_count=len(rows))
    result['simulation_valid']=False
    if (folder/'trace.txt').exists():
        a=np.loadtxt(folder/'trace.txt',skiprows=1);result.update(actual_stop_ns=float(a[-1,0]*1e9),trace_sha256=sha(folder/'trace.txt'))
        stop=re.search(r'(?im)^\.tran\s+\S+\s+(\S+)',original)
        units={'f':1e-15,'p':1e-12,'n':1e-9,'u':1e-6,'m':1e-3,'':1.}
        token=re.fullmatch(r'([\d.eE+-]+)([fpnum]?)',stop[1]) if stop else None
        expected=float(token[1])*units[token[2]] if token else None
        result['expected_stop_ns']=expected*1e9 if expected else None
        result['simulation_valid']=bool(expected and a[-1,0]>=expected*(1-1e-8) and np.isfinite(a).all() and result['first_transient_data'] is not None and final=='complete')
    if final=='complete' and not result['simulation_valid']:result['status']='INVALID_COMPLETION'
    write(folder/'summary.json',result);print(json.dumps(result,indent=2),flush=True)
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--worker');ap.add_argument('--label');ap.add_argument('--source');ap.add_argument('--timeout',type=int,default=360);ap.add_argument('--rc',action='store_true');args=ap.parse_args()
    if args.worker:worker(args.worker)
    elif args.rc:
        folder=OUT/'rc_input';folder.mkdir(parents=True,exist_ok=True)
        path=folder/'case.cir';path.write_text('* callback wiring check\nVin in 0 PULSE(0 1 10n .2n .2n 20n 40n)\nR1 in out 1k\nC1 out 0 1p\n'+footer(100,['v(in)','v(out)'],.1),encoding='ascii')
        probe(path,args.label or 'rc_pilot_v2',60)
    else:probe(args.source,args.label,args.timeout)
