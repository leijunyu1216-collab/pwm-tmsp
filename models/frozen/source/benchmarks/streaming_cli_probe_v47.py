"""Bounded RC feasibility test for ngspice -r on the bundled KLU build.

This is a small engineering pilot, not an acceptance test or a physical-system
speed benchmark. The transistor network requires its own same-input comparison.
"""
import json,os,re,subprocess,time
from pathlib import Path
import numpy as np
import psutil
from benchmarks import spice_stream_probe_v34 as probe
from benchmarks.physical_common import ROOT
from benchmarks.live_spice import now,write

OUT=ROOT/'artifacts/efficiency_probe_v47'
EXE=ROOT/'design/tools/ngspice_klu/ucrt64/bin/ngspice.exe'
VECTORS=['v(in)','v(out)','i(vin)']
BODY=('streaming rc pilot\nVin in 0 PULSE(0 1.8 1n .1n .1n 4n 10n)\n'
      'R1 in out 1k\nC1 out 0 1p\n'
      '.options method=gear maxord=2 reltol=1e-4 vntol=1e-7 abstol=1e-13 chgtol=1e-16\n')


def raw_read(path):
    with path.open('rb') as f:
        header=b''
        while not header.endswith(b'Binary:\n'):
            part=f.read(1)
            if not part or len(header)>100_000:raise ValueError('Missing binary raw header')
            header+=part
        offset=f.tell()
    text=header.decode('ascii');columns=int(re.search(r'No. Variables:\s*(\d+)',text)[1])
    points=int(re.search(r'No. Points:\s*(\d+)',text)[1])
    names=re.findall(r'(?m)^\s*\d+\s+(\S+)\s+\S+\s*$',text)
    assert len(names)==columns and path.stat().st_size-offset==points*columns*8
    return np.memmap(path,dtype='<f8',mode='r',offset=offset,shape=(points,columns)),names


def cli(name,stop_ns):
    folder=OUT/name;assert not folder.exists();folder.mkdir(parents=True)
    net=folder/'case.cir';raw=(folder/'transient.raw').resolve();log=(folder/'ngspice.log').resolve()
    net.write_text(BODY+f'.save {" ".join(VECTORS)}\n.tran 0.01n {stop_ns}n 0 0.01n\n'
        '.control\nset klu\nset num_threads=8\n.endc\n.end\n',encoding='ascii')
    env=os.environ.copy();env['PATH']=str(EXE.parent)+os.pathsep+'D:/MinGW-w64/mingw64/bin'+os.pathsep+env['PATH']
    env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    command=[str(EXE),'-b','-n','-r',str(raw),'-o',str(log),str(net.resolve())]
    start=time.perf_counter();proc=subprocess.Popen(command,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW);tracked=psutil.Process(proc.pid);peak=0
    while proc.poll() is None:
        try:peak=max(peak,tracked.memory_info().rss)
        except psutil.Error:pass
        if time.perf_counter()-start>120:proc.kill();proc.wait();raise TimeoutError(name)
        time.sleep(.01)
    elapsed=time.perf_counter()-start
    logtext=log.read_text(errors='replace');assert proc.returncode==0 and 'Using KLU as Direct Linear Solver' in logtext
    data,names=raw_read(raw);t=data[:,names.index('time')]*1e9
    assert np.isfinite(data).all() and (np.diff(t)>=0).all() and abs(t[-1]-stop_ns)<1e-6
    row=dict(name=name,stop_ns=stop_ns,rows=len(data),columns=len(names),raw_bytes=raw.stat().st_size,
        peak_RSS_bytes=peak,wall_s=elapsed,raw_completed=True,klu_selected=True,
        binary_raw_streaming_route=True,code_model_load_warnings='code models not needed in RC;real-network dependencies must be checked')
    write(folder/'summary.json',row);return row,data,names


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    short,data,names=cli('rc_short',100.)
    folder=OUT/'dll_input';assert not folder.exists();folder.mkdir()
    deck=folder/'case.cir';deck.write_text(BODY+'.tran 0.01n 100n 0 0.01n\n.control\nrun\nwrdata trace.txt '
        +' '.join(VECTORS)+'\n.endc\n.end\n',encoding='ascii')
    probe.OUT=OUT/'dll_stage';summary=probe.probe(deck,'rc_short',30,retain_all=False,
        max_rss_bytes=300*2**20,stream_vectors=True,num_threads=8,raw_only=True)
    assert summary['simulation_valid']
    stage=probe.OUT/'rc_short';schema=json.loads((stage/'stream_schema.json').read_text())
    dll=np.memmap(stage/'transient_stream.f64',dtype='<f8',mode='r').reshape(-1,schema['columns'])
    translate=lambda x:x[2:-1] if x.startswith('v(') else x[2:-1]+'#branch' if x.startswith('i(') else x
    indices=[schema['names'].index(translate(x)) for x in names]
    same_shape=data.shape==dll.shape
    exact=bool(same_shape and np.array_equal(data.view(np.uint64),dll[:,indices].view(np.uint64)))
    max_delta=float(np.max(np.abs(data-dll[:,indices]))) if same_shape else None
    long,_,_=cli('rc_long',10000.)
    result=dict(utc=now(),scope='RC toy only;not transistor-network equivalence or system speed',short=short,long=long,
        same_small_RC_shape=same_shape,CLI_and_DLL_all_values_bitwise_identical=exact,
        max_value_delta_in_native_units=max_delta,physical_network_tested=False,
        memory_conclusion='measured CLI peak RSS at two durations;does not establish full network peak',
        next_gate='on frozen physical network: same stimuli/models/solver settings, compare completed outputs and internal observations, timing, memory and hashes before promotion')
    write(OUT/'summary.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
