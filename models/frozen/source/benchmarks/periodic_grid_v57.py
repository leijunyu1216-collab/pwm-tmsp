"""Constant-size clock-only periodic numerical mesh,qualified before freezing."""
import argparse,json,os,re,subprocess,time
import numpy as np
from benchmarks.retimed_run_v56 import OUT
from benchmarks.physical_common import ROOT,edges
from benchmarks.streaming_cli_probe_v47 import EXE,raw_read
from benchmarks.live_spice import now,write


def grid(s):
    pattern=r'(?m)^V\S+_reset\d \S+_drive 0 PULSE\(0 1.8 ([\d.]+)n .2n .2n ([\d.]+)n 600n (\d+)\)$'
    clocks=list(re.finditer(pattern,s))
    if len(clocks)!=18:raise ValueError('reset inventory changed')
    phases={round(float(m[1])*1000)%200000 for m in clocks};ticks={0,200000}
    for p in phases:ticks.update(p+1+j*8 for j in range(101))
    if max(ticks)>200000:raise ValueError('periodic grid interval wraps')
    text='Vnumerical_grid numerical_grid 0 PWL(\n'+'\n'.join(f'+ {t/1000:.6f}n 0' for t in sorted(ticks))+' ) R=0\nRnumerical_grid numerical_grid 0 1\n'
    return text,dict(period_ns=200,reset_phase_ps=sorted(phases),points_per_period=len(ticks),mesh_ps=8,uses_input_widths=False)


def toy():
    folder=ROOT/'artifacts/campaign_qualification_v57/pwl_repeat_toy';folder.mkdir(parents=True,exist_ok=False)
    raw=folder/'trace.raw';log=folder/'ngspice.log';net=folder/'case.cir'
    net.write_text('* native PWL repeat semantic test\nVg g 0 PWL(0 0 4n 1.8 8n 0 10n 0) R=0\nRg g 0 1k\n.save v(g)\n.tran .125n 100n 0 .125n\n.control\nset klu\nrun '+raw.as_posix()+'\nquit\n.endc\n.end\n',encoding='ascii')
    env=os.environ.copy();env['PATH']=str(EXE.parent)+os.pathsep+'D:/MinGW-w64/mingw64/bin'+os.pathsep+env['PATH']
    start=time.perf_counter();p=subprocess.run([str(EXE),'-b','-n','-o',str(log),str(net)],env=env,cwd=folder,capture_output=True,timeout=20)
    r=dict(utc=now(),returncode=p.returncode,runtime_s=time.perf_counter()-start,gate_pass=False,new_seed_opened=False)
    if p.returncode==0 and raw.exists():
        a,n=raw_read(raw);t=a[:,n.index('time')]*1e9;rr,ff=edges(t,a[:,n.index('v(g)')])
        r.update(rises=rr.tolist(),falls=ff.tolist(),stop_ns=float(t[-1]))
        r['gate_pass']=bool(len(rr)==len(ff)==10 and np.allclose(rr,2+10*np.arange(10),atol=1e-8,rtol=0) and np.allclose(ff,6+10*np.arange(10),atol=1e-8,rtol=0))
    write(folder/'summary.json',r);print(json.dumps(r,indent=2))
    if not r['gate_pass']:raise RuntimeError('native repeated PWL semantics failed')


def build():
    if not json.loads((ROOT/'artifacts/campaign_qualification_v57/pwl_repeat_toy/summary.json').read_text())['gate_pass']:raise ValueError('repeat syntax unqualified')
    folder=OUT/'periodic_grid';folder.mkdir(exist_ok=False);source=OUT/'tail_development/case.cir';s=source.read_text();helper,meta=grid(s)
    at=s.index('.tran');(folder/'case.cir').write_text(s[:at]+helper+s[at:],encoding='ascii')
    write(folder/'contract.json',dict(utc=now(),**meta,physical_circuit_unchanged=True,new_seed_opened=False,qualification_pending=True))
    print(meta)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['toy','build']);a=p.parse_args();toy() if a.mode=='toy' else build()
