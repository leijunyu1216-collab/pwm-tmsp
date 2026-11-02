"""SPICE sources, measured energy and cell-area adapters for physical campaigns."""
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path
import numpy as np
from tmsp.cards import ROOT

TB=ROOT/'design/baseline/testbench'
PDK=ROOT/'design/pdk/skywater-pdk-libs-sky130_fd_pr-main'
NG=ROOT/'design/tools/ngspice/Library/bin/ngspice_con.exe'
SC=ROOT/'third_party/sky130_fd_sc_hd'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def standard_cells(folder):
    """Official extracted transistor topology; convert micron-valued W/L to SI.

    Upstream '650000u' evaluates to .65 in its scale=1u convention. Existing
    local device models use SI lengths; .65um becomes 6.5e-7m explicitly.
    """
    folder.mkdir(parents=True,exist_ok=True);parts=[];areas={};sources={}
    for kind in ['and2','or2']:
        path=SC/f'cells/{kind}/sky130_fd_sc_hd__{kind}_1.spice'
        raw=path.read_text();sources[str(path.relative_to(ROOT))]=sha(path)
        def units(m):return m[1]+'='+format(float(m[2])*1e-12,'.12g')
        converted=re.sub(r'\b(w|l)=([\deE+.-]+)u\b',units,raw)
        assert len(re.findall(r'\bl=1\.5e-07',converted))==6
        parts.append(converted)
        lef=SC/f'cells/{kind}/sky130_fd_sc_hd__{kind}_1.lef'
        x,y=map(float,re.search(r'SIZE\s+([\d.]+)\s+BY\s+([\d.]+)',lef.read_text()).groups())
        lib=SC/f'cells/{kind}/sky130_fd_sc_hd__{kind}_1__tt_025C_1v80.lib.json'
        area=json.loads(lib.read_text())['area'];assert abs(x*y-area)<1e-9
        areas[kind]=area
        sources[str(lef.relative_to(ROOT))]=sha(lef);sources[str(lib.relative_to(ROOT))]=sha(lib)
    path=folder/'standard_cells_si.cir';path.write_text('\n'.join(parts),encoding='ascii')
    meta=dict(area_um2=areas,source_sha256=sources,unit_conversion='original W,L in microns -> explicit SI meters;scale=1;topology and model names unchanged',
        license='Apache-2.0',local_adapted_sha256=sha(path))
    (folder/'standard_cells.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    return path,meta


def header(sc_path=None):
    s=f'''* Physical TMSP experiment, SKY130 tt/27C/1.8V; schematic, no PEX
.lib "{(TB/'sky130_tmsp.lib.spice').as_posix()}" tt
.include "{(TB/'cells.cir').as_posix()}"
.temp 27
Vdd vdd 0 1.8
Vss vss 0 0
'''
    if sc_path:
        # The upstream corner file includes its pm3 model internally.
        for suffix in ['mismatch.corner','tt.corner']:
            s+=f'.include "{(PDK/f"cells/pfet_01v8_hvt/sky130_fd_pr__pfet_01v8_hvt__{suffix}.spice").as_posix()}"\n'
        s+=f'.include "{sc_path.as_posix()}"\n'
    return s


def pwl(name,node,starts,widths,slew=.2,high=1.8):
    pts=['0 0']
    for start,width in zip(starts,widths):
        pts.append(f'{start:.9f}n 0 {start+slew:.9f}n {high} {start+slew+width:.9f}n {high} {start+2*slew+width:.9f}n 0')
    return f'V{name} {node} 0 PWL('+ '\n+ '.join(pts)+')\n'


def footer(stop,columns,step=.2):
    return f'''.tran {step}n {stop:.9f}n 0 {step}n
.control
set wr_singlescale
set wr_vecnames
set numdgt=17
run
wrdata trace.txt {' '.join(columns)}
quit
.endc
.end
'''


def run(directory,deck,columns,timeout=240):
    directory=Path(directory).resolve()
    directory.mkdir(parents=True,exist_ok=True);path=directory/'case.cir';path.write_text(deck,encoding='ascii')
    command=[str(NG),'-b',str(path)];start=time.perf_counter()
    try:
        result=subprocess.run(command,cwd=directory,capture_output=True,text=True,timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        decode=lambda x:x.decode('utf-8',errors='replace') if isinstance(x,bytes) else (x or '')
        (directory/'ngspice.log').write_text(decode(exc.stdout)+decode(exc.stderr),encoding='utf-8')
        meta=dict(command=command,execution_status='TIMEOUT',timeout_limit_s=timeout,
            runtime_s=time.perf_counter()-start,netlist_sha256=sha(path),columns=columns)
        (directory/'run.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
        raise RuntimeError(f'transient timed out:{directory};partial evidence retained') from exc
    elapsed=time.perf_counter()-start
    (directory/'ngspice.log').write_text(result.stdout+result.stderr,encoding='utf-8')
    meta=dict(command=command,returncode=result.returncode,runtime_s=elapsed,netlist_sha256=sha(path),columns=columns)
    (directory/'run.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    if result.returncode:raise RuntimeError(f'failed:{directory};see ngspice.log')
    a=np.loadtxt(directory/'trace.txt',skiprows=1)
    if a.shape[1]!=len(columns)+1:raise AssertionError('trace schema')
    expected=float(re.search(r'\.tran\s+\S+\s+([\d.eE+-]+)n',deck)[1])
    meta.update(trace_sha256=sha(directory/'trace.txt'),expected_stop_ns=expected,actual_stop_ns=float(a[-1,0]*1e9))
    complete=np.isfinite(a).all() and meta['actual_stop_ns']>=expected-1e-5 and 'run simulation(s) aborted' not in result.stdout+result.stderr
    meta['execution_status']='complete' if complete else 'TRUNCATED_TRANSIENT'
    (directory/'run.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    if not complete:raise RuntimeError(f'transient did not complete:{directory};exitcode alone is insufficient')
    return a,meta


def edges(t,v,threshold=.9):
    found=[]
    for rising in [True,False]:
        ids=np.flatnonzero(((v[:-1]<threshold)&(v[1:]>=threshold)) if rising else ((v[:-1]>threshold)&(v[1:]<=threshold)))
        found.append(t[ids]+(threshold-v[ids])*(t[ids+1]-t[ids])/(v[ids+1]-v[ids]))
    return found


def pulse_measure(t,v,starts,period=200):
    r,f=edges(t,v);width=np.full(len(starts),np.nan);rise=width.copy();fall=width.copy();status=[]
    for k,start in enumerate(starts):
        rr=r[(r>=start)&(r<start+period)];ff=f[(f>=start)&(f<start+period)]
        ok=len(rr)==len(ff)==1 and ff[0]>rr[0]
        status.append('VALID' if ok else 'EDGE_COUNT_FAILURE')
        if ok:rise[k]=rr[0];fall[k]=ff[0];width[k]=ff[0]-rr[0]
    return dict(width=width,rise=rise,fall=fall,status=np.array(status),total_edges=[len(r),len(f)])


def integrate(t_seconds,value,start_ns,stop_ns):
    a,b=start_ns*1e-9,stop_ns*1e-9
    mask=(t_seconds>a)&(t_seconds<b)
    grid=np.r_[a,t_seconds[mask],b];v=np.r_[np.interp(a,t_seconds,value),value[mask],np.interp(b,t_seconds,value)]
    trapezoid=np.trapezoid if hasattr(np,'trapezoid') else np.trapz
    return float(trapezoid(v,grid))


def energy(a,column,start_ns,stop_ns,cycles,vdd=1.8):
    # Current sign is the SPICE voltage-source reference direction.
    joules=integrate(a[:,0],-vdd*a[:,column],start_ns,stop_ns)
    return dict(energy_J=joules,energy_per_sample_fJ=joules/cycles*1e15,mean_power_uW=joules/((stop_ns-start_ns)*1e-9)*1e6)
