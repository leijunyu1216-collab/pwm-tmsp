"""Measured paired startup metrics; unsupported system metrics remain absent."""
import json,re
from pathlib import Path
import numpy as np
import pandas as pd
from benchmarks.physical_common import ROOT,sha,pulse_measure,integrate
from benchmarks.live_spice import write
BASE=ROOT/'artifacts/stable_network_v11';OUT=ROOT/'artifacts/paired_network_v12'
def load(p):
 s=json.loads((p/'stream_schema.json').read_text());b=np.fromfile(p/'transient_stream.f64',dtype='<f8');assert len(b)%s['columns']==0
 a=b.reshape(-1,s['columns']);assert np.isfinite(a).all();names=s['names'];t=a[:,names.index('time')]
 return t,{n:a[:,i] for i,n in enumerate(names)}
def main():
 r=BASE/'stream_stage/n5_prefix500';c=OUT/'stage/candidate21'
 assert json.loads((c/'summary.json').read_text())['simulation_valid']
 t0,x0=load(r);t1,x1=load(c);assert min(t0[-1],t1[-1])>=500e-9*(1-1e-9)
 src0=(BASE/'prefix500/case.cir').read_text();src1=(OUT/'prefix21.cir').read_text()
 sources=lambda s:re.findall(r'(?m)^[VB][^\n]*(?:\n\+[^\n]*)*',s)
 assert sources(src0)==sources(src1),'paired sources must be identical'
 removed=json.loads((BASE/'candidate21/contract.json').read_text())['removed_words']
 nodes=['history'+str(i) for i in range(1,5)]+[f's{st}c{ch}' for st in range(5) for ch in range(5) if f's{st}c{ch}' not in removed]
 rows=[]
 for node in nodes:
  a=pulse_measure(t0*1e9,x0[node],[240]);b=pulse_measure(t1*1e9,x1[node],[240]);valid=a['status'][0]==b['status'][0]=='VALID'
  rows.append(dict(node=node,status29=a['status'][0],status21=b['status'][0],width29_ns=float(a['width'][0]) if a['status'][0]=='VALID' else None,width21_ns=float(b['width'][0]) if b['status'][0]=='VALID' else None,width_delta_ps=float((b['width'][0]-a['width'][0])*1000) if valid else None,rise_delta_ps=float((b['rise'][0]-a['rise'][0])*1000) if valid else None))
 pd.DataFrame(rows).to_csv(OUT/'paired_pulses.csv',index=False)
 energies=[integrate(t,-1.8*x['vdd#branch'],200,400)*1e12 for t,x in [(t0,x0),(t1,x1)]]
 steps=[]
 for t in [t0,t1]:
  ns=t*1e9;dt=np.diff(ns);m=(ns[:-1]>=440)&(ns[1:]<=442);steps.append(dict(points=int(m.sum()),median_step_ps=float(np.median(dt[m])*1000),min_step_ps=float(dt[m].min()*1000)))
 result=dict(complete_prefixes=True,identical_input_and_control_sources=True,common_outputs=len(nodes),status_changes=[v for v in rows if v['status29']!=v['status21']],paired_complete_pulses=[v for v in rows if v['width_delta_ps'] is not None],supply_energy_window_ns=[200,400],supply_energy29_pJ=energies[0],supply_energy21_pJ=energies[1],supply_energy_reduction_pct=(1-energies[1]/energies[0])*100,clock_source_energy=None,clock_energy_reason='reference currents not retained',switch_window_440_442ns=steps,source_sha256=sha(__file__),raw_sha256={'29':sha(r/'transient_stream.f64'),'21':sha(c/'transient_stream.f64')},scope='500ns cold-start schematic comparison; no steady-state/noise/layout/speed acceptance',global_acceptance=False)
 write(OUT/'comparison.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
