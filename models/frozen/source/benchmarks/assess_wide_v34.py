"""Port envelopes and measured current/comparator handoff margins.

Margins are evaluated on the measured physical input and local controls.
Source MOS drain currents are not a full displacement-charge decomposition.
"""
import argparse,json
import numpy as np
from benchmarks.wide_network_v34 import OUT
from benchmarks.physical_common import edges,sha
from benchmarks.live_spice import write

def read(label):
    p=OUT/'stage'/label;s=json.loads((p/'stream_schema.json').read_text());raw=p/'transient_stream.f64'
    n=raw.stat().st_size//(8*s['columns']);a=np.memmap(raw,dtype='<f8',mode='r',shape=(n,s['columns']));z={n:a[:,i] for i,n in enumerate(s['names'])}
    return z['time']*1e9,z

def assess(label,live=False):
    c=json.loads((OUT/'inputs'/label/'contract.json').read_text());folder=OUT/'stage'/label
    t,z=read(label);cache={};rows=[];timings=[];envelopes=[]
    def crossings(name):
        if name not in cache:cache[name]=edges(t,z[name])
        return cache[name]
    def pulse(name,left,right):
        r,f=crossings(name);rr=r[(r>=left)&(r<right)];ff=f[(f>=left)&(f<right)]
        return (float(rr[0]),float(ff[0])) if len(rr)==len(ff)==1 and ff[0]>rr[0] else None
    def nearest(name,pol,expected,tol=2.5):
        a=crossings(name)[pol];a=a[abs(a-expected)<=tol]
        return float(a[np.argmin(abs(a-expected))]) if len(a) else None
    def on_level(current,centers):
        levels=[]
        for center in centers:
            m=(t>=center+.35)&(t<=center+.8)
            if m.any():levels.append(float(np.median(current[m])))
        return max(levels,default=0.)
    def off_gap(current,level,off,on,fraction):
        if off is None or on is None or level<=0:return None
        m=(t>=min(off,on)-1)&(t<=max(off,on)+1.5);tt=t[m];v=current[m]
        up,down=edges(tt,v,level*fraction)
        down=down[(down>=off-.3)&(down<=off+1.3)];up=up[(up>=on-.3)&(up<=on+1.3)]
        if not len(down) or not len(up):return None
        end=float(down[np.argmin(abs(down-off))]);start=float(up[np.argmin(abs(up-on))])
        return dict(off_ns=end,on_ns=start,gap_ns=start-end)
    def decision(name,reset_end,replay_end):
        if reset_end is None or replay_end is None:return None
        r,_=crossings(name);r=r[(r>=reset_end+.5)&(r<replay_end)]
        return float(r[0]) if len(r)==1 else None
    for name,w in c['words'].items():
        input_widths=[];output_widths=[];cap_widths=[]
        for k,expected in enumerate(w['expected']):
            F=w['phase']+.1+200*k;outF=F+200
            if live and t[-1]<outF+1:continue
            parent=pulse(w['input'],39+200*k,239+200*k) if name=='front' else pulse(w['input'],F-199,F+1)
            local=pulse(f'x{name}.captured_input',39+200*k,239+200*k) if name=='front' else pulse(f'x{name}.captured_input',F-199,F+1)
            out=pulse(name,outF-199,outF+1);ok=bool(out and parent and local)
            b=k%3;prefix=f'x{name}.xbank{b}'
            row=dict(node=name,sample=k,expected_ns=expected,valid=ok,
                width_error_ns=float(out[1]-out[0]-expected) if out else None,anchor_error_ns=float(out[1]-outF) if out else None,
                input_width_ns=float(parent[1]-parent[0]) if parent else None,receiver_width_ns=float(local[1]-local[0]) if local else None,
                output_width_ns=float(out[1]-out[0]) if out else None,window_covered=bool(t[-1]>=outF+1))
            row['port_within_bounds']=bool(ok and abs(row['width_error_ns'])<=2 and abs(row['anchor_error_ns'])<=.2)
            rows.append(row)
            if parent:input_widths.append(parent[1]-parent[0])
            if out:output_widths.append(out[1]-out[0])
            ped_fall=nearest(f'x{name}.ped{b}',1,F-(220.4 if name=='front' else 196.4))
            ped_rise=nearest(f'x{name}.ped{b}',0,F-(244.6 if name=='front' else 220.6))
            reset_fall=nearest(f'x{name}.reset{b}',1,F-(245.8 if name=='front' else 225.8))
            replay_rise=nearest(f'x{name}.replay{b}',0,F+(.2 if name=='front' else 1.2))
            replay_fall=nearest(f'x{name}.replay{b}',1,F+(350.4 if name=='front' else 351.4))
            aa_rise=nearest(prefix+'.aa',0,local[0]) if local else None
            aa_fall=nearest(prefix+'.aa',1,local[1]) if local else None
            if aa_rise is not None and aa_fall is not None:cap_widths.append(aa_fall-aa_rise)
            en_rise=nearest(f'x{name}.en{b}',0,F-(221. if name=='front' else 200.6))
            en_fall=nearest(f'x{name}.en{b}',1,F-(21.2 if name=='front' else -.8))
            data_cur=abs(z[f'@mflat_{prefix}.xca.xsrc[id]'])+abs(z[f'@mflat_{prefix}.xcb.xsrc[id]'])
            ref_cur=abs(z[f'@mflat_{prefix}.xcr.xsrc[id]'])
            reset_cur=abs(z[f'@mflat_{prefix}.xresetd[id]'])+abs(z[f'@mflat_{prefix}.xresetr[id]'])
            centers=[v for v in [ped_rise,aa_rise,replay_rise] if v is not None];level=on_level(data_cur,centers)
            gaps=[]
            for frac in c['gate']['current_threshold_fractions']:
                gaps.append(dict(fraction=frac,precharge_to_data=off_gap(data_cur,level,ped_fall,aa_rise,frac),capture_to_replay=off_gap(data_cur,level,aa_fall,replay_rise,frac)))
            cd=decision(prefix+'.cd',reset_fall,replay_fall);cf=decision(prefix+'.cf',reset_fall,replay_fall)
            next_reset=nearest(f'x{name}.reset{b}',0,F+600-(249 if name=='front' else 229),tol=2.5)
            reuse=[]
            if next_reset is not None and replay_rise is not None:
                source=np.maximum(data_cur,ref_cur);lev=max(level,on_level(ref_cur,[replay_rise]));m=(t>=next_reset-.5)&(t<=next_reset+1);peak=float(max(reset_cur[m]))
                for frac in c['gate']['current_threshold_fractions']:
                    mm=(t>=replay_rise)&(t<=next_reset+1);tt=t[mm];up,down=edges(tt,source[mm],frac*lev)
                    _,rdown=edges(tt,reset_cur[mm],frac*peak);rup,_=edges(tt,reset_cur[mm],frac*peak)
                    starts=rup[(rup>=next_reset-.5)&(rup<=next_reset+.7)];start=float(starts[0]) if len(starts) else None
                    ends=down[down<(start if start is not None else next_reset)]
                    end=float(ends[-1]) if len(ends) else None
                    reuse.append(dict(fraction=frac,replay_current_end_ns=end,next_reset_current_on_ns=start,gap_ns=start-end if start is not None and end is not None else None))
            timing=dict(node=name,sample=k,bank=b,capture_reference_phase_ns=F,local_on_current_A=level,
                precharge_control_fall_ns=ped_fall,data_control_rise_ns=aa_rise,data_control_fall_ns=aa_fall,
                replay_control_rise_ns=replay_rise,replay_control_fall_ns=replay_fall,
                precharge_to_data_control_margin_ns=aa_rise-ped_fall if aa_rise is not None and ped_fall is not None else None,
                capture_to_replay_control_margin_ns=replay_rise-aa_fall if replay_rise is not None and aa_fall is not None else None,
                capture_enable_start_margin_ns=local[0]-en_rise if local and en_rise is not None else None,
                capture_enable_end_margin_ns=en_fall-local[1] if local and en_fall is not None else None,
                data_decision_ns=cd,reference_decision_ns=cf,
                data_decision_after_replay_ns=cd-replay_rise if cd is not None and replay_rise is not None else None,
                reference_after_data_decision_ns=cf-cd if cf is not None and cd is not None else None,
                replay_end_after_reference_ns=replay_fall-cf if replay_fall is not None and cf is not None else None,
                current_gaps=gaps,reuse_observed=next_reset is not None,reuse_control_margin_ns=next_reset-replay_fall if next_reset is not None and replay_fall is not None else None,reuse_current_gaps=reuse,
                data_voltage_at_replay_V=float(np.interp(replay_rise,t,z[prefix+'.data'])) if replay_rise is not None else None,
                reference_voltage_at_replay_V=float(np.interp(replay_rise,t,z[prefix+'.reference'])) if replay_rise is not None else None)
            required=['capture_enable_start_margin_ns','capture_enable_end_margin_ns','data_decision_after_replay_ns','reference_after_data_decision_ns','replay_end_after_reference_ns']
            timing['control_and_decision_margins_positive']=all(timing[key] is not None and timing[key]>0 for key in required)
            timing['current_gaps_positive']=all(g[which] is not None and g[which]['gap_ns']>0 for g in gaps for which in ['precharge_to_data','capture_to_replay'])
            timing['reuse_margins_positive']=bool(next_reset is None or (timing['reuse_control_margin_ns'] is not None and timing['reuse_control_margin_ns']>0 and all(g['gap_ns'] is not None and g['gap_ns']>0 for g in reuse)))
            required_end=F+(352. if name=='front' else 372.) if k+3<w['count'] else F+(351.4 if name=='front' else 352.4)
            timing['phase_evaluation_complete']=bool(t[-1]>=required_end)
            timing['phase_gate_pass']=bool(timing['phase_evaluation_complete'] and timing['control_and_decision_margins_positive'] and timing['current_gaps_positive'] and timing['reuse_margins_positive']);timings.append(timing)
        def span(v):return [float(min(v)),float(max(v))] if v else None
        envelopes.append(dict(node=name,input_width_ns=span(input_widths),receiver_width_ns=span([r['receiver_width_ns'] for r in rows if r['node']==name and r['receiver_width_ns'] is not None]),capture_control_width_ns=span(cap_widths),output_width_ns=span(output_widths),
            actual_inputs_outside_prior_card=sum(v<4 or v>196 for v in input_widths)))
    summary=json.loads((folder/'summary.json').read_text()) if (folder/'summary.json').exists() else dict(simulation_valid=False,status='running')
    all_events=all(len(crossings(node)[0])==len(crossings(node)[1])==w['count'] for node,w in c['words'].items())
    med=[r for r in rows if r['node']=='s4c2'];good=[r for r in rows if r['valid']]
    phase_fail=[dict(node=r['node'],sample=r['sample'],control=r['control_and_decision_margins_positive'],current=r['current_gaps_positive'],reuse=r['reuse_margins_positive']) for r in timings if r['phase_evaluation_complete'] and not r['phase_gate_pass']]
    out=dict(label=label,scope=c['scope'],execution=summary['status'],simulation_valid=summary['simulation_valid'],last_ns=float(t[-1]),expected_outputs=c['expected_initialized_outputs'],evaluated_outputs=len(rows),complete_outputs=len(good),
        event_count_exact=all_events,port_gate_pass=bool(summary['simulation_valid'] and len(rows)==c['expected_initialized_outputs'] and all_events and all(r['port_within_bounds'] for r in rows)),
        all_measured_phase_margins_positive=all(r['phase_gate_pass'] for r in timings),
        max_width_error_ns=max((abs(r['width_error_ns']) for r in good),default=None),max_anchor_error_ns=max((abs(r['anchor_error_ns']) for r in good),default=None),
        median_complete=sum(r['valid'] for r in med),median_endpoint4_covered=any(r['expected_ns']==4 and r['valid'] for r in med),median_endpoint196_covered=any(r['expected_ns']==196 and r['valid'] for r in med),
        phase_failures=phase_fail,envelopes=envelopes,rows=rows,timings=timings,unobserved_final_reuses=sum(not x['reuse_observed'] for x in timings),
        independent_seed_opened=False,analysis_source_sha256=sha(__file__),raw_sha256=sha(folder/'transient_stream.f64') if not live else None,global_acceptance=False)
    out['gate_pass']=bool(out['port_gate_pass'] and out['all_measured_phase_margins_positive'] and out['median_endpoint4_covered'] and out['median_endpoint196_covered'])
    write(OUT/(label+('_live' if live else '_assessment')+'.json'),out)
    print({k:v for k,v in out.items() if k not in ['envelopes','rows','timings','phase_failures']});print('phase failures',phase_fail[:12]);return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('label',nargs='?',default='endpoint_development12');p.add_argument('--live',action='store_true');a=p.parse_args();assess(a.label,a.live)
