"""Existing port/current gates plus the actual capture-cap switch phase."""
import argparse,json
import numpy as np
from benchmarks.network_capture_cap_v36 import OUT
from benchmarks.physical_common import edges,sha
from benchmarks.live_spice import write

def assess(label,live=False):
    from benchmarks import assess_wide_v34 as base
    old=base.OUT
    try:
        base.OUT=OUT;result=base.assess(label,live);t,z=base.read(label)
    finally:base.OUT=old
    rows=[];cache={}
    def near(name,pol,target):
        if name not in cache:cache[name]=edges(t,z[name])
        a=cache[name][pol];a=a[abs(a-target)<2]
        return float(a[np.argmin(abs(a-target))]) if len(a) else None
    for phase in result['timings']:
        name=phase['node']
        if name=='front':continue
        b=phase['bank'];F=phase['capture_reference_phase_ns'];prefix=f'x{name}.xbank{b}'
        off=near(f'x{name}.captrack{b}',1,F+.8);bar=near(prefix+'.capbar',0,F+.8)
        completed=phase['phase_evaluation_complete'];capend=phase['data_control_fall_ns'];read=phase['replay_control_rise_ns'];cf=phase['reference_decision_ns']
        after_capture=min(off,bar)-capend if None not in [off,bar,capend] else None
        before_replay=read-max(off,bar) if None not in [off,bar,read] else None
        gate_n=gate_p=None;current_peak=None;aux_drift=None;on_current=None;off_fraction=None
        if read is not None and cf is not None:
            m=(t>=read+.1)&(t<=cf+.1)
            if m.any():
                values=np.vstack([z[prefix+'.'+k][m] for k in ['data','reference','auxdata','auxref']])
                gate_n=float(np.min(np.min(values,axis=0)-z[f'x{name}.captrack{b}'][m]))
                gate_p=float(np.min(z[prefix+'.capbar'][m]-np.max(values,axis=0)))
                current=sum(abs(z[f'@mflat_{prefix}.xcap{branch}.x{kind}[id]']) for branch in ['d','r'] for kind in ['n','p'])
                current_peak=float(max(current[m]));aux_drift={k:float(np.ptp(z[prefix+'.'+k][m])) for k in ['auxdata','auxref']}
                active=(t>=F-222.1)&(t<=F-221.6)
                if active.any():
                    on_current=float(np.median(current[active]));off_fraction=current_peak/on_current if on_current>0 else None
        margin_ok=all(v is not None and v>0 for v in [after_capture,before_replay])
        no_enhancement=bool(gate_n is not None and gate_p is not None and gate_n>=0 and gate_p>=0)
        # Zero Vgs is sufficient, not necessary, for an enhancement MOS to be
        # off. Retain it as a diagnostic. The physical off criterion uses the
        # same1% local on-current threshold as the original phase contract.
        current_off=bool(off_fraction is not None and off_fraction<=.01)
        rows.append(dict(node=name,sample=phase['sample'],phase_complete=completed,cap_disconnect_after_capture_ns=after_capture,
            replay_after_cap_disconnect_ns=before_replay,N_zero_overdrive_margin_V=gate_n,P_zero_overdrive_margin_V=gate_p,
            zero_overdrive_sufficient_certificate=no_enhancement,aux_switch_on_current_A=on_current,useful_read_off_current_fraction=off_fraction,
            off_current_fraction_bound=.01,useful_read_switch_drain_current_peak_A=current_peak,useful_read_aux_voltage_span_V=aux_drift,
            gate_pass=bool(completed and margin_ok and current_off)))
    required_all=bool(rows and all(r['gate_pass'] for r in rows));completed=[r for r in rows if r['phase_complete']]
    base_pass=result['gate_pass'];result.update(auxiliary_switch_timings=rows,base_gate_pass=base_pass,
        auxiliary_switch_gate_pass=required_all,auxiliary_switch_failures=[r for r in completed if not r['gate_pass']],
        switch_current_scope='sum of MOS drain currents;not complete displacement-charge decomposition;1% local on-current criterion;zero-overdrive is an additional sufficient diagnostic',
        zero_overdrive_certificate_failures=[r for r in completed if not r['zero_overdrive_sufficient_certificate']],
        gain_control_scope='constant-capacitance proportional-field ratio;parasitic and switch-boundary residuals are measured,not assumed zero',
        capture_analysis_sha256=sha(__file__))
    result['gate_pass']=bool(base_pass and required_all)
    target=OUT/(label+('_live' if live else '_assessment')+'.json');write(target,result)
    print(dict(base_gate_pass=base_pass,auxiliary_switch_gate_pass=required_all,completed_auxiliary_phases=len(completed),auxiliary_failures=len(result['auxiliary_switch_failures']),gate_pass=result['gate_pass']))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('label',nargs='?',default='capture15ff_ref13434_run1');p.add_argument('--live',action='store_true');a=p.parse_args();assess(a.label,a.live)
