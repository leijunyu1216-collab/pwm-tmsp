"""Independent, fail-closed system-port scoring and immutable-file checks.

Only completed physical edge observations enter the scorer. Golden widths are
used here after simulation, never to drive output nodes or update parameters.
"""
import hashlib
import re
import numpy as np


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def verify_frozen(root, files):
    for name, expected in files.items():
        path=root/name
        if not path.is_file() or digest(path)!=expected:
            raise ValueError(f'Frozen file changed: {name}')


def electrical_body(text):
    body=text.split('.tran')[0]
    return re.sub(r'(?m)^V(?:input tin 0 PWL\([^\n]*(?:\n\+[^\n]*)*\)|\S+ \S+_drive 0 PULSE\([^\n]*\))\n','',body)


def independent_medians(widths):
    # Deliberately independent of the network generator and behavioral model.
    values=list(widths)
    return [sorted(values[k:k+5])[2] for k in range(len(values)-4)]


def evaluate(contract, observations, last_ns, simulation_valid):
    widths=contract['input_widths_ns']
    expected=independent_medians(widths)
    if not np.allclose(expected,contract['words']['s4c2']['expected'],rtol=0,atol=1e-12):
        raise ValueError('Oracle differs from network metadata')
    count=contract['target_count']
    first=contract['warmup']
    finite=bool(np.isfinite(last_ns))
    exact=True
    rows=[]
    input_rows=[]
    for name in ['tin',*contract['words']]:
        rises,falls=(np.asarray(x,dtype=float) for x in observations.get(name,([],[])))
        finite=finite and bool(np.isfinite(rises).all() and np.isfinite(falls).all())
        samples=len(widths) if name=='tin' else len(contract['words'][name]['expected'])
        exact=exact and len(rises)==len(falls)==samples
        if name=='tin':
            for k,width in enumerate(widths):
                anchor=40.1+200*k
                rr=rises[(rises>=anchor-1)&(rises<anchor+199)]
                ff=falls[(falls>=anchor-1)&(falls<anchor+199)]
                valid=bool(len(rr)==len(ff)==1 and ff[0]>rr[0])
                error=float(ff[0]-rr[0]-width) if valid else None
                input_rows.append(dict(sample=k,valid=valid,width_error_ns=error,
                    within_bounds=bool(valid and abs(error)<=contract['input_gate_ns'] and abs(rr[0]-anchor)<=contract['input_gate_ns'])))
            continue
        word=contract['words'][name]
        truth=expected if name=='s4c2' else word['expected']
        begin=first+4 if name=='front' else first+4-int(name[7:]) if name.startswith('history') else first
        for k,width in enumerate(truth):
            F=word['phase']+200*k+200.1
            rr=rises[(rises>=F-199)&(rises<F+1)]
            ff=falls[(falls>=F-199)&(falls<F+1)]
            valid=bool(len(rr)==len(ff)==1 and ff[0]>rr[0])
            err=float(ff[0]-rr[0]-width) if valid else None
            anchor=float(ff[0]-F) if valid else None
            rows.append(dict(node=name,sample=k,valid=valid,width_error_ns=err,anchor_error_ns=anchor,
                scored=bool(begin<=k<begin+count),
                within_bounds=bool(valid and abs(err)<=contract['width_gate_ns'] and abs(anchor)<=contract['anchor_gate_ns'])))
    scored=[x for x in rows if x['scored']]
    med=[x for x in scored if x['node']=='s4c2']
    good=[x for x in med if x['valid']]
    warm=[x for x in rows if x['node']=='s4c2' and x['sample']<first]
    tail=[x for x in rows if x['node']=='s4c2' and first+count<=x['sample']]
    inputs_ok=bool(len(input_rows)==len(widths) and all(x['within_bounds'] for x in input_rows))
    all_scored=bool(len(scored)==len(contract['words'])*count and all(x['within_bounds'] for x in scored))
    completed=bool(finite and simulation_valid and last_ns>=contract['expected_stop_ns']*(1-1e-8))
    endpoint_truth=[expected[k] for k in range(first,first+count)]
    both_endpoints=4. in endpoint_truth and 196. in endpoint_truth
    passed=bool(completed and exact and inputs_ok and all_scored and len(good)==count and
        len(warm)==first and all(x['valid'] for x in warm) and len(tail)>=contract['tail'] and all(x['valid'] for x in tail) and both_endpoints)
    return dict(port_gate_pass=passed,execution_complete=completed,finite_observations=finite,
        event_count_exact=bool(exact),all_physical_inputs_match=inputs_ok,
        all_scored_ports_valid=all_scored,scored_port_transactions=len(scored),
        scored_targets=len(med),complete_targets=len(good),valid_warmup_outputs=sum(x['valid'] for x in warm),
        valid_tail_outputs=sum(x['valid'] for x in tail),both_final_median_endpoints_in_scored_targets=both_endpoints,
        max_width_error_ns=max((abs(x['width_error_ns']) for x in good),default=None),
        max_anchor_error_ns=max((abs(x['anchor_error_ns']) for x in good),default=None),
        failed_scored_port_transactions=sum(not x['within_bounds'] for x in scored),
        internal_current_holdout_qualified=False,model_L1_L2_validated=False,
        system_speed_qualified=False,global_acceptance=False,rows=rows,input_rows=input_rows)
