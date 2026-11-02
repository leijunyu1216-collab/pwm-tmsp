"""Terminal independent endpoint audit;strictly readonly predictions and inputs.

This provides same-task model errors after the continuous physical run. It
does not fit model parameters, time shifts or scales, nor promote full D1-D6.
"""
import json
import numpy as np
from benchmarks.retimed_run_v56 import OUT
from benchmarks.physical_common import ROOT
from benchmarks.validation_guard_v41 import verify_frozen
from benchmarks.audit_cost_diagnosis_v56 import stream_sha
from benchmarks.crossing_index_v56 import build
from benchmarks.live_spice import now,write


def main():
    folder=OUT/'joint_independent';summary=folder/'retry1_summary.json'
    if not summary.exists():raise RuntimeError('joint activity has no terminal success;no model acceptance assertion')
    done=json.loads(summary.read_text())
    if not done.get('final_gate_pass') or not done.get('terminal_complete'):raise RuntimeError('terminal physical gate missing')
    record=json.loads((folder/'final_freeze.json').read_text());verify_frozen(ROOT,record['assets_sha256'])
    commit=json.loads((folder/'prediction_commit.json').read_text())
    for filename,key in [('final_freeze.json','official_freeze_sha256'),('predictions.json','prediction_sha256'),
                         ('frozen_inputs.npz','input_sha256'),('evaluation_contract.json','evaluation_contract_sha256')]:
        if stream_sha(folder/filename)!=commit[key]:raise RuntimeError('committed data changed:'+filename)
    result_path=folder/'terminal_model_comparison.json'
    if result_path.exists():raise FileExistsError('immutable terminal audit already exists')
    _,index,_=build(folder/'physical_retry1/transient.raw')
    predictions=json.loads((folder/'predictions.json').read_text());rows=[]
    for layer,forecast in predictions.items():
        errors=[];widths=[];pair_rows=[]
        for p in forecast:
            if p['node']!='s4c2' or not 100<=p['sample']<5100:continue
            k=p['sample'];F=1460.1+200*k;rr,ff=index['s4c2']
            r=rr[np.searchsorted(rr,F-199):np.searchsorted(rr,F+1)]
            f=ff[np.searchsorted(ff,F-199):np.searchsorted(ff,F+1)]
            if len(r)!=len(f) or len(r)!=1:raise RuntimeError('final target event mismatch')
            er=float(r[0]-p['rise_ns']);ef=float(f[0]-p['fall_ns']);ew=ef-er
            errors.extend([er,ef]);widths.append(ew)
            pair_rows.append(dict(target=k-100,rise_error_ns=er,fall_error_ns=ef,width_error_ns=ew))
        if len(pair_rows)!=5000:raise RuntimeError('exact5000-target comparison missing')
        rows.append(dict(layer=layer,targets=5000,edges=10000,edge_RMS_percent_FS=float(np.sqrt(np.mean(np.square(errors)))/2),
            edge_P99_percent_FS=float(np.percentile(np.abs(errors),99)/2),
            width_RMS_ns=float(np.sqrt(np.mean(np.square(widths)))),paired_errors=pair_rows))
    l1=next(r for r in rows if r['layer']=='L1');l2=next(r for r in rows if r['layer']=='L2_static_candidate')
    write(result_path,dict(utc=now(),scope='frozen new-input full five-point schematic,not unseen topology or layout/PEX',
        official_freeze_sha256=commit['official_freeze_sha256'],all_layers_same_targets_and_physical_reference=True,
        rows=rows,L1_final_edge_RMS_within5percent=l1['edge_RMS_percent_FS']<=5,
        L2_final_edge_P99_within1percent=l2['edge_P99_percent_FS']<=1,
        cubic_final_width_RMS_lower_than_affine=l2['width_RMS_ns']<l1['width_RMS_ns'],
        paired_comparison_is_descriptive_not_iid_significance=True,test_set_parameter_scale_or_delay_fit=False,
        full_dynamic_L2_library_qualified=False,full_system_PPA_qualified=False,global_D1_D6_acceptance=False))
    print(result_path)


if __name__=='__main__':main()
