"""Explicit development admission amendment;never relabel incomplete transient."""
import json
from benchmarks.physical_common import ROOT
from benchmarks.audit_cost_diagnosis_v56 import stream_sha
from benchmarks.validation_guard_v41 import verify_frozen
from benchmarks.live_spice import now,write

OUT=ROOT/'artifacts/snapshot_campaign_v61'
PARENT=ROOT/'artifacts/ordered_observations_v60'


def certify():
    stage=PARENT/'regression';freeze=json.loads((stage/'final_freeze.json').read_text())
    verify_frozen(ROOT,freeze['assets_sha256']);commit=json.loads((stage/'prediction_commit.json').read_text())
    for name,key in [('final_freeze.json','official_freeze_sha256'),('predictions.json','prediction_sha256'),
        ('frozen_inputs.npz','input_sha256'),('evaluation_contract.json','evaluation_contract_sha256'),
        ('generated_electrical.cir','electrical_sha256')]:
        if stream_sha(stage/name)!=commit[key]:raise RuntimeError('old committed asset changed:'+name)
    scored=json.loads((OUT/'regression_scoring/checkpoint.json').read_text())
    full=json.loads((OUT/'regression_scoring/system_live.json').read_text())
    c=json.loads((stage/'evaluation_contract.json').read_text());selected=[];warm=c['warmup'];limit=214
    for row in full['timings']:
        n=row['node'];begin=warm+(4 if n=='front' else 4-int(n[7:]) if n.startswith('history') else 0)
        if not begin<=row['sample']<begin+limit:continue
        F=row['capture_reference_phase_ns']
        required=F+(352. if n=='front' else 372.)
        selected.append(dict(node=n,sample=row['sample'],required_observation_end_ns=required,
            phase_complete=row['phase_evaluation_complete'],phase_gate_pass=row['phase_gate_pass'],reuse_observed=row['reuse_observed']))
    required=max(r['required_observation_end_ns'] for r in selected)
    fidelity=json.loads((PARENT/'checkpoint214_result_review.json').read_text())
    equivalence=json.loads((OUT/'scoring_equivalence.json').read_text())
    passed=bool(scored['gate_pass'] and scored['valid_warmup_outputs']==100 and scored['selected_word_outputs']==1284
        and scored['selected_phase_transactions']==1284 and scored['selected_aux_transactions']==1070
        and len(selected)==1284 and all(x['phase_complete'] and x['phase_gate_pass'] and x['reuse_observed'] for x in selected)
        and scored['actual_snapshot_last_ns']>=required and fidelity['numerical_fidelity_prefix_pass']
        and equivalence['all_scientific_fields_exact'])
    record=dict(utc=now(),gate_pass=passed,scope='214-target complete-network development regression scoring completed',
        admission_amendment_explicit=True,decision_stage='exposed development,prior to fresh seed opening',
        scientific_thresholds_unchanged=True,original_64310ns_transient_completed=False,
        original_execution_failure_retained=True,original_freeze_sha256=stream_sha(stage/'final_freeze.json'),
        selected_target_count=214,scheduled_valid_warmups=100,selected_word_outputs=1284,
        required_observation_end_ns=required,actual_observation_end_ns=scored['actual_snapshot_last_ns'],
        observation_coverage_margin_ns=scored['actual_snapshot_last_ns']-required,
        former_transient_stop_ns=64310,unexecuted_padding_ns=64310-scored['actual_snapshot_last_ns'],
        claims_not_made=['64310ns native terminal success','new independent validation','full dynamicL2library','full systemPPA'],
        selected_completion_checks=selected,metrics=scored['metrics'],numerical_fidelity=fidelity,
        source_checkpoint_sha256=stream_sha(OUT/'regression_scoring/checkpoint.json'),
        original_failure_sha256=stream_sha(stage/'failure.json'),new_independent_seed_opened=False)
    write(OUT/'regression_gate.json',record)
    if not passed:raise RuntimeError('scientific coverage or scoring gate failed;seed remains sealed')
    return record


if __name__=='__main__':
    print(json.dumps({k:v for k,v in certify().items() if k not in ['selected_completion_checks','metrics','numerical_fidelity']},indent=2))
