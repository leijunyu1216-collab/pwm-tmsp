"""New audited reader version:old regression first,then untouched fresh seed.

The electrical deck and all scientific criteria are inherited unchanged from
v59/v57. Only the numerical observation reader is replaced explicitly in the
generated controller. All previous failed freezes and raw rows remain evidence.
"""
import copy,inspect,json,sys,types
from pathlib import Path
from benchmarks.physical_common import ROOT
from benchmarks.audit_cost_diagnosis_v56 import stream_sha
from benchmarks.validation_guard_v41 import verify_frozen
from benchmarks.live_spice import now
from benchmarks.ordered_observations_v60 import POLICY
from benchmarks.ground_alias_v59 import eliminate
from benchmarks import run_corrected_campaign_v58 as previous

OUT=ROOT/'artifacts/ordered_observations_v60'
PARENT=ROOT/'artifacts/ground_reduction_v59'
ORIGINAL_EXECUTE=previous.execute
ORIGINAL_REWRITE=previous.rewrite
SOURCES=['benchmarks/ordered_observations_v60.py','benchmarks/campaign_observations_v60.py',
    'benchmarks/audit_current_events_v60.py',
    'benchmarks/qualify_observations_v60.py','benchmarks/run_ordered_campaign_v60.py',
    'benchmarks/status_ordered_campaign_v60.py','tests/test_ordered_observations_v60.py']


def common_assets():
    if stream_sha(previous.BASE/'final_freeze.json')!=previous.ORIGINAL_FREEZE_SHA or stream_sha(previous.BASE/'prediction_commit.json')!=previous.ORIGINAL_PREDICTION_COMMIT_SHA:
        raise RuntimeError('original freeze or prediction commitment changed')
    parent=json.loads((PARENT/'regression/final_freeze.json').read_text());verify_frozen(ROOT,parent['assets_sha256'])
    old=json.loads((previous.BASE/'final_freeze.json').read_text());assets=dict(parent['assets_sha256'])
    paths=[ROOT/s for s in SOURCES]+[OUT/s for s in ['observation_policy.json','physical_qualification.json',
        'software_qualification.json','short_equivalence.json','current_event_qualification.json']]+[PARENT/'regression/final_freeze.json',PARENT/'regression/failure.json']
    for p in paths:assets[str(p.resolve().relative_to(ROOT))]=stream_sha(p)
    verify_frozen(ROOT,assets);return old,assets


def prerequisites():
    for name in ['physical_qualification.json','software_qualification.json','current_event_qualification.json']:
        if not json.loads((OUT/name).read_text())['gate_pass']:raise RuntimeError('ordered observation prerequisite failed:'+name)
    if json.loads((OUT/'observation_policy.json').read_text())['policy']!=POLICY:raise RuntimeError('observation policy drift')
    verify_frozen(ROOT,json.loads((OUT/'software_qualification.json').read_text())['tested_sources_sha256'])
    common_assets()


def rewrite(body):
    clocks,inventory=ORIGINAL_REWRITE(body);reduced,_=eliminate(clocks)
    return reduced,inventory


def ensure_space(required):
    import psutil
    from benchmarks.lossless_raw_archive_v58 import archive
    candidates=[PARENT/'regression/physical/transient.raw',OUT/'regression/physical/transient.raw']
    candidates += [ROOT/'artifacts/observation_recovery_v45/retry1/stage'/n/'transient_stream.f64'
        for n in ['distributed_challenge24_halfstep','distributed_challenge24','shared_challenge24']]
    candidates += [ROOT/'artifacts'/p/'transient_stream.f64' for p in [
        'revision_exploration_v44/stage/distributed_challenge24',
        'capture_network_v36/stage/capture15ff_ref13434_run1','capture_network_v36/stage/capture1ff_ref13434',
        'shared_network_v38/stage/shared_endpoints12']]
    for path in candidates:
        if psutil.disk_usage(str(OUT)).free>=required:return
        if path.is_file() and path.stat().st_nlink==1:
            print('逐位保存并校验完整解码后归档旧/已完成波形：',path,flush=True);archive(path)
    if psutil.disk_usage(str(OUT)).free<required:
        raise RuntimeError('磁盘准入未过；新种子保持封存，全部旧数据保留')


def stage_record(stage,kind,extra_assets=None):
    old,assets=common_assets()
    for p in extra_assets or []:assets[str(p.relative_to(ROOT))]=stream_sha(p)
    record=copy.deepcopy(old);record.update(utc=now(),activity=kind,assets_sha256=assets,
        original_physical_freeze_sha256=stream_sha(previous.BASE/'final_freeze.json'),
        previous_failed_numerical_version_sha256=stream_sha(PARENT/'regression/final_freeze.json'),
        numerical_clock_representation='v59 finite integer-fsPWL plus exact Vss ground elimination unchanged',
        observation_policy=POLICY,observation_policy_change_explicit=True,eliminated_dynamic_states=0,
        model_cards_or_physical_parameters_refit=False,new_independent_seed_opened=False)
    if kind=='exposed_failure_regression':
        record.update(sealed_seed=None,independent_activity=False)
        record['budgets'].update(max_wall_s=4*3600,checkpoint_wall_s={'214':4*3600},max_raw_bytes=12*2**30)
    else:
        record.update(sealed_seed=previous.NEW_SEED,independent_activity=True)
        gate=OUT/'regression_gate.json';record['assets_sha256'][str(gate.relative_to(ROOT))]=stream_sha(gate)
        record['prerequisite_evidence']['old_failures_and_reader']={'path':str(gate.relative_to(ROOT)),
            'sha256':stream_sha(gate),'passed':True,'previous_failure_times_ns':[38660.8,62031,64240.11386235955]}
    verify_frozen(ROOT,record['assets_sha256'])
    with (stage/'final_freeze.json').open('x',encoding='utf-8') as f:json.dump(record,f,indent=2,ensure_ascii=False)
    return record


def execute(stage,regression=False,resume=False):
    # Clone the frozen wrapper's execution code with one explicit import change.
    source=inspect.getsource(ORIGINAL_EXECUTE)
    old="    code=pre+'def main():'+mainpart"
    new=old+"\n    code=once(code,'from benchmarks.campaign_live_v57 import contract,evaluate','from benchmarks.campaign_live_v57 import contract\\nfrom benchmarks.campaign_observations_v60 import evaluate')"
    source=previous.once(source,old,new)
    namespace=dict(previous.__dict__);namespace['__file__']=__file__
    exec(compile(source,str(Path(__file__)), 'exec'),namespace)
    namespace['execute'](stage,regression=regression,resume=resume)


def terminal_audit():
    stage=OUT/'joint_independent'
    if not (stage/'terminal_model_comparison.json').exists():
        path=ROOT/'benchmarks/post_joint_v57.py';source=path.read_text()
        source=source.replace('retry1_summary.json','summary.json').replace('physical_retry1','physical')
        source=previous.once(source,'from benchmarks.crossing_index_v56 import build','from benchmarks.ordered_observations_v60 import build')
        module=types.ModuleType('ordered_terminal_audit_v60');module.__file__=str(path)
        exec(compile(source,str(path),'exec'),module.__dict__);module.OUT=OUT;module.main()
    result=json.loads((stage/'terminal_model_comparison.json').read_text())
    if not result['L1_final_edge_RMS_within5percent'] or not result['L2_final_edge_P99_within1percent']:
        raise RuntimeError('最终10000边沿模型原门未过；不回填')


def configure():
    previous.OUT=OUT;previous.common_assets=common_assets;previous.prerequisites=prerequisites
    previous.rewrite=rewrite;previous.stage_record=stage_record;previous.ensure_space=ensure_space
    previous.execute=execute;previous.terminal_audit=terminal_audit


def main():
    if '--plan' in sys.argv:
        print('v60：全部原始微步保留并审计→64310ns/214旧回归→新冻结并打开20270108→100有效预热+128/1000/5000→三噪声和最终10000边沿；原门限保持。')
        for name in ['physical_qualification.json','software_qualification.json','regression_gate.json']:
            p=OUT/name;print(name,json.loads(p.read_text()).get('gate_pass') if p.exists() else 'pending')
        return
    configure();previous.main()


if __name__=='__main__':main()
