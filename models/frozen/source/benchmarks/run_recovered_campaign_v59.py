"""Qualified exact-ground reduction;preservev58failure and restart its stage."""
import copy,json,sys
from pathlib import Path
from benchmarks.physical_common import ROOT
from benchmarks.audit_cost_diagnosis_v56 import stream_sha
from benchmarks.validation_guard_v41 import verify_frozen
from benchmarks.live_spice import now
from benchmarks.ground_alias_v59 import eliminate
from benchmarks import run_corrected_campaign_v58 as previous

OUT=ROOT/'artifacts/ground_reduction_v59'
PARENT=ROOT/'artifacts/clock_repair_v58'
ORIGINAL_COMMON=previous.common_assets
ORIGINAL_REWRITE=previous.rewrite
ORIGINAL_SPACE=previous.ensure_space


def common_assets():
    current=previous.OUT;previous.OUT=PARENT
    try:old,assets=ORIGINAL_COMMON()
    finally:previous.OUT=current
    parent=json.loads((PARENT/'regression/final_freeze.json').read_text())
    verify_frozen(ROOT,parent['assets_sha256'])
    for p in [Path(__file__),ROOT/'benchmarks/ground_alias_v59.py',ROOT/'tests/test_ground_reduction_v59.py',
              OUT/'physical_qualification.json',OUT/'software_qualification.json',OUT/'equivalence_proof.json',
              PARENT/'regression/final_freeze.json',PARENT/'regression/failure.json',PARENT/'regression/prediction_commit.json']:
        assets[str(p.resolve().relative_to(ROOT))]=stream_sha(p)
    return old,assets


def rewrite(body):
    clocks,inventory=ORIGINAL_REWRITE(body);reduced,_=eliminate(clocks)
    return reduced,inventory


def prerequisites():
    for name in ['physical_qualification.json','software_qualification.json']:
        if not json.loads((OUT/name).read_text())['gate_pass']:raise RuntimeError('ground-reduction prerequisite failed:'+name)
    common_assets()
    verify_frozen(ROOT,json.loads((OUT/'software_qualification.json').read_text())['tested_sources_sha256'])


def ensure_space(required):
    import psutil
    try:ORIGINAL_SPACE(required);return
    except RuntimeError as e:
        if '磁盘准入未过' not in str(e):raise
    extra=[PARENT/'regression/physical/transient.raw']
    extra += [ROOT/'artifacts'/p/'transient_stream.f64' for p in [
        'revision_exploration_v44/stage/distributed_challenge24',
        'capture_network_v36/stage/capture15ff_ref13434_run1',
        'capture_network_v36/stage/capture1ff_ref13434',
        'shared_network_v38/stage/shared_endpoints12']]
    for path in extra:
        if psutil.disk_usage(str(OUT)).free>=required:return
        if path.is_file() and path.stat().st_nlink==1:
            print('逐位保留旧失败/已完成数据并归档：',path,flush=True);previous.archive(path)
    ORIGINAL_SPACE(required)


def stage_record(stage,kind,extra_assets=None):
    old,assets=common_assets()
    for p in extra_assets or []:assets[str(p.relative_to(ROOT))]=stream_sha(p)
    record=copy.deepcopy(old);record.update(utc=now(),activity=kind,assets_sha256=assets,
        original_physical_freeze_sha256=stream_sha(previous.BASE/'final_freeze.json'),
        previous_failed_numerical_version_sha256=stream_sha(PARENT/'regression/final_freeze.json'),
        numerical_clock_representation='v58finite integer-fsPWL unchanged;Vss=0 alias algebraically eliminated',
        eliminated_dynamic_states=0,eliminated_algebraic_unknowns=2,
        model_cards_or_physical_parameters_refit=False,new_independent_seed_opened=False)
    if kind=='exposed_failure_regression':
        record.update(sealed_seed=None,independent_activity=False)
        record['budgets'].update(max_wall_s=4*3600,checkpoint_wall_s={'214':4*3600},max_raw_bytes=12*2**30)
    else:
        record.update(sealed_seed=previous.NEW_SEED,independent_activity=True)
        gate=OUT/'regression_gate.json';record['assets_sha256'][str(gate.relative_to(ROOT))]=stream_sha(gate)
        record['prerequisite_evidence']['both_old_failures']={'path':str(gate.relative_to(ROOT)),'sha256':stream_sha(gate),'passed':True,
            'failure_times_ns':[38660.8,62031]}
    verify_frozen(ROOT,record['assets_sha256'])
    with (stage/'final_freeze.json').open('x',encoding='utf-8') as f:json.dump(record,f,indent=2,ensure_ascii=False)
    return record


def configure():
    previous.OUT=OUT;previous.common_assets=common_assets;previous.prerequisites=prerequisites
    previous.rewrite=rewrite;previous.stage_record=stage_record
    previous.ensure_space=ensure_space


def main():
    if '--plan' in sys.argv:
        print('v59：严格地约束消元开发门→从零回归64310ns并跨越38660.8/62031ns→新冻结/20270108→100预热及128/1000/5000；模型/门限保持。')
        for name in ['physical_qualification.json','software_qualification.json','regression_gate.json']:
            p=OUT/name;print(name,json.loads(p.read_text()).get('gate_pass') if p.exists() else 'pending')
        return
    configure();previous.main()


if __name__=='__main__':main()
