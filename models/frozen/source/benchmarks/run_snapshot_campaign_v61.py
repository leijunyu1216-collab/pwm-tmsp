"""Freeze-first fresh independent activity after explicit regression score audit."""
import argparse,copy,json,msvcrt,os,sys,types
from pathlib import Path
import psutil
from benchmarks.physical_common import ROOT
from benchmarks.live_spice import write,now
from benchmarks.audit_cost_diagnosis_v56 import stream_sha
from benchmarks.validation_guard_v41 import verify_frozen
from benchmarks import run_corrected_campaign_v58 as legacy
from benchmarks.run_ordered_campaign_v60 import rewrite
from benchmarks.snapshot_controller_v61 import run

OUT=ROOT/'artifacts/snapshot_campaign_v61'
PARENT=ROOT/'artifacts/ordered_observations_v60'
SOURCES=['benchmarks/snapshot_observations_v61.py','benchmarks/snapshot_controller_v61.py',
    'benchmarks/regression_admission_v61.py','benchmarks/qualify_snapshot_v61.py',
    'benchmarks/run_snapshot_campaign_v61.py','benchmarks/status_snapshot_campaign_v61.py','tests/test_snapshot_v61.py']


def common_assets():
    parent=json.loads((PARENT/'regression/final_freeze.json').read_text());verify_frozen(ROOT,parent['assets_sha256'])
    original=json.loads((legacy.BASE/'final_freeze.json').read_text());assets=dict(parent['assets_sha256'])
    paths=[ROOT/p for p in SOURCES]+[OUT/p for p in ['regression_gate.json','scoring_equivalence.json','software_qualification.json']]
    paths += [PARENT/'regression/final_freeze.json',PARENT/'regression/failure.json',
        PARENT/'regression/checkpoint_214/checkpoint.json',PARENT/'checkpoint214_result_review.json']
    for p in paths:assets[str(p.relative_to(ROOT))]=stream_sha(p)
    return original,assets


def prerequisites():
    gate=json.loads((OUT/'regression_gate.json').read_text());software=json.loads((OUT/'software_qualification.json').read_text())
    if not gate['gate_pass'] or not gate['scientific_thresholds_unchanged'] or not software['gate_pass']:
        raise RuntimeError('regression score or snapshot executor qualification failed;seed sealed')
    verify_frozen(ROOT,software['tested_sources_sha256']);common_assets()


def ensure_space(required):
    if psutil.disk_usage(str(ROOT)).free<required:raise RuntimeError('original140GiB disk admission failed;seed remains sealed')


def stage_record(stage,kind,extra_assets=None):
    original,assets=common_assets()
    for p in extra_assets or []:assets[str(p.relative_to(ROOT))]=stream_sha(p)
    record=copy.deepcopy(original);record.update(utc=now(),activity='fresh20270108_fixed_frontier_continuous_activity',
        assets_sha256=assets,sealed_seed=20270108,independent_activity=True,new_independent_seed_opened=False,
        observation_policy='v60 ordered equal-time rows;v61 common complete-row frontier without process suspension',
        physical_parameters_and_model_cards_refit=False,
        regression_admission_amendment=dict(explicit=True,scientific_score_scope='100warm+214targets plus all selected reuse/availability checks',
            original_64310ns_transient_completed=False,certificate_sha256=stream_sha(OUT/'regression_gate.json')))
    verify_frozen(ROOT,assets)
    with (stage/'final_freeze.json').open('x',encoding='utf-8') as f:json.dump(record,f,indent=2,ensure_ascii=False)
    return record


def prepare(stage):
    prerequisites()
    # The frozen legacy generator commits all predictions before physics.
    legacy.OUT=OUT;legacy.common_assets=common_assets;legacy.stage_record=stage_record
    legacy.ensure_space=ensure_space;legacy.rewrite=rewrite
    legacy.prepare_independent(stage)


def terminal_audit():
    stage=OUT/'joint_independent';target=stage/'terminal_model_comparison.json'
    if not target.exists():
        path=ROOT/'benchmarks/post_joint_v57.py';source=path.read_text()
        source=source.replace('retry1_summary.json','summary.json').replace('physical_retry1','physical')
        source=legacy.once(source,'from benchmarks.crossing_index_v56 import build','from benchmarks.ordered_observations_v60 import build')
        module=types.ModuleType('snapshot_terminal_audit_v61');module.__file__=str(path)
        exec(compile(source,str(path),'exec'),module.__dict__);module.OUT=OUT;module.main()
    r=json.loads(target.read_text())
    if not r['L1_final_edge_RMS_within5percent'] or not r['L2_final_edge_P99_within1percent']:
        raise RuntimeError('terminal original model gates failed;no refit')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');parser.add_argument('--plan',action='store_true');a=parser.parse_args()
    if a.plan:
        print('v61:explicit214score completion certificate→snapshot/no-pause qualification→new freeze→20270108→100warm+128/1000/5000 continuous→three noise and10000finaledges;original physical thresholds unchanged.')
        return
    prerequisites();lock=(OUT/'execution.lock').open('a+b');lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
    try:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        write(OUT/'launcher.json',dict(utc=now(),pid=os.getpid(),state='RUNNING'))
        stage=OUT/'joint_independent'
        if not stage.exists():prepare(stage)
        if (stage/'physical').exists():
            if (stage/'summary.json').exists() and a.resume:
                r=json.loads((stage/'summary.json').read_text())
                if not r['terminal_complete'] or not r['final_gate_pass']:raise RuntimeError('existing summary not accepted')
            else:raise RuntimeError('existing attempt preserved;no blind numerical/scoring restart')
        else:run(stage)
        terminal_audit()
        write(OUT/'pipeline_complete.json',dict(utc=now(),fresh_independent_targets=5000,three_noise_scored=True,
            global_D1_D6_acceptance=False,full_system_PPA_complete=False))
    except BaseException as e:
        write(OUT/'launcher_failure.json',dict(utc=now(),error=repr(e),new_seed_opened=(OUT/'joint_independent/seed_opened.json').exists(),
            parameters_refit=False,old_records_preserved=True));raise
    finally:lock.close()


if __name__=='__main__':main()
