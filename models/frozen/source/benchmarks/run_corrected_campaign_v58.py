"""User-run chain:exposed numerical regression,then fresh frozen independent run.

No analog state is reconstructed. Interrupted runs restart at time zero with
the same committed inputs; numerical failures stop the chain without retries.
"""
import argparse,copy,hashlib,json,msvcrt,os,shutil,sys,types
from pathlib import Path
import numpy as np
import psutil
from benchmarks.physical_common import ROOT
from benchmarks.retimed_run_v56 import OUT as OLD
from benchmarks.audit_cost_diagnosis_v56 import stream_sha
from benchmarks.validation_guard_v41 import verify_frozen
from benchmarks.clock_table_v58 import rewrite
from benchmarks.lossless_raw_archive_v58 import archive
from benchmarks.live_spice import now,write

OUT=ROOT/'artifacts/clock_repair_v58'
BASE=OLD/'joint_independent'
NEW_SEED=20270108
ORIGINAL_FREEZE_SHA='cfe4e51cb59063820d3e4de29e2723d1114454cc5c4fcb33dc7aa589bb9c3a90'
ORIGINAL_PREDICTION_COMMIT_SHA='bd12ecf41e56ef32172cd80a518c6d045beba2d57d1b737e22db0f4cd8f59cc4'


def common_assets():
    if stream_sha(BASE/'final_freeze.json')!=ORIGINAL_FREEZE_SHA or stream_sha(BASE/'prediction_commit.json')!=ORIGINAL_PREDICTION_COMMIT_SHA:
        raise RuntimeError('original freeze/prediction commitment changed')
    old=json.loads((BASE/'final_freeze.json').read_text());verify_frozen(ROOT,old['assets_sha256'])
    result=dict(old['assets_sha256'])
    for p in [Path(__file__),ROOT/'benchmarks/clock_table_v58.py',ROOT/'benchmarks/lossless_raw_archive_v58.py',
              ROOT/'benchmarks/post_joint_v57.py',ROOT/'tests/test_clock_repair_v58.py',
              OUT/'physical_qualification.json',OUT/'source_semantics.json']:
        result[str(p.resolve().relative_to(ROOT))]=stream_sha(p)
    for p in [BASE/'prediction_commit.json',BASE/'retry1_checkpoint_128/system_live.json',BASE/'retry1_checkpoint_128/checkpoint.json']:
        result[str(p.relative_to(ROOT))]=stream_sha(p)
    return old,result


def prerequisites():
    for name in ['physical_qualification.json','source_semantics.json']:
        if not json.loads((OUT/name).read_text())['gate_pass']:raise RuntimeError('unqualified prerequisite:'+name)
    common_assets()
    proof=json.loads((OUT/'source_semantics.json').read_text())
    verify_frozen(ROOT,proof.get('tested_sources_sha256',{}))


def ensure_space(required):
    candidates=[BASE/'physical_retry1/transient.raw',OUT/'regression/physical/transient.raw',
        OUT/'runs/full_horizon_clock_equivalence_v2/transient.raw']
    # Completed historical merged waveforms are kept bit-for-bit in the
    # reversible archive; their scripts,schemas and results stay in place.
    candidates += [ROOT/'artifacts/observation_recovery_v45/retry1/stage'/name/'transient_stream.f64'
        for name in ['distributed_challenge24_halfstep','distributed_challenge24','shared_challenge24']]
    candidates += [p/'transient.raw' for p in sorted((OLD/'runs').glob('*1510')) if (p/'summary.json').exists()]
    for p in candidates:
        if psutil.disk_usage(str(OUT)).free>=required:return
        if p.is_file() and p.stat().st_nlink==1:
            print('保留全部位模式，校验后归档：',p,flush=True);archive(p)
    free=psutil.disk_usage(str(OUT)).free
    if free<required:raise RuntimeError(f'磁盘准入未过：需{required/2**30:.2f}GiB，现有{free/2**30:.2f}GiB；新种子保持未打开')


def stage_record(stage,kind,extra_assets=None):
    old,assets=common_assets()
    if extra_assets:
        for p in extra_assets:assets[str(p.relative_to(ROOT))]=stream_sha(p)
    record=copy.deepcopy(old);record.update(utc=now(),activity=kind,assets_sha256=assets,
        original_physical_freeze_sha256=stream_sha(BASE/'final_freeze.json'),
        numerical_clock_representation='finite integer-fs PWL,all87 clocks;no PULSE recurrence',
        model_cards_or_physical_parameters_refit=False,new_independent_seed_opened=False)
    if kind=='exposed_failure_regression':
        record.update(sealed_seed=None,independent_activity=False)
        record['budgets'].update(max_wall_s=4*3600,checkpoint_wall_s={'214':4*3600},max_raw_bytes=12*2**30)
    else:
        record.update(sealed_seed=NEW_SEED,independent_activity=True)
        record['assets_sha256'][str((OUT/'regression_gate.json').relative_to(ROOT))]=stream_sha(OUT/'regression_gate.json')
        record['prerequisite_evidence']['past_old_failure']={'path':str((OUT/'regression_gate.json').relative_to(ROOT)),
            'sha256':stream_sha(OUT/'regression_gate.json'),'passed':True}
    verify_frozen(ROOT,assets)
    with (stage/'final_freeze.json').open('x',encoding='utf-8') as f:json.dump(record,f,indent=2,ensure_ascii=False)
    return record


def commit(stage,record,c,predictions,widths,clean,body,protocol):
    (stage/'generated_electrical.cir').write_text(body,encoding='ascii')
    np.savez(stage/'frozen_inputs.npz',widths_ns=widths,clean_ns=clean)
    write(stage/'evaluation_contract.json',c);write(stage/'predictions.json',predictions);write(stage/'input_protocol.json',protocol)
    value=dict(utc=now(),official_freeze_sha256=stream_sha(stage/'final_freeze.json'),
        prediction_sha256=stream_sha(stage/'predictions.json'),input_sha256=stream_sha(stage/'frozen_inputs.npz'),
        evaluation_contract_sha256=stream_sha(stage/'evaluation_contract.json'),
        electrical_sha256=stream_sha(stage/'generated_electrical.cir'),physical_reference_started=False,
        test_states_or_outputs_used_in_prediction=False,model_refit=False)
    with (stage/'prediction_commit.json').open('x',encoding='utf-8') as f:json.dump(value,f,indent=2)


def prepare_regression(stage):
    stage.mkdir(exist_ok=False);old,assets=common_assets()
    previous=json.loads((BASE/'prediction_commit.json').read_text())
    for filename,key in [('final_freeze.json','official_freeze_sha256'),('frozen_inputs.npz','input_sha256'),
                         ('predictions.json','prediction_sha256'),('evaluation_contract.json','evaluation_contract_sha256')]:
        if stream_sha(BASE/filename)!=previous[key]:raise RuntimeError('original committed data changed:'+filename)
    record=stage_record(stage,'exposed_failure_regression')
    with np.load(BASE/'frozen_inputs.npz') as z:widths=z['widths_ns'].copy();clean=z['clean_ns'].copy()
    c=json.loads((BASE/'evaluation_contract.json').read_text());c.update(expected_stop_ns=64310,
        scope='exposed old numerical-failure regression;not a fresh holdout',independent_seed_opened=False,new_independent_seed_opened=False)
    predictions=json.loads((BASE/'predictions.json').read_text())
    body,inventory=rewrite((BASE/'generated_electrical.cir').read_text())
    commit(stage,record,c,predictions,widths,clean,body,dict(kind='old20270107 exposed failure replay',clocks=inventory))


def prepare_independent(stage):
    if not json.loads((OUT/'regression_gate.json').read_text())['gate_pass']:raise RuntimeError('old failure has not been crossed and validated')
    original=json.loads((BASE/'final_freeze.json').read_text());b=original['budgets']
    ensure_space(b['max_raw_bytes']+b['min_free_disk_bytes'])
    stage.mkdir(exist_ok=False);record=stage_record(stage,'fresh_independent_numerical_version')
    with (stage/'seed_opened.json').open('x',encoding='utf-8') as f:
        json.dump(dict(utc=now(),seed=NEW_SEED,freeze_sha256=stream_sha(stage/'final_freeze.json')),f)
    from benchmarks.retimed_workload_v56 import inputs,physical_case
    from benchmarks.retimed_prediction_v56 import predict
    from benchmarks.campaign_live_v57 import contract
    widths,clean,protocol=inputs(NEW_SEED);electrical,physical=physical_case(widths);body,inventory=rewrite(electrical)
    c=contract(widths,physical,100,5000,3);c['scope']='fresh independent20270108,integer-fs finite clocks;one continuous5000target trajectory'
    card=json.loads((ROOT/record['active_card']).read_text())
    predictions={name:predict(widths,card,name) for name in ['L0','L1','L2_static_candidate']}
    commit(stage,record,c,predictions,widths,clean,body,dict(protocol=protocol,physical=physical,clocks=inventory))


def load_committed(stage,record):
    verify_frozen(ROOT,record['assets_sha256']);r=json.loads((stage/'prediction_commit.json').read_text())
    for filename,key in [('final_freeze.json','official_freeze_sha256'),('predictions.json','prediction_sha256'),
                         ('frozen_inputs.npz','input_sha256'),('evaluation_contract.json','evaluation_contract_sha256'),
                         ('generated_electrical.cir','electrical_sha256')]:
        if stream_sha(stage/filename)!=r[key]:raise RuntimeError('committed stage asset changed:'+filename)
    with np.load(stage/'frozen_inputs.npz') as data:w=data['widths_ns'].copy();clean=data['clean_ns'].copy()
    return json.loads((stage/'evaluation_contract.json').read_text()),json.loads((stage/'predictions.json').read_text()),w,clean


def once(s,old,new):
    if s.count(old)!=1:raise RuntimeError('frozen controller source drift:'+old)
    return s.replace(old,new)


def execute(stage,regression=False,resume=False):
    load_committed(stage,json.loads((stage/'final_freeze.json').read_text()))
    if (stage/'summary.json').exists():
        if not resume:raise RuntimeError('completed stage exists;use --resume to reuse it')
        completed=json.loads((stage/'summary.json').read_text())
        if not completed.get('final_gate_pass') or not completed.get('terminal_complete'):raise RuntimeError('existing stage summary is not accepted')
        print('复用已完成阶段：',stage,flush=True);return
    if (stage/'physical').exists():
        failure=stage/'failure.json'
        r=json.loads(failure.read_text()) if failure.exists() else {}
        if failure.exists() and r.get('error') not in ['','KeyboardInterrupt']:
            raise RuntimeError('原失败已保留，本入口不盲目重复数值/评分失败：'+r.get('error',''))
        if not resume:raise RuntimeError('unfinished stage retained;use --resume for a cold restart')
        prior=stage/'interrupted_attempts';prior.mkdir(exist_ok=True)
        dest=prior/f'attempt_{len(list(prior.iterdir()))+1:03d}';dest.mkdir()
        # Both resolved targets are verified within this exact stage directory.
        for p in list(stage.iterdir()):
            if p.name in ['physical','failure.json','runner.json','runner_final.json','progress.json','status.json','monitor.jsonl'] or p.name.startswith('checkpoint_') or p.name=='final_evaluation':
                if not p.resolve().is_relative_to(stage.resolve()) or not dest.resolve().is_relative_to(stage.resolve()):raise ValueError('restart path outside stage')
                p.rename(dest/p.name)
        print('中断尝试已保留；按原输入/预测从零重跑：',dest,flush=True)
    source=(ROOT/'benchmarks/joint_campaign_v57.py').read_text();pre,mainpart=source.split('def main():',1)
    mainpart=once(mainpart,'\\nset klu\\nset num_threads=8','\\nset klu\\nset no_mem_check\\nset num_threads=8')
    mainpart=once(mainpart,'if elapsed-last>=5:','if elapsed-last>=30:')
    mainpart=once(mainpart,"journal.write(json.dumps(value)+'\\n');last=elapsed", "journal.write(json.dumps(value)+'\\n');print(json.dumps(value),flush=True);last=elapsed")
    if regression:
        mainpart=once(mainpart,'[128,1000,5000]','[214]')
        mainpart=once(mainpart,"c,forecasts,5000,live=False","c,forecasts,214,live=True")
        mainpart=once(mainpart,"write(CAMPAIGN/'three_noise.json',snr(final,c,forecasts,widths,clean))",'pass # regression does not score three-noise or claim5000targets')
        mainpart=mainpart.replace('COMPLETE_JOINT_SYSTEM_ACTIVITY','NUMERICAL_FAILURE_REGRESSION_PASSED')
    code=pre+'def main():'+mainpart
    generated=stage/'controller_generated.py';generated.write_text(code,encoding='utf-8')
    module=types.ModuleType('qualified_clock_table_controller_v58');module.__file__=str(generated)
    exec(compile(code,str(generated),'exec'),module.__dict__);module.CAMPAIGN=stage;module.prepare=lambda r:load_committed(stage,r)
    original_verify=module.verify;commit_hash=stream_sha(stage/'prediction_commit.json')
    def verify_stage(record):
        original_verify(record)
        if stream_sha(stage/'prediction_commit.json')!=commit_hash:raise RuntimeError('prediction commit changed during execution')
        r=json.loads((stage/'prediction_commit.json').read_text())
        for name,key in [('final_freeze.json','official_freeze_sha256'),('predictions.json','prediction_sha256'),
                         ('frozen_inputs.npz','input_sha256'),('evaluation_contract.json','evaluation_contract_sha256'),
                         ('generated_electrical.cir','electrical_sha256')]:
            if stream_sha(stage/name)!=r[key]:raise RuntimeError('committed stage changed:'+name)
    module.verify=verify_stage
    write(stage/'controller_compile.json',dict(utc=now(),generated_sha256=stream_sha(generated),
        frozen_controller_source_sha256=stream_sha(ROOT/'benchmarks/joint_campaign_v57.py'),
        wrapper_sha256=stream_sha(Path(__file__)),regression_only=regression,prediction_commit_sha256=commit_hash))
    print('启动阶段：',stage.name,'；从时间零保留完整历史；检查点不复位。',flush=True)
    module.main()


def regression_gate():
    if (OUT/'regression_gate.json').exists():
        r=json.loads((OUT/'regression_gate.json').read_text())
        if not r['gate_pass']:raise RuntimeError('existing old failure regression gate is failed')
        for name,key in [('summary.json','summary_sha256'),('final_evaluation/checkpoint.json','assessment_sha256')]:
            if stream_sha(OUT/'regression'/name)!=r[key]:raise RuntimeError('accepted regression evidence changed:'+name)
        return
    stage=OUT/'regression';summary=json.loads((stage/'summary.json').read_text());r=json.loads((stage/'final_evaluation/checkpoint.json').read_text())
    previous=json.loads((BASE/'retry1_checkpoint_128/system_live.json').read_text());current=json.loads((stage/'final_evaluation/system_live.json').read_text())
    lookup={(x['node'],x['sample']):x for x in current['rows']};width=anchor=state=0.
    for old in previous['rows']:
        new=lookup[(old['node'],old['sample'])]
        if not old['valid'] or not new['valid']:raise RuntimeError('replay comparison event missing')
        width=max(width,abs(old['output_width_ns']-new['output_width_ns']));anchor=max(anchor,abs(old['anchor_error_ns']-new['anchor_error_ns']))
    phase={(x['node'],x['sample']):x for x in current['timings']}
    for old in previous['timings']:
        if not old['phase_evaluation_complete']:continue
        new=phase[(old['node'],old['sample'])]
        for key in ['data_voltage_at_replay_V','reference_voltage_at_replay_V']:
            if old[key] is not None and new[key] is not None:state=max(state,abs(old[key]-new[key]))
    passed=bool(summary['final_gate_pass'] and r['gate_pass'] and r['actual_snapshot_last_ns']>=64310-1e-6
        and width<=.01 and anchor<=.01 and state<=.001)
    result=dict(utc=now(),gate_pass=passed,scope='exposed old failure regression;not fresh independent evidence',
        last_ns=r['actual_snapshot_last_ns'],old_abort_ns=62031,regression_scored_targets=214,
        previous128_max_width_delta_ns=width,previous128_max_anchor_delta_ns=anchor,previous128_max_replay_state_delta_V=state,
        original_model_or_prediction_refit=False,new_seed_opened=False,
        summary_sha256=stream_sha(stage/'summary.json'),assessment_sha256=stream_sha(stage/'final_evaluation/checkpoint.json'))
    write(OUT/'regression_gate.json',result)
    if not passed:raise RuntimeError('old failure regression or previous128 numerical fidelity gate failed;new seed stays unopened')


def terminal_audit():
    stage=OUT/'joint_independent'
    if not (stage/'terminal_model_comparison.json').exists():
        source=(ROOT/'benchmarks/post_joint_v57.py').read_text()
        source=source.replace('retry1_summary.json','summary.json').replace('physical_retry1','physical')
        module=types.ModuleType('readonly_terminal_audit_v58');module.__file__=str(ROOT/'benchmarks/post_joint_v57.py')
        exec(compile(source,module.__file__,'exec'),module.__dict__);module.OUT=OUT;module.main()
    result=json.loads((stage/'terminal_model_comparison.json').read_text())
    if not result['L1_final_edge_RMS_within5percent'] or not result['L2_final_edge_P99_within1percent']:
        raise RuntimeError('最终10000边沿模型门未过；结果保留，不回填')


def main():
    p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');p.add_argument('--plan',action='store_true');a=p.parse_args()
    if a.plan:
        print('只读计划：开发门→旧20270107回归至64310ns/214目标→新冻结/seed20270108→100预热+128/1000/5000→三噪声/最终10000边沿；本命令不生成输入。')
        for name in ['physical_qualification.json','source_semantics.json','regression_gate.json']:
            q=OUT/name;print(name, json.loads(q.read_text()).get('gate_pass') if q.exists() else 'pending')
        return
    prerequisites()
    lock=(OUT/'execution.lock').open('a+b');lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
    try:
        try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:raise RuntimeError('另一执行器持有任务锁；不启动重复实验')
        write(OUT/'launcher.json',dict(utc=now(),pid=os.getpid(),state='RUNNING',formal_seed_still_sealed=not (OUT/'joint_independent/seed_opened.json').exists()))
        regression=OUT/'regression'
        if not regression.exists():prepare_regression(regression)
        execute(regression,regression=True,resume=a.resume);regression_gate()
        independent=OUT/'joint_independent'
        if not independent.exists():prepare_independent(independent)
        execute(independent,resume=a.resume);terminal_audit()
        write(OUT/'pipeline_complete.json',dict(utc=now(),regression_passed=True,fresh_independent_targets=5000,
            three_noise_scored=True,global_D1_D6_acceptance=False,full_PPA_complete=False,matched_system_speedup_qualified=False))
        print('正式连续活动及终检完成；完整PPA、未见拓扑排名和匹配速度按各自门继续验收。',flush=True)
    except BaseException as e:
        write(OUT/'launcher_failure.json',dict(utc=now(),error=repr(e),new_seed_opened=(OUT/'joint_independent/seed_opened.json').exists(),
            parameters_refit=False,old_records_preserved=True));raise
    finally:lock.close()


if __name__=='__main__':main()
