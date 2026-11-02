import json,time
from pathlib import Path
import numpy as np
import pytest
from benchmarks.snapshot_observations_v61 import Snapshot


def raw(path,values):
    a=np.asarray(values,dtype='<f8')
    header=f'Title: snapshot fixture\nPlotname: Transient Analysis\nFlags: real\nNo. Variables: 2\nNo. Points: {len(a)}\nVariables:\n0 time time\n1 v(n) voltage\nBinary:\n'.encode()
    path.write_bytes(header+a.tobytes());return a


def test_prefix_snapshot_excludes_append_including_previously_incomplete_row(tmp_path):
    p=tmp_path/'trace.raw';a=raw(p,[[0,0],[1e-9,1.8],[2e-9,0]])
    future=np.array([[3e-9,1.8],[4e-9,np.nan]],dtype='<f8').tobytes()
    with p.open('ab') as f:f.write(future[:8])
    snap=Snapshot(p)
    with p.open('ab') as f:f.write(future[8:])
    t,idx,n=snap.build(p)
    assert len(t)==3 and snap.record()['future_rows_used'] is False
    assert np.array_equal(snap.read(p)[0],a)
    with pytest.raises(ValueError,match='nonfinite'):Snapshot(p).build(p)


def test_snapshot_rejects_source_truncation(tmp_path):
    p=tmp_path/'trace.raw';raw(p,[[0,0],[1e-9,1.8],[2e-9,0]])
    snap=Snapshot(p)
    # Windows itself forbids truncating a live mapping. Close this test mapping
    # to exercise our independent source-size rejection as well.
    snap.array._mmap.close()
    with p.open('r+b') as f:f.truncate(snap.byte_frontier-1)
    with pytest.raises(ValueError,match='truncated'):snap.build(p)


def test_new_freeze_preserves_all_scientific_criteria_and_long_resource_budget(tmp_path,monkeypatch):
    import copy
    from benchmarks import run_snapshot_campaign_v61 as runner
    original=json.loads((runner.legacy.BASE/'final_freeze.json').read_text())
    monkeypatch.setattr(runner,'OUT',tmp_path);monkeypatch.setattr(runner,'common_assets',lambda:(copy.deepcopy(original),{}))
    (tmp_path/'regression_gate.json').write_text('{"gate_pass":true}')
    stage=tmp_path/'fresh';stage.mkdir();r=runner.stage_record(stage,'fresh_independent_numerical_version')
    assert r['criteria']==original['criteria'] and r['budgets']==original['budgets']
    assert r['sealed_seed']==20270108 and r['active_card']==original['active_card']
    assert r['regression_admission_amendment']['original_64310ns_transient_completed'] is False
    assert not (stage/'seed_opened.json').exists()


def test_new_seed_cannot_open_without_score_and_executor_qualification(tmp_path,monkeypatch):
    from benchmarks import run_snapshot_campaign_v61 as runner
    monkeypatch.setattr(runner,'OUT',tmp_path)
    (tmp_path/'regression_gate.json').write_text('{"gate_pass":false}')
    (tmp_path/'software_qualification.json').write_text('{"gate_pass":true}')
    with pytest.raises(RuntimeError,match='seed sealed'):runner.prepare(tmp_path/'joint_independent')
    assert not (tmp_path/'joint_independent').exists()


def test_native_controller_keeps_producer_running_and_records_exit(tmp_path,monkeypatch):
    import os,platform,psutil
    from benchmarks import snapshot_controller_v61 as controller
    from benchmarks.run_corrected_campaign_v58 import commit
    if os.name!='nt':pytest.skip('qualified Windows backend required')
    stage=tmp_path/'native_fixture';stage.mkdir()
    record=dict(workspace_root=str(controller.ROOT),assets_sha256={},environment=dict(python=platform.python_version(),numpy=np.__version__),
        budgets=dict(max_raw_bytes=128*2**20,min_free_disk_bytes=0,max_wall_s=90,max_child_RSS_bytes=2*2**30,checkpoint_wall_s={'128':90}))
    (stage/'final_freeze.json').write_text(json.dumps(record))
    body='* native concurrency software fixture\nVinput input 0 PULSE(0 1.8 .001n .0001n .0001n .0008n .002n 100000)\nR1 input n 1k\nC1 n 0 1p\n'
    c=dict(expected_stop_ns=200,output_columns=['v(n)']);commit(stage,record,c,{},np.array([4.]),np.array([4.]),body,{})
    def forbidden(*a,**kw):raise AssertionError('native process must never be suspended or resumed')
    monkeypatch.setattr(psutil.Process,'suspend',forbidden);monkeypatch.setattr(psutil.Process,'resume',forbidden)
    monkeypatch.setattr(controller,'deadline',lambda k:1.)
    calls=[]
    def evaluate(raw,folder,c,pred,limit,live):
        snap=Snapshot(raw);before=snap.record();time.sleep(.03);after=snap.record();t,_,_=snap.build(raw)
        calls.append(dict(live=live,rows=len(t),before=before['complete_byte_frontier'],after=after['source_bytes_after_evaluation']))
        return dict(gate_pass=True,software_fixture_only=True)
    monkeypatch.setattr(controller,'evaluate',evaluate)
    controller.run(stage,checkpoints=(128,),final_targets=128,noise=False)
    result=json.loads((stage/'summary.json').read_text())
    assert result['native_exit_code']==0 and result['terminal_complete'] and result['completed_checkpoints']==[128]
    assert calls[0]['live'] and not calls[-1]['live'] and calls[0]['after']>calls[0]['before']
    assert result['physical_process_suspended'] is False
