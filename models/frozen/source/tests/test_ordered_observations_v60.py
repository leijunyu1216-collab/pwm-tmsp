import json
import numpy as np
import pytest
from benchmarks.ordered_observations_v60 import build
from benchmarks.physical_common import edges


def raw(tmp_path,rows):
    p=tmp_path/'fixture.raw';a=np.asarray(rows,dtype='<f8')
    head=f'Title: ordered-time fixture\nPlotname: Transient Analysis\nFlags: real\nNo. Variables: 3\nNo. Points: {len(a)}\nVariables:\n0 time time\n1 v(n) voltage\n2 i(vdd) current\nBinary:\n'.encode()
    p.write_bytes(head+a.tobytes());return p,a


def test_equal_time_rows_are_preserved_and_all_edges_equal_original_formula(tmp_path):
    p,a=raw(tmp_path,[[0,0,0],[1e-9,.2,-1],[1e-9,.20000000001,-1.000001],[2e-9,1.2,-2],[3e-9,0,-1]])
    original=p.read_bytes();audit=tmp_path/'audit.json';t,index,n=build(p,block_rows=2,audit_path=audit)
    assert len(t)==len(a) and p.read_bytes()==original
    assert all(np.array_equal(x,y) for x,y in zip(index['n'],edges(a[:,0]*1e9,a[:,1])))
    r=json.loads(audit.read_text());assert r['equal_time_pairs']==1 and r['original_rows_removed']==0
    assert r['equal_time_pairs_inventory'][0]['left_row']==1
    assert np.trapezoid(a[1:3,2],a[1:3,0])==0


@pytest.mark.parametrize('rows,match',[
    ([[0,0,0],[1e-9,0,0],[.9e-9,0,0]],'reversal'),
    ([[0,0,0],[1e-9,.89,0],[1e-9,.91,0]],'threshold crossing'),
    ([[0,0,0],[1e-9,.1,0],[1e-9,.102,0]],'1mV'),
    ([[0,0,0],[1e-9,.1,0],[1e-9,.1006,0],[1e-9,.1012,0]],'1mV'),
    ([[0,0,0],[1e-9,np.nan,0],[2e-9,0,0]],'nonfinite'),
])
def test_invalid_zero_time_or_backward_observations_fail_closed(tmp_path,rows,match):
    p,_=raw(tmp_path,rows)
    with pytest.raises(ValueError,match=match):build(p,block_rows=2)


def test_strict_grid_matches_original_across_block_boundaries(tmp_path):
    from benchmarks.crossing_index_v56 import build as strict
    p,_=raw(tmp_path,[[0,0,0],[1e-9,1.8,0],[2e-9,0,0],[3e-9,1.8,0],[4e-9,0,0]])
    t,x,n=build(p,block_rows=2);u,y,m=strict(p,block_rows=3)
    assert n==m and np.array_equal(t,u)
    assert all(np.array_equal(a,b) for a,b in zip(x['n'],y['n']))


def test_new_freeze_keeps_physical_model_criteria_and_sealed_seed(tmp_path,monkeypatch):
    import copy
    from benchmarks import run_ordered_campaign_v60 as runner
    original=json.loads((runner.previous.BASE/'final_freeze.json').read_text())
    monkeypatch.setattr(runner,'OUT',tmp_path)
    monkeypatch.setattr(runner,'common_assets',lambda:(copy.deepcopy(original),{}))
    (tmp_path/'regression_gate.json').write_text('{"gate_pass":true}')
    stage=tmp_path/'fresh';stage.mkdir();r=runner.stage_record(stage,'fresh_independent_numerical_version')
    assert r['criteria']==original['criteria'] and r['budgets']==original['budgets']
    assert r['active_card']==original['active_card'] and r['sealed_seed']==20270108
    assert r['observation_policy']['original_rows_removed']==0
    assert not (stage/'seed_opened.json').exists()


def test_generated_controller_uses_declared_reader_in_real_native_software_fixture(tmp_path,monkeypatch):
    import builtins,os,platform
    from benchmarks import run_ordered_campaign_v60 as wrapper
    runner=wrapper.previous
    from benchmarks.streaming_cli_probe_v47 import EXE,raw_read
    if os.name!='nt' or not EXE.exists():pytest.skip('qualified native runtime required')
    stage=tmp_path/'software_only';stage.mkdir()
    record=dict(workspace_root=str(runner.ROOT),assets_sha256={},environment=dict(python=platform.python_version(),numpy=np.__version__),
        budgets=dict(max_raw_bytes=10*2**20,min_free_disk_bytes=0,max_wall_s=60,max_child_RSS_bytes=2*2**30,checkpoint_wall_s={'214':60}))
    (stage/'final_freeze.json').write_text(json.dumps(record))
    c=dict(expected_stop_ns=40,output_columns=['v(n)'],scope='software-only controller fixture')
    body='* software-only RC fixture\nVin input 0 PULSE(0 1.8 1n .2n .2n 3n 20n 1)\nR input n 1k\nC n 0 1p\n'
    runner.commit(stage,record,c,{},np.array([4.]),np.array([4.]),body,{'software_fixture_only':True})
    calls=[]
    def intercept(code,namespace):
        builtins.exec(code,namespace)
        def fixture(raw,folder,contract,predictions,limit,live):
            a,n=raw_read(raw);calls.append((limit,live))
            return dict(gate_pass=abs(float(a[-1,n.index('time')]*1e9)-40)<1e-6,software_fixture_only=True)
        namespace['evaluate']=fixture
    monkeypatch.setattr(runner,'exec',intercept,raising=False)
    wrapper.execute(stage,regression=True)
    assert calls==[(214,True)]
    generated=(stage/'controller_generated.py').read_text()
    assert 'from benchmarks.campaign_observations_v60 import evaluate' in generated
    assert 'from benchmarks.campaign_live_v57 import contract,evaluate' not in generated
    assert json.loads((stage/'summary.json').read_text())['terminal_complete']
    assert not (stage/'seed_opened.json').exists()
