import json
import numpy as np
import pytest
from benchmarks.clock_table_v58 import PATTERN,points,rewrite,fs
from benchmarks.retimed_workload_v56 import inputs,physical_case
from benchmarks.lossless_raw_archive_v58 import encode_block,decode_block


def test_all_finite_clocks_preserve_every_corner_count_and_interp_without_recurrence():
    widths,_,_=inputs(1);s,_=physical_case(widths);actual,inventory=rewrite(s)
    assert len(inventory)==87 and not PATTERN.findall(actual)
    for m in PATTERN.finditer(s):
        p=points(m.groups());td,tr,tf,pw,period=map(fs,m.groups()[2:7]);count=int(m[8])
        knots=np.asarray([v[0] for v in p],dtype=np.int64);levels=np.asarray([v[1] for v in p])
        assert len(knots)==4*count+1 and np.all(np.diff(knots)>0)
        probes=np.unique(np.r_[knots,np.maximum(0,knots-1),knots+1,(knots[:-1]+knots[1:])//2,knots[-1]+period])
        dt=probes-td;cycles=dt//period;phase=dt%period;active=(dt>=0)&(cycles<count)
        expected=np.zeros(len(probes));rise=active&(phase<tr);expected[rise]=1.8*phase[rise]/tr
        high=active&(phase>=tr)&(phase<tr+pw);expected[high]=1.8
        fall=active&(phase>=tr+pw)&(phase<tr+pw+tf);expected[fall]=1.8*(1-(phase[fall]-tr-pw)/tf)
        assert np.max(np.abs(np.interp(probes,knots,levels)-expected))<1e-12


def test_raw_codec_preserves_nan_payloads_signed_zero_and_all_random_bits():
    a=np.random.default_rng(1).integers(0,2**64,(4096,11),dtype=np.uint64)
    a[0,:4]=[0,0x8000000000000000,0x7ff8000000000042,0x7ff0000000000000]
    assert decode_block(encode_block(a.tobytes(),11),11,len(a))==a.tobytes()


def test_clock_rewrite_rejects_unrecognized_inventory():
    with pytest.raises(ValueError,match='clock inventory'):rewrite('Vbad a 0 PULSE(0 1.8 0 .2n .2n 3n 600n)\n')


def test_archive_full_native_header_rows_and_partial_tail_restore_exactly(tmp_path):
    from benchmarks.lossless_raw_archive_v58 import archive,restore
    header=b'Title: lossless fixture\nPlotname: Transient Analysis\nFlags: real\nNo. Variables: 2\nNo. Points: 3\nVariables:\n0 time time\n1 v(n) voltage\nBinary:\n'
    data=np.array([[0,0],[1,0x8000000000000000],[2,0x7ff8000000000042]],dtype='<u8').tobytes()
    original=header+data+b'xyz';source=tmp_path/'test.raw';source.write_bytes(original)
    record=archive(source);assert record['decoded_all_bytes_bit_exact'] and not source.exists()
    restored=tmp_path/'restored.raw';restore(source.with_suffix('.raw.tmraw'),restored)
    assert restored.read_bytes()==original and record['partial_tail_bytes']==3


def test_new_seed_is_not_opened_when_failure_regression_fails(tmp_path,monkeypatch):
    import benchmarks.run_corrected_campaign_v58 as runner
    monkeypatch.setattr(runner,'OUT',tmp_path)
    (tmp_path/'regression_gate.json').write_text('{"gate_pass":false}')
    with pytest.raises(RuntimeError,match='old failure'):
        runner.prepare_independent(tmp_path/'joint_independent')
    assert not (tmp_path/'joint_independent').exists()


def test_headerless_observation_archive_preserves_schema_bytes_and_data(tmp_path):
    from benchmarks.lossless_raw_archive_v58 import archive,restore
    schema=tmp_path/'stream_schema.json';schema.write_text('{"columns":3,"names":["time","a","b"]}')
    before=schema.read_bytes();original=np.arange(90,dtype='<u8').tobytes()+b'end'
    source=tmp_path/'transient_stream.f64';source.write_bytes(original);r=archive(source)
    restored=tmp_path/'restored.f64';restore(source.with_suffix('.f64.tmraw'),restored)
    assert restored.read_bytes()==original and schema.read_bytes()==before
    assert r['columns']==3 and r['partial_tail_bytes']==3


def test_numerical_version_preserves_all_physical_and_model_thresholds(tmp_path,monkeypatch):
    import copy
    import benchmarks.run_corrected_campaign_v58 as runner
    original=json.loads((runner.BASE/'final_freeze.json').read_text())
    monkeypatch.setattr(runner,'OUT',tmp_path)
    monkeypatch.setattr(runner,'common_assets',lambda:(copy.deepcopy(original),{}))
    (tmp_path/'regression_gate.json').write_text('{"gate_pass":true}')
    stage=tmp_path/'fresh';stage.mkdir();record=runner.stage_record(stage,'fresh_independent_numerical_version')
    assert record['criteria']==original['criteria'] and record['budgets']==original['budgets']
    assert record['active_card']==original['active_card'] and record['sealed_seed']==20270108
    assert not (stage/'seed_opened.json').exists()


def test_controller_starts_real_native_file_sink_and_finishes_software_fixture(tmp_path,monkeypatch):
    import builtins,os,platform
    import benchmarks.run_corrected_campaign_v58 as runner
    from benchmarks.streaming_cli_probe_v47 import EXE,raw_read
    if os.name!='nt' or not EXE.exists():pytest.skip('qualified Windows native runtime required')
    stage=tmp_path/'software_controller_fixture';stage.mkdir()
    record=dict(workspace_root=str(runner.ROOT),assets_sha256={},environment=dict(python=platform.python_version(),numpy=np.__version__),
        budgets=dict(max_raw_bytes=10*2**20,min_free_disk_bytes=0,max_wall_s=60,max_child_RSS_bytes=2*2**30,checkpoint_wall_s={'214':60}))
    (stage/'final_freeze.json').write_text(json.dumps(record))
    c=dict(expected_stop_ns=40,output_columns=['v(n)'],scope='software fixture only;not scientific acceptance')
    body='* software-only RC controller fixture\nVin input 0 PULSE(0 1.8 1n .2n .2n 3n 20n 1)\nR input n 1k\nC n 0 1p\n'
    runner.commit(stage,record,c,{},np.array([4.]),np.array([4.]),body,{'software_fixture_only':True})
    real_exec=builtins.exec;calls=[]
    def intercept(code,namespace):
        real_exec(code,namespace)
        def fixture_evaluation(raw,folder,contract,predictions,limit,live):
            a,n=raw_read(raw);last=float(a[-1,n.index('time')]*1e9)
            calls.append((limit,live,last))
            return dict(gate_pass=abs(last-40)<1e-6,software_fixture_only=True)
        namespace['evaluate']=fixture_evaluation
    monkeypatch.setattr(runner,'exec',intercept,raising=False)
    runner.execute(stage,regression=True)
    assert calls==[(214,True,40.0)] or (len(calls)==1 and calls[0][:2]==(214,True) and abs(calls[0][2]-40)<1e-6)
    summary=json.loads((stage/'summary.json').read_text())
    assert summary['terminal_complete'] and summary['global_D1_D6_acceptance'] is False
    assert 'set no_mem_check' in (stage/'physical/case.cir').read_text()
    assert not (stage/'seed_opened.json').exists()
