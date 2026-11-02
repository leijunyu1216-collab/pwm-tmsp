"""Run the campaign's meaningful software checks without opening the seed."""
import json,os,subprocess,sys,time,tempfile
from pathlib import Path
from benchmarks.physical_common import ROOT
from benchmarks.live_spice import now,write
from benchmarks.retimed_workload_v56 import inputs,physical_case
from benchmarks.retimed_prediction_v56 import predict,OUT


def main():
    folder=ROOT/'artifacts/campaign_qualification_v57';report=folder/'software_tests.json'
    if report.exists():raise RuntimeError('immutable software qualification exists')
    tests=['test_campaign_v57','test_retimed_workload_v56','test_retimed_prediction_v56',
        'test_retimed_lattice_v56','test_crossing_index_v56','test_validation_guard','test_joint_independent_gate_v55']
    local_tmp=Path(tempfile.mkdtemp(prefix='pytest_v57_',dir=folder))/'tmp'
    command=[sys.executable,'-X','utf8','-m','pytest','-q','--basetemp',str(local_tmp),*[f'tests/{x}.py' for x in tests]]
    start=time.perf_counter();p=subprocess.run(command,cwd=ROOT,capture_output=True,timeout=120)
    number=len(list(folder.glob('software_tests_attempt*.log')))+1
    (folder/f'software_tests_attempt{number}.log').write_bytes(p.stdout+p.stderr)
    if p.returncode:raise RuntimeError('continuous campaign software checks failed')
    card=json.loads((OUT/'periodic_short27_prediction_cards.json').read_text())
    widths,_,protocol=inputs(1);electrical,physical=physical_case(widths)
    sample={k:predict(widths,card,k) for k in ['L0','L1','L2_static_candidate']}
    counts={k:len(v) for k,v in sample.items()}
    if any(v!=30628 for v in counts.values()):raise RuntimeError('full fixture output counts invalid')
    write(report,dict(utc=now(),command=command,returncode=p.returncode,gate_pass=True,
        runtime_s=time.perf_counter()-start,fixture_seed=1,new_independent_seed_opened=False,
        forecast_word_records=counts,physical_stop_ns=physical['stop_ns'],input_count=len(widths),
        physical_network_acceptance_claimed=False,system_speed_claimed=False))
    print(report.read_text())


if __name__=='__main__':main()
