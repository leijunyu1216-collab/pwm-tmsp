"""Complete old214 scoring with fixed frontier;an explicit scope certificate."""
import json
from benchmarks.physical_common import ROOT
from benchmarks.live_spice import write,now
from benchmarks.snapshot_observations_v61 import evaluate
from benchmarks.qualify_observations_v60 import compare_scientific
from benchmarks.regression_admission_v61 import OUT,certify


def main():
    OUT.mkdir(exist_ok=True)
    parent=ROOT/'artifacts/ordered_observations_v60/regression'
    c=json.loads((parent/'evaluation_contract.json').read_text());pred=json.loads((parent/'predictions.json').read_text())
    result=evaluate(parent/'physical/transient.raw',OUT/'regression_scoring',c,pred,214,live=True)
    a=json.loads((OUT/'regression_scoring/system_live.json').read_text())
    b=json.loads((parent/'checkpoint_214/system_live.json').read_text());q=compare_scientific(a,b)
    exact=bool(q['gate_pass'] and q['maximum_numeric_field_difference']==0)
    write(OUT/'scoring_equivalence.json',dict(utc=now(),all_scientific_fields_exact=exact,
        comparison=q,new_seed_opened=False,original_execution_failure_overwritten=False))
    if not exact or not result['gate_pass']:raise RuntimeError('snapshot changes a scientific score')
    certificate=certify()
    print(json.dumps({k:v for k,v in certificate.items() if k not in ['selected_completion_checks','metrics','numerical_fidelity']},indent=2))


if __name__=='__main__':main()
