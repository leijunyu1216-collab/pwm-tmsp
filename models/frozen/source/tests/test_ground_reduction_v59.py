import json
from pathlib import Path
import pytest
from benchmarks.ground_alias_v59 import eliminate
from benchmarks.physical_common import ROOT


def test_elimination_only_relabels_node_positions_and_removes_zero_constraint():
    s=(ROOT/'artifacts/clock_repair_v58/runs/table_short27/case.cir').read_text().split('\n.save ')[0]
    actual,proof=eliminate(s)
    old={x.split()[0]:x.split() for x in s.splitlines() if x and x[0].lower() in 'mrcxv'}
    new={x.split()[0]:x.split() for x in actual.splitlines() if x and x[0].lower() in 'mrcxv'}
    assert set(old)-set(new)=={'Vss'} and not set(new)-set(old)
    for name,tokens in new.items():
        a=old[name]
        assert len(a)==len(tokens)
        for i,(before,after) in enumerate(zip(a,tokens)):
            if before!=after:assert before.lower()=='vss' and after=='0' and 1<=i<=4
    assert proof['eliminated_algebraic_unknowns']==2 and proof['stored_dynamic_states_removed'] is False


def test_current_feedback_or_nonzero_reference_is_not_silently_eliminated():
    with pytest.raises(ValueError,match='observed'):
        eliminate('Vss vss 0 0\nR1 a vss 1\n.save i(Vss)\n')
    with pytest.raises(ValueError,match='zero-volt'):
        eliminate('Vss vss 0 0.1\nR1 a vss 1\n')


def test_new_numerical_version_has_no_new_physical_parameters_or_new_inputs():
    from benchmarks.run_recovered_campaign_v59 import rewrite
    from benchmarks.clock_table_v58 import rewrite as clocks
    s=(ROOT/'artifacts/retimed_lattice_v56/tail_development/case.cir').read_text().split('.tran')[0]
    expected,_=eliminate(clocks(s)[0]);actual,_=rewrite(s)
    assert actual==expected and 'Vinput tin 0 PWL' in actual
    assert '.options method=gear maxord=2 reltol=1e-4 vntol=1e-7 abstol=1e-13 chgtol=1e-16' in actual
