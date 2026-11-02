import json, re
from pathlib import Path
import numpy as np
from benchmarks.periodic_grid_v57 import grid
from benchmarks.retimed_workload_v56 import inputs,physical_case,OUT
from benchmarks.freeze_joint_v57 import source_closure,electrical_closure
from benchmarks.physical_common import ROOT


def test_repeating_mesh_is_independent_of_input_values_and_horizon():
    widths,_,_=inputs(1)
    short,_=physical_case(widths[:27]);long,_=physical_case(widths)
    alternate,_=physical_case(np.full(27,100.))
    assert grid(short)==grid(long)==grid(alternate)
    assert grid(short)[1]['points_per_period']==204
    assert 'R=0' in grid(short)[0]


def test_all_clock_counts_cover_full_word_trajectory_without_new_checkpoint_reset():
    widths,_,_=inputs(1);s,c=physical_case(widths)
    assert c['stop_ns']==1022310
    clocks=re.findall(r'(?m)^V(\S+) \S+_drive 0 PULSE\([^\n]* 600n (\d+)\)$',s)
    assert len(clocks)==87
    for name,count in clocks:
        word,control=name.rsplit('_',1);bank=int(control[-1])
        assert int(count)==len(range(bank,c['words'][word]['count'],3))


def test_frozen_source_closure_includes_predictor_generator_and_scorers():
    files=source_closure(['joint_campaign_v57'])
    for name in ['retimed_prediction_v56','retimed_workload_v56','periodic_grid_v57',
                 'campaign_live_v57','bounded_phase_scorer_v56','assess_wide_v34','assess_capture_network_v36',
                 'validation_guard_v41','campaign_resources_v57','crossing_index_v56']:
        assert (ROOT/'benchmarks'/f'{name}.py').resolve() in files


def test_frozen_electrical_closure_contains_models_and_parameters():
    files=electrical_closure([OUT/'periodic_grid/case.cir'])
    assert (OUT/'pruned/geometry_models.cir').resolve() in files
    assert (ROOT/'artifacts/aux_cap_electrical_v54/nominal_mim.cir').resolve() in files
    assert any(p.name=='invariant.spice' for p in files)
