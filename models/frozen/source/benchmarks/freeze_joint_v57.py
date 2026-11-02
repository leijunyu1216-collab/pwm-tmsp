"""One-time qualified schematic freeze. Does not read or generate sealed inputs."""
import ast, json, platform, re, sys, os
from pathlib import Path
import numpy as np
import psutil
from benchmarks.physical_common import ROOT
from benchmarks.retimed_run_v56 import OUT
from benchmarks.streaming_cli_probe_v47 import EXE
from benchmarks.audit_cost_diagnosis_v56 import stream_sha
from benchmarks.live_spice import now
from benchmarks.validation_guard_v41 import verify_frozen


def source_closure(start):
    todo=[ROOT/'benchmarks'/f'{x}.py' for x in start];found=set()
    while todo:
        p=todo.pop().resolve()
        if p in found:continue
        if not p.is_file():raise FileNotFoundError(p)
        found.add(p)
        tree=ast.parse(p.read_text(encoding='utf-8-sig'))
        for n in ast.walk(tree):
            modules=[]
            if isinstance(n,ast.ImportFrom) and n.module:
                if n.module=='benchmarks':modules=['benchmarks.'+x.name for x in n.names]
                elif n.module.startswith('benchmarks.'):modules=[n.module]
            elif isinstance(n,ast.Import):modules=[x.name for x in n.names if x.name.startswith('benchmarks.')]
            for mod in modules:
                target=ROOT/Path(*mod.split('.')).with_suffix('.py')
                if target.is_file():todo.append(target)
    return found


def electrical_closure(start):
    todo=list(start);found=set()
    while todo:
        p=todo.pop().resolve()
        if p in found:continue
        if not p.is_file():raise FileNotFoundError(p)
        found.add(p)
        for m in re.finditer(r'(?mi)^\s*\.(?:include|inc|lib)\s+"([^"]+)"',p.read_text()):
            dependency=Path(m[1]);todo.append(dependency if dependency.is_absolute() else p.parent/dependency)
    return found


def native_closure(start):
    import pefile
    directories=[EXE.parent,Path('D:/MinGW-w64/mingw64/bin'),Path(sys.executable).parent,
        Path(sys.executable).parent/'DLLs',Path(sys.executable).parent/'Library/bin',
        Path(np.__file__).parent/'_core',Path(np.__file__).parent.parent/'numpy.libs']
    system=Path(os.environ['SystemRoot'])/'System32';todo=list(start);found=set();os_imports=set()
    while todo:
        p=todo.pop().resolve()
        if p in found:continue
        found.add(p)
        image=pefile.PE(str(p),fast_load=True)
        try:
            image.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_IMPORT']])
            for entry in getattr(image,'DIRECTORY_ENTRY_IMPORT',[]):
                name=entry.dll.decode('ascii')
                candidates=[d/name for d in [p.parent,*directories]]
                match=next((q for q in candidates if q.is_file()),None)
                if match is not None:todo.append(match)
                elif (system/name).is_file() or name.lower().startswith(('api-ms-','ext-ms-')):os_imports.add(name)
                else:raise RuntimeError('unresolved non-system native dependency:'+name)
        finally:image.close()
    return found,sorted(os_imports)


def main():
    campaign=OUT/'joint_independent';campaign.mkdir(exist_ok=True)
    target=campaign/'final_freeze.json'
    if target.exists() or (campaign/'seed_opened.json').exists():raise RuntimeError('one-time freeze already exists or seed was opened')
    requirements={
        'periodic_physical_development':(OUT/'periodic_grid/qualification.json','all_gates_pass'),
        'full_horizon_clock_fixture':(ROOT/'artifacts/campaign_qualification_v57/clock_periodic_full_horizon/summary.json','gate_pass'),
        'checkpoint_fault_rejection':(ROOT/'artifacts/campaign_qualification_v57/checkpoint_fault_controls/summary.json','gate_pass'),
        'bounded_scientific_field_equivalence':(OUT/'bounded_scorer/equivalence.json','all_scientific_fields_exactly_equal'),
        'same_process_pause_resume':(OUT/'pause_resume_qualification.json','gate_pass'),
        'parent_death_containment':(ROOT/'artifacts/campaign_qualification_v57/resource_job.json','parent_death_job_kills_child'),
        'continuous_asset_tests':(ROOT/'artifacts/campaign_qualification_v57/software_tests.json','gate_pass')}
    evidence={}
    for key,(p,flag) in requirements.items():
        r=json.loads(p.read_text())
        if r.get(flag) is not True:raise RuntimeError('unqualified prerequisite:'+key)
        evidence[key]=dict(path=str(p.relative_to(ROOT)),sha256=stream_sha(p),passed=True)
    card=OUT/'periodic_short27_prediction_cards.json';trained=json.loads(card.read_text())
    if trained['training_run']!='periodic_short27' or trained['new_independent_seed_opened']:
        raise RuntimeError('training provenance invalid')
    assets=source_closure(['joint_campaign_v57','freeze_joint_v57'])
    assets |= electrical_closure([OUT/'tail_development/case.cir',OUT/'periodic_grid/case.cir'])
    assets |= {card,OUT/'tail_development/contract.json',OUT/'effective_contract.json',EXE,Path(sys.executable)}
    assets |= set(EXE.parent.glob('*.dll'))
    # Lock numerical package implementation as well as the reported versions.
    assets |= {Path(np.__file__),Path(np.polynomial.polynomial.__file__),Path(np.linalg.__file__)}
    compiled=set(Path(np.__file__).parent.rglob('*.pyd'))
    native,os_imports=native_closure([EXE,Path(sys.executable),*compiled]);assets |= native
    for p in [ROOT/'benchmarks/qualify_software_v57.py',*ROOT.glob('tests/test_campaign_v57.py')]:assets.add(p)
    def name(p):
        try:return str(p.relative_to(ROOT))
        except ValueError:return str(p)
    hashes={name(p.resolve()):stream_sha(p) for p in sorted(assets)}
    budget=dict(max_wall_s=60*3600,checkpoint_wall_s={'128':3*3600,'1000':14*3600,'5000':60*3600},
        max_child_RSS_bytes=2*2**30,max_evaluator_RSS_bytes=int(1.6*2**30),max_raw_bytes=120*2**30,min_free_disk_bytes=20*2**30)
    free=psutil.disk_usage(str(OUT)).free
    if free<budget['max_raw_bytes']+budget['min_free_disk_bytes']:
        raise RuntimeError('disk envelope unavailable;seed remains sealed')
    record=dict(utc=now(),activity='six-word common-fall retimed median;one continuous joint independent schematic test',
        workspace_root=str(ROOT),freeze_kind='schematic;not layout/PEX',all_prerequisites_pass=True,
        sealed_seed=20270107,active_card=str(card.relative_to(ROOT)),assets_sha256=hashes,prerequisite_evidence=evidence,
        environment=dict(python=platform.python_version(),numpy=np.__version__,platform=platform.platform(),
            operating_system_imports=os_imports,non_system_native_dependencies_hashed=True),
        scope=dict(corner='TT',temperature_C=27,supply_V=1.8,period_ns=200,input_slew_ns=.2,external_width_ns=[4,196],
            local_card_width_ns=[2,198],physical_words=6,storage_banks=18,external_finite_R_clock_ports=87,
            external_clock_generator_included=False,full_layout_PEX=False,noise='input Gaussian/salt-pepper/mixed;not transistor device noise'),
        criteria=dict(width_max_ns=2,fall_anchor_max_ns=.2,model_L2_edge_P99_percent_FS=1,
            numerical_edge_ns=.01,numerical_state_V=.001,numerical_energy_relative=.001,
            physical_event_counts_exact=True,all_internal_phase_gaps_positive=True,aux_off_current_fraction=.01,
            scheduled_valid_warmups=100,nested_targets=[128,1000,5000],final_target_edges=10000,
            tail_transactions=3,checkpoint_resets=False,test_refit=False,MM_profile_separate_percent_FS=1.5),
        budgets=budget,disk_free_bytes_at_freeze=free,
        reporting=dict(model='locked development-only static cubic candidate;does not qualify full dynamic L2 library',
            first_predict_then_physics=True,failed_activity_preserved=True,global_D1_D6_acceptance=False,
            system_PPA_complete=False,matched_system_speedup_qualified=False,public_release=False))
    verify_frozen(ROOT,hashes)
    with target.open('x',encoding='utf-8') as f:json.dump(record,f,indent=2,ensure_ascii=False)
    print(json.dumps(dict(freeze=str(target),sha256=stream_sha(target),assets=len(hashes),new_seed_opened=False),indent=2))


if __name__=='__main__':main()
