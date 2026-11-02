"""New six-word physical development branch; not a relabeling of old acceptance.

All existing transistor AND/OR gates remain. Remove16intermediate sorting
words; retain real frontend,four history words,and one final retiming word.
The exposed24input source and retained cell parameters are unchanged.
"""
import itertools,json,re
from pathlib import Path
import numpy as np
from benchmarks.physical_common import ROOT,sha,footer
from benchmarks.live_spice import write,now
from benchmarks.prune_model_bins_v56 import build as prune

OUT=ROOT/'artifacts/retimed_lattice_v56'
BASE=ROOT/'artifacts/observation_recovery_v45/retry1'
KEEP={'front','history1','history2','history3','history4','s4c2'}


def logic(contract):
    mapping={k:v['input'] for k,v in contract['words'].items() if k not in KEEP}
    def root(n):
        seen=set()
        while n in mapping:
            if n in seen:raise ValueError('alias cycle')
            seen.add(n);n=mapping[n]
        return n
    gates={k:dict(v,inputs=[root(n) for n in v['inputs']]) for k,v in contract['gates'].items()}
    def evaluate(values):
        state=dict(zip(['front','history1','history2','history3','history4'],values));pending=dict(gates)
        while pending:
            advanced=False
            for name,g in list(pending.items()):
                if all(n in state for n in g['inputs']):
                    state[name]=(min if g['kind']=='and' else max)(state[n] for n in g['inputs']);del pending[name];advanced=True
            if not advanced:raise ValueError('combinational cycle')
        return state[root(contract['words']['s4c2']['input'])]
    booleans=list(itertools.product([0,1],repeat=5))
    permutations=list(itertools.permutations([4,52,100,148,196]))
    if not all(evaluate(v)==sorted(v)[2] for v in booleans+permutations):raise ValueError('lattice not median')
    return root,gates,dict(exhaustive_boolean_cases=32,distinct_width_permutations=120,all_pass=True,
        proof='Thresholding a min/max lattice commutes with its gates;Boolean majority at every threshold gives median width for common-fall PWM. Finite transistor delays still require physical validation.')


def build():
    OUT.mkdir(parents=True,exist_ok=False);folder=OUT/'inputs/development24';folder.mkdir(parents=True)
    src=BASE/'flat/distributed_challenge24/case.cir';old=src.read_text();c=json.loads((BASE/'inputs/distributed_challenge24/contract.json').read_text())
    root,gates,proof=logic(c);removed=set(c['words'])-KEEP;lines=[];removed_counts={}
    for line in old.split('.tran')[0].splitlines():
        tokens=line.split()
        if not tokens:lines.append(line);continue
        name=tokens[0].lower()
        if name[0] in 'mrcv' and any(name.startswith((f'mflat_x{w}.',f'rflat_x{w}.',f'cflat_x{w}.',f'v{w}_')) for w in removed):
            removed_counts[name[0]]=removed_counts.get(name[0],0)+1;continue
        if name.startswith('mflat_'):
            tokens[1:5]=[root(n) for n in tokens[1:5]];line=' '.join(tokens)
        if name.startswith('vs4c2_'):
            def earlier(m):return m[1]+format(float(m[2])-800,'.9f')+'n'
            line=re.sub(r'(PULSE\(0 1.8 )([\d.]+)n',earlier,line)
        lines.append(line)
    net='\n'.join(lines)+'\n';words={k:dict(v) for k,v in c['words'].items() if k in KEEP}
    words['s4c2']['phase']=1260
    cols=[]
    for expression in c['output_columns']:
        if any(re.search(r'(?<![a-z0-9_])x?'+re.escape(w)+r'(?:\.|\)|$)',expression) for w in removed):continue
        cols.append(expression)
    # Whole physical components, including local bias/reference and finite-R
    # controls, are retained for every surviving word. No ideal output oracle.
    assert len(re.findall(r'(?m)^mflat_',net))==sum(1 for x in old.splitlines() if x.startswith('mflat_'))-removed_counts.get('m',0)
    assert not any(re.search(r'\bx'+re.escape(w)+r'\.',net) for w in removed)
    # Add process-associated auxiliary implementation directly to this branch.
    mimroot=ROOT/'artifacts/aux_cap_electrical_v54';mim=json.loads((mimroot/'contract.json').read_text())
    net+='\n.param '+' '.join(k+'='+v for k,v in mim['parameters'].items())+'\n'
    net+=f'.include "{(mimroot/"nominal_mim.cir").as_posix()}"\n'
    def cap(m):
        if abs(float(m[3])-.75e-15)>1e-25:raise ValueError('unexpected auxiliary value')
        return f'Xphysical_{m[1]} {m[2]} 0 physical_aux'
    net,cap_count=re.subn(r'(?m)^cflat_(\S+\.caux(?:data|ref)) (\S+) 0 (\S+)$',cap,net)
    assert cap_count==30,cap_count
    net+='\n.subckt physical_aux port bottom\n'
    for i in range(4):
        a='port' if i==0 else f'n{i}';b='bottom' if i==3 else f'n{i+1}'
        net+=f'Xc{i} {a} {b} sky130_fd_pr__cap_mim_m3_1 w=1.075 l=1.075\n'
    for i in range(1,4):net+=f'Rregular{i} n{i} bottom 1e15\n'
    net+='.ends physical_aux\n'
    path=folder/'case.cir';path.write_text(net+footer(5710,cols,.125),encoding='ascii')
    c.update(label='development24',scope='new six-word retimed lattice plus nominal series-MIM schematic development;not independent',
        words=words,gates=gates,expected_stop_ns=5710.,expected_initialized_outputs=sum(w['count'] for w in words.values()),
        output_columns=cols,physical_words=6,storage_banks=18,removed_sorting_words=16,
        new_independent_seed_opened=False,independent_seed_opened=False,target_count=20,warmup=0,tail=0,
        physical_parameters_unchanged=False,physical_layout_qualified=False,internal_current_monitoring_complete=True,
        process_auxiliary_count=30,process_MIM_instances=120,regularizing_leakage_is_declared_assumption=True,
        architecture_proof=proof,source_sha256=sha(src),netlist_sha256=sha(path),builder_sha256=sha(__file__),
        schedule_change='only final word phase2060->1260;remove16sorting-word banks and their finite-R clocks;retained inputs/cells unchanged',
        no_claim_of_equivalence_to_old_internal_states=True,full_PPA=False)
    write(folder/'contract.json',c);write(OUT/'architecture_proof.json',proof)
    prune(path,OUT/'pruned')
    write(OUT/'prelaunch.json',dict(utc=now(),question='Can static common-fall lattice retiming reduce true transistor reference cost while retaining full endpoint and phase gates?',
        tier='exposed development',new_independent_seed_opened=False,formal_freeze_executed=False,
        reference_cards_refit=False,execution_precision='original Gear2/reltol1e-4/vntol1e-7/abstol1e-13/chgtol1e-16/maxstep0.125ns',
        endpoint_width_gate_ns=2.,falling_anchor_gate_ns=.2,internal_phase_gates_unchanged=True,
        cost_stop_ns=1510.,cost_budget_s=360.,full_short_stop_ns=5710.,scope='new schematic implementation;full layout,routing,PEX and system noise remain pending'))
    print(json.dumps({k:c[k] for k in ['physical_words','storage_banks','expected_initialized_outputs','process_MIM_instances','removed_sorting_words']},indent=2))


if __name__=='__main__':build()
