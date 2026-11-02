"""Remove unused geometry bins only; retain all matching-bin physics verbatim."""
import json,re
from pathlib import Path
from benchmarks.physical_common import ROOT,sha
from benchmarks.live_spice import write,now

OUT=ROOT/'artifacts/cost_diagnosis_v56/model_pruning'


def build(source=None,folder=None):
    destination=Path(folder) if folder is not None else OUT
    destination.mkdir(parents=True,exist_ok=False)
    source=Path(source) if source is not None else ROOT/'artifacts/observation_recovery_v45/retry1/flat/shared_challenge24/case.cir'
    text=source.read_text();geos={}
    for line in text.splitlines():
        if not line.startswith('mflat_'):continue
        model=line.split()[5]
        wl=tuple(float(re.search(r'\b'+p+r'=(\S+)',line)[1]) for p in ['w','l'])
        if model in geos:assert geos[model]==wl
        geos[model]=wl
    model_paths=[Path(p) for p in re.findall(r'(?m)^\.include "([^"]*geometry_models.cir)"',text)]
    if len(model_paths)!=1:raise ValueError('need one geometry model include')
    old=model_paths[0];body=old.read_text()
    bins=[s for s in re.split(r'(?m)(?=^\.model )',body) if s.startswith('.model')]
    kept=[];mapping={}
    for model,(w,l) in geos.items():
        matches=[]
        for block in bins:
            name=re.match(r'\.model (\S+)',block)[1]
            if not name.startswith(model+'.'):continue
            bounds={p:float(re.search(r'\b'+p+r'\s*=\s*(\S+)',block)[1]) for p in ['lmin','lmax','wmin','wmax']}
            if bounds['lmin']-1e-18<=l<=bounds['lmax']+1e-18 and bounds['wmin']-1e-18<=w<=bounds['wmax']+1e-18:
                matches.append((name,block))
        # Ambiguous boundary bins are all retained; do not guess native choice.
        if not matches:raise ValueError('no geometry bin: '+model)
        mapping[model]=dict(w=w,l=l,retained_bins=[name for name,_ in matches])
        kept.extend(block for _,block in matches)
    target=destination/'geometry_models.cir';target.write_text(''.join(kept),encoding='ascii')
    # Keep names and all numerical body parameters; strip only unused bins.
    result=text.replace(old.as_posix(),target.as_posix());assert result!=text
    case=destination/'case.cir';case.write_text(result,encoding='ascii')
    contract=dict(utc=now(),tier='exposed exact-bin-pruning candidate;equivalence not yet qualified',
        original_source_sha256=sha(source),original_models_sha256=sha(old),pruned_source_sha256=sha(case),
        pruned_models_sha256=sha(target),original_bin_count=len(bins),retained_bin_count=len(kept),mapping=mapping,
        kept_blocks_verbatim=True,instances_sources_parameters_unchanged=True,new_independent_seed_opened=False,
        required_validation='same native active bin,edges,state,current and energy;no gate inheritance before validation')
    write(destination/'contract.json',contract)
    print(json.dumps({k:v for k,v in contract.items() if k!='mapping'},indent=2))
    return case


if __name__=='__main__':build()
