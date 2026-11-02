"""Pure workload builder plus guarded sealed-seed entry. No automatic seed use."""
import hashlib,json,re
from pathlib import Path
import numpy as np
from benchmarks.retimed_run_v56 import OUT
from benchmarks.periodic_grid_v57 import grid as periodic_grid
from benchmarks.validation_guard_v41 import verify_frozen
from benchmarks.live_spice import write,now


def inputs(seed):
    # Always generate one fixed5000target workload. Prefix checkpoints do not
    # alter RNG draw counts or create independent datasets.
    rng=np.random.Generator(np.random.PCG64(seed));warm=100;targets=5000;tail=3
    count=warm+targets+4+tail;widths=rng.uniform(4,196,count)
    clean=100+80*np.sin(2*np.pi*np.arange(count)/128)
    widths[warm:warm+12]=[4,4,4,196,196,196,4,196,100,4,196,4]
    blocks={'gaussian':(900,2051),'salt_pepper':(2200,3351),'mixed':(3500,4651)}
    for kind,(lo,hi) in blocks.items():
        sl=slice(warm+lo,warm+hi+1);n=hi-lo+1;values=clean[sl].copy()
        if kind in ['gaussian','mixed']:values+=rng.normal(0,6 if kind=='gaussian' else 4,n)
        if kind in ['salt_pepper','mixed']:
            mask=rng.random(n)<(.1 if kind=='salt_pepper' else .05)
            replacements=np.where(rng.random(n)<.5,4.,196.);values[mask]=replacements[mask]
        widths[sl]=np.clip(values,4,196)
    widths[-3:]=[4,196,100]
    # The physical source and prediction receive the same integer-fs command.
    widths=np.rint(widths*1e6)/1e6
    return widths,clean,dict(warmup=warm,targets=targets,tail=tail,input_count=count,
        target_checkpoints=[128,1000,5000],checkpoints_nested=True,reset_between_checkpoints=False,
        noise_target_blocks=blocks,noise_scored_centers={'gaussian':[964,1987],'salt_pepper':[2264,3287],'mixed':[3564,4587]},
        clean_reference_center_input_offset=warm+2,new_input_generation_only_after_freeze=True,
        source_noise='input corruption;not device thermal/flicker/jitter qualification')


def physical_case(widths):
    source=OUT/'tail_development/case.cir';s=source.read_text();c=json.loads((source.parent/'contract.json').read_text())
    lines=['Vinput tin 0 PWL(0 0']
    for k,width in enumerate(widths):
        t=40+200*k;lines.append(f'+ {t:.9f}n 0 {t+.2:.9f}n 1.8 {t+width:.9f}n 1.8 {t+width+.2:.9f}n 0')
    lines[-1]+=')'
    s,n=re.subn(r'(?m)^Vinput tin 0 PWL\([^\n]*(?:\n\+[^\n]*)*\)','\n'.join(lines),s)
    if n!=1:raise ValueError('input source inventory changed')
    for name,w in c['words'].items():
        w['count']=len(widths)-(0 if name=='front' else int(name[7:]) if name.startswith('history') else 4)
        for bank in range(3):
            count=len(range(bank,w['count'],3))
            pattern=r'(?m)^(V'+name+r'_\S+'+str(bank)+r' \S+_drive 0 PULSE\([^\n]* 600n )\d+(\))$'
            s,n=re.subn(pattern,lambda m:m[1]+str(count)+m[2],s)
            expected=4 if name=='front' else 5
            if n!=expected:raise ValueError('clock inventory changed: '+name)
    stop=6310+200*(len(widths)-27)
    helper,mesh=periodic_grid(s)
    electrical=s.split('.tran')[0]+helper
    # Expected output widths are evaluation data only, absent from this deck.
    return electrical,dict(stop_ns=stop,words={k:{p:v for p,v in w.items() if p!='expected'} for k,w in c['words'].items()},
        clock_sources=87,physical_words=6,uniform_state_grid_ps=8,reset_grid_depends_on_input_values=False,
        periodic_numerical_mesh=mesh)


def open_sealed():
    folder=OUT/'joint_independent';freeze=folder/'final_freeze.json'
    if not freeze.is_file():raise RuntimeError('Sealed input refused:official freeze does not exist')
    record=json.loads(freeze.read_text());verify_frozen(Path(record['workspace_root']),record['assets_sha256'])
    if not record.get('all_prerequisites_pass'):raise RuntimeError('Sealed input refused:prerequisites not qualified')
    # Exclusive record precedes any RNG call and prevents accidental reopening.
    with (folder/'seed_opened.json').open('x',encoding='utf-8') as f:json.dump(dict(utc=now(),freeze_sha256=hashlib.sha256(freeze.read_bytes()).hexdigest()),f)
    widths,clean,protocol=inputs(record['sealed_seed'])
    np.savez(folder/'frozen_inputs.npz',widths_ns=widths,clean_ns=clean)
    electrical,physical=physical_case(widths);(folder/'generated_electrical.cir').write_text(electrical,encoding='ascii')
    write(folder/'input_protocol.json',dict(protocol=protocol,physical=physical,numpy_version=np.__version__,no_physics_started=True))


if __name__=='__main__':open_sealed()
