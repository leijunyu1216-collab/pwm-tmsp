"""Complete live21-word median mapping using physically connected fall storage.

An additional real frontend word converts the original rising-aligned input.
Distributed qualified word-bias macros are explicitly retained and costed in
the flattened inventory. This development control does not qualify shared bias,
layout, noise, long-run accuracy, or system speed.
"""
import argparse,json,re
import numpy as np
from tmsp.networks import median_network
from benchmarks.physical_common import ROOT,pwl,footer,sha,edges
from benchmarks.live_spice import write
from benchmarks.flatten_nominal_v30 import convert
from benchmarks.analyze_pair_v12 import load
from benchmarks import spice_stage_probe as sp
OUT=ROOT/'artifacts/median_fall_v33'
SOURCE=ROOT/'artifacts/fall_repair_v32/inputs/independent128/case.cir'

def topology(targets):
    net=median_network(5);n=len(targets);words={};gates={};wires=['front']
    for j in range(1,5):
        out=f'history{j}';words[out]=dict(input=wires[-1],phase=460+200*(j-1),count=n-j,expected=list(map(float,targets[:n-j])),kind='history',delay=j)
        wires.append(out)
    levels=[-1]*5;operations={}
    for ix,(i,j,lo,hi) in enumerate(net.operations):
        stage=max(levels[i],levels[j])+1;operations.setdefault(stage,[]).append((ix,i,j,lo,hi))
        if lo:levels[i]=stage
        if hi:levels[j]=stage
    ref=np.array([targets[k-np.arange(5)] for k in range(4,n)])
    for stage in range(net.depth):
        comb=wires.copy();before=ref.copy()
        for ix,i,j,lo,hi in operations.get(stage,[]):
            for keep,ch,kind in [(lo,i,'and'),(hi,j,'or')]:
                if keep:
                    name=f'g{ix}_{kind}';gates[name]=dict(inputs=[wires[i],wires[j]],kind=kind)
                    comb[ch]=name;ref[:,ch]=np.minimum(before[:,i],before[:,j]) if kind=='and' else np.maximum(before[:,i],before[:,j])
        wires=[]
        for ch,node in enumerate(comb):
            out=f's{stage}c{ch}';words[out]=dict(input=node,phase=1260+200*stage,count=n-4,expected=ref[:,ch].tolist(),kind='sort',stage=stage,channel=ch)
            wires.append(out)
    needed=set()
    def visit(node):
        if node in needed:return
        needed.add(node)
        if node in words:visit(words[node]['input'])
        if node in gates:
            for parent in gates[node]['inputs']:visit(parent)
    visit('s4c2');words={k:v for k,v in words.items() if k in needed};gates={k:v for k,v in gates.items() if k in needed}
    assert len(words)==21
    assert np.allclose(words['s4c2']['expected'],np.median(np.lib.stride_tricks.sliding_window_view(targets,5),axis=1))
    return words,gates

def prepare(label):
    pair=ROOT/'artifacts/cascade_handoff_v33';lock=json.loads((pair/'model_lock.json').read_text())
    assert json.loads((pair/'independent16_assessment.json').read_text())['gate_pass']
    assert sha(SOURCE)==lock['source_sha256']
    targets=np.array([30.,80.,45.,90.,60.,35.,70.,30.,80.,45.,90.,60.]);words,gates=topology(targets)
    raw=SOURCE.read_text().split('.tran')[0]
    definitions='\n'.join(m[0] for m in re.finditer(r'(?ims)^\.subckt\s+\S+[^\n]*\n.*?^\.ends[^\n]*',raw))+'\n'
    top=re.sub(r'(?ims)^\.subckt\s+\S+[^\n]*\n.*?^\.ends[^\n]*','',raw)
    top=re.sub(r'(?m)^V(?:input|en[012]|ped[012]|reset[012]|replay[012])[^\n]*(?:\n\+[^\n]*)*\n','',top)
    head='\n'.join(line for line in top.splitlines() if not re.match(r'^[XRC]',line))+'\n'
    body='\n'.join(line for line in top.splitlines() if re.match(r'^[XRC]',line))+'\n'
    pins=[f'{name}{b}_drive' for b in range(3) for name in ['en','ped','reset','replay']]
    deck=head+definitions+'''.subckt logic_and a b out vdd vss
Xn a b n vdd vss nand2
Xi n out vdd vss inv
.ends
.subckt logic_or a b out vdd vss
Xn a b n vdd vss nor2
Xi n out vdd vss inv
.ends
'''
    for kind in ['front','store']:
        b=body
        if kind=='store':
            rtop=(1.8/lock['reference_V']-1)*100000
            b=re.sub(r'(?m)^(Rtop vdd fixed )\S+',lambda m:m[1]+f'{rtop:.12f}',b)
        deck+=f'.subckt {kind}_word tin out vdd vss '+' '.join(pins)+'\n'+b+'.ends\n'
    allwords={'front':dict(input='tin',phase=260,count=len(targets),expected=targets.tolist(),kind='frontend'),**words}
    clock_count=0
    for name,row in allwords.items():
        ext=[f'{name}_{pin}' for pin in pins];kind='front' if name=='front' else 'store'
        deck+=f'X{name} {row["input"]} {name} vdd vss '+' '.join(ext)+f' {kind}_word\n'
        for b in range(3):
            phases=row['phase']+200*np.arange(row['count'])[b::3]
            if name=='front':settings=[('en',phases-221.,199.6),('ped',phases-244.6,24.),('reset',phases-249.,3.),('replay',phases+.2,350.)]
            else:settings=[('en',phases-200.6,201.2),('ped',phases-220.6,24.),('reset',phases-229.,3.),('replay',phases+.2+lock['replay_guard_ns'],350.)]
            for signal,start,width in settings:
                assert len(start)>0
                deck+=f'V{name}_{signal}{b} {name}_{signal}{b}_drive 0 PULSE(0 1.8 {start[0]:.9f}n .2n .2n {width:.9f}n 600n {len(start)})\n';clock_count+=1
    for name,row in gates.items():deck+=f'X{name} '+ ' '.join(row['inputs'])+f' {name} vdd vss logic_{row["kind"]}\n'
    deck+=pwl('input','tin',40+200*np.arange(len(targets)),targets-.2)
    cols=['v(tin)']+[f'v({name})' for name in allwords]+[f'v({name})' for name in gates]+['i(vdd)']
    folder=OUT/'inputs'/label;folder.mkdir(parents=True,exist_ok=True);path=folder/'case.cir';assert not path.exists()
    stop=4110.;path.write_text(deck+footer(stop,cols,.125),encoding='ascii');flat=convert(path,OUT/'flat'/label)
    assert len(re.findall(r'(?m)^V\S+ (?!vdd|vss)\S+ 0 PULSE',flat.read_text()))==clock_count==264
    contract=dict(label=label,scope='complete21-word median plus real frontend;development;distributed word bias macros',targets_ns=targets.tolist(),
        words=allwords,gates=gates,sorting_history_words=21,frontend_words=1,total_physical_words=22,physical_banks=66,
        bias_macros=22,shared_bias_qualification=False,source_sha256=sha(SOURCE),builder_sha256=sha(__file__),model_lock_sha256=sha(pair/'model_lock.json'),
        netlist_sha256=sha(path),flat_sha256=sha(flat),expected_stop_ns=stop,
        gate=dict(all_initialized_ports=True,width_max_ns=2.,anchor_max_ns=.2,min_median_outputs=5),no_network_refit=True,
        resource_note='all22 finite bias/reference macros,66 banks and14 six-transistor logic gates retained;no layout area or SNR claim')
    write(folder/'contract.json',contract);return flat

def score(label):
    c=json.loads((OUT/'inputs'/label/'contract.json').read_text());folder=OUT/'stage'/label;execution=json.loads((folder/'summary.json').read_text())
    t,z=load(folder);tn=t*1e9;cache={name:edges(tn,z[name]) for name in z if name not in ['time','vdd#branch']};rows=[];groups=[]
    def measure(name,F):
        r,f=cache[name];r=r[(r>=F-199)&(r<F+1)];f=f[(f>=F-199)&(f<F+1)]
        return (float(f[0]-r[0]),float(f[0])) if len(r)==len(f)==1 and f[0]>r[0] else None
    for name,w in c['words'].items():
        rr=[]
        for k,expected in enumerate(w['expected']):
            F=w['phase']+200+200*k+.1;m=measure(name,F);parent=measure(w['input'],F-200)
            row=dict(node=name,sample=k,expected_ns=expected,valid=m is not None,width_ns=m[0] if m else None,
                width_error_ns=m[0]-expected if m else None,anchor_error_ns=m[1]-F if m else None,
                local_input_width_ns=parent[0] if parent else None,local_width_change_ns=m[0]-parent[0] if m and parent else None)
            rr.append(row);rows.append(row)
        good=[r for r in rr if r['valid']];r,f=cache[name]
        groups.append(dict(node=name,targets=len(rr),valid=len(good),event_count_exact=len(r)==len(f)==len(rr),
            max_width_error_ns=max((abs(r['width_error_ns']) for r in good),default=None),max_anchor_error_ns=max((abs(r['anchor_error_ns']) for r in good),default=None)))
    good=[r for r in rows if r['valid']];med=[r for r in rows if r['node']=='s4c2'];maxw=max((abs(r['width_error_ns']) for r in good),default=None);maxa=max((abs(r['anchor_error_ns']) for r in good),default=None)
    portpass=all(g['valid']==g['targets'] and g['event_count_exact'] and g['max_width_error_ns']<=2 and g['max_anchor_error_ns']<=.2 for g in groups)
    result=dict(label=label,scope=c['scope'],simulation_valid=execution['simulation_valid'],initialized_output_targets=len(rows),complete_outputs=len(good),
        max_width_error_ns=maxw,max_anchor_error_ns=maxa,median_targets=len(med),median_valid=sum(r['valid'] for r in med),
        median_max_width_error_ns=max((abs(r['width_error_ns']) for r in med if r['valid']),default=None),
        gate_pass=bool(execution['simulation_valid'] and portpass and len(med)>=5),groups=groups,rows=rows,raw_sha256=sha(folder/'transient_stream.f64'),global_acceptance=False)
    write(OUT/(label+'_assessment.json'),result);print({k:v for k,v in result.items() if k not in ['groups','rows']},flush=True);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('label');p.add_argument('--score-only',action='store_true');a=p.parse_args()
    if a.score_only:score(a.label)
    else:
        flat=prepare(a.label);sp.OUT=OUT/'stage';sp.probe(flat,a.label,3600,retain_all=False,max_rss_bytes=2200000000,stream_vectors=True);score(a.label)
