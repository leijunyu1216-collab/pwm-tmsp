"""Whole-word connectivity inventory before physical passive implementation."""
import json
import re
from collections import Counter
from pathlib import Path
from benchmarks.physical_common import ROOT,TB,sha
from benchmarks.live_spice import write

OUT=ROOT/'artifacts/register_implementation_v8'


def scalar(token):
    match=re.fullmatch(r'([\d.eE+-]+)(meg|[fpnumk]?)',token.lower())
    if not match:raise ValueError(token)
    return float(match[1])*{'':1,'f':1e-15,'p':1e-12,'n':1e-9,'u':1e-6,'m':1e-3,'k':1e3,'meg':1e6}[match[2]]


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source=next((ROOT/'artifacts/wide_register_validation_v8').glob('n228_*/case.cir'))
    # Electrical definitions are independent of the random input waveform.
    files=[TB/'cells.cir',source.parent/'capture_ratio.cir',source.parent/'scaled_replay.cir']
    definitions={}
    for file in files:
        for m in re.finditer(r'(?ims)^\.subckt\s+(\S+)\s+([^\n]+)\n(.*?)^\.ends[^\n]*',file.read_text()):
            definitions[m[1].lower()]=(m[2].lower().split(),m[3].splitlines())
    devices=[]
    def expand(lines,prefix,mapping):
        def net(n):return mapping.get(n.lower(),prefix+n.lower()) if n!='0' else '0'
        for line in lines:
            line=line.strip()
            if not line or line[0] not in 'XxRrCc':continue
            words=line.split();kind=words[0][0].upper();name=prefix+words[0].lower()
            if kind=='X':
                index=next((i for i,w in enumerate(words[1:],1) if '=' in w),len(words))
                model=words[index-1].lower();nodes=[net(n) for n in words[1:index-1]]
                if model in definitions:
                    pins,body=definitions[model];assert len(pins)==len(nodes),(name,pins,nodes)
                    expand(body,name+'.',dict(zip(pins,nodes)))
                else:
                    assert model.startswith('sky130_fd_pr__') and len(nodes)==4,model
                    params=dict(w.split('=',1) for w in words[index:])
                    devices.append(dict(name=name,kind='MOS',model=model,nodes=nodes,w_um=scalar(params['w'])*1e6,l_um=scalar(params['l'])*1e6))
            else:
                devices.append(dict(name=name,kind=kind,nodes=[net(n) for n in words[1:3]],value_si=scalar(words[3]),physical_model_bound=False))
    expand(source.read_text().split('.control')[0].splitlines(),'',{n:n for n in ['vdd','vss','pbias','nbias','tin','y']})
    write(OUT/'whole_word_devices.json',dict(source_schematic=str(source.relative_to(ROOT)),source_schematic_sha256=sha(source),
        definition_sha256={str(p.relative_to(ROOT)):sha(p) for p in files},devices=devices,
        ground_equivalence='0 and vss connected by the original zero-volt Vss source;not silently merged in inventory'))
    groups={}
    for d in devices:
        group=('bias' if d['name'].startswith('xbias.') else 'VTC' if d['name'].startswith(('xrep0.','xrep1.','xrep2.'))
               else 'capture' if d['name'].startswith(('xcap0.','xcap1.','xcap2.')) else 'demux_mask_merge_and_load')
        groups.setdefault(group,Counter())[d['kind']]+=1
    totals=Counter(d['kind'] for d in devices)
    capsum=sum(d['value_si'] for d in devices if d['kind']=='C')
    result=dict(bank_count=3,counts=dict(totals),blocks={k:dict(v) for k,v in groups.items()},
        ideal_capacitance_total_pf=capsum*1e12,
        ideal_resistance_values_ohm=sorted(d['value_si'] for d in devices if d['kind']=='R'),
        full_word_gds_completed=False,full_word_drc_pass=False,full_word_lvs_pass=False,qualified_area_um2=None,
        pending_physical_binding=['VTC level-shifter and bias resistors:physical model/geometry/corner/parasitics',
            'all MIM capacitor geometries and model correspondence',
            'whole-word placement/routing including clock pins/distribution and bias',
            'whole-word process DRC/LVS and matched electrical revalidation'],
        current_power_binding='same schematic supply/control energy recorded in wide_register_validation_v8;functional gate failed',
        external_clock_generator_included=False,complete_frontier_accepted=False)
    write(OUT/'summary.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
