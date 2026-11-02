"""TT geometry sharing with explicit hierarchical parameters/multiplicity.
No PVT/mismatch/noise equivalence is claimed by this deterministic adapter.
"""
import json,re
from pathlib import Path
from benchmarks.physical_common import PDK,TB,sha
from benchmarks.register_inventory_v8 import scalar
from benchmarks.live_spice import write
def number(token,env):
    text=token.strip('{}').lower()
    return env[text] if text in env else scalar(text)
def convert(source,folder):
    # Windows may return an extended-length prefix during concurrent mkdir.
    # ngspice treats //?/ in an include as a comment; use drive-letter paths.
    source=Path(Path(source).resolve().as_posix().removeprefix('//?/'))
    folder=Path(Path(folder).resolve().as_posix().removeprefix('//?/'));folder.mkdir(exist_ok=True,parents=True)
    raw=source.read_text();files=[TB/'cells.cir',source]
    for m in re.finditer(r'(?m)^\.include "([^"]+)"',raw):
        p=Path(m[1])
        if p.name in ['standard_cells_si.cir','capture_ratio.cir','replay.cir']:files.append(p)
    defs={}
    for file in files:
        for m in re.finditer(r'(?ims)^\.subckt\s+(\S+)\s+([^\n]+)\n(.*?)^\.ends[^\n]*',file.read_text()):
            header=m[2].lower().split();pins=[];params={}
            for token in header:
                if token=='params:':continue
                if '=' in token:
                    k,v=token.split('=',1);params[k]=v
                else:pins.append(token)
            defs[m[1].lower()]=(pins,params,m[3].splitlines())
    leaves=[];duplicated_bulk_tokens=[]
    def expand(lines,prefix,mapping,env,ancestor_m=1.):
        def node(s):return '0' if s=='0' else mapping.get(s.lower(),prefix+s.lower())
        for text in lines:
            text=text.strip()
            if not text or text[0].lower() not in 'xrc':continue
            w=text.split();kind=w[0][0].lower();name=prefix+w[0].lower()
            if kind!='x':leaves.append(dict(name=name,kind=kind,nodes=[node(s) for s in w[1:3]],value=number(w[3],env)));continue
            j=next((i for i,token in enumerate(w[1:],1) if '=' in token),len(w));model=w[j-1].lower();nodes=[node(s) for s in w[1:j-1]]
            assignments={k.lower():number(v,env) for k,v in (token.split('=',1) for token in w[j:])}
            if model in defs:
                pins,defaults,body=defs[model];assert len(pins)==len(nodes),(model,pins,nodes)
                child={**env,**{k:number(v,env) for k,v in defaults.items()},**assignments}
                expand(body,name+'.',dict(zip(pins,nodes)),child,ancestor_m*assignments.get('m',1.))
            else:
                if model.startswith('sky130_fd_pr__') and len(nodes)==5 and nodes[-1]==nodes[-2]:
                    duplicated_bulk_tokens.append(dict(name=name,raw_nodes=nodes.copy(),normalized_nodes=nodes[:-1]));nodes=nodes[:-1]
                assert model.startswith('sky130_fd_pr__') and len(nodes)==4,(name,model,nodes,w)
                assert set(assignments)<=set(['w','l','m','mult']),assignments
                leaves.append(dict(name=name,kind='mos',nodes=nodes,model=model,w=assignments['w'],l=assignments['l'],m=ancestor_m*assignments.get('m',1.)))
    top=re.sub(r'(?ims)^\.subckt\s+\S+[^\n]*\n.*?^\.ends[^\n]*','',raw)
    expand(top.split('.control')[0].splitlines(),'',{},{});geometries={};cards=[];body=[];provenance={}
    for d in leaves:
        name=('m' if d['kind']=='mos' else d['kind'])+'flat_'+d['name'];nodes=' '.join(d['nodes'])
        if d['kind']!='mos':body.append(f'{name} {nodes} {d["value"]:.17g}');continue
        key=(d['model'],d['w'],d['l'])
        if key not in geometries:
            alias='shape'+str(len(geometries));geometries[key]=alias
            family=d['model'].replace('sky130_fd_pr__','');path=PDK/f'cells/{family}/{d["model"]}__tt.pm3.spice'
            text=path.read_text();text=text[text.index('.model '):];text=re.sub(r'(?m)^\.ends[^\n]*','',text).replace(d['model']+'__model',alias)
            def substitute(m):
                s=m[1]
                for var,val in [('w',d['w']),('l',d['l']),('mult',1.)]:s=re.sub(r'(?<![\w])'+var+r'(?![\w])',f'({val:.17g})',s)
                assert not re.search(r'(?<![\w])(?:l|w|mult|ad|as|pd|ps|nf|sa|sb|sd)(?![\w])',s),s
                return '{'+s+'}'
            cards.append(re.sub(r'\{([^{}]*)\}',substitute,text));provenance[str(path)]=sha(path)
        body.append(f'{name} {nodes} {geometries[key]} w={d["w"]:.17g} l={d["l"]:.17g} m={d["m"]:.17g} ad=0 as=0 pd=0 ps=0 nrd=0 nrs=0 sa=0 sb=0 sd=0 nf=1')
    models=folder/'geometry_models.cir';models.write_text('\n'.join(cards),encoding='ascii')
    electrical,control=top.split('.control',1)
    top=re.sub(r'(?im)^[XRC][^\n]*\n','',electrical)+'.control'+control
    # Expanded project definitions are unnecessary; PDK corner includes remain.
    top=re.sub(r'(?m)^\.include "[^"]*/(?:cells|standard_cells_si|capture_ratio|replay)\.cir"\n','',top)
    at=top.index('.tran');text=top[:at]+f'.include "{models.as_posix()}"\n'+'\n'.join(body)+'\n'+top[at:]
    path=folder/'case.cir';path.write_text(text,encoding='ascii')
    write(folder/'contract.json',dict(source_sha256=sha(source),flat_sha256=sha(path),model_cards_sha256=sha(models),source_code_sha256=sha(__file__),
        leaf_count=len(leaves),mos_instances=sum(d['kind']=='mos' for d in leaves),equivalent_mos=sum(d['m'] for d in leaves if d['kind']=='mos'),
        explicit_cap_pF=sum(d['value'] for d in leaves if d['kind']=='c')*1e12,multiplicity_preserved=True,parameters_preserved=True,
        duplicated_bulk_token_normalization=duplicated_bulk_tokens,
        pdk_sources=provenance,qualification='TT nominal zero mismatch; waveform equivalence required separately; no layout or MC claim'))
    return path
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('folder');a=p.parse_args();print(convert(a.source,a.folder))
