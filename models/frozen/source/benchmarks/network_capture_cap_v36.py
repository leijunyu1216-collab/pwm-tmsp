"""Complete registered network with phase-selective capture capacitance.

This is development, not independent validation. The original frontend and
all old bias/reference/clock resources are retained. Added switches, clock
receivers, auxiliary capacitors and clock sources are explicitly counted.
"""
import argparse,json,re
from benchmarks.physical_common import ROOT,footer,sha
from benchmarks.live_spice import write
from benchmarks.flatten_nominal_v30 import convert
from benchmarks import spice_stream_probe_v34 as probe
BASE=ROOT/'artifacts/wide_network_v34';LOCAL=ROOT/'artifacts/capture_cap_v36';OUT=ROOT/'artifacts/capture_network_v36'
LOCAL_LABEL='smalltg_cap15ff_ref13434'

def prepare(label,local_label=LOCAL_LABEL):
    a=json.loads((LOCAL/(local_label+'_assessment.json')).read_text());assert a['execution']['simulation_valid'] and a['exact_events']
    assert a['max_width_error_ns']<.5 and max(abs(a['anchor_min_ns']),abs(a['anchor_max_ns']))<.2
    local=(LOCAL/'inputs'/local_label/'case.cir').read_text();lc=json.loads((LOCAL/'inputs'/local_label/'contract.json').read_text());assert lc['cap_ff'] in [1.,1.5] and lc['tg_width_um']==.42 and lc['reference_V']==1.3434
    cap=lc['cap_ff']
    src=BASE/'inputs/endpoint_development12/case.cir';c=json.loads((src.parent/'contract.json').read_text());s=src.read_text().split('.tran')[0]
    block=lambda text,name:re.search(r'(?ims)^\.subckt '+name+r' [^\n]*\n.*?^\.ends[^\n]*',text)[0]
    oldfront=block(s,'front_word');bank=block(local,'flow_bank').replace('.subckt flow_bank ','.subckt flow_bank_store ',1)
    at=s.index('.subckt front_word');s=s[:at]+bank+'\n'+block(local,'cap_tg')+'\n'+s[at:]
    old=block(s,'store_word');head,body=old.split('\n',1);head+=' captrack0_drive captrack1_drive captrack2_drive'
    body,n=re.subn(r'(?m)^(Xbank([012]) [^\n]* )flow_bank$',lambda m:m[1]+f'captrack{m[2]} flow_bank_store',body);assert n==3
    rtop=(1.8/1.3434-1)*100000;body,n=re.subn(r'(?m)^(Rtop vdd fixed )\S+',lambda m:m[1]+f'{rtop:.12f}',body);assert n==1
    receivers=''.join(f'Rcaptrack{b} captrack{b}_drive captrack{b} 100\n' for b in range(3));body=body.replace('.ends',receivers+'.ends',1)
    s=s.replace(old,head+'\n'+body,1)
    def connect(m):return m[1]+''.join(f' {m[2]}_captrack{b}_drive' for b in range(3))+' store_word'
    s,n=re.subn(r'(?m)^(X(\S+) [^\n]*) store_word$',connect,s);assert n==21
    def advance(m):return m[1]+f'{float(m[2])-2:.9f}n'+m[3]
    s,n=re.subn(r'(?m)^(V(?!front_)\S+_ped[012] .*?PULSE\(0 1.8 )([\d.]+)n([^\n]*)$',advance,s);assert n==63
    for name,w in c['words'].items():
        if name=='front':continue
        for b in range(3):
            count=len(range(b,w['count'],3));phase=w['phase']+200*b
            s+=f'V{name}_captrack{b} {name}_captrack{b}_drive 0 PULSE(0 1.8 {phase-232:.9f}n .2n .2n 232.6n 600n {count})\n'
    assert block(s,'front_word')==oldfront
    cols=c['output_columns']+[f'v(x{name}.{node})' for name in c['words'] for node in ['pbias','nbias']]
    for name in c['words']:
        if name=='front':continue
        for b in range(3):
            prefix=f'x{name}.xbank{b}';cols += [f'v(x{name}.captrack{b})']+[f'v({prefix}.{n})' for n in ['capbar','auxdata','auxref']]
            cols += [f'@mflat_{prefix}.xcap{branch}.x{kind}[id]' for branch in ['d','r'] for kind in ['n','p']]
    folder=OUT/'inputs'/label;folder.mkdir(parents=True,exist_ok=True);path=folder/'case.cir';assert not path.exists();path.write_text(s+footer(4110,cols,.125),encoding='ascii');flat=convert(path,OUT/'flat'/label)
    inventory=json.loads((OUT/'flat'/label/'contract.json').read_text());assert inventory['mos_instances']==7370 and inventory['equivalent_mos']==8888
    assert abs(inventory['explicit_cap_pF']-(286.44+126*cap*.001))<1e-8
    assert len(re.findall(r'(?m)^V\S+ \S+_drive 0 PULSE',s))==327
    c.update(label=label,scope='full-network capture-only capacitance development;not independent;not released card',
        capture_only_cap_ff=cap,capture_switch_width_um=.42,storage_reference_V=1.3434,storage_pedestal_advance_ns=2.,storage_data_capacitance_pf=1.,
        netlist_sha256=sha(path),flat_sha256=sha(flat),builder_sha256=sha(__file__),local_development_sha256=sha(LOCAL/(local_label+'_assessment.json')),
        output_columns=cols,clock_sources=327,added_clock_sources=63,added_MOS_instances=378,mos_instances=7370,equivalent_mos=8888.,explicit_cap_pF=inventory['explicit_cap_pF'],
        physical_parameters_unchanged=False,frontend_parameters_unchanged=True,full_layout_area_um2=None,physical_clock_generation_cost=None,
        capture_cap_phase_checks_required=True,independent_seed_opened=False)
    write(folder/'contract.json',c);write(OUT/'progress.json',dict(stage='capture_cap_network_development',active_label=label,gate_pass=None,independent_seed_opened=False,system_speed_benchmark=False));return flat

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('label',nargs='?',default='capture15ff_ref13434');p.add_argument('--local-label',default=LOCAL_LABEL);p.add_argument('--prepare-only',action='store_true');a=p.parse_args();flat=prepare(a.label,a.local_label)
    if not a.prepare_only:
        probe.OUT=OUT/'stage';probe.probe(flat,a.label,7200,retain_all=False,max_rss_bytes=2400000000,stream_vectors=True,num_threads=8,raw_only=True)
