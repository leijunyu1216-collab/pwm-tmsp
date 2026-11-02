"""Endpoint development with dense physical state/timing observation.

No new independent network seed is opened. Frozen v33 physical parameters are
unchanged; only the explicitly requested development input and observation set
change. Uses binary-only post-run output, qualified by exact RC-pilot parity.
"""
import argparse,json,re
import numpy as np
from benchmarks.physical_common import ROOT,pwl,footer,sha
from benchmarks.median_fall_v33 import topology
from benchmarks.flatten_nominal_v30 import convert
from benchmarks.live_spice import write
from benchmarks import spice_stream_probe_v34 as probe
OUT=ROOT/'artifacts/wide_network_v34'
PARENT=ROOT/'artifacts/median_fall_v33'
TARGETS=np.array([4.,4.,4.,196.,196.,196.,4.,196.,100.,4.,196.,4.])

def prepare(label):
    assert json.loads((PARENT/'development4110_47klu_8threads_assessment.json').read_text())['gate_pass']
    assert json.loads((OUT/'raw_output_validation.json').read_text())['raw_identical']
    source=PARENT/'inputs/development4110/case.cir';old=source.read_text();deck=old.split('.tran')[0]
    deck,n=re.subn(r'(?m)^Vinput tin 0 PWL\([^\n]*(?:\n\+[^\n]*)*\)\n',lambda m:pwl('input','tin',40+200*np.arange(12),TARGETS-.2),deck);assert n==1
    deck=deck.replace('.temp 27','.temp 27\n.options klu')
    original=json.loads((PARENT/'inputs/development4110/contract.json').read_text());words,gates=topology(TARGETS)
    allwords={'front':{**original['words']['front'],'expected':TARGETS.tolist()},**words}
    cols=['v(tin)']+[f'v({name})' for name in allwords]+[f'v({name})' for name in gates]+['i(vdd)']
    for name in allwords:
        cols += [f'v(x{name}.fixed)',f'v(x{name}.captured_input)']
        for bank in range(3):
            prefix=f'x{name}.xbank{bank}'
            cols += [f'v(x{name}.{signal}{bank})' for signal in ['en','ped','reset','replay']]
            cols += [f'v({prefix}.{node})' for node in ['aa','ac0','ac','data','reference','cd','cf']]
            cols += [f'@mflat_{prefix}.xc{branch}.xsrc[id]' for branch in ['a','b','r']]
            cols += [f'@mflat_{prefix}.xreset{branch}[id]' for branch in ['d','r']]
    # The electrical statements, except input, match the accepted v33 design.
    strip_input=lambda s:re.sub(r'(?m)^Vinput tin 0 PWL\([^\n]*(?:\n\+[^\n]*)*\)\n','',s).replace('\n.options klu','')
    assert strip_input(deck)==strip_input(old.split('.tran')[0])
    folder=OUT/'inputs'/label;folder.mkdir(parents=True,exist_ok=True);path=folder/'case.cir';assert not path.exists()
    path.write_text(deck+footer(4110,cols,.125),encoding='ascii');flat=convert(path,OUT/'flat'/label)
    assert sha(OUT/'flat'/label/'geometry_models.cir')==sha(PARENT/'flat/development4110/geometry_models.cir')
    write(folder/'contract.json',dict(label=label,scope='4-196ns endpoint development;not independent or long continuous qualification',
        targets_ns=TARGETS.tolist(),words=allwords,gates=gates,parent_physical_source_sha256=sha(source),
        physical_parameters_unchanged=True,geometry_models_identical=True,netlist_sha256=sha(path),flat_sha256=sha(flat),builder_sha256=sha(__file__),
        output_columns=cols,thread_count=8,expected_stop_ns=4110.,period_ns=200.,independent_seed_opened=False,
        expected_initialized_outputs=sum(w['count'] for w in allwords.values()),expected_medians=words['s4c2']['expected'],
        gate=dict(all_outputs=True,width_max_ns=2.,anchor_max_ns=.2,required_median_endpoints_ns=[4,196],
            current_threshold_fractions=[.01,.05,.1],precharge_to_data_gap_positive=True,capture_to_replay_gap_positive=True,
            replay_before_data_decision=True,data_before_reference_decision=True,reference_before_replay_end=True,
            replay_before_next_reset_when_reuse_present=True),
        known_primitive_input_envelope_ns=[4,196],uncovered_condition_policy='report internal excursions and unobserved final bank reuse;no extrapolated qualification'))
    write(OUT/'progress.json',dict(stage='endpoint_development_running',active_label=label,gate_pass=None,independent_seed_opened=False,system_speed_benchmark=False))
    return flat

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('label',nargs='?',default='endpoint_development12');p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    path=prepare(a.label);print('prepared:',path,flush=True)
    if not a.prepare_only:
        probe.OUT=OUT/'stage';probe.probe(path,a.label,7200,retain_all=False,max_rss_bytes=2200000000,stream_vectors=True,num_threads=8,raw_only=True)
