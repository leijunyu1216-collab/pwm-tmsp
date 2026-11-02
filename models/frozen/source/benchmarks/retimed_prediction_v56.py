"""Train static candidate endpoint cards on development only; predict open-loop.

This is a candidate model for the changed architecture, not a qualification of
the complete dynamic L2 library. No physical test-state trajectory is accepted
by predict(). No sample index or expected median is a fitted feature.
"""
import argparse,json,time
import numpy as np
from benchmarks.retimed_run_v56 import OUT
from benchmarks.physical_common import edges
from benchmarks.streaming_cli_probe_v47 import raw_read
from benchmarks.live_spice import now,write
from benchmarks.audit_cost_diagnosis_v56 import stream_sha


def fit(label):
    folder=OUT/'runs'/label;summary=json.loads((folder/'summary.json').read_text())
    phase=json.loads((folder/'phase_assessment.json').read_text())
    if not summary['terminal_complete'] or not phase['full_phase_contract_complete']:raise ValueError('training circuit not qualified')
    c=json.loads((OUT/'tail_development/contract.json').read_text());a,names=raw_read(folder/'transient.raw');t=a[:,names.index('time')]*1e9;cache={}
    def pulse(node,F):
        if node not in cache:cache[node]=edges(t,a[:,names.index('v('+node+')')])
        r,f=cache[node];r=r[(r>=F-199)&(r<F+1)];f=f[(f>=F-199)&(f<F+1)]
        if len(r)!=1 or len(f)!=1 or f[0]<=r[0]:raise ValueError('training event failure: '+node)
        return float(r[0]),float(f[0])
    cards={}
    # Only the original24input challenge is training. Added tail samples are
    # used to qualify bank reuse, not to increase the fitted sample count.
    for name,w in c['words'].items():
        x=[];y=[];anchor=[]
        for k in range(c['score_main_counts'][name]):
            F=w['phase']+.1+200*k;parent=pulse(w['input'],239+200*k) if name=='front' else pulse(w['input'],F)
            output=pulse(name,F+200);x.append(parent[1]-parent[0]);y.append(output[1]-output[0]);anchor.append(output[1]-(F+200))
        x=np.asarray(x);u=(x-100)/100;y=np.asarray(y);anchor=np.asarray(anchor)
        degrees={}
        for degree,layer in [(1,'L1'),(3,'L2_static_candidate')]:
            matrix=np.polynomial.polynomial.polyvander(u,degree)
            width=np.linalg.lstsq(matrix,y,rcond=None)[0];fall=np.linalg.lstsq(matrix,anchor,rcond=None)[0]
            degrees[layer]=dict(width_coefficients=width.tolist(),fall_anchor_coefficients=fall.tolist(),
                development_width_fit_RMS_ns=float(np.sqrt(np.mean((matrix@width-y)**2))),design_matrix_condition=float(np.linalg.cond(matrix)))
        cards[name]=dict(training_samples=len(x),training_width_range_ns=[float(min(x)),float(max(x))],
            inferred_input_width_domain_ns=[2.,198.],domain_status='proposed operating envelope;independent validation pending',models=degrees)
    gates={}
    for name,g in c['gates'].items():
        dr=[];df=[]
        for k in range(c['score_main_counts']['s4c2']):
            F=1260.1+200*k;p=[pulse(n,F) for n in g['inputs']];o=pulse(name,F)
            ideal=((max(v[0] for v in p),min(v[1] for v in p)) if g['kind']=='and' else (min(v[0] for v in p),max(v[1] for v in p)))
            dr.append(o[0]-ideal[0]);df.append(o[1]-ideal[1])
        gates[name]=dict(kind=g['kind'],inputs=g['inputs'],rise_delay_ns=float(np.median(dr)),fall_delay_ns=float(np.median(df)),training_samples=len(dr))
        if min(gates[name]['rise_delay_ns'],gates[name]['fall_delay_ns'])<0:raise ValueError('noncausal gate-delay card')
    result=dict(utc=now(),status='development_trained_candidate_not_independently_qualified',training_run=label,
        training_raw_sha256=summary['raw_sha256'],training_phase_assessment_sha256=stream_sha(folder/'phase_assessment.json'),
        fit_features='local physical input width only;fixed per-gate propagation medians',normalization_center_ns=100,normalization_scale_ns=100,
        words=cards,gates=gates,architecture={n:{k:w[k] for k in ['input','phase','kind'] if k in w} for n,w in c['words'].items()},external_width_domain_ns=[4,196],PVT=dict(corner='tt',temperature_C=27,supply_V=1.8),
        detailed_dynamic_state_predictions=False,static_card_does_not_qualify_full_dynamic_L2=True,
        sampled_test_internal_states_allowed=False,new_independent_seed_opened=False)
    path=OUT/f'{label}_prediction_cards.json'
    if path.exists():raise FileExistsError('candidate card is immutable: '+str(path))
    write(path,result);print(dict(cards=str(path),words=len(cards),gates=len(gates),new_independent_seed_opened=False));return result


def predict(widths,card,layer):
    if layer not in ['L0','L1','L2_static_candidate']:raise ValueError('unknown model layer')
    # API deliberately accepts only input widths and locked cards. Physical
    # measured widths, output timestamps and hidden analog states are absent.
    previous=[None]*5;rows=[]
    def registered(name,parent,Fout,sample):
        if parent is None:return None
        width=parent[1]-parent[0]
        if not 2<=width<=198:raise ValueError('frozen local input-width domain exceeded')
        if layer=='L0':out_width,delta=width,0.
        else:
            model=card['words'][name]['models'][layer];u=(width-100)/100
            out_width=float(np.polynomial.polynomial.polyval(u,model['width_coefficients']));delta=float(np.polynomial.polynomial.polyval(u,model['fall_anchor_coefficients']))
        fall=round((Fout+delta)*1e6)/1e6;rise=round((fall-out_width)*1e6)/1e6
        if not 0<fall-rise<200 or rise<parent[1]:raise ValueError('invalid or noncausal modeled output')
        rows.append(dict(node=name,sample=sample,rise_ns=rise,fall_ns=fall,width_ns=fall-rise,available_ns=fall))
        return rise,fall
    for i,width in enumerate(widths):
        if not 4<=float(width)<=196:raise ValueError('external domain exceeded')
        current=[None]*5;F=460.1+200*i
        current[0]=registered('front',(40.1+200*i,40.1+200*i+float(width)),F,i)
        for j in range(1,5):
            if i>=j:current[j]=registered('history'+str(j),previous[j-1],F,i-j)
        previous=current
        if i<4:continue
        values=dict(zip(['front','history1','history2','history3','history4'],current));pending=dict(card['gates'])
        while pending:
            progressed=False
            for name,g in list(pending.items()):
                if not all(n in values for n in g['inputs']):continue
                p=[values[n] for n in g['inputs']]
                if g['kind']=='and':rise,fall=max(v[0] for v in p),min(v[1] for v in p)
                else:rise,fall=min(v[0] for v in p),max(v[1] for v in p)
                if layer!='L0':rise+=g['rise_delay_ns'];fall+=g['fall_delay_ns']
                if rise>=fall:raise ValueError('modeled combinational event lost')
                values[name]=(rise,fall);del pending[name];progressed=True
            if not progressed:raise ValueError('combinational dependency cycle')
        registered('s4c2',values['g8_and'],F+200,i-4)
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fit',required=True);a=p.parse_args();card=fit(a.fit)
    widths=json.loads((OUT/'tail_development/contract.json').read_text())['input_widths_ns']
    start=time.perf_counter();predictions={layer:predict(widths,card,layer) for layer in ['L0','L1','L2_static_candidate']}
    write(OUT/(a.fit+'_development_predictions.json'),dict(training_set_diagnostic_only=True,predictions=predictions,runtime_s=time.perf_counter()-start,
        new_independent_seed_opened=False,independent_accuracy_claimed=False))
