"""Experimental charge-coordinate models; separate from frozen DD-02 v1.1.

MM has two physical capture states and one selected-node charge state.
Mechanisms: (1) charge sharing, (2) finite/selective switch conduction.
No arbitrary polynomial of input skew is included.
"""
import math
import numpy as np
from numba import njit


@njit(cache=True)
def mm_charge(ra, fa, rb, fb, observe, csel_ff, gain_v_ns, dead_ns,
              droop_v_ns, decision_offset_v, cpar_ff, resistance_ohm,
              selection_band_v, dt_ns=.1):
    # Units: pF, ns, V. Chold=1pF in the original transistor cell.
    ca=1.;cb=1.;cs=(csel_ff+cpar_ff)/1000.
    g=1000./resistance_ohm
    va=0.;vb=0.;vs=0.
    start=min(ra,rb)
    steps=int(math.ceil((observe-start)/dt_ns))
    h=(observe-start)/steps
    for i in range(steps):
        t=start+(i+.5)*h
        ia=gain_v_ns if ra+dead_ns<=t<fa else droop_v_ns
        ib=gain_v_ns if rb+dead_ns<=t<fb else droop_v_ns
        ah=va+h*ia;bh=vb+h*ib
        z=(va-vb+decision_offset_v)/selection_band_v
        z=max(-40.,min(40.,z))
        gate=1./(1.+math.exp(-z))
        ga=g*gate;gb=g*(1.-gate)
        aa=ga/(1.+ga*h/ca);bb=gb/(1.+gb*h/cb)
        new_s=(cs/h*vs+aa*ah+bb*bh)/(cs/h+aa+bb)
        va=(ca/h*ah+ga*new_s)/(ca/h+ga)
        vb=(cb/h*bh+gb*new_s)/(cb/h+gb)
        vs=new_s
    return np.array([va,vb,vs])


@njit(cache=True)
def mm_charge_lag(ra,fa,rb,fb,observe,csel_ff,gain_v_ns,dead_ns,droop_v_ns,
                  decision_offset_v,cpar_ff,resistance_ohm,selection_band_v,
                  selection_tau_ns,dt_ns=.05):
    """Charge model with one filtered comparator-input difference state.

    The new state is part of finite selection response, the second mechanism.
    Returns only observable capture/selected voltages. No skew regressor.
    """
    cs=(csel_ff+cpar_ff)/1000.;g=1000./resistance_ohm
    va=0.;vb=0.;vs=0.;delta=0.
    start=min(ra,rb);steps=int(math.ceil((observe-start)/dt_ns));h=(observe-start)/steps
    decay=math.exp(-h/selection_tau_ns)
    for i in range(steps):
        t=start+(i+.5)*h
        ia=gain_v_ns if ra+dead_ns<=t<fa else droop_v_ns
        ib=gain_v_ns if rb+dead_ns<=t<fb else droop_v_ns
        ah=va+h*ia;bh=vb+h*ib
        delta=decay*delta+(1-decay)*(va-vb)
        z=max(-40.,min(40.,(delta+decision_offset_v)/selection_band_v))
        gate=1./(1.+math.exp(-z));ga=g*gate;gb=g*(1-gate)
        aa=ga/(1+ga*h);bb=gb/(1+gb*h)
        vs=(cs/h*vs+aa*ah+bb*bh)/(cs/h+aa+bb)
        va=(ah/h+ga*vs)/(1/h+ga);vb=(bh/h+gb*vs)/(1/h+gb)
    return np.array([va,vb,vs])


def vtc_cubic(parameters, voltage):
    p=parameters
    return ((p['kappa_v3']*voltage+p['kappa_v'])*voltage+
            p['K_v']+p['gain_delta'])*voltage+p['w0']+p['offset_delta']


class ChargeMM:
    def __init__(self,card):
        if len(card['active_mechanisms'])>2:
            raise ValueError('at most two mechanisms')
        self.card=card

    def predict(self,rise_a_ns,width_a_ns,rise_b_ns,width_b_ns,csel_ff,observe_ns=299.,dt_ns=.1):
        c=self.card
        for w in [width_a_ns,width_b_ns]:
            if not c['envelope']['width_ns'][0]<=w<=c['envelope']['width_ns'][1]:
                raise ValueError('width outside card envelope')
        if not c['envelope']['csel_ff'][0]<=csel_ff<=c['envelope']['csel_ff'][1]:
            raise ValueError('load outside card envelope')
        p=c['parameters'];a=c['capture']
        kernel=mm_charge_lag if 'selection_tau_ns' in p else mm_charge
        extra=[p['selection_tau_ns'],dt_ns] if 'selection_tau_ns' in p else [dt_ns]
        state=kernel(rise_a_ns+.1,rise_a_ns+width_a_ns+.3,
                        rise_b_ns+.1,rise_b_ns+width_b_ns+.3,observe_ns,csel_ff,
                        a['gain_v_ns'],a['dead_ns'],a['droop_v_ns'],a['decision_offset_v'],
                        p['cpar_ff'],p['resistance_ohm'],p['selection_band_v'],*extra)
        return dict(capture_a_v=float(state[0]),capture_b_v=float(state[1]),
                    selected_v=float(state[2]),width_fs=float(vtc_cubic(c['vtc'],state[2])))
