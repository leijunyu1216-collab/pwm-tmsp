"""L0 aligned-edge MIN/MAX networks and causal median window reference."""
from dataclasses import dataclass
import numpy as np
from .core import Pulse,Status


def aligned_minmax(a,b,anchor='rise'):
    if any(not isinstance(x,Pulse) or x.validate()!=Status.VALID for x in [a,b]):raise ValueError('valid pulses required')
    if anchor not in ('rise','fall'):raise ValueError('anchor must be rise or fall')
    edge_a=a.rise.t if anchor=='rise' else a.fall.t
    edge_b=b.rise.t if anchor=='rise' else b.fall.t
    if a.sample!=b.sample or edge_a!=edge_b:raise ValueError('logical AND/OR requires common anchor and sample')
    # Intersection/union implement MIN/MAX for either explicitly chosen anchor.
    low,high=min(a.width_fs,b.width_fs),max(a.width_fs,b.width_fs)
    low_start=edge_a if anchor=='rise' else edge_a-low
    high_start=edge_a if anchor=='rise' else edge_a-high
    return (Pulse.at(low_start,low,a.sample,'min'),
            Pulse.at(high_start,high,a.sample,'max'))


@dataclass(frozen=True)
class MedianNetwork:
    n:int
    topology:str
    pruned:bool
    operations:tuple
    depth:int

    @property
    def minmax_cells(self):return sum(int(lo)+int(hi) for i,j,lo,hi in self.operations)
    @property
    def word_registers(self):return self.n-1+self.n*self.depth

    def evaluate(self,windows):
        x=np.asarray(windows).copy()
        if x.shape[-1]!=self.n:raise ValueError('wrong window width')
        for i,j,lo,hi in self.operations:
            a=x[...,i].copy();b=x[...,j].copy()
            if lo:x[...,i]=np.minimum(a,b)
            if hi:x[...,j]=np.maximum(a,b)
        return x[...,self.n//2]


def median_network(n,topology='odd_even',pruned=True):
    if type(n) is not int or n<3 or n%2!=1:raise ValueError('odd window>=3 required')
    raw=[]
    if topology=='insertion':
        for end in range(1,n):
            for j in range(end,0,-1):raw.append((j-1,j))
    elif topology=='odd_even':
        for phase in range(n):
            for j in range(phase%2,n-1,2):raw.append((j,j+1))
    else:raise ValueError('unknown sorting topology')
    needed={n//2};ops=[]
    for i,j in reversed(raw):
        lo=i in needed if pruned else True;hi=j in needed if pruned else True
        if lo or hi:
            ops.append((i,j,lo,hi));needed.update([i,j])
    ops.reverse();levels=[-1]*n
    for i,j,lo,hi in ops:
        stage=max(levels[i],levels[j])+1
        if lo:levels[i]=stage
        if hi:levels[j]=stage
    depth=(levels[n//2] if pruned else max(levels))+1
    return MedianNetwork(n,topology,pruned,tuple(ops),depth)


def median_stream(values,network):
    """Output labels are newest samples; each window's reference center is explicit."""
    values=np.asarray(values)
    windows=np.lib.stride_tricks.sliding_window_view(values,network.n)
    newest=np.arange(network.n-1,len(values),dtype=np.int64)
    return dict(width=network.evaluate(windows),sample=newest,reference_center=newest-network.n//2,
        available_cycle=newest+network.depth)
