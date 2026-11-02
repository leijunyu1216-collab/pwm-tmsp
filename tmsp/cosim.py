"""Fixed-cycle component graph with causal, timestamped observation delivery."""
from dataclasses import dataclass
import copy
import hashlib
import heapq
import json
from .core import FS,is_time


@dataclass
class CycleContext:
    cycle:int
    tick_fs:int
    period_fs:int
    events:tuple
    external:dict
    _graph:object

    def emit(self,destination,payload,available_at_fs):
        if destination not in self._graph.nodes or not isinstance(payload,dict):
            raise ValueError('unknown destination or non-dictionary message')
        if not is_time(available_at_fs) or available_at_fs<self.tick_fs:
            raise ValueError('message precedes its emitting cycle')
        self._graph._sequence+=1
        heapq.heappush(self._graph._pending,(available_at_fs,self._graph._sequence,destination,copy.deepcopy(payload)))


class CycleGraph:
    """Acyclic frame computation plus a queue of delayed feedback observations.

    All queued observations are latched once at the start of a cycle. A frame
    may contain planned future pulse edges; feedback is released only when
    its explicit validity timestamp has passed. No analog event solver runs.
    """
    def __init__(self,period_fs=FS,t0_fs=0):
        if not is_time(period_fs) or not period_fs or not is_time(t0_fs):raise ValueError('invalid cycle clock')
        self.period=period_fs;self.t0=t0_fs;self.cycle=0
        self.nodes={};self.bindings={};self._pending=[];self._sequence=0;self._order=None

    def add(self,name,component,bindings=None):
        if self._order is not None or not name or '.' in name or name in self.nodes:
            raise ValueError('graph frozen or invalid/duplicate component')
        self.nodes[name]=component;self.bindings[name]=dict(bindings or {})
        return self

    def _compile(self):
        remaining=set(self.nodes);order=[]
        while remaining:
            ready=[]
            for name in self.nodes:
                if name not in remaining:continue
                dependencies=[]
                for ref in self.bindings[name].values():
                    source,port=ref.split('.',1)
                    if source not in self.nodes or not port:raise ValueError('unknown graph binding')
                    dependencies.append(source)
                if all(d in order for d in dependencies):ready.append(name)
            if not ready:raise ValueError('frame graph contains a cycle; feedback must use timed messages')
            order.extend(ready);remaining.difference_update(ready)
        self._order=order
        descriptor=dict(period=self.period,t0=self.t0,nodes=[dict(name=n,bindings=self.bindings[n],
            contract=self.nodes[n].contract()) for n in order])
        self.checksum=hashlib.sha256(json.dumps(descriptor,sort_keys=True).encode()).hexdigest()

    def step(self,external=None):
        if self._order is None:self._compile()
        tick=self.t0+self.cycle*self.period
        if not is_time(tick):raise ValueError('cycle time overflow')
        inbox={n:[] for n in self.nodes}
        while self._pending and self._pending[0][0]<=tick:
            at,seq,dest,payload=heapq.heappop(self._pending)
            inbox[dest].append(dict(available_at_fs=at,delivered_at_fs=tick,payload=payload,sequence=seq))
        values={}
        for name in self._order:
            inputs={port:values[ref.split('.',1)[0]][ref.split('.',1)[1]] for port,ref in self.bindings[name].items()}
            ctx=CycleContext(self.cycle,tick,self.period,tuple(inbox[name]),external or {},self)
            values[name]=self.nodes[name].step(ctx,inputs)
        self.cycle+=1
        return values

    def snapshot(self):
        if self._order is None:self._compile()
        return copy.deepcopy(dict(checksum=self.checksum,cycle=self.cycle,sequence=self._sequence,
            pending=self._pending,nodes={n:b.snapshot() for n,b in self.nodes.items()}))

    def restore(self,snapshot):
        if self._order is None:self._compile()
        if snapshot['checksum']!=self.checksum:raise ValueError('snapshot belongs to another graph')
        self.cycle=snapshot['cycle'];self._sequence=snapshot['sequence']
        self._pending=copy.deepcopy([tuple(x) for x in snapshot['pending']]);heapq.heapify(self._pending)
        for name,state in snapshot['nodes'].items():self.nodes[name].restore(copy.deepcopy(state))
