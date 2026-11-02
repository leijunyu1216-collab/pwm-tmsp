"""Compile L0 median networks into causal, registered CycleGraph components.

Widths are integer femtoseconds. Register cost follows the explicit full-wire
retiming policy in networks.py; no cell area, power or physical card is implied.
"""
import copy
from .cosim import CycleGraph
from .networks import median_network


def scheduled_layers(network):
    levels = [-1] * network.n
    layers = [[] for _ in range(network.depth)]
    for i, j, lo, hi in network.operations:
        stage = max(levels[i], levels[j]) + 1
        layers[stage].append((i, j, lo, hi))
        if lo:
            levels[i] = stage
        if hi:
            levels[j] = stage
    return tuple(tuple(layer) for layer in layers)


class _History:
    def __init__(self, n):
        self.n = n
        self.values = []
        self.sample = -1

    def contract(self):
        return dict(kind='median_history_L0', window=self.n)

    def step(self, ctx, inputs):
        width = ctx.external.get('width_fs')
        if width is None:
            return dict(frame=None)
        if type(width) is not int or width <= 0:
            raise ValueError('positive integer-fs width required')
        self.sample += 1
        self.values = (self.values + [width])[-self.n:]
        frame = dict(widths=tuple(self.values), sample=self.sample) if len(self.values) == self.n else None
        return dict(frame=frame)

    def snapshot(self):
        return dict(values=self.values.copy(), sample=self.sample)

    def restore(self, state):
        self.values = list(state['values'])
        self.sample = state['sample']


class _RegisteredLayer:
    def __init__(self, operations):
        self.operations = operations
        self.pending = None

    def contract(self):
        return dict(kind='registered_minmax_layer_L0', operations=self.operations)

    def step(self, ctx, inputs):
        prior = self.pending
        frame = inputs['frame']
        self.pending = None
        if frame is not None:
            values = list(frame['widths'])
            # Operations at one layer use that layer's common input registers.
            source = values.copy()
            for i, j, lo, hi in self.operations:
                if lo:
                    values[i] = min(source[i], source[j])
                if hi:
                    values[j] = max(source[i], source[j])
            self.pending = dict(widths=tuple(values), sample=frame['sample'])
        return dict(frame=prior)

    def snapshot(self):
        return copy.deepcopy(self.pending)

    def restore(self, state):
        self.pending = copy.deepcopy(state)


class RegisteredMedianPipeline:
    """Each layer delays one frame; drain with step() after the last input.

Output k is available at cycle k+depth, and represents center k-floor(n/2).
This class is L0 only. Physical validity requires independently measured cards.
"""
    def __init__(self, n, topology='odd_even', pruned=True):
        self.network = median_network(n, topology, pruned)
        self.graph = CycleGraph()
        self.graph.add('history', _History(n))
        previous = 'history'
        for index, operations in enumerate(scheduled_layers(self.network)):
            name = f'layer{index}'
            self.graph.add(name, _RegisteredLayer(operations), {'frame': previous + '.frame'})
            previous = name
        self.output = previous

    def step(self, width_fs=None):
        frame = self.graph.step({'width_fs': width_fs})[self.output]['frame']
        if frame is None:
            return None
        return dict(width_fs=frame['widths'][self.network.n // 2],
                    sample=frame['sample'], reference_center=frame['sample'] - self.network.n // 2,
                    available_cycle=self.graph.cycle - 1)

    def snapshot(self):
        return self.graph.snapshot()

    def restore(self, state):
        self.graph.restore(state)
