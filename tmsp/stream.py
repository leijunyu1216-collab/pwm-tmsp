"""Deterministic same-time edge assembly and sample-label joins (DD-02 3.3).

This layer validates input streams. It does not emit a primitive's output
retroactively or implement a mixed-signal event scheduler.
"""
import copy
import hashlib
import json
import math
from dataclasses import asdict
from .core import Edge, Pulse, Voltage, Result, Status, NUMERIC, is_time, primary


def pack_result(result):
    value=result.value
    kind='pulse' if isinstance(value,Pulse) else 'voltage' if isinstance(value,Voltage) else None
    if value is not None and kind is None:
        raise ValueError('unsupported snapshot result domain')
    return dict(status=result.status.value,value_kind=kind,
                value=asdict(value) if value is not None else None,
                flags=[str(x) for x in result.flags],obs=copy.deepcopy(result.obs),
                diagnostics=copy.deepcopy(result.diagnostics))


def unpack_result(data):
    value=data['value']
    if data['value_kind']=='pulse':
        value=Pulse(Edge(**value['rise']),Edge(**value['fall']))
    elif data['value_kind']=='voltage':
        value=Voltage(value['t'],value['v'],value['sample'],Status(value['status']))
    return Result(Status(data['status']),value,tuple(Status(x) for x in data['flags']),
                  data['obs'],tuple(data['diagnostics']))


def diagnostic(status, chan, sample, t, detail, flags=(), edges=()):
    return Result(status, flags=tuple(flags), diagnostics=(dict(
        status=status.value, flags=[str(x) for x in flags], port=chan,
        sample=sample, edge=None, t=t, error_id='T-LBL-01', detail=detail,
        repro={'edges': [asdict(e) for e in edges if isinstance(e, Edge)]}),))


class PulseAssembler:
    """push() accepts exactly one complete simultaneous event group.

    Same-time groups must not be split across calls. Across distinct channels,
    result order is canonical, independent of the group's input ordering.
    """
    def __init__(self, channels):
        if not channels or any(not isinstance(c, str) or not c for c in channels) or len(set(channels)) != len(channels):
            raise ValueError('nonempty unique channel names required')
        self.channels = tuple(sorted(channels))
        self.checksum = hashlib.sha256(json.dumps(self.channels).encode()).hexdigest()
        self.reset()

    def reset(self, t0=0):
        if not is_time(t0):
            raise ValueError('invalid reset time')
        self.state = dict(t0=t0, last_group=None, channels={
            c: dict(last_sample=-1, pending=None, closed=False) for c in self.channels})

    def snapshot(self):
        return dict(schema='tmsp-edge-assembler/v1', checksum=self.checksum, state=copy.deepcopy(self.state))

    def restore(self, snapshot):
        if snapshot.get('schema') != 'tmsp-edge-assembler/v1' or snapshot.get('checksum') != self.checksum:
            raise ValueError('incompatible edge-assembler snapshot')
        self.state = copy.deepcopy(snapshot['state'])

    def push(self, group):
        group = tuple(group)
        if not group:
            return []
        if any(not isinstance(e, Edge) or not is_time(e.t)
               or type(e.sample) is not int or e.sample < 0
               or e.pol not in {'R', 'F'} or not isinstance(e.chan, str)
               or (e.slew is not None and (not isinstance(e.slew, (int, float))
                   or not math.isfinite(e.slew) or e.slew <= 0)) for e in group):
            return [diagnostic(Status.UNDEF, None, None, None, 'invalid edge fields', edges=group)]
        times = {e.t for e in group}
        if len(times) != 1:
            return [diagnostic(Status.UNDEF, None, None, None, 'push requires one same-time event group', edges=group)]
        t = group[0].t
        if t < self.state['t0'] or self.state['last_group'] is not None and t <= self.state['last_group']:
            return [diagnostic(Status.UNDEF, None, None, t, 'time reversal or split same-time group', edges=group)]
        self.state['last_group'] = t
        results = []
        for chan in sorted({e.chan for e in group}):
            edges = [e for e in group if e.chan == chan]
            if chan not in self.state['channels']:
                results.append(diagnostic(Status.EXTRA, chan, edges[0].sample, t, 'unexpected channel', edges=edges))
                continue
            s = self.state['channels'][chan]
            if len(edges) != 1:
                # Same-channel co-timed transitions cannot form a positive pulse.
                # Poison pending data rather than manufacture a corrected pair.
                s['pending'] = None
                s['last_sample'] = max(s['last_sample'], *(e.sample for e in edges))
                s['closed'] = True
                results.append(diagnostic(Status.UNDEF, chan, min(e.sample for e in edges), t,
                                          'duplicate/simultaneous same-channel edges', edges=edges))
                continue
            e = edges[0]
            pending = s['pending']
            if e.sample < s['last_sample'] or e.sample == s['last_sample'] and s['closed']:
                results.append(diagnostic(Status.UNDEF, chan, e.sample, t, 'repeated/backward sample label', edges=edges))
                continue
            if e.pol == 'R':
                if pending is not None:
                    if e.sample == s['last_sample']:
                        s.update(pending=None, closed=True)
                        results.append(diagnostic(Status.UNDEF, chan, e.sample, t, 'second rising edge', edges=edges))
                        continue
                    results.append(diagnostic(Status.MISS, chan, pending['sample'], t,
                                              'next rising edge arrived before falling edge', edges=(Edge(**pending),)))
                s.update(pending=asdict(e), last_sample=e.sample, closed=False)
            else:
                if pending is None:
                    s.update(last_sample=e.sample, closed=True)
                    results.append(diagnostic(Status.EXTRA, chan, e.sample, t, 'falling edge without rise', edges=edges))
                elif e.sample != pending['sample']:
                    results.append(diagnostic(Status.MISS, chan, pending['sample'], t,
                                              'falling label does not match pending rise', flags=(Status.EXTRA,),
                                              edges=(Edge(**pending), e)))
                    s.update(pending=None, last_sample=e.sample, closed=True)
                else:
                    pulse = Pulse(Edge(**pending), e)
                    status = pulse.validate()
                    results.append(Result(Status.VALID, pulse) if status == Status.VALID else
                                   diagnostic(status, chan, e.sample, t, 'invalid pulse pair', edges=(pulse.rise, pulse.fall)))
                    s.update(pending=None, closed=True)
        return results

    def flush(self):
        results = []
        for chan, s in self.state['channels'].items():
            if s['pending'] is not None:
                e = Edge(**s['pending'])
                results.append(diagnostic(Status.MISS, chan, e.sample, e.t, 'stream ended before fall', edges=(e,)))
                s.update(pending=None, closed=True)
        return results


class SampleJoin:
    """Join complete port results by label, with explicit per-port watermarks.

    push(port, result) returns zero or more (sample, input_map, status) triples.
    A newer sample on a port proves an older missing label on that port absent;
    flush() closes all remaining samples. No missing input is replaced by zero.
    """
    def __init__(self, ports):
        if not ports or len(set(ports)) != len(ports):
            raise ValueError('unique ports required')
        self.ports = tuple(sorted(ports))
        self.reset()

    def reset(self):
        self.state = dict(watermarks={p: -1 for p in self.ports}, pending={}, emitted=-1)

    def push(self, port, result, sample=None):
        if port not in self.ports:
            raise ValueError('unexpected join port')
        if not isinstance(result, Result):
            raise ValueError('join requires explicit Result')
        if sample is None:
            sample = result.value.sample if result.value is not None else (
                result.diagnostics[0]['sample'] if result.diagnostics else None)
        if type(sample) is not int or sample < 0:
            raise ValueError('sample label required for invalid input')
        if sample <= self.state['watermarks'][port] or sample <= self.state['emitted']:
            return [(sample, {port: diagnostic(Status.UNDEF, port, sample, None, 'join label repeated/backward')}, Status.UNDEF)]
        self.state['watermarks'][port] = sample
        self.state['pending'].setdefault(sample, {})[port] = result
        return self._drain(False)

    def _drain(self, final):
        out = []
        for k in sorted(self.state['pending']):
            inputs = self.state['pending'][k]
            missing = [p for p in self.ports if p not in inputs]
            if missing and not (final or all(self.state['watermarks'][p] > k for p in missing)):
                break
            for p in missing:
                inputs[p] = diagnostic(Status.MISS, p, k, None, 'port missing sample label')
            status = primary([r.status for r in inputs.values()])
            out.append((k, inputs, status))
        for k, _, _ in out:
            del self.state['pending'][k]
            self.state['emitted'] = k
        return out

    def flush(self):
        return self._drain(True)

    def snapshot(self):
        state=copy.deepcopy(self.state)
        state['pending']={str(k):{p:pack_result(r) for p,r in row.items()}
                          for k,row in state['pending'].items()}
        return dict(schema='tmsp-sample-join/v1',ports=self.ports,state=state)

    def restore(self, snapshot):
        if snapshot.get('schema')!='tmsp-sample-join/v1' or tuple(snapshot['ports']) != self.ports:
            raise ValueError('incompatible join snapshot')
        self.state = copy.deepcopy(snapshot['state'])
        self.state['pending']={int(k):{p:unpack_result(r) for p,r in row.items()}
                               for k,row in self.state['pending'].items()}
