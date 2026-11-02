"""Composable TR/encoder/fixed-word-controller/observer research components."""
import copy
from .core import FS,Pulse,Status
from .calibration import WidthCalibrator
from .cosim import CycleGraph
from .dynamic import TwoBankTR
from .physical import triggered_card


RECORD_FIELDS=['cycle','epoch','role','target_fs','command_fs','revision','gain_q14','offset_word',
               'status','reference_status','model_width_fs','reference_width_fs','output_end_fs','reference_end_fs']


class FixedWordController:
    def __init__(self,total_cycles,warmup=100,refresh=128):
        self.total=total_cycles;self.warmup=warmup;self.refresh=refresh
        self.state=dict(gain_q14=16384,offset_word=0,revision=0,epoch=0,next_cal=warmup,
            batch_seq=0,active=None,pending=None,log=[],stale_observations=0,invalid_observations=0)

    def contract(self):return dict(type='two_point_fixed_word',cycles=self.total,warmup=self.warmup,refresh=self.refresh,
        calibration_targets=[[60_000_000,100_000_000],[55_000_000,95_000_000]],threshold_fs=20_000,control_delay_cycles=1)

    def snapshot(self):return copy.deepcopy(self.state)
    def restore(self,state):
        self.state=copy.deepcopy(state)
        if self.state['active'] is not None:
            observations=self.state['active']['observations']
            self.state['active']['observations']={int(k):v for k,v in observations.items()}

    def step(self,ctx,inputs):
        s=self.state;k=ctx.cycle;epoch=int(k>=self.total//2)
        if epoch!=s['epoch']:
            s.update(epoch=epoch,active=None,pending=None,next_cal=k)
            s['log'].append(dict(event='epoch',cycle=k,epoch=epoch))
        pending=s['pending']
        if pending is not None and pending['cycle']<=k:
            s.update(gain_q14=pending['gain_q14'],offset_word=pending['offset_word'],
                     revision=s['revision']+1,pending=None,active=None,
                     next_cal=k+(self.refresh if pending['settled'] else 0))
            s['log'].append(dict(event='latch',cycle=k,epoch=epoch,revision=s['revision'],gain_q14=s['gain_q14'],offset_word=s['offset_word']))
        for event in ctx.events:
            p=event['payload']
            if p['role']!=1:continue
            active=s['active']
            if active is None or p['epoch']!=epoch or p['batch']!=active['id'] or p['revision']!=active['revision']:
                s['stale_observations']+=1;continue
            if p['status']!='VALID':s['invalid_observations']+=1;continue
            if p['slot'] in active['observations']:raise ValueError('duplicate calibration observation')
            active['observations'][p['slot']]=p
            if len(active['observations'])==2:
                a,b=[active['observations'][i] for i in [0,1]]
                error=max(abs(a['width_fs']-a['target_fs']),abs(b['width_fs']-b['target_fs']))
                c=WidthCalibrator(s['gain_q14'],s['offset_word'])
                changed=c.update([a['target_fs'],b['target_fs']],[a['command_fs'],b['command_fs']],[a['width_fs'],b['width_fs']])
                settled=error<=20_000
                s['log'].append(dict(event='batch',cycle=k,epoch=epoch,batch=active['id'],revision=s['revision'],
                    max_endpoint_error_fs=error,within_tolerance=settled,word_unchanged=not changed,
                    sample_ids=[a['sample'],b['sample']],observation_end_fs=[a['t_valid_fs'],b['t_valid_fs']],
                    gain_q14=s['gain_q14'],offset_word=s['offset_word']))
                if changed and not settled:
                    s['pending']=dict(cycle=k+1,gain_q14=c.gain_q14,offset_word=c.offset_word,settled=False)
                else:s.update(active=None,next_cal=k+self.refresh)
        if k>=self.warmup and s['active'] is None and s['pending'] is None and k>=s['next_cal']:
            s['active']=dict(id=s['batch_seq'],revision=s['revision'],issued=0,observations={})
            s['batch_seq']+=1
        active=s['active'];slot=-1;batch=-1;role=0 if k<self.warmup else 2
        if active is not None and active['issued']<2:
            slot=active['issued'];batch=active['id'];active['issued']+=1;role=1
            target=([60_000_000,100_000_000] if not epoch else [55_000_000,95_000_000])[slot]
        else:target=(65_000_000 if not epoch else 60_000_000)+5_000_000*(k%7)
        return dict(request=dict(sample=k,epoch=epoch,role=role,target_fs=target,slot=slot,batch=batch,
            gain_q14=s['gain_q14'],offset_word=s['offset_word'],revision=s['revision']))


class WidthEncoder:
    def contract(self):return dict(type='programmed_width_encoder',grid_fs=1000,ramp_crossing_width_fs=200_000,phase_fs=40_100_000)
    def snapshot(self):return {}
    def restore(self,state):pass
    def step(self,ctx,inputs):
        request=dict(inputs['request'])
        c=WidthCalibrator(request['gain_q14'],request['offset_word'])
        command=c.command(request['target_fs'])
        request.update(command_fs=command,pulse=Pulse.at(ctx.tick_fs+40_100_000,command+200_000,ctx.cycle,'encoder'))
        return dict(frame=request)


class TRBankComponent:
    def __init__(self,config,warmup=100):
        self.config=copy.deepcopy(config);self.warmup=warmup;self.model=TwoBankTR(config,warmup=warmup)
    def contract(self):return dict(type='two_store_TR',config=self.config,warmup=self.warmup)
    def snapshot(self):return self.model.snapshot()
    def restore(self,state):self.model.restore(state)
    def step(self,ctx,inputs):return dict(result=self.model.process(inputs['frame']['pulse']))


class WidthObserver:
    def __init__(self,delay_cycles=1):self.delay=delay_cycles
    def contract(self):return dict(type='threshold_width_observer',delay_cycles=self.delay,reference_feedback=False)
    def snapshot(self):return {}
    def restore(self,state):pass
    def step(self,ctx,inputs):
        request=inputs['frame'];out=inputs['result'];ref=inputs['reference'];primary=out.value;reference=ref.value
        if out.status!=Status.RESET:
            payload={k:request[k] for k in ['sample','epoch','role','target_fs','slot','batch','command_fs','revision']}
            payload.update(status=out.status.value,width_fs=primary.width_fs if primary else 0,
                t_valid_fs=primary.fall.t if primary else ctx.tick_fs)
            ctx.emit('controller',payload,payload['t_valid_fs']+self.delay*ctx.period_fs)
        row=(ctx.cycle,request['epoch'],request['role'],request['target_fs'],request['command_fs'],
            request['revision'],request['gain_q14'],request['offset_word'],
            1 if out.status==Status.VALID else 0 if out.status==Status.RESET else -1,
            1 if ref.status==Status.VALID else 0 if ref.status==Status.RESET else -1,
            primary.width_fs if primary else 0,reference.width_fs if reference else 0,
            primary.fall.t if primary else 0,reference.fall.t if reference else 0)
        return dict(record=row)


def tr_fidelity_config(layer):
    if layer=='L0':
        return dict(kind='TR',layer='L0',params=dict(trigger_mode='clk',latency=0.),
            model_id='TR.ideal.design',input_width=[20_200_000,100_200_000],output_width=[1,FS])
    config,_=triggered_card('TR',layer)
    config['input_width']=[20_200_000,100_200_000]
    return config


def calibration_graph(layer,total_cycles,warmup=100,refresh=128,observation_delay=1):
    graph=CycleGraph()
    graph.add('controller',FixedWordController(total_cycles,warmup,refresh))
    graph.add('encoder',WidthEncoder(),{'request':'controller.request'})
    graph.add('plant',TRBankComponent(tr_fidelity_config(layer),warmup),{'frame':'encoder.frame'})
    graph.add('reference',TRBankComponent(tr_fidelity_config('L2'),warmup),{'frame':'encoder.frame'})
    graph.add('observer',WidthObserver(observation_delay),{'frame':'encoder.frame','result':'plant.result','reference':'reference.result'})
    return graph
