"""Frozen continuous reference:predict first,pause/evaluate/resume,no refitting."""
import json,os,subprocess,time,traceback,platform
from pathlib import Path
import numpy as np
import psutil
from benchmarks.retimed_run_v56 import OUT
from benchmarks.retimed_workload_v56 import open_sealed
from benchmarks.retimed_prediction_v56 import predict
from benchmarks.streaming_cli_probe_v47 import EXE
from benchmarks.network_streaming_pilot_v52 import raw_progress
from benchmarks.campaign_resources_v57 import ProcessJob
from benchmarks.campaign_live_v57 import contract,evaluate
from benchmarks.validation_guard_v41 import verify_frozen
from benchmarks.audit_cost_diagnosis_v56 import stream_sha
from benchmarks.live_spice import now,write

CAMPAIGN=OUT/'joint_independent'


def verify(record):
    verify_frozen(Path(record['workspace_root']),record['assets_sha256'])
    if np.__version__!=record['environment']['numpy'] or platform.python_version()!=record['environment']['python']:
        raise RuntimeError('frozen Python/numpy environment changed')


def prepare(record):
    verify(record);open_sealed()
    with np.load(CAMPAIGN/'frozen_inputs.npz') as data:widths=data['widths_ns'].copy();clean=data['clean_ns'].copy()
    info=json.loads((CAMPAIGN/'input_protocol.json').read_text());c=contract(widths,info['physical'],100,5000,3)
    card=json.loads((Path(record['workspace_root'])/record['active_card']).read_text())
    forecasts={layer:predict(widths,card,layer) for layer in ['L0','L1','L2_static_candidate']}
    prediction=CAMPAIGN/'predictions.json';write(prediction,forecasts);write(CAMPAIGN/'evaluation_contract.json',c)
    commit=dict(utc=now(),official_freeze_sha256=stream_sha(CAMPAIGN/'final_freeze.json'),prediction_sha256=stream_sha(prediction),
        input_sha256=stream_sha(CAMPAIGN/'frozen_inputs.npz'),evaluation_contract_sha256=stream_sha(CAMPAIGN/'evaluation_contract.json'),
        predictor_inputs='frozen input widths and development-only locked cards;no measured test states',physical_reference_started=False,
        forecast_word_records={k:len(v) for k,v in forecasts.items()})
    with (CAMPAIGN/'prediction_commit.json').open('x',encoding='utf-8') as f:json.dump(commit,f,indent=2)
    return c,forecasts,widths,clean


def snr(final,c,predictions,widths,clean):
    measured=np.asarray(final['final_measured_widths_ns']);result={};warm=c['warmup']
    for kind,(lo,hi) in [('gaussian',(964,1987)),('salt_pepper',(2264,3287)),('mixed',(3564,4587))]:
        target=np.arange(lo,hi+1);source=warm+target+2;reference=clean[source];power=float(np.mean((reference-reference.mean())**2))
        def score(v):
            mse=float(np.mean((np.asarray(v)-reference)**2));return dict(samples=len(target),MSE_ns2=mse,SNR_dB=float(10*np.log10(power/mse)) if mse>0 else None)
        values=dict(input=score(widths[source]),physical=score(measured[target]),ideal_median=score(np.asarray(c['words']['s4c2']['expected'])[warm+target]))
        for layer,rows in predictions.items():
            lookup={r['sample']:r['width_ns'] for r in rows if r['node']=='s4c2'};values[layer]=score([lookup[warm+int(k)] for k in target])
        result[kind]=values
    return dict(noise_is_input_corruption=True,signal_power='variance of fixed clean common-center signal',DC_error_retained_in_MSE=True,
        test_set_scale_or_delay_fit=False,noise=result)


def main():
    freeze=CAMPAIGN/'final_freeze.json'
    if not freeze.exists():raise RuntimeError('formal freeze missing;seed remains unopened')
    record=json.loads(freeze.read_text());verify(record)
    budget=record['budgets'];free=psutil.disk_usage(str(OUT)).free
    if free<budget['max_raw_bytes']+budget['min_free_disk_bytes']:raise RuntimeError('registered disk budget not available before seed opening')
    folder=CAMPAIGN/'physical';folder.mkdir(exist_ok=False)
    proc=None;job=None;paused=False;begin=time.perf_counter();status='PREPARING';completed=[];peak=0;last=-5;pause_total=0.
    write(CAMPAIGN/'runner.json',dict(utc=now(),pid=os.getpid(),state=status))
    try:
        c,forecasts,widths,clean=prepare(record);verify(record)
        raw=folder/'transient.raw';log=folder/'ngspice.log';deck=folder/'case.cir';stop=c['expected_stop_ns']
        body=(CAMPAIGN/'generated_electrical.cir').read_text();columns=c['output_columns']
        deck.write_text(body+f'\n.save {" ".join(columns)}\n.tran .125n {stop:.9f}n 0 .125n\n.control\nset klu\nset num_threads=8\nrun {raw.as_posix()}\nrusage all\nquit\n.endc\n.end\n',encoding='ascii')
        cmd=[str(EXE),'-b','-n','-o',str(log.resolve()),str(deck.resolve())]
        write(folder/'launch.json',dict(utc=now(),command=cmd,netlist_sha256=stream_sha(deck),prediction_commit_sha256=stream_sha(CAMPAIGN/'prediction_commit.json'),
            num_device_threads=8,not_a_matched_system_speed_benchmark=True))
        env=os.environ.copy();env['PATH']=str(EXE.parent)+os.pathsep+'D:/MinGW-w64/mingw64/bin'+os.pathsep+env['PATH'];env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        proc=subprocess.Popen(cmd,cwd=folder,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
        job=ProcessJob(proc);tracked=psutil.Process(proc.pid);status='RUNNING'
        with (CAMPAIGN/'monitor.jsonl').open('w',encoding='utf-8',buffering=1) as journal:
            while proc.poll() is None:
                elapsed=time.perf_counter()-begin
                try:peak=max(peak,tracked.memory_info().rss)
                except psutil.Error:pass
                progress=raw_progress(raw);last_ns=progress.get('last_ns') or 0.
                if elapsed-last>=5:
                    value=dict(utc=now(),status=status,controller_pid=os.getpid(),physical_pid=proc.pid,elapsed_s=elapsed,
                        peak_child_RSS_bytes=peak,completed_checkpoints=completed,paused_evaluation_s=pause_total,**progress)
                    # Cache replacement avoids a transient sharing violation killing a reference run.
                    from benchmarks.spice_stream_probe_v34 import write_status_cache
                    write_status_cache(CAMPAIGN/'progress.json',value);journal.write(json.dumps(value)+'\n');last=elapsed
                if elapsed>budget['max_wall_s'] or peak>budget['max_child_RSS_bytes'] or progress.get('raw_bytes',0)>budget['max_raw_bytes'] or psutil.disk_usage(str(OUT)).free<budget['min_free_disk_bytes']:
                    raise RuntimeError('registered time/memory/raw/disk resource gate exceeded')
                remaining=[k for k in [128,1000,5000] if k not in completed]
                if remaining:
                    k=remaining[0];deadline=1460.1+200*(100+k-1)+180
                    if elapsed>budget['checkpoint_wall_s'][str(k)]:raise RuntimeError('checkpoint wall budget exceeded:'+str(k))
                    if last_ns>=deadline:
                        tracked.suspend();paused=True;pause_start=time.perf_counter();verify(record)
                        write(CAMPAIGN/'status.json',dict(utc=now(),state='CHECKPOINT_EVALUATION',targets=k,physical_pid=proc.pid,same_process_paused=True))
                        evaluation=evaluate(raw,CAMPAIGN/f'checkpoint_{k}',c,forecasts,k,live=True)
                        verify(record);pause_total+=time.perf_counter()-pause_start
                        if not evaluation['gate_pass']:raise RuntimeError('frozen checkpoint failed:'+str(k))
                        completed.append(k);tracked.resume();paused=False
                        write(CAMPAIGN/'status.json',dict(utc=now(),state='RUNNING',completed_checkpoints=completed,same_physical_pid=proc.pid))
                time.sleep(.1)
        if proc.returncode!=0:raise RuntimeError('physical simulator exit:'+str(proc.returncode))
        text=log.read_text(errors='replace')
        if 'run simulation(s) aborted' in text.lower():raise RuntimeError('physical numerical abort')
        verify(record);final=evaluate(raw,CAMPAIGN/'final_evaluation',c,forecasts,5000,live=False)
        if not final['gate_pass']:raise RuntimeError('terminal complete-network gate failed')
        write(CAMPAIGN/'three_noise.json',snr(final,c,forecasts,widths,clean))
        status='COMPLETE_JOINT_SYSTEM_ACTIVITY'
        write(CAMPAIGN/'summary.json',dict(utc=now(),status=status,completed_checkpoints=completed,terminal_complete=True,
            wall_s=time.perf_counter()-begin,paused_evaluation_s=pause_total,peak_child_RSS_bytes=peak,
            raw_sha256=stream_sha(raw),final_gate_pass=True,global_D1_D6_acceptance=False,
            full_system_PPA_complete=False,matched_speedup_100x_qualified=False,public_release=False))
    except BaseException as e:
        status='FAILED_OR_INTERRUPTED'
        write(CAMPAIGN/'failure.json',dict(utc=now(),status=status,error=str(e),traceback=traceback.format_exc(),
            completed_checkpoints=completed,wall_s=time.perf_counter()-begin,new_seed_exposed=(CAMPAIGN/'seed_opened.json').exists(),
            partial_results_not_acceptance=True,parameters_refit=False))
        raise
    finally:
        if proc is not None and proc.poll() is None:proc.kill();proc.wait()
        if job is not None:job.close()
        write(CAMPAIGN/'runner_final.json',dict(utc=now(),status=status,controller_pid=os.getpid(),physical_pid=proc.pid if proc else None))


if __name__=='__main__':main()
