"""Continuous native execution,readonly frontier scoring and live resource guard."""
import json,os,platform,subprocess,threading,time,traceback
from pathlib import Path
import numpy as np
import psutil
from benchmarks.physical_common import ROOT
from benchmarks.live_spice import now,write
from benchmarks.audit_cost_diagnosis_v56 import stream_sha
from benchmarks.validation_guard_v41 import verify_frozen
from benchmarks.network_streaming_pilot_v52 import raw_progress
from benchmarks.streaming_cli_probe_v47 import EXE
from benchmarks.campaign_resources_v57 import ProcessJob
from benchmarks.snapshot_observations_v61 import evaluate
from benchmarks.joint_campaign_v57 import snr
from benchmarks.run_corrected_campaign_v58 import load_committed


def deadline(k):return 1460.1+200*(100+k-1)+180


def run(stage,*,checkpoints=(128,1000,5000),final_targets=5000,noise=True):
    stage=Path(stage);record=json.loads((stage/'final_freeze.json').read_text());budget=record['budgets']
    def verify():
        verify_frozen(ROOT,record['assets_sha256'])
        if platform.python_version()!=record['environment']['python'] or np.__version__!=record['environment']['numpy']:
            raise RuntimeError('frozen runtime changed')
        return load_committed(stage,record)
    c,forecast,widths,clean=verify()
    if psutil.disk_usage(str(ROOT)).free<budget['max_raw_bytes']+budget['min_free_disk_bytes']:
        raise RuntimeError('original disk admission failed')
    folder=stage/'physical';folder.mkdir(exist_ok=False);raw=folder/'transient.raw';log=folder/'ngspice.log'
    deck=folder/'case.cir';body=(stage/'generated_electrical.cir').read_text();stop=c['expected_stop_ns']
    deck.write_text(body+f'\n.save {" ".join(c["output_columns"])}\n.tran .125n {stop:.9f}n 0 .125n\n.control\nset klu\nset no_mem_check\nset num_threads=8\nrun {raw.as_posix()}\nrusage all\nquit\n.endc\n.end\n',encoding='ascii')
    command=[str(EXE),'-b','-n','-o',str(log.resolve()),str(deck.resolve())]
    write(folder/'launch.json',dict(utc=now(),command=command,deck_sha256=stream_sha(deck),num_device_threads=8,
        not_a_matched_speed_benchmark=True,no_process_suspend_or_resume=True,prediction_commit_sha256=stream_sha(stage/'prediction_commit.json')))
    env=os.environ.copy();env['PATH']=str(EXE.parent)+os.pathsep+'D:/MinGW-w64/mingw64/bin'+os.pathsep+env['PATH']
    env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    proc=None;job=None;watch=None;done=threading.Event();reason=[];completed=[];peak=[0];state=['PREPARING'];started=time.perf_counter()
    def monitor():
        last=-30
        with (stage/'monitor.jsonl').open('x',encoding='utf-8',buffering=1) as journal:
            while not done.is_set():
                elapsed=time.perf_counter()-started
                try:peak[0]=max(peak[0],psutil.Process(proc.pid).memory_info().rss)
                except psutil.Error:pass
                progress=raw_progress(raw);free=psutil.disk_usage(str(ROOT)).free
                pending=[k for k in checkpoints if k not in completed]
                failure=(elapsed>budget['max_wall_s'] or peak[0]>budget['max_child_RSS_bytes']
                    or progress.get('raw_bytes',0)>budget['max_raw_bytes'] or free<budget['min_free_disk_bytes'])
                if pending and elapsed>budget['checkpoint_wall_s'][str(pending[0])]:failure=True
                if failure and not reason:
                    reason.append('original time/memory/raw/disk/checkpoint resource gate exceeded')
                    if proc.poll() is None:proc.kill()
                if elapsed-last>=30:
                    value=dict(utc=now(),status=state[0],controller_pid=os.getpid(),physical_pid=proc.pid,
                        elapsed_s=elapsed,peak_child_RSS_bytes=peak[0],completed_checkpoints=list(completed),
                        physical_process_suspended=False,free_disk_bytes=free,**progress)
                    from benchmarks.spice_stream_probe_v34 import write_status_cache
                    write_status_cache(stage/'progress.json',value);journal.write(json.dumps(value)+'\n')
                    print(json.dumps(value),flush=True);last=elapsed
                done.wait(.5)
    def guarded_monitor():
        try:monitor()
        except BaseException as error:
            reason.append('resource monitor failed:'+repr(error))
            if proc.poll() is None:proc.kill()
            write(stage/'monitor_failure.json',dict(utc=now(),error=repr(error)))
    try:
        write(stage/'runner.json',dict(utc=now(),pid=os.getpid(),state='STARTING',no_suspend=True))
        proc=subprocess.Popen(command,cwd=folder,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
        job=ProcessJob(proc);state[0]='RUNNING';watch=threading.Thread(target=guarded_monitor,daemon=True);watch.start()
        while True:
            if reason:raise RuntimeError(reason[0])
            pending=[k for k in checkpoints if k not in completed];progress=raw_progress(raw)
            if pending and (progress.get('last_ns') or 0)>=deadline(pending[0]):
                k=pending[0];verify();state[0]='SNAPSHOT_EVALUATION'
                write(stage/'status.json',dict(utc=now(),state=state[0],targets=k,physical_pid=proc.pid,physical_process_suspended=False))
                result=evaluate(raw,stage/f'checkpoint_{k}',c,forecast,k,live=True);verify()
                if not result['gate_pass']:raise RuntimeError('original scientific checkpoint gate failed:'+str(k))
                if reason:raise RuntimeError(reason[0])
                completed.append(k);state[0]='RUNNING';continue
            if proc.poll() is not None:break
            time.sleep(.01)
        if reason:raise RuntimeError(reason[0])
        if proc.returncode!=0:raise RuntimeError('native process exit:'+str(proc.returncode))
        if any(s in log.read_text(errors='replace').lower() for s in ['run simulation(s) aborted','timestep too small']):
            raise RuntimeError('native numerical abort recorded')
        progress=raw_progress(raw)
        if (progress.get('last_ns') or 0)<stop-1e-6:raise RuntimeError('native endpoint incomplete')
        if list(completed)!=list(checkpoints):raise RuntimeError('nested checkpoint coverage incomplete')
        state[0]='TERMINAL_EVALUATION';verify()
        final=evaluate(raw,stage/'final_evaluation',c,forecast,final_targets,live=False)
        if reason:raise RuntimeError(reason[0])
        if not final['gate_pass']:raise RuntimeError('original terminal scientific gate failed')
        if noise:write(stage/'three_noise.json',snr(final,c,forecast,widths,clean))
        digest=stream_sha(raw);verify()
        if reason:raise RuntimeError(reason[0])
        state[0]='COMPLETE_JOINT_SYSTEM_ACTIVITY'
        write(stage/'summary.json',dict(utc=now(),status=state[0],completed_checkpoints=completed,terminal_complete=True,
            final_gate_pass=True,native_exit_code=proc.returncode,physical_process_suspended=False,
            wall_s=time.perf_counter()-started,peak_child_RSS_bytes=peak[0],raw_sha256=digest,
            global_D1_D6_acceptance=False,full_system_PPA_complete=False,matched_speedup_100x_qualified=False))
    except BaseException as e:
        state[0]='FAILED_OR_INTERRUPTED'
        write(stage/'failure.json',dict(utc=now(),status=state[0],error=repr(e),traceback=traceback.format_exc(),
            completed_checkpoints=completed,native_exit_code=proc.poll() if proc else None,
            native_alive_before_controller_cleanup=proc.poll() is None if proc else False,
            wall_s=time.perf_counter()-started,resource_reason=reason,new_seed_exposed=(stage/'seed_opened.json').exists(),
            physical_process_suspended=False,parameters_refit=False));raise
    finally:
        done.set()
        if watch is not None:watch.join(timeout=3)
        killed=bool(proc is not None and proc.poll() is None)
        if killed:proc.kill();proc.wait()
        if job is not None:job.close()
        write(stage/'runner_final.json',dict(utc=now(),status=state[0],controller_pid=os.getpid(),
            physical_pid=proc.pid if proc else None,native_exit_code=proc.poll() if proc else None,
            native_terminated_by_controller=killed,physical_process_suspended=False))
