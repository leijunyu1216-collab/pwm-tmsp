"""Same scientific evaluator with an explicit audited numerical reader version."""
import types
from pathlib import Path
from benchmarks.physical_common import ROOT
from benchmarks.ordered_observations_v60 import build


def module(*,audit_path=None):
    path=ROOT/'benchmarks/campaign_live_v57.py';source=path.read_text()
    old='from benchmarks.crossing_index_v56 import build as crossing_build'
    new='from benchmarks.ordered_observations_v60 import build as crossing_build'
    if source.count(old)!=1:raise RuntimeError('original evaluator source drift')
    result=types.ModuleType('ordered_campaign_v60');result.__file__=str(path)
    exec(compile(source.replace(old,new),str(path),'exec'),result.__dict__)
    if audit_path is not None:
        result.crossing_build=lambda path,**kw:build(path,audit_path=audit_path,**kw)
    return result


def evaluate(raw,folder,c,predictions,limit,*,live=True,cut_ns=None):
    folder=Path(folder)
    if folder.exists():raise FileExistsError(folder)
    # evaluate creates the exclusive result directory before indexing.
    return module(audit_path=folder/'native_time_audit.json').evaluate(
        raw,folder,c,predictions,limit,live=live,cut_ns=cut_ns)
