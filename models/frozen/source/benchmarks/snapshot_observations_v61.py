"""One immutable complete-row frontier while the native producer keeps running."""
import json,types
from pathlib import Path
from benchmarks.audit_cost_diagnosis_v56 import partial_read
from benchmarks.campaign_observations_v60 import module
from benchmarks.ordered_observations_v60 import build as ordered_build
from benchmarks.live_spice import now,write


class Snapshot:
    def __init__(self,path):
        self.path=Path(path);self.array,self.names,self.tail=partial_read(self.path)
        self.rows,self.columns=self.array.shape;self.offset=self.array.offset
        st=self.path.stat();self.identity=(st.st_dev,st.st_ino)
        self.byte_frontier=self.offset+self.rows*self.columns*8
        if self.rows<2:raise ValueError('snapshot has insufficient complete rows')
        self.capture_utc=now()
    def read(self,path):
        if Path(path).resolve()!=self.path.resolve():raise ValueError('snapshot path changed')
        st=self.path.stat()
        if (st.st_dev,st.st_ino)!=self.identity or st.st_size<self.byte_frontier:
            raise ValueError('snapshot source replaced or truncated')
        return self.array,self.names,self.tail
    def build(self,path,*,live=True,audit_path=None):
        self.read(path)
        namespace=dict(ordered_build.__globals__);namespace['partial_read']=self.read
        builder=types.FunctionType(ordered_build.__code__,namespace,ordered_build.__name__,ordered_build.__defaults__)
        builder.__kwdefaults__=ordered_build.__kwdefaults__
        # Both immutable terminal and growing-prefix reads use the same frontier.
        return builder(path,live=True,audit_path=audit_path)
    def record(self):
        self.read(self.path)
        return dict(capture_utc=self.capture_utc,source=str(self.path.resolve()),rows=self.rows,columns=self.columns,
            data_offset_bytes=self.offset,complete_byte_frontier=self.byte_frontier,
            excluded_incomplete_bytes_at_capture=self.tail,source_bytes_after_evaluation=self.path.stat().st_size,
            all_reads_use_identical_frontier=True,physical_process_suspended=False,
            future_rows_used=False,rows_removed_within_frontier=0)


def evaluate(raw,folder,c,predictions,limit,*,live=True,cut_ns=None):
    if cut_ns is not None:raise ValueError('v61 frontier is based on complete rows,not a fitted time cut')
    folder=Path(folder);snap=Snapshot(raw);m=module()
    m.partial_read=snap.read
    m.crossing_build=lambda p,**kw:snap.build(p,audit_path=folder/'native_time_audit.json')
    # Live scoring bounds to this complete prefix; terminal source is immutable.
    result=m.evaluate(raw,folder,c,predictions,limit,live=live)
    write(folder/'snapshot.json',snap.record());return result
