"""Parent-death containment and bounded readonly evaluation memory on Windows."""
import ctypes,os
from ctypes import wintypes as w
import psutil


class Basic(ctypes.Structure):
    _fields_=[('process_time',ctypes.c_int64),('job_time',ctypes.c_int64),('flags',w.DWORD),
        ('min_ws',ctypes.c_size_t),('max_ws',ctypes.c_size_t),('active',w.DWORD),
        ('affinity',ctypes.c_size_t),('priority',w.DWORD),('scheduling',w.DWORD)]
class IO(ctypes.Structure):_fields_=[(n,ctypes.c_uint64) for n in ['read_ops','write_ops','other_ops','read_bytes','write_bytes','other_bytes']]
class Extended(ctypes.Structure):
    _fields_=[('basic',Basic),('io',IO),('process_memory',ctypes.c_size_t),('job_memory',ctypes.c_size_t),('peak_process',ctypes.c_size_t),('peak_job',ctypes.c_size_t)]


class ProcessJob:
    def __init__(self,process):
        if os.name!='nt':raise RuntimeError('qualified runtime requires Windows')
        k=ctypes.windll.kernel32
        k.CreateJobObjectW.argtypes=[ctypes.c_void_p,w.LPCWSTR];k.CreateJobObjectW.restype=w.HANDLE
        k.SetInformationJobObject.argtypes=[w.HANDLE,ctypes.c_int,ctypes.c_void_p,w.DWORD]
        k.AssignProcessToJobObject.argtypes=[w.HANDLE,w.HANDLE]
        k.CloseHandle.argtypes=[w.HANDLE]
        self.handle=k.CreateJobObjectW(None,None)
        if not self.handle:raise OSError('CreateJobObject failed')
        info=Extended();info.basic.flags=0x2000|0x100;info.process_memory=2*2**30
        if not k.SetInformationJobObject(self.handle,9,ctypes.byref(info),ctypes.sizeof(info)) or not k.AssignProcessToJobObject(self.handle,int(process._handle)):
            k.CloseHandle(self.handle);self.handle=None;raise OSError('job containment failed')
    def close(self):
        if self.handle:ctypes.windll.kernel32.CloseHandle(self.handle);self.handle=None


def trim_readonly_cache():
    # Reclaim file-backed working-set pages; immutable data stay in the file.
    proc=psutil.Process();rss=proc.memory_info().rss
    if rss>int(1.1*2**30):
        k=ctypes.windll.kernel32;k.GetCurrentProcess.restype=w.HANDLE
        p=ctypes.windll.psapi;p.EmptyWorkingSet.argtypes=[w.HANDLE]
        if not p.EmptyWorkingSet(k.GetCurrentProcess()):raise OSError('working-set trim failed')
    if proc.memory_info().rss>int(1.6*2**30):raise MemoryError('evaluator memory budget exceeded')
