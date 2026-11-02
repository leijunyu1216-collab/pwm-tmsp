"""Correct inherited metadata/removed-device observations without changing physics."""
import json,re
from benchmarks.retimed_lattice_v56 import OUT,KEEP
from benchmarks.live_spice import write,now
from benchmarks.audit_cost_diagnosis_v56 import stream_sha


def effective():
    source=OUT/'inputs/development24/contract.json';c=json.loads(source.read_text())
    text=(OUT/'pruned/case.cir').read_text();parent_words={'s0c0','s0c1','s0c2','s0c3','s0c4','s1c0','s1c1','s1c2','s1c3','s1c4','s2c1','s2c2','s2c3','s2c4','s3c2','s3c3'}
    cols=[e for e in c['output_columns'] if not any('x'+w+'.' in e or e.lower()==f'v({w})' for w in parent_words)]
    c['output_columns']=list(dict.fromkeys(cols));mos=[line for line in text.splitlines() if line.startswith('mflat_')]
    c.update(clock_sources=len(re.findall(r'(?m)^V\S+ \S+_drive 0 PULSE',text)),mos_instances=len(mos),
        equivalent_mos=sum(float(re.search(r'\bm=(\S+)',line)[1]) for line in mos),
        metadata_source_sha256=stream_sha(source),corrected_removed_device_current_vectors=True,
        corrections='Inherited22word MOS/clock counts and orphan current expressions replaced by actual6word inventory. Physical circuit unchanged.')
    return c


if __name__=='__main__':
    c=effective();write(OUT/'effective_contract.json',c)
    print({k:c[k] for k in ['clock_sources','mos_instances','equivalent_mos']});print('observations',len(c['output_columns']))
