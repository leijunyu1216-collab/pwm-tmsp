"""Bit-exact native-raw archive:per-column XOR,byte planes,Zstandard blocks.

The native header,all float64 bit patterns and any partial trailing row are
preserved. A complete decode/hash check precedes removal of the source file.
"""
import gc,hashlib,json,struct
from pathlib import Path
import numpy as np
import zstandard as zs
from benchmarks.physical_common import ROOT
from benchmarks.audit_cost_diagnosis_v56 import partial_read,stream_sha
from benchmarks.live_spice import now,write

MAGIC=b'TMSP_XOR_ZSTD_RAW_1\n'


def encode_block(raw,columns):
    a=np.frombuffer(raw,dtype='<u8').reshape(-1,columns);x=a.copy();x[1:]=a[1:]^a[:-1]
    return x.view(np.uint8).reshape(len(x),columns,8).transpose(2,1,0).copy().tobytes()


def decode_block(raw,columns,rows):
    x=np.frombuffer(raw,np.uint8).reshape(8,columns,rows).transpose(2,1,0).copy().view('<u8').reshape(rows,columns)
    np.bitwise_xor.accumulate(x,axis=0,out=x);return x.tobytes()


def chunks(path):
    with Path(path).open('rb') as f:
        if f.read(len(MAGIC))!=MAGIC:raise ValueError('archive format mismatch')
        n,hlen=struct.unpack('<IQ',f.read(12));yield f.read(hlen)
        decompressor=zs.ZstdDecompressor()
        while True:
            rows,size=struct.unpack('<IQ',f.read(12))
            if rows==0:
                yield f.read(size)
                if f.read(1):raise ValueError('unexpected archive tail')
                break
            data=f.read(size)
            if len(data)!=size:raise ValueError('truncated archive')
            yield decode_block(decompressor.decompress(data,max_output_size=rows*n*8),n,rows)


def archive(path):
    path=Path(path).resolve()
    if not path.is_relative_to((ROOT/'artifacts').resolve()):raise ValueError('archive target outside workspace artifacts')
    destination=path.with_suffix(path.suffix+'.tmraw');manifest=path.with_suffix(path.suffix+'.archive.json')
    if manifest.exists():
        m=json.loads(manifest.read_text())
        if not destination.exists() or stream_sha(destination)!=m['archive_sha256']:raise RuntimeError('existing archive changed')
        if path.exists() and path.stat().st_nlink==1:
            if stream_sha(path)!=m['original_sha256']:raise RuntimeError('source changed after archive verification')
            path.unlink()
        return m
    if path.stat().st_nlink!=1:raise ValueError('shared hard-link raw is not an archive target')
    if path.suffix=='.f64':
        schema=json.loads((path.parent/'stream_schema.json').read_text());columns=int(schema['columns'])
        total,tail=divmod(path.stat().st_size,columns*8);offset=0;family='headerless_float64'
    else:
        mapped,names,tail=partial_read(path);offset=mapped.offset;total=len(mapped);columns=len(names);del mapped;gc.collect();family='native_raw'
    temporary=destination.with_suffix(destination.suffix+'.part')
    if temporary.exists():
        # This is a regenerable unfinished compressed copy. The complete
        # source is still present;no experimental data are unique to it.
        if not temporary.resolve().is_relative_to((ROOT/'artifacts').resolve()) or temporary.resolve().parent!=path.parent:
            raise ValueError('temporary archive outside intended artifact directory')
        temporary.unlink()
    digest=hashlib.sha256();count=0;compressor=zs.ZstdCompressor(level=3)
    with path.open('rb') as src,temporary.open('xb') as out:
        header=src.read(offset);out.write(MAGIC+struct.pack('<IQ',columns,len(header))+header);digest.update(header);count+=len(header)
        for first in range(0,total,4096):
            rows=min(4096,total-first);data=src.read(rows*columns*8);digest.update(data);count+=len(data)
            packed=compressor.compress(encode_block(data,columns));out.write(struct.pack('<IQ',rows,len(packed))+packed)
        remaining=src.read();digest.update(remaining);count+=len(remaining);out.write(struct.pack('<IQ',0,len(remaining))+remaining)
    checked=hashlib.sha256();decoded=0
    for data in chunks(temporary):checked.update(data);decoded+=len(data)
    if checked.hexdigest()!=digest.hexdigest() or decoded!=count:raise RuntimeError('archive bit-exact verification failed;source retained')
    temporary.rename(destination)
    result=dict(utc=now(),source=str(path.relative_to(ROOT)),archive=str(destination.relative_to(ROOT)),
        original_sha256=digest.hexdigest(),archive_sha256=stream_sha(destination),original_bytes=count,archive_bytes=destination.stat().st_size,
        decoded_all_bytes_bit_exact=True,columns=columns,complete_rows=total,partial_tail_bytes=tail,
        source_family=family,
        format='TMSP_XOR_ZSTD_RAW_1;little-endian uint64 XOR by column,8byte-plane transpose,Zstd blocks;header and tail verbatim',
        source_removed_only_after_full_decode_verification=True)
    write(manifest,result);path.unlink();return result


def restore(path,destination):
    destination=Path(destination)
    with destination.open('xb') as f:
        for data in chunks(path):f.write(data)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['archive','restore']);p.add_argument('path');p.add_argument('destination',nargs='?');a=p.parse_args()
    print(archive(a.path)) if a.mode=='archive' else restore(a.path,a.destination)
