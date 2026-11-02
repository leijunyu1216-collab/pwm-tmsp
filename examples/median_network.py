from tmsp.networks import median_network,median_stream

network=median_network(5,'odd_even',pruned=True)
result=median_stream([60_000_000,30_000_000,100_000_000,50_000_000,70_000_000,40_000_000],network)
print(dict(minmax_cells=network.minmax_cells,width_register_words=network.word_registers,
    sample=result['sample'].tolist(),available_cycle=result['available_cycle'].tolist(),width_fs=result['width'].tolist()))
