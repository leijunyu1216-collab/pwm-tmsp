"""Readonly status of the fixed-frontier fresh independent activity."""
from benchmarks import status_corrected_campaign_v58 as status
from benchmarks.run_snapshot_campaign_v61 import OUT


if __name__=='__main__':status.OUT=OUT;status.main()
