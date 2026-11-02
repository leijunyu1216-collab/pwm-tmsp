"""Readonly terminal-aware status for the audited observation version."""
from benchmarks import status_corrected_campaign_v58 as status
from benchmarks.run_ordered_campaign_v60 import OUT


if __name__=='__main__':
    status.OUT=OUT;status.main()
