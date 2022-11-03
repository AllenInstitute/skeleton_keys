from skeleton_keys.hist_tasks import create_layer_histograms

from taskqueue import TaskQueue
import sys


queue = sys.argv[1]
timeout = int(sys.argv[2])
with TaskQueue(qurl=queue, n_threads=0) as tq:
    tq.poll(lease_seconds=timeout)
