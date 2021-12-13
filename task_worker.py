from skeleton_keys.tasks import *

from taskqueue import TaskQueue
import sys


queue = sys.argv[1]
timeout = int(sys.argv[2])
with TaskQueue(qurl=queue, n_threads=0) as tq:
    tq.poll(lease_seconds=timeout)
