#!/usr/bin/env python
# coding: utf-8

import os
import hostlist


class IDR:
    def __init__(self):
        # get SLURM variables
        self.rank = int(os.environ['SLURM_PROCID'])
        self.local_rank = int(os.environ['SLURM_LOCALID'])
        self.size = int(os.environ['SLURM_NTASKS'])
        self.cpus_per_task = int(os.environ['SLURM_CPUS_PER_TASK'])

        # get node list from slurm
        self.hostnames = hostlist.expand_hostlist(os.environ['SLURM_JOB_NODELIST'])

        # get IDs of reserved GPU   
        self.gpu_ids = os.environ['SLURM_STEP_GPUS'].split(",")

        # define MASTER_ADD & MASTER_PORT
        os.environ['MASTER_ADDR'] = self.hostnames[0]
        os.environ['MASTER_PORT'] = str(12345)  # to avoid port conflict on the same node