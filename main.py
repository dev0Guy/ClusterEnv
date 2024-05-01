import pyximport
import numpy as np
pyximport.install(
    reload_support=True, language_level=3,
    setup_args={"include_dirs": np.get_include()},
)
from clusterEnv.common import *


if __name__ == '__main__':
    j = Job(np.array([[1,2,3]], dtype=np.float64))
    n = Node(np.array([[1,2,3]], dtype=np.float64))
    j_collection = Jobs(
        spec=np.ones((5, 3, 2), dtype=np.float64),
        submission=np.zeros(5, dtype=np.uint)
    )
    n_collection = Nodes(
        spec=np.ones((5, 3, 2), dtype=np.float64),
    )
    cluster = Cluster(
        5, 4, 3, 2
    )
    print("Job=", j)
    print("Jobs=", j_collection[0])
    print("Node=", n)
    print("Nodes=", n_collection[0])
    # print(cluster.schedule(4, 3))
    print(cluster.step(0))
    print(cluster.step(0))
    print(cluster.reset())



# import pyximport
# import numpy as np
#
#
# pyximport.install(setup_args={"include_dirs":np.get_include()},
#                   reload_support=True)
#
# from src.common.types import JobStatus, Job, Node, Nodes, Jobs
# # from src.common.cluster import CClusterEnv
# # from src.common import JobStatus, Job, Node, Nodes, Jobs
# from src.envs.cluster import CCluster
# # import numpy as np
#
# # Create an instance of the Job class
# status = JobStatus.NONEXISTENT
# print(status)
# submission = 123
# usage = np.array([
#     [0,1],
#     [1, 0]
# ]).astype(np.float64)
# wait_time = 10
# run_time = 20
#
# job_instance = Job(status, submission, usage, wait_time, run_time)
# job_instance.change_status(JobStatus.PENDING)
# print(job_instance)
# spec = np.array([[1,2],[3,4]]).astype(np.float64)
# print(type(spec))
# node_instance = Node(spec)
# print(node_instance)
# nodes_spec = np.array([[[1,2],[3,4]]]).astype(np.float64)
# nodes_instance = Nodes(nodes_spec)
# print(nodes_instance)
# nodes_instance.tick()
# print(nodes_instance)
# jobs_spec = np.ones((3,2,1)).astype(np.float64)
# jobs_sub = np.arange(3, dtype=np.uint32)
# jobs_instance = Jobs(jobs_spec, jobs_sub)
# print(jobs_instance)
# jobs_instance.update(1)
#
# #
#
# cluster_instance = CClusterEnv(2, 3,  3)
# print(cluster_instance)
