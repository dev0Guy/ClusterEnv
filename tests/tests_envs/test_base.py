from clusterenv.envs.base import Jobs, JobStatus, ClusterObject
import numpy as np
import pytest


@pytest.fixture
def sample_jobs():
    arrival = np.array([1, 2, 3, 4], dtype=np.uint32)
    usage = np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float64)
    return Jobs(arrival, usage)


class TestJobs:
    def test_length(self, sample_jobs):
        assert len(sample_jobs) == len(sample_jobs.arrival) == len(sample_jobs.usage)

    @pytest.mark.parametrize(
        "idx, expected_result", [(0, (1, 0.5, 0)), (1, (2, 0.6, 0))]
    )
    def test_get_item(self, sample_jobs, idx, expected_result):
        assert sample_jobs[idx] == expected_result

    @pytest.mark.parametrize(
        "idx, status", [(0, JobStatus.RUNNING), (1, JobStatus.COMPLETE)]
    )
    def test_set_item(self, sample_jobs, idx, status):
        sample_jobs[idx] = status
        assert sample_jobs.status[idx] == status.value


@pytest.fixture
def sample_cluster_object():
    nodes = np.array([[10, 20], [15, 25], [20, 30]], dtype=np.float64)
    arrival = np.array([0, 0, 1, 2, 3], dtype=np.uint32)
    usage = np.array(
        [
            [2 for _ in range(10)],
            [3 for _ in range(10)],
            [4 for _ in range(10)],
            [5 for _ in range(10)],
            [6 for _ in range(10)],
        ],
        dtype=np.float64,
    )
    jobs = Jobs(arrival, usage)
    return ClusterObject(nodes, jobs)


class TestClusterObject:
    def test_n_jobs(self, sample_cluster_object):
        assert sample_cluster_object.n_jobs == 5

    def test_n_nodes(self, sample_cluster_object):
        assert sample_cluster_object.n_nodes == 3

    def test_all_jobs_complete_false(self, sample_cluster_object):
        assert not sample_cluster_object.all_jobs_complete()

    def test_tick(self, sample_cluster_object):
        sample_cluster_object.tick()
        assert sample_cluster_object._time == 1
        assert not sample_cluster_object.all_jobs_complete()

    # @pytest.mark.parametrize("n_idx, j_idx, expected_result", [(0, 0, True), (1, 1, True), (2, 2, False), (0, 3, False)])
    # def test_schedule(self, sample_cluster_object, n_idx, j_idx, expected_result):
    #     assert sample_cluster_object.schedule(n_idx, j_idx) == expected_result
