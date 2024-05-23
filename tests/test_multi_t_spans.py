import numpy as np

from pyorerun.multi_frame_rate_phase_rerun import MultiFrameRatePhaseRerun


class PhaseRerun:
    def __init__(self, t_span):
        self.t_span = t_span


MOCK_PHASE_RERUN_1 = PhaseRerun(t_span=np.linspace(0, 1, 11))
MOCK_PHASE_RERUN_2 = PhaseRerun(t_span=np.linspace(0, 2, 41))

MOCK_MULTI_PHASE_RERUN = MultiFrameRatePhaseRerun((MOCK_PHASE_RERUN_1, MOCK_PHASE_RERUN_2))


def test_multi_frame_rate_phase_rerun():
    # Test t_spans property
    assert np.allclose(MOCK_MULTI_PHASE_RERUN.t_spans[0], np.linspace(0, 1, 11))
    assert np.allclose(MOCK_MULTI_PHASE_RERUN.t_spans[1], np.linspace(0, 2, 41))

    assert len(MOCK_MULTI_PHASE_RERUN.t_spans) == 2

    # Test merged_t_span property
    assert np.allclose(MOCK_MULTI_PHASE_RERUN.merged_t_span, np.linspace(0, 2, 41))

    assert MOCK_MULTI_PHASE_RERUN.frame_t_span_idx == [
        [0, 1],
        [1],
        [0, 1],
        [1],
        [0, 1],
        [1],
        [0, 1],
        [1],
        [0, 1],
        [1],
        [0, 1],
        [1],
        [0, 1],
        [1],
        [0, 1],
        [1],
        [0, 1],
        [1],
        [0, 1],
        [1],
        [0, 1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
    ]

    assert len(MOCK_MULTI_PHASE_RERUN.cumulative_frames_in_merged_t_span) == 2
    cumulative_frames_in_merged_t_span_1 = MOCK_MULTI_PHASE_RERUN.cumulative_frames_in_merged_t_span[0]
    cumulative_frames_in_merged_t_span_2 = MOCK_MULTI_PHASE_RERUN.cumulative_frames_in_merged_t_span[1]

    assert cumulative_frames_in_merged_t_span_1 == [
        0,
        1,
        1,
        2,
        2,
        3,
        3,
        4,
        4,
        5,
        5,
        6,
        6,
        7,
        7,
        8,
        8,
        9,
        9,
        10,
        10,
        11,
        11,
        11,
        11,
        11,
        11,
        11,
        11,
        11,
        11,
        11,
        11,
        11,
        11,
        11,
        11,
        11,
        11,
        11,
        11,
    ]
    assert cumulative_frames_in_merged_t_span_2 == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
    ]

    assert len(cumulative_frames_in_merged_t_span_1) == 41
    assert len(cumulative_frames_in_merged_t_span_2) == 41
