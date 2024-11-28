import numpy as np
import rerun as rr

from .phase_rerun import PhaseRerun


class MultiFrameRatePhaseRerun:
    """
    A class to animate a biorbd model in rerun.

    Attributes
    ----------
    phase_reruns : list[PhaseRerun]
        The phases to animate.
    """

    def __init__(self, phase_reruns: list[PhaseRerun]):
        """
        Parameters
        ----------
        phase_reruns: list[PhaseRerun]
            The phases to animate.
        """
        self.phase_reruns = phase_reruns

    @property
    def nb_phases(self) -> int:
        """
        Get the number of phases in parallel.
        """
        return len(self.phase_reruns)

    @property
    def t_spans(self) -> list[np.ndarray]:
        """
        Get the time spans of the phases.
        """
        rounding_to_nano = 9
        return [np.round(phase_rerun.t_span, rounding_to_nano) for phase_rerun in self.phase_reruns]

    @property
    def merged_t_span(self) -> np.ndarray:
        """
        Merge and sort the time spans of the phases, so that redundant time framed are removed.
        """
        # concatenate all time spans
        all_t_spans = np.concatenate(self.t_spans)
        sorted_all_t_spans = np.sort(all_t_spans)

        # remove duplicates
        return np.unique(sorted_all_t_spans)

    @property
    def frame_t_span_idx(self) -> list[list[int]]:
        """
        Get the index of the time spans for each frame.
        """
        frame_t_span_idx = []
        for t in self.merged_t_span:
            idx = []
            for i, t_span in enumerate(self.t_spans):
                if t in t_span:
                    idx.append(i)
            frame_t_span_idx.append(idx)

        return frame_t_span_idx

    @property
    def cumulative_frames_in_merged_t_span(self) -> list[list[int]]:
        """
        Get the cumulative frames in the merged time span.
        """
        frame_t_span_idx = self.frame_t_span_idx
        return [calculate_cumulative_frames(p, frame_t_span_idx) for p in range(self.nb_phases)]

    def rerun_by_frame(
        self, name: str = "animation_phase", init: bool = True, clear_last_node: bool = False, notebook: bool = False
    ) -> None:
        if self.nb_phases == 1:
            self.phase_reruns[0].rerun_by_frame(name, init, clear_last_node, notebook)
            return

        if init:
            rr.init(f"{name}_{0}", spawn=True if not notebook else False)

        for phase_rerun in self.phase_reruns:
            frame = 0
            rr.set_time_seconds("stable_time", phase_rerun.t_span[frame])
            phase_rerun.timeless_components.to_rerun()
            phase_rerun.biorbd_models.to_rerun(frame)
            phase_rerun.xp_data.to_rerun(frame)

        cumulative_frames_in_merged_t_span = self.cumulative_frames_in_merged_t_span
        for frame, (t, idx) in enumerate(zip(self.merged_t_span[1:], self.frame_t_span_idx[1:])):
            rr.set_time_seconds("stable_time", t)
            for i in idx:
                frame_i = cumulative_frames_in_merged_t_span[i][frame + 1]
                self.phase_reruns[i].biorbd_models.to_rerun(frame_i)
                self.phase_reruns[i].xp_data.to_rerun(frame_i)

        if clear_last_node:
            for phase_rerun in self.phase_reruns:
                for component in [
                    *phase_rerun.biorbd_models.component_names,
                    *phase_rerun.xp_data.component_names,
                    *phase_rerun.timeless_components.component_names,
                ]:
                    rr.log(component, rr.Clear(recursive=False))

    def rerun(
        self, name: str = "animation_phase", init: bool = True, clear_last_node: bool = False, notebook: bool = False
    ) -> None:
        if self.nb_phases == 1:
            self.phase_reruns[0].rerun_with_chunks(name, init, clear_last_node, notebook)
            return

        if init:
            rr.init(f"{name}_{0}", spawn=True if not notebook else False)

        for phase_rerun in self.phase_reruns:
            frame = 0
            rr.set_time_seconds("stable_time", phase_rerun.t_span[frame])
            phase_rerun.timeless_components.to_rerun()
            phase_rerun.biorbd_models.initialize()
            phase_rerun.xp_data.initialize()

            times = [rr.TimeSecondsColumn("stable_time", phase_rerun.t_span)]

            for name, chunk in phase_rerun.xp_data.to_chunk().items():
                rr.send_columns(
                    name,
                    times=times,
                    components=chunk,
                )

            for name, chunk in phase_rerun.biorbd_models.to_chunk().items():
                rr.send_columns(
                    name,
                    times=times,
                    components=chunk,
                )

        # cumulative_frames_in_merged_t_span = self.cumulative_frames_in_merged_t_span
        # for frame, (t, idx) in enumerate(zip(self.merged_t_span[1:], self.frame_t_span_idx[1:])):
        #     rr.set_time_seconds("stable_time", t)
        #     for i in idx:
        #         frame_i = cumulative_frames_in_merged_t_span[i][frame + 1]
        #         self.phase_reruns[i].biorbd_models.to_rerun(frame_i)
        #         self.phase_reruns[i].xp_data.to_rerun(frame_i)
        #
        # if clear_last_node:
        #     for phase_rerun in self.phase_reruns:
        #         for component in [
        #             *phase_rerun.biorbd_models.component_names,
        #             *phase_rerun.xp_data.component_names,
        #             *phase_rerun.timeless_components.component_names,
        #         ]:
        #             rr.log(component, rr.Clear(recursive=False))


def calculate_cumulative_frames(id: int, frame_t_span_idx) -> list[int]:
    """
    Calculate the cumulative frames for a given id in the frame_t_span_idx list.
    Example output for frame_t_span_idx = [0, 0, 1, 1, 2, 2, 2, 3, 3]
    """
    cumulative_frames = []
    counter = 0
    for frame_idx in frame_t_span_idx:
        cumulative_frames.append(counter)
        if id in frame_idx:
            counter += 1
    return cumulative_frames
