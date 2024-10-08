import numpy as np
import rerun as rr
import time

from .biorbd_components.model_updapter import ModelUpdater


class LiveModelIntegration:
    """
    A class to animate a biorbd model in rerun by integrating the dynamics over time.
    """

    def __init__(self, model_path: str, dt: float = 0.01, total_time: float = 5.0):
        """
        Parameters
        ----------
        model_path : str
            The path to the bioMod file.
        dt : float
            Time step for integration.
        total_time : float
            Total simulation time.
        """
        self.model_updater = ModelUpdater.from_file(model_path)
        self.model = self.model_updater.model
        self.biorbd_model = self.model.model

        self.dt = dt
        self.total_time = total_time
        self.time_vector = np.arange(0, self.total_time, self.dt)

        self.nb_q = self.biorbd_model.nbQ()
        self.nb_qdot = self.biorbd_model.nbQdot()

        # Initial conditions
        self.q = np.zeros((self.nb_q, len(self.time_vector)))
        self.qdot = np.zeros((self.nb_qdot, len(self.time_vector)))
        self.qddot = np.zeros((self.nb_qdot, len(self.time_vector)))

        # Set initial position (can be modified as needed)
        self.q[:, 0] = np.array([0.0 for _ in range(self.nb_q)])  # Replace with initial joint positions

        # External forces (if any)
        self.tau = np.zeros(self.nb_q)  # Control torques (assuming zero for passive motion)

    def simulate(self, q: np.ndarray, qdot: np.ndarray, tau: np.ndarray):
        """
        Simulate the model dynamics over time using RK1 integration.
        """
        rr.init(application_id=f"{self.model.name}_simulation", spawn=True)

        self.q[:, 0] = q
        self.qdot[:, 0] = qdot
        self.tau = tau

        for i in range(0, len(self.time_vector) - 1):
            # Compute joint accelerations
            self.qddot[:, i] = self.biorbd_model.ForwardDynamics(self.q[:, i], self.qdot[:, i], self.tau).to_array()

            # Integrate using RK1 (Explicit Euler)
            self.qdot[:, i + 1] = self.qdot[:, i] + self.qddot[:, i] * self.dt
            self.q[:, i + 1] = self.q[:, i] + self.qdot[:, i] * self.dt

            # Update the model visualization
            self.update_model(self.q[:, i + 1])

            # Optional: Sleep to control the simulation speed (adjust as needed)
            time.sleep(self.dt)

    def update_model(self, q: np.ndarray):
        """
        Update the model visualization in rerun.

        Parameters
        ----------
        q : np.ndarray
            The current joint angles.
        """
        self.model_updater.to_rerun(q)

    def run(self, q, qdot, tau):
        """
        Run the simulation.
        """
        self.simulate(q, qdot, tau)


# Usage example
if __name__ == "__main__":
    model_path = "path_to_your_model.bioMod"  # Replace with your model path
    animation = LiveModelIntegration(model_path=model_path)
    animation.run(q=np.array([0,.5, 0]), qdot=np.zeros(animation.nb_qdot), tau=np.zeros(animation.nb_q))
