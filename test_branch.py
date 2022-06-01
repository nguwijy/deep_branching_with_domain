import unittest
import math
import torch
import numpy as np
from branch import Net
from functools import partial

torch.manual_seed(0)  # set seed for reproducibility

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = {
    "device" : device,
    "nu" : 1,
    "T" : .5,
    "epochs" : 3000,
    "branch_nb_states" : 100,
    "branch_nb_states_per_batch" : 1000,
    "branch_patches" : 1,
    "lr_gamma" : 0.1,
    "branch_lr" : 1e-2,
    "branch_activation" : "tanh",
    "save_for_best_model" : False,
    "verbose": False,
    # the model learns quicker without derivatives constraint
    # otherwise we need epochs=10000
    "div_condition_coeff" : 0,
    "poisson_coeff": 0,
}
deriv_map = {
    "allen_cahn_1d" : np.array([0]).reshape(-1, 1),
    "allen_cahn_5d": np.array([0] * 5).reshape(-1, 5),
    "taylor_green" : np.array (
        [
            [1, 0],  # for nabla p
            [0, 1],
            [0, 0],  # for u
            [0, 0],
            [1, 0],  # for nabla u1
            [0, 1],
            [1, 0],  # for nabla u2
            [0, 1],
        ]
    ),
}
zeta_map = {
    "allen_cahn_1d" : None,
    "allen_cahn_5d": None,
    "taylor_green" : np.array([-1, -1, 0, 1, 0, 0, 1, 1]),
}
deriv_condition_deriv_map = {
    "allen_cahn_1d" : None,
    "allen_cahn_5d": None,
    "taylor_green" : np.array(
        [
            [1, 0],
            [0, 1],
        ]
    ),
}
deriv_condition_zeta_map = {
    "allen_cahn_1d" : None,
    "allen_cahn_5d": None,
    "taylor_green" : np.array([0, 1]),
}


def f_example(y, coordinate, problem_name):
    if problem_name.startswith("allen_cahn"):
        return y[0] - y[0] ** 3
    elif problem_name == "taylor_green":
        dim = 2
        f = -y[coordinate]
        for j in range(dim):
            f += -y[dim + j] * y[2 * dim + dim * coordinate + j]
        return f

def phi_example(x, coordinate, problem_name):
    dim = x.shape[0]
    if problem_name.startswith("allen_cahn"):
        return -0.5 - 0.5 * torch.nn.Tanh()(-.5 * x.sum(dim=0) / math.sqrt(dim))
    elif problem_name == "taylor_green":
        if coordinate == 0:
            return -torch.cos(x[0]) * torch.sin(x[1])
        else:
            return torch.sin(x[0]) * torch.cos(x[1])

def exact_example(t, x, T, coordinate, problem_name, p_or_u):
    dim = x.shape[0]
    if problem_name.startswith("allen_cahn"):
        return -0.5 - 0.5 * np.tanh(-.5 * x.sum(axis=0) / math.sqrt(dim) + 3 * (T - t) / 4)
    elif problem_name == "taylor_green":
        nu = config["nu"]
        if p_or_u == "u":
            if coordinate == 0:
                return -np.cos(x[0]) * np.sin(x[1]) * np.exp(-nu * (T - t))
            else:
                return np.sin(x[0]) * np.cos(x[1]) * np.exp(-nu * (T - t))
        else:
            return (
                -1 / 4
                * np.exp(-2 * nu * (T - t))
                * (np.cos(2 * x[0]) + np.cos(2 * x[1]))
            )


class TestBranch(unittest.TestCase):

    def test_allen_cahn_1d(self):
        self.update_with_name("allen_cahn_1d")
        config["x_lo"] = -8
        config["x_hi"] = 8
        config["branch_nb_path_per_state"] = 10000
        config["neurons"] = 20
        config["layers"] = 5
        config["outlier_multiplier"] = 1000
        self.run_test()

    def test_allen_cahn_5d(self):
        self.update_with_name("allen_cahn_5d")
        config["x_lo"] = -8
        config["x_hi"] = 8
        config["branch_nb_path_per_state"] = 100000
        config["neurons"] = 20
        config["layers"] = 5
        config["outlier_multiplier"] = 1000
        self.run_test()

    def test_taylor_green_for_u(self):
        self.update_with_name("taylor_green")
        config["x_lo"] = 0
        config["x_hi"] = math.pi
        config["branch_nb_path_per_state"] = 1000
        config["neurons"] = 100
        config["layers"] = 2
        config["outlier_multiplier"] = 10
        config["continue_from_checkpoint"] = "../logs/20220521-212107-taylor_green_2d-T0.25-nu2"
        self.run_test()
        config["continue_from_checkpoint"] = None

    def test_taylor_green_for_p(self):
        self.update_with_name("taylor_green", p_or_u="p")
        config["x_lo"] = 0
        config["x_hi"] = math.pi
        config["branch_nb_path_per_state"] = 1000
        config["neurons"] = 100
        config["layers"] = 2
        config["outlier_multiplier"] = 10
        config["overtrain_rate_for_p"] = 0.
        config["train_for_u"] = False
        self.run_test(p_or_u="p")
        config["train_for_u"] = True

    @staticmethod
    def update_with_name(problem_name, p_or_u="u"):
        config["problem_name"] = problem_name
        config["deriv_map"] = deriv_map[problem_name]
        config["zeta_map"] = zeta_map[problem_name]
        config["deriv_condition_deriv_map"] = deriv_condition_deriv_map[problem_name]
        config["deriv_condition_zeta_map"] = deriv_condition_zeta_map[problem_name]
        config["f_fun"] = partial(f_example, problem_name=problem_name)
        config["phi_fun"] = partial(phi_example, problem_name=problem_name)
        config["exact_fun"] = partial(
            exact_example,
            problem_name=problem_name,
            p_or_u=p_or_u,
        )

    def run_test(self, p_or_u="u"):
        model = Net(**config)
        model.train_and_eval()
        error = model.compare_with_exact(
            config["exact_fun"],
            return_error=True,
            show_plot=False,
            p_or_u=p_or_u,
        )
        is_error_small = np.isclose(error, 0, atol=1e-2)
        self.assertTrue(is_error_small.all(), f"Some L1 errors {error} are not small")

if __name__ == '__main__':
    unittest.main()
