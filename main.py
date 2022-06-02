import math
import numpy as np
import torch
from scipy.stats import norm
from branch.branch import Net

lower_bound, upper_bound = -2, 2
nu = 1
y, eps = 0, 1
a, b = y - eps, y + eps


def f_example(y, coodinate=0):
    """
    idx 0 -> no deriv
    """
    return torch.zeros_like(y[0])


def phi_example(x, coodinate=0):
    return torch.logical_and(x[0] <= b, x[0] >= a).float()


def exact_example(t, x, T, coordinate=0, k_arr=range(-5, 5)):
    if t == T:
        return np.logical_and(x[0] <= b, x[0] >= a)
    else:
        normal_std = math.sqrt(nu * (T - t))
        ans = 0
        for k in k_arr:
            mu = x[0] - 2 * k * (upper_bound - lower_bound)
            ans += (norm.cdf((b - mu) / normal_std) - norm.cdf((a - mu) / normal_std))
            mu = 2 * lower_bound - 2 * k * (upper_bound - lower_bound) - x[0]
            ans -= (norm.cdf((b - mu) / normal_std) - norm.cdf((a - mu) / normal_std))
        return ans


def conditional_probability_to_survive(t, x, y, k_arr=range(-5, 5)):
    ans = 0
    for k in k_arr:
        ans += (
                torch.exp(((y - x) ** 2 - (y - x + 2 * k * (upper_bound - lower_bound)) ** 2) / (2 * t))
                - torch.exp(
            ((y - x) ** 2 - (y + x - 2 * lower_bound + 2 * k * (upper_bound - lower_bound)) ** 2) / (2 * t)
        )
        )
    return ans.prod(dim=0)


def is_x_inside(x):
    return torch.logical_and(lower_bound <= x, x <= upper_bound).all(dim=0)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    deriv_map = np.array([0]).reshape(-1, 1)
    problem_name = "heat_equation"
    t_lo, x_lo, x_hi, n = 0., lower_bound, upper_bound, 0

    patches = 2
    T = patches * 1.0
    model = Net(
        problem_name=problem_name,
        f_fun=f_example,
        deriv_map=deriv_map,
        phi_fun=phi_example,
        conditional_probability_to_survive=conditional_probability_to_survive,
        is_x_inside=is_x_inside,
        device=device,
        x_lo=x_lo,
        x_hi=x_hi,
        T=T,
        verbose=True,
        nu=nu,
        branch_patches=patches,
        overtrain_rate=0.,
        save_as_tmp=True,
    )
    model.train_and_eval()
    model.compare_with_exact(exact_fun=exact_example)

if __name__ == "__main__":
    main()