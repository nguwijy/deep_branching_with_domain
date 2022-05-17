import time
import math
import torch
from torch.distributions.exponential import Exponential
import matplotlib.pyplot as plt
import numpy as np
from fdb import fdb_nd
from scipy.stats import norm

torch.manual_seed(0)  # set seed for reproducibility


class Net(torch.nn.Module):
    """
    deep branching approach to solve PDE with utility functions
    """
    def __init__(
        self,
        f_fun,
        deriv_map,
        phi_fun=(lambda x: x),
        phi0=0,
        conditional_probability_to_survive=(lambda t, x, y: torch.ones_like(x[0])),
        is_x_inside=(lambda x: torch.ones_like(x[0]).bool()),
        x_lo=-10.0,
        x_hi=10.0,
        t_lo=0.0,
        t_hi=0.0,
        T=1.0,
        nu=0.5,
        branch_exponential_lambda=None,
        neurons=20,
        layers=5,
        branch_lr=1e-2,
        lr_milestones=[1000, 2000],
        lr_gamma=0.1,
        weight_decay=0,
        branch_nb_path_per_state=10000,
        branch_nb_states=1000,
        branch_nb_states_per_batch=1000,
        epochs=3000,
        batch_normalization=True,
        antithetic=True,
        overtrain_rate=0.,
        device="cpu",
        branch_activation="tanh",
        verbose=False,
        fix_all_dim_except_first=True,
        branch_patches=1,
        outlier_percentile=1,
        outlier_multiplier=1000,
        code=None,
        **kwargs,
    ):
        super(Net, self).__init__()
        self.f_fun = f_fun
        self.phi_fun = phi_fun
        self.phi0 = phi0
        self.conditional_probability_to_survive = conditional_probability_to_survive
        self.is_x_inside = is_x_inside
        self.deriv_map = deriv_map
        self.n, self.dim = deriv_map.shape
        # patching is used for calculating the target expected value of the tree in branch_patches steps
        #
        # for example, when t_lo=t_hi=0 and branch_patches=2
        # the algorithm calculates the function u(T/2, x) with terminal condition phi
        # then, the algorithm calculates the function u(0, x) with terminal condition of u(T/2, x)
        #
        # such approach relies on precise approximation of u(T/2, x)
        # which is very time-consuming in high dimensional case
        self.patches = branch_patches
        self.code = np.array([-1] * self.dim) if code is None else np.array(code)

        # store the (faa di bruno) fdb results for quicker lookup
        start = time.time()
        self.fdb_lookup = {
            tuple(deriv): fdb_nd(self.n, tuple(deriv)) for deriv in deriv_map
        }
        self.fdb_runtime = time.time() - start
        self.mechanism_tot_len = self.dim * self.n ** 2 + sum(
            [len(self.fdb_lookup[tuple(deriv)]) for deriv in deriv_map]
        )

        self.layer = torch.nn.ModuleList(
            [
                torch.nn.ModuleList(
                    [torch.nn.Linear(self.dim + 1, neurons, device=device)]
                    + [
                        torch.nn.Linear(neurons, neurons, device=device)
                        for _ in range(layers)
                    ]
                    + [torch.nn.Linear(neurons, 1, device=device)]
                )
                for _ in range(branch_patches)
            ]
        )
        self.bn_layer = torch.nn.ModuleList(
            [
                torch.nn.ModuleList(
                    [torch.nn.BatchNorm1d(self.dim + 1, device=device)]
                    + [
                        torch.nn.BatchNorm1d(neurons, device=device)
                        for _ in range(layers + 1)
                    ]
                )
                for _ in range(branch_patches)
            ]
        )
        self.lr = branch_lr
        self.lr_milestones = lr_milestones
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay

        self.loss = torch.nn.MSELoss()
        self.activation = {
            "tanh": torch.nn.Tanh(),
            "relu": torch.nn.ReLU(),
            "softplus": torch.nn.ReLU(),
        }[branch_activation]
        self.batch_normalization = batch_normalization
        self.nb_states = branch_nb_states
        self.nb_states_per_batch = branch_nb_states_per_batch
        self.nb_path_per_state = branch_nb_path_per_state
        self.x_lo = x_lo
        self.x_hi = x_hi
        # slight overtrain the domain of x for higher precision near boundary
        self.adjusted_x_boundaries = (
            x_lo - overtrain_rate * (x_hi - x_lo),
            x_hi + overtrain_rate * (x_hi - x_lo),
        )
        self.t_lo = t_lo
        self.t_hi = t_hi
        self.T = T
        self.nu = nu
        self.delta_t = (T - t_lo) / branch_patches
        self.outlier_percentile = outlier_percentile
        self.outlier_multiplier = outlier_multiplier

        self.exponential_lambda = branch_exponential_lambda if branch_exponential_lambda is not None else -math.log(.95)/T
        self.epochs = epochs
        self.antithetic = antithetic
        self.device = device
        self.verbose = verbose
        self.fix_all_dim_except_first = fix_all_dim_except_first
        self.t_boundaries = torch.tensor(
            ([t_lo + i * self.delta_t for i in range(branch_patches)] + [T])[::-1],
            device=device,
        )
        self.adjusted_t_boundaries = [
            (lo, hi) for hi, lo in zip(self.t_boundaries[1:], self.t_boundaries[1:])
        ]

    def forward(self, x, patch=None):
        """
        self(x) evaluates the neural network approximation NN(x)
        """
        if patch is not None:
            if self.batch_normalization:
                x = self.bn_layer[patch][0](x)
            y = x
            for idx, (f, bn) in enumerate(
                zip(self.layer[patch][:-1], self.bn_layer[patch][1:])
            ):
                tmp = f(y)
                tmp = self.activation(tmp)
                if self.batch_normalization:
                    tmp = bn(tmp)
                if idx == 0:
                    y = tmp
                else:
                    # resnet
                    y = tmp + y

            y = self.layer[patch][-1](y).reshape(-1)
        else:
            yy = []
            for p in range(self.patches):
                if self.batch_normalization:
                    x = self.bn_layer[p][0](x)
                y = x
                for idx, (f, bn) in enumerate(
                    zip(self.layer[p][:-1], self.bn_layer[p][1:])
                ):
                    tmp = f(y)
                    tmp = self.activation(tmp)
                    if self.batch_normalization:
                        tmp = bn(tmp)
                    if idx == 0:
                        y = tmp
                    else:
                        # resnet
                        y = tmp + y
                yy.append(self.layer[p][-1](y).reshape(-1))
            idx = self.bisect_left(x[:, 0])
            y = torch.gather(torch.stack(yy, dim=-1), -1, idx.reshape(-1, 1)).squeeze(
                -1
            )
        return y

    def bisect_left(self, val):
        """
        find the index of val based on the discretization of self.t_boundaries
        it is only used when branch_patches > 1
        """
        idx = (
            torch.max(self.t_boundaries <= (val + 1e-8).reshape(-1, 1), dim=1)[
                1
            ].reshape(val.shape)
            - 1
        )
        # t_boundaries[0], use the first network
        idx = idx.where(~(val == self.t_boundaries[0]), torch.zeros_like(idx))
        # t_boundaries[-1], use the last network
        idx = idx.where(
            ~(val == self.t_boundaries[-1]),
            (self.t_boundaries.shape[0] - 2) * torch.ones_like(idx),
        )
        return idx

    @staticmethod
    def nth_derivatives(order, y, x):
        """
        calculate the derivatives of y wrt x with order `order`
        """
        for cur_dim, cur_order in enumerate(order):
            for _ in range(int(cur_order)):
                try:
                    grads = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
                except RuntimeError:
                    # when very high order derivatives are taken for polynomial function
                    # it has 0 gradient but torch has difficulty knowing that
                    # hence we handle such error separately
                    return torch.zeros_like(y)

                # update y
                y = grads[cur_dim]
        return y

    def adjusted_phi(self, x, T, patch):
        """
        find the suitable terminal condition based on the value of patch
        when branch_patches=1, this function always outputs self.phi_fun(x)
        """
        if patch == 0:
            return self.phi_fun(x)
        else:
            is_training = self.training
            self.eval()
            xx = torch.stack((T.reshape(-1), x.reshape(-1)), dim=-1)
            ans = self(xx, patch=patch - 1)
            if is_training:
                self.train()
            return ans

    def code_to_function(self, code, x, T, patch=0):
        """
        calculate the functional of tree based on code and x

        there are two ways of representing the code
        1. negative code of size d
                (neg_num_1, ..., neg_num_d) -> d/dx1^{-neg_num_1 - 1} ... d/dxd^{-neg_num_d - 1} phi(x1, ..., xd)
        2. positive code of size n
                (pos_num_1, ..., pos_num_n) -> d/dy1^{pos_num_1 - 1} ... d/dyd^{-pos_num_1 - 1} phi(y1, ..., yn)
                    y_i is the derivatives of phi wrt x with order self.deriv_map[i-1]

        shape of x      -> d x batch
        shape of output -> batch
        """
        x = x.detach().clone().requires_grad_(True)
        fun_val = torch.zeros_like(x[0])

        # negative code of size d
        if code[0] < 0:
            return self.nth_derivatives(
                -code - 1, self.adjusted_phi(x, T, patch), x
            ).detach()

        # positive code of size d
        if code[0] > 0:
            y = []
            for order in self.deriv_map:
                y.append(
                    self.nth_derivatives(
                        order, self.adjusted_phi(x, T, patch), x
                    ).detach()
                )
            y = torch.stack(y[: self.n]).requires_grad_()

            return self.nth_derivatives(code - 1, self.f_fun(y), y).detach()

        return fun_val

    def gen_bm(self, dt, nb_states):
        """
        generate brownian motion sqrt{dt} x Gaussian

        when self.antithetic=true, we generate
        dw = sqrt{dt} x Gaussian of size nb_states//2
        and return (dw, -dw)
        """
        dt = dt.clip(min=0.0)  # so that we can safely take square root of dt

        if self.antithetic:
            # antithetic variates
            dw = torch.sqrt(self.nu * dt) * torch.randn(
                self.dim, nb_states, self.nb_path_per_state // 2, device=self.device
            ).repeat(1, 1, 2)
            dw[:, :, : (self.nb_path_per_state // 2)] *= -1
        else:
            # usual generation
            dw = torch.sqrt(self.nu * dt) * torch.randn(
                self.dim, nb_states, self.nb_path_per_state, device=self.device
            )
        return dw

    def gen_sample_batch(self, t, T, x, mask, H, code, patch):
        """
        recursive function to calculate E[ H(t, x, code) ]

        t    -> current time
             -> shape of nb_states x nb_paths_per_state
        T    -> terminal time
             -> shape of nb_states x nb_paths_per_state
        x    -> value of brownian motion at time t
             -> shape of d x nb_states x nb_paths_per_state
        mask -> mask[idx]=1 means the state at index idx is still alive
             -> mask[idx]=0 means the state at index idx is dead
             -> shape of nb_states x nb_paths_per_state
        H    -> cummulative value of the product of functional H
             -> shape of nb_states x nb_paths_per_state
        code -> determine the operation to be taken on the functions f and phi
             -> negative code of size d or positive code of size n
        """
        # return zero tensor when no operation is needed
        ans = torch.zeros_like(t)
        if ~mask.any():
            return ans

        nb_states, _ = t.shape
        tau = Exponential(
            self.exponential_lambda
            * torch.ones(nb_states, self.nb_path_per_state, device=self.device)
        ).sample()
        dw = self.gen_bm(T - t, nb_states)
        x_is_inside = self.is_x_inside(x + dw)
        survive_prob = self.conditional_probability_to_survive(self.nu * (T - t), x, x + dw)

        ############################### for t + tau >= T
        mask_now = mask.bool() * (t + tau >= T)
        if mask_now.any():
            tmp = (
                    H[mask_now]
                    / torch.exp(-self.exponential_lambda * (T - t)[mask_now])
                    * (
                        x_is_inside[mask_now] * survive_prob[mask_now]
                            * self.code_to_function(code, (x + dw)[:, mask_now], T[mask_now], patch)
                        + (1 - x_is_inside[mask_now] * survive_prob[mask_now]) * self.phi0
                    )
            )
            ans[mask_now] = tmp

        ############################### for t + tau < T
        dw = self.gen_bm(tau, nb_states)
        x_is_inside = self.is_x_inside(x + dw)
        survive_prob = self.conditional_probability_to_survive(self.nu * (tau - t), x, x + dw)
        mask_now = mask.bool() * (t + tau < T) * x_is_inside
        H = H * survive_prob

        # return when all processes die
        if ~mask_now.any():
            return ans

        # uniform distribution to choose from the set of mechanism
        unif = torch.rand(nb_states, self.nb_path_per_state, device=self.device)

        # identity code (-1, ..., -1) of size d
        if (len(code) == self.dim) and (code == [-1] * self.dim).all():
            tmp = self.gen_sample_batch(
                t + tau,
                T,
                x + dw,
                mask_now,
                H / self.exponential_lambda / torch.exp(-self.exponential_lambda * tau),
                np.array([1] * self.n),
                patch,
            )
            ans = ans.where(~mask_now, tmp)

        # negative code of size d
        elif code[0] < 0:
            order = tuple(-code - 1)
            # if c is not in the lookup, add it
            if order not in self.fdb_lookup.keys():
                start = time.time()
                self.fdb_lookup[order] = fdb_nd(self.n, order)
                self.fdb_runtime += (time.time() - start)
            L = self.fdb_lookup[order]
            idx = (unif * len(L)).long()
            idx_counter = 0

            # loop through all fdb elements
            for fdb in L:
                mask_tmp = mask_now * (idx == idx_counter)
                A = fdb.coeff * self.gen_sample_batch(
                    t + tau,
                    T,
                    x + dw,
                    mask_tmp,
                    len(L)
                    * H
                    / self.exponential_lambda
                    / torch.exp(-self.exponential_lambda * tau),
                    np.array(fdb.lamb) + 1,
                    patch,
                )

                for ll, k_arr in fdb.l_and_k.items():
                    for q in range(self.n):
                        for _ in range(k_arr[q]):
                            A = A * self.gen_sample_batch(
                                t + tau,
                                T,
                                x + dw,
                                mask_tmp,
                                torch.ones_like(t),
                                -self.deriv_map[q] - ll - 1,
                                patch,
                            )
                ans = ans.where(~mask_tmp, A)
                idx_counter += 1

        # positive code of size n
        elif code[0] > 0:
            idx = (unif * self.mechanism_tot_len).long()
            idx_counter = 0

            # positive code part 1
            for i in range(self.n):
                for j in range(self.n):
                    for k in range(self.dim):
                        mask_tmp = mask_now * (idx == idx_counter)
                        code_increment = np.zeros_like(self.deriv_map[i])
                        code_increment[k] += 1
                        A = self.gen_sample_batch(
                            t + tau,
                            T,
                            x + dw,
                            mask_tmp,
                            torch.ones_like(t),
                            -self.deriv_map[i] - code_increment - 1,
                            patch,
                        )
                        B = self.gen_sample_batch(
                            t + tau,
                            T,
                            x + dw,
                            mask_tmp,
                            torch.ones_like(t),
                            -self.deriv_map[j] - code_increment - 1,
                            patch,
                        )
                        # only code + 1 in the dimension j and l
                        code_increment = np.zeros_like(code)
                        code_increment[i] += 1
                        code_increment[j] += 1
                        tmp = self.gen_sample_batch(
                            t + tau,
                            T,
                            x + dw,
                            mask_tmp,
                            -self.nu/2
                            * self.mechanism_tot_len
                            * A
                            * B
                            * H
                            / self.exponential_lambda
                            / torch.exp(-self.exponential_lambda * tau),
                            code + code_increment,
                            patch,
                        )
                        ans = ans.where(~mask_tmp, tmp)
                        idx_counter += 1

            # positive code part 2
            for k in range(self.n):
                for fdb in self.fdb_lookup[tuple(self.deriv_map[k])]:
                    mask_tmp = mask_now * (idx == idx_counter)
                    A = fdb.coeff * self.gen_sample_batch(
                        t + tau,
                        T,
                        x + dw,
                        mask_tmp,
                        torch.ones_like(t),
                        np.array(fdb.lamb) + 1,
                        patch,
                    )
                    for ll, k_arr in fdb.l_and_k.items():
                        for q in range(self.n):
                            for _ in range(k_arr[q]):
                                A = A * self.gen_sample_batch(
                                    t + tau,
                                    T,
                                    x + dw,
                                    mask_tmp,
                                    torch.ones_like(t),
                                    -self.deriv_map[q] - ll - 1,
                                    patch,
                                )
                    code_increment = np.zeros_like(code)
                    code_increment[k] += 1
                    tmp = self.gen_sample_batch(
                        t + tau,
                        T,
                        x + dw,
                        mask_tmp,
                        self.mechanism_tot_len
                        * A
                        * H
                        / self.exponential_lambda
                        / torch.exp(-self.exponential_lambda * tau),
                        code + code_increment,
                        patch,
                    )
                    ans = ans.where(~mask_tmp, tmp)
                    idx_counter += 1
        return ans

    def gen_sample(self, patch, t=None):
        """
        generate sample based on the (t, x) boundary and the function gen_sample_batch
        """
        if t is None:
            nb_states = self.nb_states
        else:
            nb_states, _ = t.shape
        states_per_batch = min(nb_states, self.nb_states_per_batch)
        batches = math.ceil(nb_states / states_per_batch)
        t_lo, t_hi = self.adjusted_t_boundaries[patch]
        x_lo, x_hi = self.adjusted_x_boundaries
        xx, yy = [], []
        for _ in range(batches):
            unif = (
                torch.rand(states_per_batch, device=self.device)
                .repeat(self.nb_path_per_state)
                .reshape(self.nb_path_per_state, -1)
                .T
            )
            t = t_lo + (t_hi - t_lo) * unif
            unif = (
                torch.rand(self.dim * states_per_batch, device=self.device)
                .repeat(self.nb_path_per_state)
                .reshape(self.nb_path_per_state, -1)
                .T.reshape(self.dim, states_per_batch, self.nb_path_per_state)
            )
            x = x_lo + (x_hi - x_lo) * unif
            # fix all dimensions (except the first) to be the middle value
            if self.dim > 1 and self.fix_all_dim_except_first:
                x[1:, :, :] = (x_hi + x_lo) / 2
            T = (t_lo + self.delta_t) * torch.ones_like(t)
            xx.append(torch.cat((t[:, :1], x[:, :, 0].T), dim=-1).detach())
            yy_tmp = self.gen_sample_batch(
                t,
                T,
                x,
                torch.ones_like(t),
                torch.ones_like(t),
                self.code,
                patch,
            ).detach()
            # let (lo, hi) be 
            # (self.outlier_percentile, 100 - self.outlier_percentile)
            # percentile of yy_tmp
            #
            # set the boundary as [lo-1000*(hi-lo), hi+1000*(hi-lo)]
            # samples out of this boundary is considered as outlier and removed
            lo, hi = (
                yy_tmp.nanquantile(self.outlier_percentile/100, dim=1, keepdim=True), 
                yy_tmp.nanquantile(1 - self.outlier_percentile/100, dim=1, keepdim=True)
            )
            lo, hi = lo - self.outlier_multiplier * (hi - lo), hi + self.outlier_multiplier * (hi - lo)
            mask = torch.logical_or(
                torch.logical_and(lo <= yy_tmp, yy_tmp <= hi), yy_tmp.isnan()
            )
            yy.append((yy_tmp.nan_to_num() * mask).sum(dim=1) / mask.sum(dim=1))

        return (
            torch.cat(xx),
            torch.cat(yy),
        )

    def train_and_eval(self, debug_mode=False):
        """
        generate sample and evaluate (plot) NN approximation when debug_mode=True
        """
        for p in range(self.patches):
            # initialize optimizer
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
            # lr *= .1 for every epochs // 3 steps
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=self.lr_milestones,
                gamma=self.lr_gamma,
            )

            start = time.time()
            x, y = self.gen_sample(patch=p)
            if self.verbose:
                print(
                    f"Patch {p}: generation of samples take {time.time() - start} seconds."
                )

            start = time.time()
            self.train()  # training mode

            # loop through epochs
            for epoch in range(self.epochs):
                # clear gradients and evaluate training loss
                optimizer.zero_grad()
                predict = self(x, patch=p)
                loss = self.loss(predict, y)

                # update model weights and schedule
                loss.backward()
                optimizer.step()
                scheduler.step()

                # print loss information every 500 epochs
                if epoch % 500 == 0 or epoch + 1 == self.epochs:
                    if debug_mode:
                        grid = np.linspace(self.x_lo, self.x_hi, 100)
                        x_mid = (self.x_lo + self.x_hi) / 2
                        t_lo = x[:, 0].min().item()
                        grid_nd = np.concatenate(
                            (
                                t_lo * np.ones((1, 100)),
                                np.expand_dims(grid, axis=0),
                                x_mid * np.ones((self.dim - 1, 100)),
                            ),
                            axis=0,
                        ).astype(np.float32)
                        self.eval()
                        nn = (
                            self(torch.tensor(grid_nd.T, device=self.device), patch=p)
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        f = plt.figure()
                        plt.plot(x[:, 1].detach().cpu(), y.cpu(), "+", label="Monte Carlo samples")
                        plt.plot(grid, nn, label="Neural network function")
                        plt.legend()
                        plt.show()
                        f.savefig("plot/debug.pdf", bbox_inches="tight")
                        # save points to csv
                        data = np.stack((x[:, 1].detach().cpu().numpy(), y.cpu().numpy()), axis=-1)
                        np.savetxt(
                            "log/debug_mc_samples.csv",
                            data,
                            delimiter=",",
                            header="x,y",
                            comments="",
                        )
                        data = np.stack((grid, nn), axis=-1)
                        np.savetxt(
                            "log/debug_nn.csv",
                            data,
                            delimiter=",",
                            header="x,nn",
                            comments="",
                        )
                        self.train()
                    if self.verbose:
                        print(f"Patch {p}: epoch {epoch} with loss {loss.detach()}")
            if self.verbose:
                print(
                    f"Patch {p}: training of neural network with {self.epochs} epochs take {time.time() - start} seconds."
                )
        self.eval()


if __name__ == "__main__":
    # configurations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lower_bound, upper_bound = -2, 2
    nu = 1
    y, eps = 0, 1
    a, b = y - eps, y + eps

    # function definition
    deriv_map = np.array([0]).reshape(-1, 1)
    def f_example(y):
        """
        idx 0 -> no deriv
        """
        return torch.zeros_like(y[0])

    def phi_example(x):
        return torch.logical_and(x[0] <= b, x[0] >= a).float()

    def exact_example(t, x, T, with_bound=False, k_arr=range(-5, 5)):
        if t == T:
            return np.logical_and(x[0] <= b, x[0] >= a)
        else:
            normal_std = math.sqrt(nu * (T - t))
            if not with_bound:
                # without bound
                return norm.cdf((b - x[0]) / normal_std) - norm.cdf((a - x[0]) / normal_std)
            else:
                # with bound
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
                    - torch.exp(((y - x) ** 2 - (y + x - 2 * lower_bound + 2 * k * (upper_bound - lower_bound)) ** 2) / (2 * t))
            )
        return ans.prod(dim=0)

    def is_x_inside(x):
        return torch.logical_and(lower_bound <= x, x <= upper_bound).all(dim=0)

    t_lo, x_lo, x_hi, n = 0., lower_bound, upper_bound, 0
    grid = np.linspace(x_lo, x_hi, 100)
    grid_d_dim = np.expand_dims(grid, axis=0)
    grid_d_dim_with_t = np.concatenate((t_lo * np.ones((1, 100)), grid_d_dim), axis=0)

    patches = 1
    T = patches * 1.0
    true = exact_example(t_lo, grid_d_dim, T)
    true_with_bound = exact_example(t_lo, grid_d_dim, T, with_bound=True)
    terminal = exact_example(T, grid_d_dim, T)
    model = Net(
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
    )
    model.train_and_eval(debug_mode=True)

    nn = (
        model(torch.tensor(grid_d_dim_with_t.astype(np.float32).T, device=model.device), patch=patches-1)
            .detach()
            .cpu()
            .numpy()
    )
    plt.plot(grid, nn, label="NN approximation")
    plt.plot(grid, true, label="True solution without bound")
    plt.plot(grid, true_with_bound, label="True solution with bound")
    plt.legend()
    plt.show()
