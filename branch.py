import os
import time
import math
import torch
from torch.distributions.exponential import Exponential
import matplotlib.pyplot as plt
import numpy as np
from fdb import fdb_nd
from scipy.stats import norm
import logging

torch.manual_seed(0)  # set seed for reproducibility


class Net(torch.nn.Module):
    """
    Deep branching approach to solve PDE with utility functions


    Attributes
    ----------
    problem_name : str
        Describe...

    f_fun : function
        Describe...

    phi_fun : function
        Describe...

    phi0 : float
        Describe...

    conditional_probability_to_survive : function
        Describe...

    is_x_inside : function
        Describe...

    deriv_map : numpy.ndarray
        Describe...

    n : int
        Describe...

    dim_in : int
        Describe...

    zeta_map : numpy.ndarray
        Describe...

    deriv_condition_deriv_map : numpy.ndarray
        Describe...

    deriv_condition_zeta_map : numpy.ndarray
        Describe...

    dim_out : int
        Describe...

    nprime : int
        Describe...

    exact_p_fun : function
        Describe...

    train_for_p : bool
        Describe...

    patches : int
        Describe...

    code : numpy.ndarray
        Describe...

    coordinate : numpy.ndarray
        Describe...

    fdb_lookup : dict
        Describe...

    fdb_runtime : float
        Describe...

    mechanism_tot_len : int
        Describe...

    u_layer : torch.nn.modules.container.ModuleList
        Describe...

    u_bn_layer : torch.nn.modules.container.ModuleList
        Describe...

    p_layer : torch.nn.modules.container.ModuleList
        Describe...

    p_bn_layer : torch.nn.modules.container.ModuleList
        Describe...

    lr : float
        Describe...

    lr_milestones : list
        Describe...

    lr_gamma : float
        Describe...

    weight_decay : float
        Describe...

    save_for_best_model : bool
        Describe...

    save_data : bool
        Describe...

    loss : torch.nn.modules.loss
        Describe...

    activation : torch.nn.modules.activation
        Describe...

    batch_normalization : bool
        Describe...

    nb_states : int
        Describe...

    nb_states_per_batch : int
        Describe...

    nb_path_per_state : int
        Describe...

    x_lo : float
        Describe...

    x_hi : float
        Describe...

    adjusted_x_boundaries : tuple
        Describe...

    t_lo : float
        Describe...

    t_hi : float
        Describe...

    T : float
        Describe...

    tau_lo : float
        Describe...

    tau_hi : float
        Describe...

    nu : float
        Describe...

    delta_t : float
        Describe...

    outlier_percentile : float
        Describe...

    outlier_multiplier : float
        Describe...

    exponential_lambda : float
        Describe...

    epochs : int
        Describe...

    antithetic : bool
        Describe...

    device : torch.device
        Describe...

    verbose : bool
        Describe...

    fix_all_dim_except_first : bool
        Describe...

    fix_t_dim : bool
        Describe...

    t_boundaries : torch.Tensor
        Describe...

    adjusted_t_boundaries : list
        Describe...

    working_dir : str
        Describe...


    Methods
    -------
    forward(x, patch=None, p_or_u="u")
        Describe

    log_config()
        Describe

    bisect_left(val)
        Describe

    pretty_print(tensor)
        Describe

    error_calculation(exact_u_fun, exact_p_fun, nb_pts_time=11, nb_pts_spatial=2*126+1, error_multiplier=1)
        Describe

    nth_derivatives(order, y, x)
        Describe

    adjusted_phi(x, T, patch, coordinate)
        Describe

    print_msg(msg)
        Describe

    code_to_function(code, x, T, patch, coordinate)
        Describe

    gen_bm(dt, nb_states, var=None)
        Describe

    helper_negative_code_on_f(t, T, x, mask, H, code, patch, coordinate)
        Describe

    gen_sample_batch(t, T, x, mask, H, code, patch, coordinate)
        Describe

    compare_with_exact(exact_fun)
        Describe

    log_plot_save(patch, epoch, loss, x, y, debug_mode=False, p_or_u="u")
        Describe

    gen_sample(patch, t=None)
        Describe

    calculate_p_from_u(x, patch)
        Describe

    gen_sample_for_p(patch, gen_y=True, overtrain_rate=.5)
        Describe

    train_and_eval(debug_mode=False, return_dict=False, reuse_x=None, reuse_y=None)
        Describe
    """
    def __init__(
        self,
        problem_name,
        f_fun,
        deriv_map,
        zeta_map=None,
        deriv_condition_deriv_map=None,
        deriv_condition_zeta_map=None,
        dim_out=None,
        phi_fun=(lambda x: x),
        exact_p_fun=None,
        phi0=0,
        conditional_probability_to_survive=(lambda t, x, y: torch.ones_like(x[0])),
        is_x_inside=(lambda x: torch.ones_like(x[0]).bool()),
        x_lo=-10.0,
        x_hi=10.0,
        t_lo=0.0,
        T=1.0,
        nu=1.0,
        branch_exponential_lambda=None,
        neurons=50,
        layers=6,
        branch_lr=1e-3,
        lr_milestones=(1000, 2000),
        lr_gamma=0.8,
        weight_decay=0,
        branch_nb_path_per_state=10000,
        branch_nb_states=1000,
        branch_nb_states_per_batch=100,
        epochs=5000,
        batch_normalization=True,
        antithetic=True,
        overtrain_rate=.1,
        device="cpu",
        branch_activation="softplus",
        verbose=False,
        fix_all_dim_except_first=True,
        fix_t_dim=True,
        branch_patches=1,
        outlier_percentile=1,
        outlier_multiplier=1000,
        code=None,
        coordinate=None,
        train_for_p=None,
        save_for_best_model=True,
        save_data=False,
        continue_from_checkpoint=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        problem_name : str
            Describe

        f_fun : function
            Describe

        deriv_map : numpy.ndarray
            Describe

        zeta_map : numpy.ndarray, optional
            Describe

        deriv_condition_deriv_map : numpy.ndarray, optional
            Describe

        deriv_condition_zeta_map : numpy.ndarray, optional
            Describe

        dim_out : int, optional
            Describe

        phi_fun : function, optional
            Describe

        exact_p_fun : function, optional
            Describe

        phi0 : float, optional
            Describe

        conditional_probability_to_survive : function, optional
            Describe

        is_x_inside : function, optional
            Describe

        x_lo : float, optional
            Describe

        x_hi : float, optional
            Describe

        t_lo : float, optional
            Describe

        T : float, optional
            Describe

        nu : float, optional
            Describe

        branch_exponential_lambda : float, optional
            Describe

        neurons : int, optional
            Describe

        layers : int, optional
            Describe

        branch_lr : float, optional
            Describe

        lr_milestones : tuple, optional
            Describe

        lr_gamma : float, optional
            Describe

        weight_decay : float, optional
            Describe

        branch_nb_path_per_state : int, optional
            Describe

        branch_nb_states : int, optional
            Describe

        branch_nb_states_per_batch : int, optional
            Describe

        epochs : int, optional
            Describe

        batch_normalization : bool, optional
            Describe

        antithetic : bool, optional
            Describe

        overtrain_rate : float, optional
            Describe

        device : str, optional
            Describe

        branch_activation : str, optional
            Describe

        verbose : bool, optional
            Describe

        fix_all_dim_except_first : bool, optional
            Describe

        fix_t_dim : bool, optional
            Describe

        branch_patches : int, optional
            Describe

        outlier_percentile : float, optional
            Describe

        outlier_multiplier : float, optional
            Describe

        code : numpy.ndarray, optional
            Describe

        coordinate : numpy.ndarray, optional
            Describe

        train_for_p : bool, optional
            Describe

        save_for_best_model : bool, optional
            Describe

        save_data : bool, optional
            Describe

        continue_from_checkpoint : str, optional
            Describe
        """
        super(Net, self).__init__()
        self.problem_name = problem_name
        self.f_fun = f_fun
        self.phi_fun = phi_fun
        self.phi0 = phi0
        self.conditional_probability_to_survive = conditional_probability_to_survive
        self.is_x_inside = is_x_inside
        self.deriv_map = deriv_map
        self.n, self.dim_in = deriv_map.shape
        self.zeta_map = zeta_map if zeta_map is not None else np.zeros(self.n, dtype=int)
        self.deriv_condition_deriv_map = deriv_condition_deriv_map
        self.deriv_condition_zeta_map = deriv_condition_zeta_map
        self.dim_out = dim_out if dim_out is not None else self.zeta_map.max() + 1
        self.nprime = sum(self.zeta_map == -1)
        self.exact_p_fun = exact_p_fun
        self.train_for_p = train_for_p if train_for_p is not None else (self.nprime > 0) and self.exact_p_fun is None

        # patching is used for calculating the target expected value of the tree in branch_patches steps
        #
        # for example, when t_lo=t_hi=0 and branch_patches=2
        # the algorithm calculates the function u(T/2, x) with terminal condition phi
        # then, the algorithm calculates the function u(0, x) with terminal condition of u(T/2, x)
        #
        # such approach relies on precise approximation of u(T/2, x)
        # which is very time-consuming in high dimensional case
        self.patches = branch_patches
        self.code = -np.ones((self.dim_out, self.dim_in), dtype=int) if code is None else np.array(code)
        self.coordinate = np.array(range(self.dim_out)) if coordinate is None else np.array(coordinate)

        # store the (faa di bruno) fdb results for quicker lookup
        start = time.time()
        self.fdb_lookup = {
            tuple(deriv): fdb_nd(self.n, tuple(deriv)) for deriv in deriv_map
        }
        self.fdb_runtime = time.time() - start
        self.mechanism_tot_len = self.dim_in * self.n ** 2 + sum(
            [len(self.fdb_lookup[tuple(deriv)]) for deriv in deriv_map]
        )

        # NN function for u
        self.u_layer = torch.nn.ModuleList(
            [
                torch.nn.ModuleList(
                    [torch.nn.Linear(self.dim_in + 1, neurons, device=device)]
                    + [
                        torch.nn.Linear(neurons, neurons, device=device)
                        for _ in range(layers)
                    ]
                    + [torch.nn.Linear(neurons, self.dim_out, device=device)]
                )
                for _ in range(branch_patches)
            ]
        )
        # set affine=False and higher eps because it may be the case that t is the constant t_lo
        self.u_bn_layer = torch.nn.ModuleList(
            [
                torch.nn.ModuleList(
                    [torch.nn.BatchNorm1d(self.dim_in + 1, eps=5e-1, affine=False, device=device)]
                    + [
                        torch.nn.BatchNorm1d(neurons, device=device)
                        for _ in range(layers + 1)
                    ]
                )
                for _ in range(branch_patches)
            ]
        )

        # NN function for p
        self.p_layer = torch.nn.ModuleList(
            [
                torch.nn.ModuleList(
                    [torch.nn.Linear(self.dim_in, neurons, device=device)]
                    + [
                        torch.nn.Linear(neurons, neurons, device=device)
                        for _ in range(layers)
                    ]
                    + [torch.nn.Linear(neurons, 1, device=device)]
                )
                for _ in range(branch_patches)
            ]
        )
        # set affine=False and higher eps because it may be the case that t is the constant t_lo
        self.p_bn_layer = torch.nn.ModuleList(
            [
                torch.nn.ModuleList(
                    [torch.nn.BatchNorm1d(self.dim_in, eps=5e-1, affine=False, device=device)]
                    + [
                        torch.nn.BatchNorm1d(neurons, device=device)
                        for _ in range(layers + 1)
                    ]
                )
                for _ in range(branch_patches)
            ]
        )

        self.lr = branch_lr
        self.lr_milestones = list(lr_milestones)
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay
        # only save for the best model if we run epochs for long enough
        self.save_for_best_model = save_for_best_model and epochs > self.lr_milestones[-1]

        self.save_data = save_data

        self.loss = torch.nn.MSELoss()
        self.activation = {
            "tanh": torch.nn.Tanh(),
            "relu": torch.nn.ReLU(),
            "softplus": torch.nn.Softplus(),
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
        self.t_hi = T
        self.T = T
        self.tau_lo, self.tau_hi = 1e-5, 10  # for negative coordinate
        self.nu = nu
        self.delta_t = (T - t_lo) / branch_patches
        self.outlier_percentile = outlier_percentile
        self.outlier_multiplier = outlier_multiplier

        self.exponential_lambda = (
            branch_exponential_lambda if branch_exponential_lambda is not None
            else -math.log(.95)/self.delta_t
        )
        self.epochs = epochs
        self.antithetic = antithetic
        self.device = device
        self.verbose = verbose
        self.fix_all_dim_except_first = fix_all_dim_except_first
        self.fix_t_dim = fix_t_dim
        self.t_boundaries = torch.tensor(
            ([t_lo + i * self.delta_t for i in range(branch_patches)] + [T])[::-1],
            device=device,
        )
        if fix_t_dim:
            self.adjusted_t_boundaries = [
                (lo, lo) for lo, hi in zip(self.t_boundaries[1:], self.t_boundaries[:-1])
            ]
        else:
            self.adjusted_t_boundaries = [
                (lo, hi) for lo, hi in zip(self.t_boundaries[1:], self.t_boundaries[:-1])
            ]
        if continue_from_checkpoint is not None:
            self.load_state_dict(torch.load(f"{continue_from_checkpoint}/checkpoint.pt"))
            self.train_for_p = False
            self.eval()

        timestr = time.strftime("%Y%m%d-%H%M%S")  # current time stamp
        self.working_dir = f"logs/{timestr}-{problem_name}-T{self.T}-nu{self.nu}"
        self.log_config()

    def forward(self, x, patch=None, p_or_u="u"):
        """
        Describe

        Parameters
        ----------
        x : torch.Tensor
            Describe

        patch : int, optional
            Describe

        p_or_u : str, optional
            Describe
        """
        if self.exact_p_fun is not None and p_or_u == "p":
            return self.exact_p_fun(x.T)

        layer = self.u_layer if p_or_u == "u" else self.p_layer
        bn_layer = self.u_bn_layer if p_or_u == "u" else self.p_bn_layer

        if patch is not None:
            if self.batch_normalization:
                x = bn_layer[patch][0](x)
            y = x
            for idx, (f, bn) in enumerate(
                zip(layer[patch][:-1], bn_layer[patch][1:])
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

            y = layer[patch][-1](y)
        else:
            yy = []
            for p in range(self.patches):
                if self.batch_normalization:
                    x = bn_layer[p][0](x)
                y = x
                for idx, (f, bn) in enumerate(
                    zip(layer[p][:-1], bn_layer[p][1:])
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
                yy.append(layer[p][-1](y))
            idx = self.bisect_left(x[:, 0])
            y = torch.gather(torch.stack(yy, dim=-1), -1, idx.reshape(-1, 1)).squeeze(
                -1
            )
        return y

    def log_config(self):
        """
        Set up configuration for log files and mkdir
        """
        os.makedirs(self.working_dir)
        os.mkdir(f"{self.working_dir}/plot")
        os.mkdir(f"{self.working_dir}/data")
        formatter = "%(asctime)s | %(name)s |  %(levelname)s: %(message)s"
        logging.getLogger().handlers = []  # clear previous loggers if any
        logging.basicConfig(
            filename=f"{self.working_dir}/run.log",
            filemode="w",
            level=logging.DEBUG,
            format=formatter,
        )
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logging.getLogger().addHandler(console)
        logging.debug(f"Current configuration: {self.__dict__}")

    def bisect_left(self, val):
        """
        Find the index of val based on the discretization of self.t_boundaries
        it is only used when branch_patches > 1

        Parameters
        ----------
        val : torch.Tensor
            Describe
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
    def pretty_print(tensor):
        """
        Describe


        Parameters
        ----------
        tensor : torch.Tensor
            Describe
        """
        mess = ""
        for i in tensor[:-1]:
            mess += f"& {i.item():.2E} "
        mess += "& --- \\\\"
        logging.info(mess)

    def error_calculation(self, exact_u_fun, exact_p_fun, nb_pts_time=11, nb_pts_spatial=2*126+1, error_multiplier=1):
        """
        Describe

        Parameters
        ----------
        exact_u_fun : function
            Describe

        exact_p_fun : function
            Describe

        nb_pts_time : int, optional
            Describe

        nb_pts_spatial : int, optional
            Describe

        error_multiplier : float, optional
            Describe
        """
        x = np.linspace(self.x_lo, self.x_hi, nb_pts_spatial)
        t = np.linspace(self.t_lo, self.t_hi, nb_pts_time)
        arr = np.array(np.meshgrid(*([x]*self.dim_in + [t]))).T.reshape(-1, self.dim_in + 1)
        arr[:, [-1, 0]] = arr[:, [0, -1]]
        arr = torch.tensor(arr, device=self.device, dtype=torch.get_default_dtype())
        error = []
        nn = []
        cur, batch_size, last = 0, 100000, arr.shape[0]
        while cur < last:
            nn.append(self(arr[cur:min(cur+batch_size, last)], patch=0).detach())
            cur += batch_size
        nn = torch.cat(nn, dim=0)

        # Lejay
        logging.info("The error as in Lejay is calculated as follows.")
        overall_error = 0
        for i in range(self.dim_in):
            error.append(error_multiplier * (nn[:, i] - exact_u_fun(arr.T, i)).reshape(nb_pts_time, -1) ** 2)
            overall_error += (error[-1])
        error.append(overall_error)
        for i in range(self.dim_in):
            logging.info(f"$\\hat{{e}}_{i}(t_k)$")
            self.pretty_print(error[i].max(dim=1)[0])
        logging.info("$\\hat{e}(t_k)$")
        self.pretty_print(error[-1].max(dim=1)[0])
        logging.info("\\hline")

        # erru
        logging.info("\nThe relative L2 error of u (erru) is calculated as follows.")
        denominator, numerator = 0, 0
        for i in range(self.dim_in):
            denominator += exact_u_fun(arr.T, i).reshape(nb_pts_time, -1) ** 2
            numerator += (nn[:, i] - exact_u_fun(arr.T, i)).reshape(nb_pts_time, -1) ** 2
        logging.info("erru($t_k$)")
        self.pretty_print((numerator.mean(dim=-1)/denominator.mean(dim=-1)).sqrt())

        del nn
        torch.cuda.empty_cache()
        grad = []
        cur, batch_size, last = 0, 100000, arr.shape[0]
        while cur < last:
            xx = arr[cur:min(cur+batch_size, last)].detach().clone().requires_grad_(True)
            tmp = []
            for i in range(self.dim_in):
                tmp.append(
                    torch.autograd.grad(
                        self(xx, patch=0)[:, i].sum(),
                        xx,
                    )[0][:, 1:].detach()
                )
            grad.append(torch.stack(tmp, dim=-1))
            cur += batch_size
        grad = torch.cat(grad, dim=0)

        # errgu
        logging.info("\nThe relative L2 error of gradient of u (errgu) is calculated as follows.")
        denominator, numerator = 0, 0
        xx = arr.detach().clone().requires_grad_(True)
        for i in range(self.dim_in):
            exact = torch.autograd.grad(
                    exact_u_fun(xx.T, i).sum(),
                    xx,
            )[0][:, 1:]
            denominator += exact.reshape(nb_pts_time, -1, self.dim_in) ** 2
            numerator += (exact - grad[:, :, i]).reshape(nb_pts_time, -1, self.dim_in) ** 2
        logging.info("errgu($t_k$)")
        self.pretty_print((numerator.mean(dim=(1, 2))/denominator.mean(dim=(1, 2))).sqrt())

        # errdivu
        logging.info("\nThe absolute divergence of u (errdivu) is calculated as follows.")
        numerator = 0
        for i in range(self.dim_in):
            numerator += (grad[:, i, i]).reshape(nb_pts_time, -1)
        numerator = numerator**2
        logging.info("errdivu($t_k$)")
        self.pretty_print(
                ((self.x_hi - self.x_lo)**self.dim_in * numerator.mean(dim=-1)).sqrt()
        )

        del grad, xx
        torch.cuda.empty_cache()
        arr = arr.reshape(nb_pts_time, -1, self.dim_in + 1)[-1].detach()
        nn = []
        cur, batch_size, last = 0, 100000, arr.shape[0]
        while cur < last:
            nn.append(
                self(
                    arr[cur:min(cur+batch_size, last), 1:],
                    patch=0,
                    p_or_u="p",
                ).squeeze().detach()
            )
            cur += batch_size
        nn = torch.cat(nn, dim=0)

        # errp
        logging.info("\nThe relative L2 error of p (errp) is calculated as follows.")
        denominator = (exact_p_fun(arr.T) - exact_p_fun(arr.T).mean()) ** 2
        numerator = (
                nn - nn.mean() - exact_p_fun(arr.T) + exact_p_fun(arr.T).mean()
        ) ** 2
        logging.info("errp($t_k$)")
        logging.info(
                "& --- " * (nb_pts_time - 1)
                + f"& {(numerator.mean()/denominator.mean()).sqrt().item():.2E} \\\\"
        )

    @staticmethod
    def nth_derivatives(order, y, x):
        """
        Calculate the derivatives of y wrt x with order `order`

        Parameters
        ----------
        order : numpy.ndarray
            Describe

        y : torch.Tensor
            Describe

        x : torch.Tensor
            Describe
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

    def adjusted_phi(self, x, T, patch, coordinate):
        """
        Find the suitable terminal condition based on the value of patch
        when branch_patches=1, this function always outputs self.phi_fun(x)

        Parameters
        ----------
        x : torch.Tensor
            Describe

        T : torch.Tensor
            Describe

        patch : int
            Describe

        coordinate : numpy.ndarray
            Describe
        """
        if patch == 0:
            return self.phi_fun(x, coordinate)
        else:
            is_training = self.training
            self.eval()
            xx = torch.stack((T.reshape(-1), x.reshape(-1)), dim=-1)
            ans = self(xx, patch=patch - 1)[:, coordinate]
            if is_training:
                self.train()
            return ans

    def print_msg(self, msg):
        """
        Describe

        Parameters
        ----------
        msg : str
            Describe
        """
        if self.verbose:
            logging.info(msg)
        else:
            logging.debug(msg)

    def code_to_function(self, code, x, T, patch, coordinate):
        """
        Calculate the functional of tree based on code and x

        There are two ways of representing the code
        1. negative code of size d
            (neg_num_1, ..., neg_num_d) -> d/dx1^{-neg_num_1 - 1} ... d/dxd^{-neg_num_d - 1} phi(x1, ..., xd)
        2. positive code of size n
            (pos_num_1, ..., pos_num_n) -> d/dy1^{pos_num_1 - 1} ... d/dyd^{-pos_num_1 - 1} phi(y1, ..., yn)
                y_i is the derivatives of phi wrt x with order self.deriv_map[i-1]

        Parameters
        ----------
        code : numpy.ndarray
            Describe

        x : torch.Tensor
            Describe

        T : torch.Tensor
            Describe

        patch : int, optional
            Describe

        coordinate : numpy.ndarray
            Describe

        """
        x = x.detach().clone().requires_grad_(True)
        fun_val = torch.zeros_like(x[0])

        # negative code of size d
        if code[0] < 0:
            return self.nth_derivatives(
                -code - 1, self.adjusted_phi(x, T, patch, coordinate), x
            ).detach()

        # positive code of size d
        if code[0] > 0:
            y = []
            for idx, order in enumerate(self.deriv_map):
                if self.zeta_map[idx] < 0:
                    y.append(
                        self.nth_derivatives(
                            order, self(x.T, p_or_u="p", patch=patch).squeeze(dim=-1), x
                        )
                    )
                else:
                    y.append(
                        self.nth_derivatives(
                            order, self.adjusted_phi(x, T, patch, self.zeta_map[idx]), x
                        ).detach()
                    )
            y = torch.stack(y[: self.n]).requires_grad_()

            return self.nth_derivatives(
                code - 1, self.f_fun(y, coordinate), y
            ).detach()

        return fun_val.detach()

    def gen_bm(self, dt, nb_states, var=None):
        """
        Generate brownian motion sqrt{dt} x Gaussian

        When self.antithetic=true, we generate
        dw = sqrt{dt} x Gaussian of size nb_states//2
        and return (dw, -dw)

        Parameters
        ----------
        dt : torch.Tensor
            Describe

        nb_states : int
            Describe

        var : float, optional
            Describe
        """
        var = self.nu if var is None else var
        dt = dt.clip(min=0.0)  # so that we can safely take square root of dt

        if self.antithetic:
            # antithetic variates
            dw = torch.sqrt(var * dt) * torch.randn(
                self.dim_in, nb_states, self.nb_path_per_state // 2, device=self.device
            ).repeat(1, 1, 2)
            dw[:, :, : (self.nb_path_per_state // 2)] *= -1
        else:
            # usual generation
            dw = torch.sqrt(var * dt) * torch.randn(
                self.dim_in, nb_states, self.nb_path_per_state, device=self.device
            )
        return dw

    def helper_negative_code_on_f(self, t, T, x, mask, H, code, patch, coordinate):
        """
        Describe


        Parameters
        ----------
        t : torch.Tensor
            Describe

        T : torch.Tensor
            Describe

        x : torch.Tensor
            Describe

        mask : torch.Tensor
            Describe

        H : torch.Tensor
            Describe

        code : numpy.ndarray
            Describe

        patch : int, optional
            Describe

        coordinate : numpy.ndarray
            Describe
        """
        ans = torch.zeros_like(t)
        order = tuple(-code - 1)
        # if c is not in the lookup, add it
        if order not in self.fdb_lookup.keys():
            start = time.time()
            self.fdb_lookup[order] = fdb_nd(self.n, order)
            self.fdb_runtime += time.time() - start
        L = self.fdb_lookup[order]
        unif = torch.rand(t.shape[0], self.nb_path_per_state, device=self.device)
        idx = (unif * len(L)).long()
        idx_counter = 0
        for fdb in self.fdb_lookup[order]:
            mask_tmp = mask * (idx == idx_counter)
            if mask_tmp.any():
                A = self.gen_sample_batch(
                    t,
                    T,
                    x,
                    mask_tmp,
                    fdb.coeff * len(L) * H,
                    np.array(fdb.lamb) + 1,
                    patch,
                    coordinate,
                    )
                for ll, k_arr in fdb.l_and_k.items():
                    for q in range(self.n):
                        for _ in range(k_arr[q]):
                            A = A * self.gen_sample_batch(
                                t,
                                T,
                                x,
                                mask_tmp,
                                torch.ones_like(t),
                                -self.deriv_map[q] - ll - 1,
                                patch,
                                self.zeta_map[q],
                                )
                ans = ans.where(~mask_tmp, A)
            idx_counter += 1
        return ans

    def gen_sample_batch(self, t, T, x, mask, H, code, patch, coordinate):
        """
        Recursive function to calculate E[ H(t, x, code) ]

        Parameters
        ----------
        t : torch.Tensor
            Current time

        T : torch.Tensor
            Terminal time

        x : torch.Tensor
            Value of brownian motion at time t

        mask : torch.Tensor
            mask[idx]=1 means the state at index idx is still alive
            mask[idx]=0 means the state at index idx is dead

        H : torch.Tensor
            Cummulative value of the product of functional H

        code : numpy.ndarray
            Determine the operation to be taken on the functions f and phi

        patch : int
            Describe

        coordinate : numpy.ndarray
            Describe
        """
        # return zero tensor when no operation is needed
        ans = torch.zeros_like(t)
        if ~mask.any():
            return ans

        nb_states, _ = t.shape
        # for the p coordinate
        if coordinate < 0:
            unif = (
                torch.rand(nb_states * self.nb_path_per_state, device=self.device)
                     .reshape(nb_states, self.nb_path_per_state)
            )
            tau = self.tau_lo + (self.tau_hi - self.tau_lo) * unif
            dw = self.gen_bm(tau, nb_states, var=1)
            unif = torch.rand(nb_states, self.nb_path_per_state, device=self.device)
            order = -code - 1
            L = [fdb for fdb in fdb_nd(2, order) if max(fdb.lamb) < 2]
            idx_counter = 0
            idx = (unif * len(L) * self.dim_in ** 2).long()
            if coordinate == -2:
                idx *= (self.dim_in + 2)
            for i in range(self.dim_in):
                for j in range(self.dim_in):
                    for fdb in L:
                        if coordinate == -1:
                            # coordinate -1 -> apply code to p
                            mask_tmp = mask.bool() * (idx == idx_counter)
                            if mask_tmp.any():
                                A = (
                                        H * fdb.coeff
                                        * len(L) * self.dim_in ** 2
                                        * self.dim_in ** 2
                                        * (dw ** 2).sum(dim=0)
                                        * (self.tau_hi - self.tau_lo)
                                        / (2 * tau)
                                )
                                if self.dim_in > 2:
                                    A = A / (self.dim_in - 2)
                                elif self.dim_in == 2:
                                    A = -A * torch.log((dw ** 2).sum(dim=0).sqrt())
                                code_increment = np.zeros_like(code)
                                code_increment[j] += 1
                                if fdb.lamb[0] == 0:
                                    A = A * self.gen_sample_batch(
                                        t,
                                        T,
                                        x + dw,
                                        mask_tmp,
                                        torch.ones_like(t),
                                        -code_increment - 1,
                                        patch,
                                        i,
                                    )
                                code_increment[j] -= 1
                                code_increment[i] += 1
                                if fdb.lamb[1] == 0:
                                    A = A * self.gen_sample_batch(
                                        t,
                                        T,
                                        x + dw,
                                        mask_tmp,
                                        torch.ones_like(t),
                                        -code_increment - 1,
                                        patch,
                                        j,
                                    )

                                for ll, k_arr in fdb.l_and_k.items():
                                    for _ in range(k_arr[1]):
                                        A = A * self.gen_sample_batch(
                                            t,
                                            T,
                                            x + dw,
                                            mask_tmp,
                                            torch.ones_like(t),
                                            -code_increment - ll - 1,
                                            patch,
                                            j,
                                        )
                                    code_increment[i] -= 1
                                    code_increment[j] += 1
                                    for _ in range(k_arr[0]):
                                        A = A * self.gen_sample_batch(
                                            t,
                                            T,
                                            x + dw,
                                            mask_tmp,
                                            torch.ones_like(t),
                                            -code_increment - ll - 1,
                                            patch,
                                            i,
                                        )
                                ans = ans.where(~mask_tmp, A)
                            idx_counter += 1

                        elif coordinate == -2:
                            # coordinate -2 -> apply code to \partial_t p + nu * \Delta p
                            for k in range(self.dim_in + 2):
                                mask_tmp = mask.bool() * (idx == idx_counter)
                                if mask_tmp.any():
                                    A = (
                                            H * fdb.coeff
                                            * len(L) * self.dim_in ** 2 * (self.dim_in + 2)
                                            * self.dim_in ** 2
                                            * (dw ** 2).sum(dim=0)
                                            * (self.tau_hi - self.tau_lo)
                                            / (2 * tau)
                                    )
                                    if self.dim_in > 2:
                                        A = A / (self.dim_in - 2)
                                    elif self.dim_in == 2:
                                        A = -A * torch.log((dw ** 2).sum(dim=0).sqrt())
                                    code_increment = np.zeros_like(code)
                                    if k < self.dim_in:
                                        A = self.nu * A
                                        code_increment[k] += 1
                                    elif k == self.dim_in + 1:
                                        # the only difference between the last two k is the indexing of i, j
                                        i, j = j, i
                                    code_increment[j] += 1
                                    if fdb.lamb[0] == 0:
                                        A = A * self.gen_sample_batch(
                                            t,
                                            T,
                                            x + dw,
                                            mask_tmp,
                                            torch.ones_like(t),
                                            -code_increment - 1,
                                            patch,
                                            i,
                                        )
                                    code_increment[j] -= 1
                                    code_increment[i] += 1
                                    if fdb.lamb[1] == 0:
                                        if k < self.dim_in:
                                            A = A * self.gen_sample_batch(
                                                t,
                                                T,
                                                x + dw,
                                                mask_tmp,
                                                torch.ones_like(t),
                                                -code_increment - 1,
                                                patch,
                                                j,
                                            )
                                        else:
                                            A = A * self.helper_negative_code_on_f(
                                                t,
                                                T,
                                                x + dw,
                                                mask_tmp,
                                                -torch.ones_like(t),
                                                -code_increment - 1,
                                                patch,
                                                j,
                                            )

                                    for ll, k_arr in fdb.l_and_k.items():
                                        if k < self.dim_in:
                                            for _ in range(k_arr[1]):
                                                A = A * self.gen_sample_batch(
                                                    t,
                                                    T,
                                                    x + dw,
                                                    mask_tmp,
                                                    torch.ones_like(t),
                                                    -code_increment - ll - 1,
                                                    patch,
                                                    j,
                                                )
                                        else:
                                            for _ in range(k_arr[1]):
                                                A = A * self.helper_negative_code_on_f(
                                                    t,
                                                    T,
                                                    x + dw,
                                                    mask_tmp,
                                                    -torch.ones_like(t),
                                                    -code_increment - ll - 1,
                                                    patch,
                                                    j,
                                                )
                                        code_increment[i] -= 1
                                        code_increment[j] += 1
                                        for _ in range(k_arr[0]):
                                            A = A * self.gen_sample_batch(
                                                t,
                                                T,
                                                x + dw,
                                                mask_tmp,
                                                torch.ones_like(t),
                                                -code_increment - ll - 1,
                                                patch,
                                                i,
                                            )
                                    ans = ans.where(~mask_tmp, A)
                                idx_counter += 1
            return ans

        tau = Exponential(
            self.exponential_lambda
            * torch.ones(nb_states, self.nb_path_per_state, device=self.device)
        ).sample()
        dw = self.gen_bm(T - t, nb_states)
        x_is_inside = self.is_x_inside(x + dw)
        survive_prob = self.conditional_probability_to_survive(self.nu * (T - t), x, x + dw).clip(0, 1)

        ############################### for t + tau >= T
        mask_now = mask.bool() * (t + tau >= T)
        if mask_now.any():
            tmp = (
                    H[mask_now]
                    / torch.exp(-self.exponential_lambda * (T - t)[mask_now])
                    * (
                        x_is_inside[mask_now] * survive_prob[mask_now]
                            * self.code_to_function(code, (x + dw)[:, mask_now], T[mask_now], patch, coordinate)
                        + (1 - x_is_inside[mask_now] * survive_prob[mask_now]) * self.phi0
                    )
            )
            ans[mask_now] = tmp

        ############################### for t + tau < T
        dw = self.gen_bm(tau, nb_states)
        x_is_inside = self.is_x_inside(x + dw)
        survive_prob = self.conditional_probability_to_survive(self.nu * tau, x, x + dw).clip(0, 1)
        mask_now = mask.bool() * (t + tau < T) * x_is_inside
        H = H * survive_prob

        # return when all processes die
        if ~mask_now.any():
            return ans

        # uniform distribution to choose from the set of mechanism
        unif = torch.rand(nb_states, self.nb_path_per_state, device=self.device)

        # identity code (-1, ..., -1) of size d
        if (len(code) == self.dim_in) and (code == [-1] * self.dim_in).all():
            tmp = self.gen_sample_batch(
                t + tau,
                T,
                x + dw,
                mask_now,
                H / self.exponential_lambda / torch.exp(-self.exponential_lambda * tau),
                np.array([1] * self.n),
                patch,
                coordinate,
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
                if mask_tmp.any():
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
                        coordinate,
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
                                    self.zeta_map[q],
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
                    for k in range(self.dim_in):
                        mask_tmp = mask_now * (idx == idx_counter)
                        if mask_tmp.any():
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
                                self.zeta_map[i],
                            )
                            B = self.gen_sample_batch(
                                t + tau,
                                T,
                                x + dw,
                                mask_tmp,
                                torch.ones_like(t),
                                -self.deriv_map[j] - code_increment - 1,
                                patch,
                                self.zeta_map[i],
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
                                coordinate,
                            )
                            ans = ans.where(~mask_tmp, tmp)
                        idx_counter += 1

            # positive code part 2
            for k in range(self.nprime, self.n):
                for fdb in self.fdb_lookup[tuple(self.deriv_map[k])]:
                    mask_tmp = mask_now * (idx == idx_counter)
                    if mask_tmp.any():
                        A = fdb.coeff * self.gen_sample_batch(
                            t + tau,
                            T,
                            x + dw,
                            mask_tmp,
                            torch.ones_like(t),
                            np.array(fdb.lamb) + 1,
                            patch,
                            self.zeta_map[k],
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
                                        self.zeta_map[q],
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
                            coordinate,
                        )
                        ans = ans.where(~mask_tmp, tmp)
                    idx_counter += 1

            # positive code part 3
            for k in range(self.nprime):
                mask_tmp = mask_now * (idx == idx_counter)
                if mask_tmp.any():
                    code_increment = np.zeros_like(code)
                    code_increment[k] += 1
                    A = -self.gen_sample_batch(
                        t + tau,
                        T,
                        x + dw,
                        mask_tmp,
                        self.mechanism_tot_len
                        * H
                        / self.exponential_lambda
                        / torch.exp(-self.exponential_lambda * tau),
                        code + code_increment,
                        patch,
                        coordinate,
                        )
                    A = A * self.gen_sample_batch(
                        t + tau,
                        T,
                        x + dw,
                        mask_tmp,
                        torch.ones_like(t),
                        -self.deriv_map[k] - 1,
                        patch,
                        -2,
                        )
                    ans = ans.where(~mask_tmp, A)
                idx_counter += 1

        return ans

    def compare_with_exact(self, exact_fun):
        """
        Describe

        Parameters
        ----------
        exact_fun : function
            Describe
        """
        nb_points = 100
        grid = np.linspace(self.x_lo, self.x_hi, nb_points)
        x_mid = (self.x_lo + self.x_hi) / 2
        grid_d_dim = np.concatenate((
            np.expand_dims(grid, axis=0),
            x_mid * np.ones((self.dim_in - 1, nb_points))
        ), axis=0)
        grid_d_dim_with_t = np.concatenate((self.t_lo * np.ones((1, nb_points)), grid_d_dim), axis=0)

        nn = (
            self(torch.tensor(
                grid_d_dim_with_t.T, device=self.device, dtype=torch.get_default_dtype()
            ), patch=self.patches-1)
            .detach()
            .cpu()
            .numpy()
        )
        for i in range(self.dim_out):
            f = plt.figure()
            true = exact_fun(self.t_lo, grid_d_dim, self.T, i)
            terminal = exact_fun(self.T, grid_d_dim, self.T, i)
            plt.plot(grid, nn[:, i], label="NN")
            plt.plot(grid, true, label="True solution")
            plt.plot(grid, terminal, label="Terminal solution")
            plt.legend()
            f.savefig(
                f"{self.working_dir}/plot/u{i}_comparison_with_exact.png", bbox_inches="tight"
            )
            plt.show()
            plt.close()

            data = np.stack((grid, true, terminal, nn[:, i])).T
            np.savetxt(
                f"{self.working_dir}/data/plt_{self.problem_name}_dim_{self.dim_in}.csv",
                data,
                delimiter=",",
                header="x,true,terminal,branch",
                comments=""
            )

    def log_plot_save(self, patch, epoch, loss, x, y, debug_mode=False, p_or_u="u"):
        """
        Describe


        Parameters
        ----------
        patch : int
            Describe

        epoch : int
            Describe

        loss : torch.Tensor
            Describe

        x : torch.Tensor
            Describe

        y : torch.Tensor
            Describe

        debug_mode : bool, optional
            Describe

        p_or_u : str, optional
            Describe
        """
        # loss info
        self.print_msg(f"Patch {patch:2.0f}: epoch {epoch:4.0f} with loss {loss.detach():.2E}")

        # if we do not always save for the best model, save it every 500 epochs
        if not self.save_for_best_model:
            torch.save(
                self.state_dict(), f"{self.working_dir}/checkpoint.pt"
            )

        first_state_idx = 1 if p_or_u == "u" else 0
        x_lo, x_hi = x.min(dim=0)[0][first_state_idx].item(), x.max(dim=0)[0][first_state_idx].item()
        grid = np.linspace(x_lo, x_hi, 100)
        if self.fix_all_dim_except_first:
            x_mid = x[0, min(first_state_idx + 1, self.dim_in)].item()
        else:
            x_mid = (self.x_lo + self.x_hi) / 2
        t_lo = x[:, 0].min().item()
        grid_nd = np.concatenate(
            (
                t_lo * np.ones((1, 100)),
                np.expand_dims(grid, axis=0),
                x_mid * np.ones((self.dim_in - 1, 100)),
            ),
            axis=0,
        )
        nn = (
            self(
                torch.tensor(
                    grid_nd.T if p_or_u == "u" else grid_nd[1:].T,
                    device=self.device,
                    dtype=torch.get_default_dtype()
                ), patch=patch, p_or_u=p_or_u,
            ).detach().cpu()
        )

        # plots only in 1d or fix_all_dim_except_first
        if (self.fix_t_dim or p_or_u == "p") and (self.dim_in == 1 or self.fix_all_dim_except_first):
            loop_range = range(self.dim_out) if p_or_u == "u" else range(1)
            for i in loop_range:
                f = plt.figure()
                plt.plot(x[:, first_state_idx].detach().cpu(), y[:, i].cpu(), "+", label="MC samples")
                plt.plot(grid, nn[:, i], label="NN")
                plt.title(f"Epoch {epoch:4.0f} and patch {patch:2.0f}")
                plt.legend()
                f.savefig(
                    f"{self.working_dir}/plot/{p_or_u}{i}_patch_{patch:02}_epoch_{epoch:04}.png", bbox_inches="tight"
                )
                if debug_mode:
                    plt.show()
                plt.close()

        # save data to csv
        if self.save_data:
            header = (
                    ("t," if p_or_u == "u" else "")
                    + "".join([f"x{i}," for i in range(self.dim_in)])
                    + "".join([f"y{i}," for i in range(self.dim_out)])
            )[:-1]
            if epoch == 0:
                data = np.concatenate((x.detach().cpu().numpy(), y.cpu().numpy()), axis=-1)
                np.savetxt(
                    f"{self.working_dir}/data/mc_samples_{p_or_u}_patch_{patch:02}.csv",
                    data,
                    delimiter=",",
                    header=header,
                    comments="",
                )
            data = np.concatenate((grid_nd.T, nn), axis=-1)
            np.savetxt(
                f"{self.working_dir}/data/nn_patch_{p_or_u}_{patch:02}_epoch_{epoch:04}.csv",
                data,
                delimiter=",",
                header=header,
                comments="",
            )

    def gen_sample(self, patch):
        """
        Generate sample based on the (t, x) boundary and the function gen_sample_batch

        Parameters
        ----------
        patch : int
            Describe
        """
        nb_states = self.nb_states
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
                torch.rand(self.dim_in * states_per_batch, device=self.device)
                .repeat(self.nb_path_per_state)
                .reshape(self.nb_path_per_state, -1)
                .T.reshape(self.dim_in, states_per_batch, self.nb_path_per_state)
            )
            x = x_lo + (x_hi - x_lo) * unif
            # fix all dimensions (except the first) to be the middle value
            if self.dim_in > 1 and self.fix_all_dim_except_first:
                x[1:, :, :] = (x_hi + x_lo) / 2
            T = (t_lo + self.delta_t) * torch.ones_like(t)
            xx.append(torch.cat((t[:, :1], x[:, :, 0].T), dim=-1).detach())
            yyy = []
            for (idx, c) in zip(self.coordinate, self.code):
                yy_tmp = self.gen_sample_batch(
                    t,
                    T,
                    x,
                    torch.ones_like(t),
                    torch.ones_like(t),
                    c,
                    patch,
                    idx,
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
                yyy.append((yy_tmp.nan_to_num() * mask).sum(dim=1) / mask.sum(dim=1))
            yy.append(torch.stack(yyy, dim=-1))

        return (
            torch.cat(xx),
            torch.cat(yy),
        )

    def calculate_p_from_u(self, x, patch):
        """
        Describe

        Parameters
        ----------
        x : torch.Tensor
            Describe

        patch : int
            Describe
        """
        x = x.detach().clone().requires_grad_(True)
        nb_mc = self.nb_path_per_state
        x = x.repeat(nb_mc, 1, 1)
        unif = (
            torch.rand(nb_mc * x.shape[1], device=self.device)
                 .reshape(nb_mc, -1, 1)
        )
        tau = self.tau_lo + (self.tau_hi - self.tau_lo) * unif
        y = self.gen_bm(tau.transpose(0, -1), x.shape[1], var=1).transpose(0, -1)
        x = x + y
        x = x.reshape(-1, self.dim_in).T
        order = np.array([0] * self.dim_in)
        ans = 0
        for i in range(self.dim_in):
            for j in range(self.dim_in):
                order[i] += 1
                # TODO: use the correct T!!
                # self.adjusted_phi(x, T, patch, j)
                tmp = self.nth_derivatives(
                    order, self.phi_fun(x, j), x
                )
                order[i] -= 1
                order[j] += 1
                tmp *= self.nth_derivatives(
                    order, self.phi_fun(x, i), x
                )
                order[j] -= 1
                ans += tmp
        ans = ans.reshape(nb_mc, -1)
        ans *= (y**2).sum(dim=-1)
        if self.dim_in > 2:
            ans /= (self.dim_in - 2)
        elif self.dim_in == 2:
            ans *= -torch.log((y**2).sum(dim=-1).sqrt())
        ans *= ((self.tau_hi - self.tau_lo) / (2 * tau[:, :, 0]))
        return ans.mean(dim=0, keepdim=True).detach()

    def gen_sample_for_p(self, patch, gen_y=True, overtrain_rate=.5):
        """
        Describe

        Parameters
        ----------
        patch : int
            Describe

        gen_y : bool, optional
            Describe

        overtrain_rate : float, optional
            Describe
        """
        self.nb_path_per_state *= 1000
        self.nb_states_per_batch //= 1000
        states_per_batch = min(self.nb_states, self.nb_states_per_batch)
        batches = math.ceil(self.nb_states / states_per_batch)
        xx, yy = [], []
        # widen the domain from [x_lo, x_hi] to [x_lo - .5*(x_hi-x_lo), x_hi + .5*(x_hi-x_lo)]
        x_lo, x_hi = self.x_lo, self.x_hi
        x_lo, x_hi = x_lo - overtrain_rate * (x_hi - x_lo), x_hi + overtrain_rate * (x_hi - x_lo)
        start = time.time()
        for batch_now in range(batches):
            unif = (
                torch.rand(self.dim_in * states_per_batch, device=self.device)
                     .reshape(states_per_batch, self.dim_in)
            )
            x = (x_lo + (x_hi - x_lo) * unif).T
            if self.dim_in > 1 and self.fix_all_dim_except_first:
                x[1:, :] = (x_hi + x_lo) / 2
            if gen_y:
                y = self.calculate_p_from_u(x.T, patch=patch)
                yy.append(y)
            xx.append(x)
            if batch_now % 1000 == 0 or batch_now == batches - 1:
                logging.debug(f"Generated {batch_now + 1} out of {batches} batches with {time.time() - start} seconds.")
                start = time.time()
        self.nb_path_per_state //= 1000
        self.nb_states_per_batch *= 1000
        return (
            torch.cat(xx, dim=-1).T,
            torch.cat(yy, dim=0) if yy else None,
        )

    def train_and_eval(self, debug_mode=False, return_dict=False, reuse_x=None, reuse_y=None):
        """
        Generate sample and evaluate (plot) NN approximation when debug_mode=True

        Parameters
        ----------
        debug_mode : bool, optional
            Describe

        return_dict : bool, optional
            Describe

        reuse_x : torch.Tensor, optional
            Describe

        reuse_y : torch.Tensor, optional
            Describe
        """
        output_dict = {}
        for p in range(self.patches):
            if self.train_for_p:
                # initialize optimizer for p
                optimizer = torch.optim.Adam(
                    (val for key, val in self.named_parameters() if f'layer.{p}' in key),
                    lr=self.lr, weight_decay=self.weight_decay
                )
                scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=self.lr_milestones,
                    gamma=self.lr_gamma,
                )

                start = time.time()
                x, y = self.gen_sample_for_p(patch=p)
                self.print_msg(
                    f"Patch {p}: generation of p samples take {time.time() - start} seconds."
                )

                best_loss = float('inf')
                start = time.time()
                for epoch in range(self.epochs):
                    # clear gradients and evaluate training loss
                    optimizer.zero_grad()
                    self.train()
                    loss = self.loss(self(x, p_or_u="p", patch=p), y)
                    self.eval()

                    # use loss related to poisson equation to train
                    poisson_rhs = 0
                    order = np.array([0] * self.dim_in)
                    xx = x.T.detach().clone().requires_grad_(True)
                    for i in range(self.dim_in):
                        for j in range(self.dim_in):
                            order[i] += 1
                            tmp = self.nth_derivatives(
                                order, self.phi_fun(xx, j), xx
                            )
                            order[i] -= 1
                            order[j] += 1
                            tmp *= self.nth_derivatives(
                                order, self.phi_fun(xx, i), xx
                            )
                            order[j] -= 1
                            poisson_rhs -= tmp
                    poisson_rhs = poisson_rhs.detach()
                    poisson_lhs = 0
                    xx = x.T.detach().clone().requires_grad_(True)
                    for i in range(self.dim_in):
                        order[i] += 2
                        poisson_lhs += self.nth_derivatives(
                            order, self(xx.T, p_or_u="p", patch=p), xx
                        )
                        order[i] -= 2
                    loss += self.loss(poisson_lhs, poisson_rhs)

                    # update model weights and schedule
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    if epoch > self.lr_milestones[-1] and loss.item() < best_loss:
                        # save the best model when NN enters stabilising zone
                        best_loss = loss.item()
                        if self.save_for_best_model:
                            torch.save(self.state_dict(), f"{self.working_dir}/checkpoint.pt")

                    # print loss information and plot every 500 epochs
                    if epoch % 500 == 0 or epoch + 1 == self.epochs:
                        self.log_plot_save(patch=p, epoch=epoch, loss=loss, x=x, y=y, debug_mode=debug_mode, p_or_u="p")

                if self.save_for_best_model:
                    self.load_state_dict(torch.load(f"{self.working_dir}/checkpoint.pt"))
                self.print_msg(
                    f"Patch{p}: pre-training of p with {self.epochs} epochs takes {time.time() - start:4.0f} seconds."
                )

            # initialize optimizer for u
            optimizer = torch.optim.Adam(
                (val for key, val in self.named_parameters() if f'layer.{p}' in key),
                lr=self.lr, weight_decay=self.weight_decay
            )
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=self.lr_milestones,
                gamma=self.lr_gamma,
            )

            start = time.time()
            if reuse_x is None:
                x, y = self.gen_sample(patch=p)
            else:
                x, y = reuse_x, reuse_y
            self.print_msg(
                f"Patch {p}: generation of u samples take {time.time() - start} seconds."
            )

            best_loss = float('inf')
            start = time.time()
            # loop through epochs
            for epoch in range(self.epochs):
                # clear gradients and evaluate training loss
                optimizer.zero_grad()
                self.train()
                loss = self.loss(self(x, patch=p), y)
                self.eval()

                # divergence free condition
                if self.deriv_condition_zeta_map is not None:
                    grad = 0
                    xx = x.T.detach().clone().requires_grad_(True)
                    for (idx, c) in zip(self.deriv_condition_zeta_map, self.deriv_condition_deriv_map):
                        # additional t coordinate
                        grad += self.nth_derivatives(
                            np.insert(c, 0, 0), self(xx.T, patch=p)[:, idx], xx
                        )
                    loss += self.loss(grad, torch.zeros_like(grad))

                # update model weights and schedule
                loss.backward()
                optimizer.step()
                scheduler.step()

                # save the best model when NN enters stabilising zone
                if epoch > self.lr_milestones[-1] and loss.item() < best_loss:
                    best_loss = loss.item()
                    if self.save_for_best_model:
                        torch.save(self.state_dict(), f"{self.working_dir}/checkpoint.pt")

                # print loss information and plot every 500 epochs
                if epoch % 500 == 0 or epoch + 1 == self.epochs:
                    self.log_plot_save(patch=p, epoch=epoch, loss=loss, x=x, y=y, debug_mode=debug_mode, p_or_u="u")

            if self.save_for_best_model:
                self.load_state_dict(torch.load(f"{self.working_dir}/checkpoint.pt"))
            self.print_msg(
                f"Patch {p}: training of u with {self.epochs} epochs take {time.time() - start} seconds."
            )
            output_dict[f"patch_{p}"] = (time.time() - start, best_loss)
        if return_dict:
            return output_dict


if __name__ == "__main__":
    # configurations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lower_bound, upper_bound = -2, 2
    nu = 1
    y, eps = 0, 1
    a, b = y - eps, y + eps

    # function definition
    deriv_map = np.array([0]).reshape(-1, 1)
    def f_example(y, coodinate=0):
        """
        idx 0 -> no deriv
        """
        return torch.zeros_like(y[0])

    def phi_example(x, coodinate=0):
        return torch.logical_and(x[0] <= b, x[0] >= a).float()

    def exact_example(t, x, T, coordinate=0, with_bound=True, k_arr=range(-5, 5)):
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
                - torch.exp(
                    ((y - x) ** 2 - (y + x - 2 * lower_bound + 2 * k * (upper_bound - lower_bound)) ** 2) / (2 * t)
                )
            )
        return ans.prod(dim=0)

    def is_x_inside(x):
        return torch.logical_and(lower_bound <= x, x <= upper_bound).all(dim=0)

    problem_name = "heat_equation"
    t_lo, x_lo, x_hi, n = 0., lower_bound, upper_bound, 0
    grid = np.linspace(x_lo, x_hi, 100)
    grid_d_dim = np.expand_dims(grid, axis=0)
    grid_d_dim_with_t = np.concatenate((t_lo * np.ones((1, 100)), grid_d_dim), axis=0)

    patches = 5
    T = patches * 1.0
    true = exact_example(t_lo, grid_d_dim, T)
    true_with_bound = exact_example(t_lo, grid_d_dim, T, with_bound=True)
    terminal = exact_example(T, grid_d_dim, T)
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
    )
    model.train_and_eval(debug_mode=True)
    model.compare_with_exact(exact_fun=exact_example)
