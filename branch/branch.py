import os
import time
import math
import torch
from torch.distributions.exponential import Exponential
import matplotlib.pyplot as plt
import numpy as np
from .fdb import fdb_nd
import logging

torch.manual_seed(0)  # set seed for reproducibility


class Net(torch.nn.Module):
    """
    Deep branching approach to solve the PDE system
    $$
    \\partial_t u_i(t, x) + \\frac{\\nu}{2} \\Delta u_i(t, x)
    + f_i((\\partial_{\\alpha^j} u_{\\zeta^j}(t, x))_{0 \\le j < n}) = 0,
    \\quad 0 \\le i < d_{out},
    $$
    $$
    \\Delta u_{-1}(t, x) =
    -\\sum\\limits_{i=0}^{d_{out}-1}
    \\partial_{1_i} u_j(t, x)
    \\partial_{1_j} u_i(t, x),
    $$
    $$
    \\sum\\limits_{i = 0}^{m - 1}
    \\partial_{\\alpha_{deriv}^i} u_{\\zeta_{deriv}^i}(t, x) = 0,
    \\quad (t, x) \\in [t_{lo}, T] \\times \\Omega,
    \\quad \\Omega \\subset \\mathbb{R}^{d_{in}}.
    $$


    Attributes
    ----------
    problem_name : str
        The name of the problem that will be used for naming
        the directory of logs and plots.

    f_fun : function
        The f function `f_fun(x, coordinate)`
        in the PDE written in PyTorch framework
        that takes x tensor of size `n x batch_size`
        and `int` as input,
        and outputs tensor of size `batch_size`.

    phi_fun : function
        The terminal condition `phi_fun(x, coordinate)`
        of PDE, i.e. phi_fun(x, i) = u_i(T, x),
        written in PyTorch framework
        that takes x tensor of size `d_in x batch_size`
        and `int` as input,
        and outputs tensor of size `batch_size`.

    phi0 : float
        The boundary condition of PDE,
        currently works only for `phi0 = 0`.

    conditional_probability_to_survive : function
        The function `conditional_probability_to_survive(t, x, y)`
        written in PyTorch framework
        that takes t tensor of size `batch_size`,
        x tensor of size `d_in x batch_size`,
        y tensor of size `d_in x batch_size`,
        and outputs tensor of size `batch_size`.
        The function should satisfy
        $$
        \\text{conditional_probability_to_survive}(t, x, y)
        = \\mathbb{P}
        (W \\text{ does not exit the domain before } t | W_0 = x, W_t = y).
        $$

    is_x_inside : function
        The function `is_x_inside(x)`
        written in PyTorch framework
        that takes x tensor of size `d_in x batch_size`
        and outputs bool tensor of size `batch_size`.
        The function checks whether x is stll inside the domain.

    deriv_map : numpy.ndarray
        Numpy array of size `n x d_in` such that
        $$
        \\text{deriv_map}[i] = \\alpha^i.
        $$

    n : int
        n in the PDE system.

    nu : float
        nu in the PDE system.

    dim_in : int
        The dimension of the input x.

    zeta_map : numpy.ndarray
        Numpy array of size `n` such that
        $$
        \\text{zeta_map}[i] = \\zeta^i.
        $$
        If coordinate -1 is involved,
        put them first in `zeta_map`.

    deriv_condition_deriv_map : numpy.ndarray
        Numpy array of size `m x d_in` such that
        $$
        \\text{deriv_condition_deriv_map}[i] = \\alpha_{deriv}^i.
        $$

    deriv_condition_zeta_map : numpy.ndarray
        Numpy array of size `m` such that
        $$
        \\text{deriv_condition_zeta_map}[i] = \\zeta_{deriv}^i.
        $$

    dim_out : int
        The dimension of the PDE system.

    nprime : int
        The number of -1 in `zeta_map`.
        Note that we assume that for all i < `nprime`,
        zeta_map[i] = -1.

    exact_p_fun : function
        The function `exact_p_fun(x)`
        written in PyTorch framework
        that takes x tensor of size `d_in x batch_size`
        and outputs tensor of size `batch_size`.
        The function should satisfy
        $$
        \\text{exact_p_fun}(x) = u_{-1}(T, x).
        $$
        If `exact_p_fun` is not passed to Net,
        Net will use neural network approximation to solve
        the Poisson equation at terminal time.

    train_for_p : bool
        Whether or not to use neural network approximation to solve
        the Poisson equation at terminal time.

    patches : int
        Split the solver into `patches` steps, i.e.
        $$
        [(patches - 1) \\frac{T}{patches}, T], ..., [0, \\frac{T}{patches}].
        $$

    delta_t : float
        The quantity `T/patches`.

    code : numpy.ndarray
        Net trains the network based on the `code = [c0, c1, ..., ck]`
        and the `coordinate = [i0, i1, ..., ik]` (see `coordinate` below),
        i.e. match c0(u_{i0})(t, x), ..., ck(u_{ik})(t, x).
        See also the deep branching paper for more details about the code.

    coordinate : numpy.ndarray
        See `code` above.

    fdb_lookup : dict
        Store the Faa di Bruno output as dictionary,
        to be used in `gen_sample_batch` and `helper_negative_code_on_f`.

    fdb_runtime : float
        Calculate the runtime used in generating Faa di Bruno output.

    mechanism_tot_len : int
        The total length of mechanism when the `fdb_lookup` is initialized.

    u_layer : torch.nn.modules.container.ModuleList
        The layer functions for the approximation of PDE system u.

    u_bn_layer : torch.nn.modules.container.ModuleList
        The batch normalization functions
        for the approximation of PDE system u.

    p_layer : torch.nn.modules.container.ModuleList
        The layer functions
        for the approximation of u_{-1} at terminal time.

    p_bn_layer : torch.nn.modules.container.ModuleList
        The batch normalization functions
        for the approximation of u_{-1} at terminal time.

    lr : float
        The initial learning rate for training the neural network.

    lr_milestones : list
        Current lr will be multiplied by `lr_gamma` below
        when epoch is in `lr_milestones`,
        see torch.optim.lr_scheduler.MultiStepLR
        for more details.

    lr_gamma : float
        See `lr_milestones` above.

    weight_decay : float
        Weight decay for the Adam optimizer,
        see torch.optim.Adam for more details.

    save_for_best_model : bool
        Whether or not to save for the best model
        based on the training loss
        when epoch > lr_milestones[-1].

    save_data : bool
        Whether or not to save the data
        generated by `gen_sample` and neural network output.

    loss : torch.nn.modules.loss
        The loss function to evaluate the neural network output,
        currently we use `MSELoss`.

    activation : torch.nn.modules.activation
        The activation function for the neural network.

    batch_normalization : bool
        Whether or not to use batch normalization.

    epochs : int
        The number of epochs for training the neural network

    nb_states : int
        The number of samples (batch size) that `gen_sample` generates.

    nb_states_per_batch : int
        Due to memory limit,
        we process only `nb_states_per_batch` each time
        until `nb_states` of samples are generated.

    nb_path_per_state : int
        The number of Monte Carlo path for each sample.

    x_lo : float
        We fit the neural network
        inside the hypercube [x_lo, x_hi]^{d_in},
        unless otherwise stated,
        see `adjusted_x_boundaries` below.

    x_hi : float
        See `x_lo` above.

    adjusted_x_boundaries : tuple
        When `overtrain_rate` > 0,
        we widen [x_lo, x_hi] accordingly
        so that the neural network fit better
        on the boundary of [x_lo, x_hi]^{d_in}.

    t_lo : float
        We fit the neural network
        inside the time horizon [t_lo, t_hi],
        unless otherwise stated,
        see `fix_t_dim` below.

    t_hi : float
        See `t_lo` above.

    T : float
        The terminal time of the problem.

    fix_all_dim_except_first : bool
        In the case of d_in > 1,
        whether or not to generate sample on
        $$
        [x_{lo}, x_{hi}]
        \\times \\{ \\frac{x_{lo} + x_{hi}}{2} \\}^{d_{in} - 1}
        $$
        instead of
        $$
        [x_{lo}, x_{hi}]^{d_{in}}.
        $$

    fix_t_dim : bool
        Whether or not to generate
        [t_lo, t_lo] instead of [t_lo, t_hi].

    t_boundaries : torch.Tensor
        The usual boundaries [t_lo, t_hi]
        with batches taken into account,
        to be used in `bisect_left`.

    adjusted_t_boundaries : list
        The boundaries for t,
        with both `fix_t_dim` and `patches` taken into account,
        used for sample generation in `gen_sample`.

    tau_lo : float
        The lower limit of time integration
        when we inverse the Laplacian,
        see the deep branching paper for more details.

    tau_hi : float
        The upper limit of time integration
        when we inverse the Laplacian,
        see the deep branching paper for more details.

    outlier_percentile : float
        The parameter used to discard outliers
        for the Monte Carlo samples. <br />
        Let (lo, hi) be
        $$
        (\\text{outlier_percentile},
        100 - \\text{outlier_percentile})
        $$
        percentile of the Monte Carlo samples.
        We set the boundary as
        $$
        [\\text{lo - outlier_multiplier} \\times \\text{(hi-lo)},
        \\text{hi + outlier_multiplier} \\times \\text{(hi-lo)}].
        $$
        Samples out of this boundary are considered as outlier and removed.

    outlier_multiplier : float
        See `outlier_percentile` above.

    exponential_lambda : float
        The parameter used for the generation of random tree time,
        see the deep branching paper for more details.

    antithetic : bool
        Whether or not to use antithetic variates.

    device : torch.device
        The device used by Net, either cpu or cuda.

    verbose : bool
        If `verbose=True`, more information will be printed.
        Otherwise, the information is saved in run.log file.

    working_dir : str
        The directory that saves all the logs and plots.
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
        exact_p_fun_full=None,
        phi0=0,
        conditional_probability_to_survive=(lambda t, x, y: torch.ones_like(x[0])),
        conditional_probability_to_survive_for_p=(lambda t, x, y: torch.ones_like(x[0])),
        is_x_inside=(lambda x: torch.ones_like(x[0]).bool()),
        is_x_inside_for_p=(lambda x: torch.ones_like(x[0]).bool()),
        x_lo=-10.0,
        x_hi=10.0,
        t_lo=0.0,
        T=1.0,
        nu=1.0,
        branch_exponential_lambda=None,
        bm_discretization_steps=1,
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
        div_condition_coeff=1.,
        poisson_coeff=1.,
        overtrain_rate=.1,
        overtrain_rate_for_p=.5,
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
        train_for_u=True,
        save_for_best_model=True,
        save_data=False,
        continue_from_checkpoint=None,
        save_as_tmp=False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        problem_name : str
            The name of the problem that will be used for naming
            the directory of logs and plots.

        f_fun : function
            The f function in the PDE written in PyTorch framework.

        deriv_map : numpy.ndarray
            Numpy array of size `n x d_in` such that
            $$
            \\text{deriv_map}[i] = \\alpha^i.
            $$

        zeta_map : numpy.ndarray, optional
            Default to be zeros array of size `n`.

        deriv_condition_deriv_map : numpy.ndarray, optional
            Default to be `None`
            so that no deriv condition is required.

        deriv_condition_zeta_map : numpy.ndarray, optional
            Default to be `None`
            so that no deriv condition is required.

        dim_out : int, optional
            Default to be the maximum value in `zeta_map`.

        phi_fun : function, optional
            Default to be identitiy function.

        exact_p_fun : function, optional
            Default to be `None`,
            so that Net will use neural network approximation to solve
            the Poisson equation at terminal time when `train_for_p = True`.

        phi0 : float, optional
            Default to be `0`.

        conditional_probability_to_survive : function, optional
            Default to always output `1`,
            which is the case when the domain is the whole space.

        is_x_inside : function, optional
            Default to always output `True`,
            which is the case when the domain is the whole space.

        x_lo : float, optional
            Default to be `-10`.

        x_hi : float, optional
            Default to be `10`.

        t_lo : float, optional
            Default to be `0`.

        T : float, optional
            Default to be `1`.

        nu : float, optional
            Default to be `1`.

        branch_exponential_lambda : float, optional
            Default to be `-log(.95)/T`.

        neurons : int, optional
            The number of neurons of the neural networks,
            default to be `50`.

        layers : int, optional
            The number of layers of the neural networks,
            default to be `6`.

        branch_lr : float, optional
            Initial learning rate, default to be `1e-3`.

        lr_milestones : tuple, optional
            Default to be `(1000, 2000)`.

        lr_gamma : float, optional
            Default to be `0.8`.

        weight_decay : float, optional
            Default to be `0`.

        branch_nb_path_per_state : int, optional
            Default to be `10000`.

        branch_nb_states : int, optional
            Default to be `1000`.

        branch_nb_states_per_batch : int, optional
            default to be `100`.

        epochs : int, optional
            Default to be `5000`.

        batch_normalization : bool, optional
            Default to be `True`.

        antithetic : bool, optional
            Default to be `True`.

        overtrain_rate : float, optional
            Default to be `.1`.

        device : str, optional
            Default to be cpu.

        branch_activation : str, optional
            Default to be softplus function.

        verbose : bool, optional
            Default to be `False`.

        fix_all_dim_except_first : bool, optional
            Default to be `True`.

        fix_t_dim : bool, optional
            Default to be `True`.

        branch_patches : int, optional
            Default to be `1`.

        outlier_percentile : float, optional
            Default to be `1`.

        outlier_multiplier : float, optional
            Default to be `1000`.

        code : numpy.ndarray, optional
            Default to be the identity code of size `d_out`.

        coordinate : numpy.ndarray, optional
            Default to be `[0, 1, ..., d_out-1]`.

        train_for_p : bool, optional
            Default to be `True` if `nprime > 0` and `exact_p_fun = None`,
            `False` otherwise.

        save_for_best_model : bool, optional
            Default to be `True`.

        save_data : bool, optional
            Default to be `False`.

        continue_from_checkpoint : str, optional
            If a directory is passed as `continue_from_checkpoint`,
            we load the model from the directory
            and set `train_for_p = False`.
        """
        super(Net, self).__init__()
        self.problem_name = problem_name
        self.f_fun = f_fun
        self.phi_fun = phi_fun
        self.phi0 = phi0
        self.conditional_probability_to_survive = conditional_probability_to_survive
        self.is_x_inside = is_x_inside
        self.conditional_probability_to_survive_for_p = conditional_probability_to_survive_for_p
        self.is_x_inside_for_p = is_x_inside_for_p
        self.deriv_map = deriv_map
        self.n, self.dim_in = deriv_map.shape
        self.zeta_map = zeta_map if zeta_map is not None else np.zeros(self.n, dtype=int)
        self.deriv_condition_deriv_map = deriv_condition_deriv_map
        self.deriv_condition_zeta_map = deriv_condition_zeta_map
        self.dim_out = dim_out if dim_out is not None else self.zeta_map.max() + 1
        self.nprime = sum(self.zeta_map == -1)
        self.exact_p_fun = exact_p_fun
        self.exact_p_fun_full = exact_p_fun_full
        self.train_for_p = (
            train_for_p if train_for_p is not None
            else (self.nprime > 0) and self.exact_p_fun is None and self.exact_p_fun_full is None
        )
        self.train_for_u = train_for_u

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
        self.overtrain_rate_for_p = overtrain_rate_for_p
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
        self.bm_discretization_steps = bm_discretization_steps
        self.epochs = epochs
        self.antithetic = antithetic
        self.div_condition_coeff = div_condition_coeff
        self.poisson_coeff = poisson_coeff
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
            continue_from_checkpoint_full_path = os.path.join(
                os.getcwd(),
                continue_from_checkpoint,
            )
            self.load_state_dict(torch.load(
                f"{continue_from_checkpoint_full_path}/checkpoint.pt")
            )
            self.train_for_p = False

        timestr = time.strftime("%Y%m%d-%H%M%S")  # current time stamp
        self.working_dir = (
            "logs/tmp" if save_as_tmp
            else f"logs/{timestr}-{problem_name}-T{self.T}-nu{self.nu}"
        )
        self.working_dir_full_path = os.path.join(
            os.getcwd(),
            self.working_dir,
        )
        self.log_config()
        self.eval()

    def forward(self, x, patch=None, p_or_u="u"):
        """
        Apply the neural network function on the input `x`.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of size `batch_size x (d_in + 1)` for u network;
            tensor of size `batch_size x d_in` for p network,

        patch : int, optional
            The current patch index.
            When `patch` is not specified,
            we use `bisect_left` on the first dimension of `x`
            to determine the correct `patch`.

        p_or_u : str, optional
            To call u network or p network,
            default to be "u".
        """
        if self.exact_p_fun is not None and p_or_u == "p":
            return self.exact_p_fun(x.T).unsqueeze(dim=-1)
        if self.exact_p_fun_full is not None and p_or_u == "p":
            return self.exact_p_fun_full(x.T, t=self.T).unsqueeze(dim=-1)

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
        Set up configuration for log files and mkdir.
        """
        if not os.path.isdir(self.working_dir_full_path):
            os.makedirs(self.working_dir_full_path)
            os.mkdir(f"{self.working_dir_full_path}/plot")
            os.mkdir(f"{self.working_dir_full_path}/data")
        formatter = "%(asctime)s | %(name)s |  %(levelname)s: %(message)s"
        logging.getLogger().handlers = []  # clear previous loggers if any
        logging.basicConfig(
            filename=f"{self.working_dir_full_path}/run.log",
            filemode="w",
            level=logging.DEBUG,
            format=formatter,
        )
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logging.getLogger().addHandler(console)
        logging.info(f"Logs are saved in {os.path.abspath(self.working_dir_full_path)}")
        logging.debug(f"Current configuration: {self.__dict__}")

    def bisect_left(self, val):
        """
        Find the index of val based on the discretization of self.t_boundaries
        it is only used when branch_patches > 1

        Parameters
        ----------
        val : torch.Tensor
            Tensor representing the current time.
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
    def latex_print(tensor):
        """
        Print the tensor in the latex format,
        to be used in `error_calculation`.

        Parameters
        ----------
        tensor : torch.Tensor
            Tensor to be printed in latex format.
        """
        mess = ""
        for i in tensor[:-1]:
            mess += f"& {i.item():.2E} "
        mess += "& --- \\\\"
        logging.info(mess)

    def error_calculation(self, exact_u_fun, exact_p_fun, nb_pts_time=11, nb_pts_spatial=2*126+1, error_multiplier=1):
        """
        The calculation of error according to the metrics
        by Lejay and Gonzalez and Angeli et al.

        Parameters
        ----------
        exact_u_fun : function
            The u function `exact_u_fun(t, x, coordinate)`
            in closed-form written in PyTorch framework
            that takes t tensor of size `batch_size`,
            x tensor of size `d_in x batch_size`,
            and int as input,
            and outputs tensor of size `batch_size`.

        exact_p_fun : function
            The p function `exact_p_fun(tx)`
            in closed-form written in PyTorch framework
            that takes tx tensor of size `(d_in + 1) x batch_size` as input,
            and outputs tensor of size `batch_size`.

        nb_pts_time : int, optional
            The number of points in [0, T],
            default to be `11`.

        nb_pts_spatial : int, optional
            The number of points in [x_lo, x_hi],
            default to be `2*126+1`.

        error_multiplier : float, optional
            Multiplier for the error metric of Lejay,
            default to be `1`.
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
            self.latex_print(error[i].max(dim=1)[0])
        logging.info("$\\hat{e}(t_k)$")
        self.latex_print(error[-1].max(dim=1)[0])
        logging.info("\\hline")

        # erru
        logging.info("\nThe relative L2 error of u (erru) is calculated as follows.")
        denominator, numerator = 0, 0
        for i in range(self.dim_in):
            denominator += exact_u_fun(arr.T, i).reshape(nb_pts_time, -1) ** 2
            numerator += (nn[:, i] - exact_u_fun(arr.T, i)).reshape(nb_pts_time, -1) ** 2
        logging.info("erru($t_k$)")
        self.latex_print((numerator.mean(dim=-1)/denominator.mean(dim=-1)).sqrt())

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
        self.latex_print((numerator.mean(dim=(1, 2))/denominator.mean(dim=(1, 2))).sqrt())

        # errdivu
        logging.info("\nThe absolute divergence of u (errdivu) is calculated as follows.")
        numerator = 0
        for i in range(self.dim_in):
            numerator += (grad[:, i, i]).reshape(nb_pts_time, -1)
        numerator = numerator**2
        logging.info("errdivu($t_k$)")
        self.latex_print(
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
            Numpy array of size `d`.

        y : torch.Tensor
            Tensor of size `batch_size`.

        x : torch.Tensor
            Tensor of size `d x batch_size`.
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
            Tensor of size `d_in x batch_size`.

        T : torch.Tensor
            Tensor of size `batch_size` that represents
            the terminal time of the current patch.

        patch : int
            The current patch index.

        coordinate : numpy.ndarray
            The current coordinate of the system.
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
        To log as info when `verbose = True`
        and to log as debug otherwise.

        Parameters
        ----------
        msg : str
            Message to be logged.
        """
        if self.verbose:
            logging.info(msg)
        else:
            logging.debug(msg)

    def code_to_function(self, code, x, T, patch, coordinate):
        """
        Calculate the functional of tree based on code, x, and coordinate.

        There are two ways of representing the code <br />
        1. negative code of size `d`
            $$
            (i_0, \\ldots, i_{d-1}) \\rightarrow
            \\partial_{x_0}^{-i_0 - 1} \\dots
            \\partial_{x_{d-1}}^{-i_{d-1} - 1}
            \\phi_{coordinate}(x_0, ..., x_{d-1}).
            $$
        2. positive code of size `n`
            $$
            (i_0, \\ldots, i_{n-1}) \\rightarrow
            \\partial_{y_0}^{i_0 - 1} \\dots
            \\partial_{y_{n-1}}^{i_{n-1} - 1}
            f_{coordinate}
            ((\\partial_{\\alpha^j} u_{\\zeta^j}(t, x))_{0 \\le j < n}).
            $$

        Parameters
        ----------
        code : numpy.ndarray
            Code to be applied on `f_fun` and `phi_fun`.

        x : torch.Tensor
            Tensor of size `d_in x batch_size`.

        T : torch.Tensor
            Tensor of size `batch_size` that represents
            the terminal time of the current patch.

        patch : int, optional
            The current patch index.

        coordinate : numpy.ndarray
            The current coordinate of the system.
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

    def gen_bm(self, x, dt, nb_states, var=None):
        """
        Generate Brownian motion var x sqrt{dt} x Gaussian.

        When self.antithetic=true, we generate
        dw = sqrt{dt} x Gaussian of size nb_states//2
        and return (dw, -dw).

        Parameters
        ----------
        dt : torch.Tensor
            Time increment of the Brownian motion.

        nb_states : int
            Number of states for Brownian motion generation.

        var : float, optional
            Default to be `nu`.
        """
        dt = dt.clip(min=0.0)  # so that we can safely take square root of dt
        if var is not None:
            # simulate BM for p, max T = self.tau_hi
            delta_dt = self.tau_hi / self.bm_discretization_steps * torch.ones_like(dt)
        else:
            # simulate BM for u, max T = self.delta_t
            delta_dt = self.delta_t / self.bm_discretization_steps * torch.ones_like(dt)
        var = self.nu if var is None else var

        x_now = x
        is_x_inside = self.is_x_inside(x_now)
        for _ in range(self.bm_discretization_steps):
            if (dt <= 0).all():
                break
            dt_now = torch.minimum(dt, delta_dt)
            dt = dt - dt_now
            if self.antithetic:
                # antithetic variates
                dw = torch.randn(
                    self.dim_in, nb_states, self.nb_path_per_state // 2, device=self.device
                ).repeat(1, 1, 2)
                dw[:, :, : (self.nb_path_per_state // 2)] *= -1
            else:
                # usual generation
                dw = torch.randn(
                    self.dim_in, nb_states, self.nb_path_per_state, device=self.device
                )
            is_x_inside = is_x_inside * self.is_x_inside(x_now)
            x_next = x_now + dw * torch.sqrt(var * dt_now)
            x_now = x_next.where(is_x_inside, x_now)
            
        return x_now

    def helper_negative_code_on_f(self, t, T, x, mask, H, code, patch, coordinate):
        """
        Helper function to deal with coordinate -2,
        which corresponds to
        $$
        \\partial_t u_{-1} + \\frac{\\nu}{2} * \\Delta u_{-1}.
        $$


        Parameters
        ----------
        t : torch.Tensor
            Current time.

        T : torch.Tensor
            Terminal time.

        x : torch.Tensor
            Value of Brownian motion at time t.

        mask : torch.Tensor
            mask[idx]=1 means the state at index idx is still alive
            mask[idx]=0 means the state at index idx is dead.

        H : torch.Tensor
            Cummulative value of the product of functional H.

        code : numpy.ndarray
            Determine the operation to be taken on the functions f and phi.

        patch : int
            The current patch index.

        coordinate : numpy.ndarray
            The current coordinate of the system.
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
        Recursive function to calculate E[ H(t, x, code) ].

        Parameters
        ----------
        t : torch.Tensor
            Current time.

        T : torch.Tensor
            Terminal time.

        x : torch.Tensor
            Value of Brownian motion at time t.

        mask : torch.Tensor
            mask[idx]=1 means the state at index idx is still alive
            mask[idx]=0 means the state at index idx is dead.

        H : torch.Tensor
            Cummulative value of the product of functional H.

        code : numpy.ndarray
            Determine the operation to be taken on the functions f and phi.

        patch : int
            The current patch index.

        coordinate : numpy.ndarray
            The current coordinate of the system.
        """
        # return zero tensor when no operation is needed
        ans = torch.zeros_like(t)
        if ~mask.any():
            return ans

        nb_states, _ = t.shape
        # for the p coordinate
        if coordinate < 0:
            assert (code < 0).all(), "coordinate p should not have positive code"
            if self.exact_p_fun_full is not None:
                mask = mask.bool()
                x = x[:, mask].detach().clone().requires_grad_(True)
                tmp = self.nth_derivatives(
                    -code - 1, self.exact_p_fun_full(x, t=t[mask]), x
                ).detach()
                ans[mask] = tmp
                return ans

            unif = (
                torch.rand(nb_states * self.nb_path_per_state, device=self.device)
                     .reshape(nb_states, self.nb_path_per_state)
            )
            tau = self.tau_lo + (self.tau_hi - self.tau_lo) * unif
            next_x = self.gen_bm(x, tau, nb_states, var=1)
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
                                        # * self.dim_in ** 2
                                        # * (dw ** 2).sum(dim=0)
                                        * (self.tau_hi - self.tau_lo)
                                        / 2
                                        # / (2 * tau)
                                )
                                # if self.dim_in > 2:
                                #     A = A / (self.dim_in - 2)
                                # elif self.dim_in == 2:
                                #     A = -A * torch.log((dw ** 2).sum(dim=0).sqrt())
                                code_increment = np.zeros_like(code)
                                code_increment[j] += 1
                                if fdb.lamb[0] == 0:
                                    A = A * self.gen_sample_batch(
                                        t,
                                        T,
                                        next_x,
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
                                        next_x,
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
                                            next_x,
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
                                            next_x,
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
                                            # * self.dim_in ** 2
                                            # * (dw ** 2).sum(dim=0)
                                            * (self.tau_hi - self.tau_lo)
                                            / 2
                                            # / (2 * tau)
                                    )
                                    # if self.dim_in > 2:
                                    #     A = A / (self.dim_in - 2)
                                    # elif self.dim_in == 2:
                                    #     A = -A * torch.log((dw ** 2).sum(dim=0).sqrt())
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
                                            next_x,
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
                                                next_x,
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
                                                next_x,
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
                                                    next_x,
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
                                                    next_x,
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
                                                next_x,
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
        next_x = self.gen_bm(x, T - t, nb_states)
        x_is_inside = self.is_x_inside(next_x)
        survive_prob = self.conditional_probability_to_survive(self.nu * (T - t), x, next_x).clip(0, 1)

        ############################### for t + tau >= T
        mask_now = mask.bool() * (t + tau >= T)
        if mask_now.any():
            tmp = (
                    H[mask_now]
                    / torch.exp(-self.exponential_lambda * (T - t)[mask_now])
                    * (
                        x_is_inside[mask_now] * survive_prob[mask_now]
                            * self.code_to_function(code, next_x[:, mask_now], T[mask_now], patch, coordinate)
                        + (1 - x_is_inside[mask_now] * survive_prob[mask_now]) * self.phi0
                    )
            )
            ans[mask_now] = tmp

        ############################### for t + tau < T
        next_x = self.gen_bm(x, tau, nb_states)
        x_is_inside = self.is_x_inside(next_x)
        survive_prob = self.conditional_probability_to_survive(self.nu * tau, x, next_x).clip(0, 1)
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
                next_x,
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
                        next_x,
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
                                    next_x,
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
                                next_x,
                                mask_tmp,
                                torch.ones_like(t),
                                -self.deriv_map[i] - code_increment - 1,
                                patch,
                                self.zeta_map[i],
                            )
                            B = self.gen_sample_batch(
                                t + tau,
                                T,
                                next_x,
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
                                next_x,
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
                            next_x,
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
                                        next_x,
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
                            next_x,
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
                        next_x,
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
                        next_x,
                        mask_tmp,
                        torch.ones_like(t),
                        -self.deriv_map[k] - 1,
                        patch,
                        -2,
                        )
                    ans = ans.where(~mask_tmp, A)
                idx_counter += 1

        return ans

    def compare_with_exact(
            self,
            exact_fun,
            return_error=False,
            nb_points=100,
            show_plot=True,
            p_or_u="u",
    ):
        """
        Plot the comparison among
        exact u solution, terminal condition,
        and the neural network approximation.

        Parameters
        ----------
        exact_fun : function
            The exact u function `exact_example(t, x, T, coordinate)`
            that takes t array of size `batch_size`,
            x array of d_in x size `batch_size`,
            T array of size `batch_size`,
            and `int` as input,
            and output array of size `batch_size`.
        """
        grid = np.linspace(self.x_lo, self.x_hi, nb_points)
        x_mid = (self.x_lo + self.x_hi) / 2
        grid_d_dim = np.concatenate((
            np.expand_dims(grid, axis=0),
            x_mid * np.ones((self.dim_in - 1, nb_points))
        ), axis=0)
        grid_d_dim_with_t = np.concatenate((self.t_lo * np.ones((1, nb_points)), grid_d_dim), axis=0)

        nn_input = grid_d_dim_with_t if p_or_u == "u" else grid_d_dim_with_t[1:]
        nn = (
            self(torch.tensor(
                nn_input.T, device=self.device, dtype=torch.get_default_dtype()
            ), patch=self.patches-1, p_or_u=p_or_u)
            .detach()
            .cpu()
            .numpy()
        )
        error = []
        for_range = self.dim_out if p_or_u == "u" else 1
        for i in range(for_range):
            f = plt.figure()
            true = exact_fun(
                self.t_lo if p_or_u == "u" else self.T,
                grid_d_dim,
                self.T,
                i
            )
            terminal = exact_fun(self.T, grid_d_dim, self.T, i)
            error.append(np.abs(true - nn[:, i]).mean())
            plt.plot(grid, nn[:, i], label="NN")
            plt.plot(grid, true, label="True solution")
            plt.plot(grid, terminal, label="Terminal solution")
            plt.title(f"Comparison with exact {p_or_u}{i} ")
            plt.legend()
            f.savefig(
                f"{self.working_dir_full_path}/plot/{p_or_u}{i}_comparison_with_exact.png", bbox_inches="tight"
            )
            if show_plot:
                plt.show()
            plt.close()

            data = np.stack((grid, true, terminal, nn[:, i])).T
            np.savetxt(
                f"{self.working_dir_full_path}/data/{p_or_u}{i}_comparison_with_exact.csv",
                data,
                delimiter=",",
                header="x,true,terminal,branch",
                comments=""
            )
        if return_error:
            return np.array(error)

    def log_plot_save(self, patch, epoch, loss, x, y, debug_mode=False, p_or_u="u"):
        """
        Log, plot, and save data in the current `epoch`.


        Parameters
        ----------
        patch : int
            The current patch index.

        epoch : int
            The current epoch index.

        loss : torch.Tensor
            The current training loss.

        x : torch.Tensor
            The generated MC samples.

        y : torch.Tensor
            The generated MC samples.

        debug_mode : bool, optional
            If `debug_mode = True`, we show the plot;
            otherwise the plot is saved in the directory `working_dir`.
            Default to be False.

        p_or_u : str, optional
            The operation on "p" or "u",
            default to be "u".
        """
        # loss info
        self.print_msg(f"Patch {patch:2.0f}: epoch {epoch:4.0f} with loss {loss.detach():.2E}")

        # if we do not always save for the best model, save it every 500 epochs
        if not self.save_for_best_model:
            torch.save(
                self.state_dict(), f"{self.working_dir_full_path}/checkpoint.pt"
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
        nn_mc = (
            self(x, patch=patch, p_or_u=p_or_u).detach().cpu()
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
                    f"{self.working_dir_full_path}/plot/{p_or_u}{i}_patch_{patch:02}_epoch_{epoch:04}.png", bbox_inches="tight"
                )
                if debug_mode:
                    plt.show()
                plt.close()

        # save data to csv
        if self.save_data:
            header = (
                    ("t," if p_or_u == "u" else "")
                    + "".join([f"x{i}," for i in range(self.dim_in)])
                    + "".join([f"mc{i}," for i in range(self.dim_out)])
                    + "".join([f"nn{i}," for i in range(self.dim_out)])
            )[:-1]
            data = np.concatenate((x.detach().cpu().numpy(), y.cpu().numpy(), nn_mc.numpy()), axis=-1)
            np.savetxt(
                f"{self.working_dir_full_path}/data/mc_samples_{p_or_u}_patch_{patch:02}_epoch_{epoch:04}.csv",
                data,
                delimiter=",",
                header=header,
                comments="",
            )
            header = (
                ("t," if p_or_u == "u" else "")
                + "".join([f"x{i}," for i in range(self.dim_in)])
                + "".join([f"nn{i}," for i in range(self.dim_out)])
            )[:-1]
            data = np.concatenate((grid_nd.T, nn), axis=-1)
            np.savetxt(
                f"{self.working_dir_full_path}/data/nn_patch_{p_or_u}_{patch:02}_epoch_{epoch:04}.csv",
                data,
                delimiter=",",
                header=header,
                comments="",
            )

    def gen_sample(self, patch, tx=None):
        """
        Generate samples for u based on `adjusted_t_boundaries`,
        `adjusted_x_boundaries` and the function `gen_sample_batch`.

        Parameters
        ----------
        patch : int
            The current patch index.
        """
        nb_states = self.nb_states
        states_per_batch = min(nb_states, self.nb_states_per_batch)
        batches = math.ceil(nb_states / states_per_batch) if tx is None else 1
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
            if tx is None:
                x = x_lo + (x_hi - x_lo) * unif
                # fix all dimensions (except the first) to be the middle value
                if self.dim_in > 1 and self.fix_all_dim_except_first:
                    x[1:, :, :] = (x_hi + x_lo) / 2
                xx.append(torch.cat((t[:, :1], x[:, :, 0].T), dim=-1).detach())
            else:
                t = tx[:, 0].unsqueeze(-1).repeat((1, self.nb_path_per_state))
                x = tx[:, 1:].T.unsqueeze(-1).repeat((1, 1, self.nb_path_per_state))
                xx.append(tx)
            T = (t_lo + self.delta_t) * torch.ones_like(t)
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
                mask = torch.logical_and(lo <= yy_tmp, yy_tmp <= hi)
                yyy.append((yy_tmp.nan_to_num() * mask).sum(dim=1) / mask.sum(dim=1))
            yy.append(torch.stack(yyy, dim=-1))

        return (
            torch.cat(xx),
            torch.cat(yy),
        )

    def gen_sample_for_p_batch(self, x, patch):
        """
        Inverse the Laplacian using Monte Carlo method.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of size `d_in x batch_size`.

        patch : int
            The current patch index.
        """
        x = x.detach().clone().requires_grad_(True)
        nb_mc = self.nb_path_per_state
        x = x.repeat(nb_mc, 1, 1)
        unif = (
            torch.rand(nb_mc * x.shape[1], device=self.device)
                 .reshape(nb_mc, -1, 1)
        )
        tau = self.tau_lo + (self.tau_hi - self.tau_lo) * unif
        next_x = self.gen_bm(x.transpose(0, -1), tau.transpose(0, -1), x.shape[1], var=1).transpose(0, -1)
        survive_prob = self.conditional_probability_to_survive_for_p(
            tau.reshape(-1),
            x.reshape(-1, self.dim_in).T,
            next_x.reshape(-1, self.dim_in).T
        ).clip(0, 1)
        survive_prob *= (
                self.is_x_inside_for_p(next_x.reshape(-1, self.dim_in).T)
                * self.is_x_inside_for_p(x.reshape(-1, self.dim_in).T)
                * (tau < self.tau_hi).squeeze(dim=-1).reshape(-1)
        )
        x = next_x.reshape(-1, self.dim_in).T
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
        ans *= survive_prob
        ans = ans.reshape(nb_mc, -1)
        # ans *= (y**2).sum(dim=-1)
        # if self.dim_in > 2:
        #     ans /= (self.dim_in - 2)
        # elif self.dim_in == 2:
        #     ans *= -torch.log((y**2).sum(dim=-1).sqrt())
        # ans *= ((self.tau_hi - self.tau_lo) / (2 * tau[:, :, 0]))
        ans *= ((self.tau_hi - self.tau_lo) / 2)

        mask = ~ans.isnan()
        return (
            ((ans.nan_to_num() * mask).sum(dim=0, keepdim=True) / mask.sum(dim=0, keepdim=True)).detach()
        )

    def gen_sample_for_p(self, patch):
        """
        Generate samples for p at terminal time based on
        the function `gen_sample_for_p_batch`.

        Parameters
        ----------
        patch : int
            The current patch index.

        overtrain_rate : float, optional
            Generate samples on
            the widened domain
            $$
            [\\text{x_lo - overtrain_rate} \\times \\text{(x_hi-x_lo)},
            \\text{x_hi + overtrain_rate} \\times \\text{(x_hi-x_lo)}].
            $$
        """
        self.nb_path_per_state *= 1000
        self.nb_states_per_batch //= 1000
        states_per_batch = min(self.nb_states, self.nb_states_per_batch)
        batches = math.ceil(self.nb_states / states_per_batch)
        xx, yy = [], []
        x_lo, x_hi = self.x_lo, self.x_hi
        x_lo, x_hi = x_lo - self.overtrain_rate_for_p * (x_hi - x_lo), x_hi + self.overtrain_rate_for_p * (x_hi - x_lo)
        start = time.time()
        for batch_now in range(batches):
            unif = (
                torch.rand(self.dim_in * states_per_batch, device=self.device)
                     .reshape(states_per_batch, self.dim_in)
            )
            x = (x_lo + (x_hi - x_lo) * unif).T
            if self.dim_in > 1 and self.fix_all_dim_except_first:
                x[1:, :] = (x_hi + x_lo) / 2
            y = self.gen_sample_for_p_batch(x.T, patch=patch)
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
        Train the neural network using the Monte Carlo samples generated by
        `gen_sample` and `gen_sample_for_p` and log the training information.

        Parameters
        ----------
        debug_mode : bool, optional
            If `debug_mode = True`, we show the plot;
            otherwise the plot is saved in the directory `working_dir`.
            Default to be False.

        return_dict : bool, optional
            Whether or not to return the dictionary
            containing the information of
            best training loss and total runtime.

        reuse_x : torch.Tensor, optional
            If a tensor is passed as `reuse_x`,
            we do not generate the Monte Carlo samples and use `x = reuse_x`.

        reuse_y : torch.Tensor, optional
            If a tensor is passed as `reuse_y`,
            we do not generate the Monte Carlo samples and use `y = reuse_y`.
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

                    if self.poisson_coeff > 0:
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
                        for i in range(self.dim_in):
                            order[i] += 2
                            poisson_lhs += self.nth_derivatives(
                                order, self(xx.T, p_or_u="p", patch=p), xx
                            )
                            order[i] -= 2
                        loss += self.poisson_coeff * self.loss(poisson_lhs, poisson_rhs)

                    # update model weights and schedule
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    if epoch > self.lr_milestones[-1] and loss.item() < best_loss:
                        # save the best model when NN enters stabilising zone
                        best_loss = loss.item()
                        if self.save_for_best_model:
                            torch.save(self.state_dict(), f"{self.working_dir_full_path}/checkpoint.pt")

                    # print loss information and plot every 500 epochs
                    if epoch % 500 == 0 or epoch + 1 == self.epochs:
                        self.log_plot_save(patch=p, epoch=epoch, loss=loss, x=x, y=y, debug_mode=debug_mode, p_or_u="p")

                if self.save_for_best_model:
                    self.load_state_dict(torch.load(f"{self.working_dir_full_path}/checkpoint.pt"))
                self.print_msg(
                    f"Patch{p}: pre-training of p with {self.epochs} epochs takes {time.time() - start:4.0f} seconds."
                )

            if self.train_for_u:
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
                    if self.div_condition_coeff > 0 and self.deriv_condition_zeta_map is not None:
                        grad = 0
                        xx = x.T.detach().clone().requires_grad_(True)
                        for (idx, c) in zip(self.deriv_condition_zeta_map, self.deriv_condition_deriv_map):
                            # additional t coordinate
                            grad += self.nth_derivatives(
                                np.insert(c, 0, 0), self(xx.T, patch=p)[:, idx], xx
                            )
                        loss += self.div_condition_coeff * self.loss(grad, torch.zeros_like(grad))

                    # update model weights and schedule
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    # save the best model when NN enters stabilising zone
                    if epoch > self.lr_milestones[-1] and loss.item() < best_loss:
                        best_loss = loss.item()
                        if self.save_for_best_model:
                            torch.save(self.state_dict(), f"{self.working_dir_full_path}/checkpoint.pt")

                    # print loss information and plot every 500 epochs
                    if epoch % 500 == 0 or epoch + 1 == self.epochs:
                        self.log_plot_save(patch=p, epoch=epoch, loss=loss, x=x, y=y, debug_mode=debug_mode, p_or_u="u")

                if self.save_for_best_model:
                    self.load_state_dict(torch.load(f"{self.working_dir_full_path}/checkpoint.pt"))
                self.print_msg(
                    f"Patch {p}: training of u with {self.epochs} epochs take {time.time() - start} seconds."
                )
            output_dict[f"patch_{p}"] = (time.time() - start, best_loss)
        if return_dict:
            return output_dict
