import os
import math
import time
import torch
import torch.nn.functional as F
import logging
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(0)  # set seed for reproducibility


class DGMNet(torch.nn.Module):
    """
    deep Galerkin approach to solve PDE with utility functions
    """
    def __init__(
        self,
        dgm_f_fun,
        dgm_deriv_map,
        problem_name="tmp",
        dgm_zeta_map=None,
        deriv_condition_deriv_map=None,
        deriv_condition_zeta_map=None,
        dim_out=None,
        phi_fun=(lambda x: x),
        x_lo=-10.0,
        x_hi=10.0,
        overtrain_rate=0.1,
        t_lo=0.0,
        t_hi=1.0,
        neurons=20,
        layers=5,
        dgm_lr=1e-3,
        batch_normalization=False,
        weight_decay=0,
        dgm_nb_states=1000,
        epochs=3000,
        device="cpu",
        dgm_activation="tanh",
        verbose=False,
        fix_all_dim_except_first=False,
        save_as_tmp=False,
        **kwargs,
    ):
        super(DGMNet, self).__init__()
        self.f_fun = dgm_f_fun
        self.n, self.dim_in = dgm_deriv_map.shape
        # add one more dimension of time to the left of deriv_map
        self.deriv_map = np.append(np.zeros((self.n, 1)), dgm_deriv_map, axis=-1)
        self.zeta_map = dgm_zeta_map if dgm_zeta_map is not None else np.zeros(self.n, dtype=int)
        self.deriv_condition_deriv_map = deriv_condition_deriv_map
        self.deriv_condition_zeta_map = deriv_condition_zeta_map
        self.dim_out = dim_out if dim_out is not None else self.zeta_map.max() + 1
        self.coordinate = np.array(range(self.dim_out))
        # add dt to the top of deriv_map
        self.deriv_map = np.append(
            np.array([[1] + [0] * self.dim_in]), self.deriv_map, axis=0
        )
        # the final deriv_map has the shape of (n + 1) x (dim + 1)

        self.phi_fun = phi_fun
        self.u_layer = torch.nn.ModuleList(
            [torch.nn.Linear(self.dim_in + 1, neurons, device=device)]
            + [torch.nn.Linear(neurons, neurons, device=device) for _ in range(layers)]
            + [torch.nn.Linear(neurons, self.dim_out, device=device)]
        )
        self.u_bn_layer = torch.nn.ModuleList(
            [torch.nn.BatchNorm1d(neurons, device=device) for _ in range(layers + 2)]
        )
        self.p_layer = torch.nn.ModuleList(
            [torch.nn.Linear(self.dim_in + 1, neurons, device=device)]
            + [torch.nn.Linear(neurons, neurons, device=device) for _ in range(layers)]
            + [torch.nn.Linear(neurons, 1, device=device)]
        )
        self.p_bn_layer = torch.nn.ModuleList(
            [torch.nn.BatchNorm1d(neurons, device=device) for _ in range(layers + 2)]
        )
        self.lr = dgm_lr
        self.weight_decay = weight_decay

        self.loss = torch.nn.MSELoss()
        self.activation = {
            "tanh": torch.nn.Tanh(),
            "relu": torch.nn.ReLU(),
            "softplus": torch.nn.ReLU(),
        }[dgm_activation]
        self.batch_normalization = batch_normalization
        self.nb_states = dgm_nb_states
        self.ori_x_lo = x_lo
        self.ori_x_hi = x_hi
        x_lo, x_hi = (
            x_lo - overtrain_rate * (x_hi - x_lo),
            x_hi + overtrain_rate * (x_hi - x_lo),
        )
        self.x_lo = x_lo
        self.x_hi = x_hi
        self.t_lo = t_lo
        self.t_hi = t_hi
        self.epochs = epochs
        self.device = device
        self.verbose = verbose
        self.fix_all_dim_except_first = fix_all_dim_except_first

        timestr = time.strftime("%Y%m%d-%H%M%S")  # current time stamp
        self.working_dir = (
            "logs/tmp" if save_as_tmp
            else f"logs/{timestr}-{problem_name}"
        )
        self.working_dir_full_path = os.path.join(
            os.getcwd(),
            self.working_dir,
        )
        self.log_config()
        self.eval()

    def forward(self, x, coordinate=0, all_u=False):
        """
        self(x) evaluates the neural network approximation NN(x)
        """
        # -1 corresponds to p_layer
        # else 0 to dim_out corresponds to u_layer
        layer = self.u_layer if coordinate >= 0 else self.p_layer
        bn_layer = self.u_bn_layer if coordinate >= 0 else self.p_bn_layer
        coordinate = 0 if coordinate < 0 else coordinate

        if self.batch_normalization:
            x = bn_layer[0](x)
        y = x
        for idx, (f, bn) in enumerate(zip(layer[:-1], bn_layer)):
            tmp = f(y)
            tmp = self.activation(tmp)
            if self.batch_normalization:
                tmp = bn(tmp)
            if idx == 0:
                y = tmp
            else:
                # resnet
                y = tmp + y

        y = layer[-1](y)
        if all_u:
            return y
        else:
            return y[:, coordinate]

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

    def pde_loss(self, x, coordinate):
        """"
        calculate the PDE loss partial_t u + f
        """
        x = x.detach().clone().requires_grad_(True)
        # recall that deriv_map has the shape of (n + 1) x (dim + 1)
        # with deriv_map[0] representing du/dt
        # zeta_map has the shape of n
        dt = self.nth_derivatives(self.deriv_map[0], self(x.T, coordinate=coordinate), x)

        fun_and_derivs = []
        for order, zeta in zip(self.deriv_map[1:], self.zeta_map):
            fun_and_derivs.append(self.nth_derivatives(order, self(x.T, coordinate=zeta), x))

        fun_and_derivs = torch.stack(fun_and_derivs)
        return self.loss(dt + self.f_fun(fun_and_derivs, coordinate=coordinate), torch.zeros_like(dt))

    def gen_sample(self):
        """
        generate (uniform) sample based on the (t_lo, t_hi) x (x_lo, x_hi)
        """
        # sample for intermediate value
        unif = torch.rand(self.nb_states, device=self.device)
        t = self.t_lo + (self.t_hi - self.t_lo) * unif
        unif = torch.rand(self.nb_states * self.dim_in, device=self.device).reshape(
            self.dim_in, -1
        )
        x = self.x_lo + (self.x_hi - self.x_lo) * unif
        tx = torch.cat((t.unsqueeze(0), x), dim=0)

        # sample for initial time, to be merged with intermediate value
        t = self.t_lo * torch.ones(self.nb_states, device=self.device)
        unif = torch.rand(self.nb_states * self.dim_in, device=self.device).reshape(
            self.dim_in, -1
        )
        x = self.x_lo + (self.x_hi - self.x_lo) * unif
        tx = torch.cat([tx, torch.cat((t.unsqueeze(0), x), dim=0)], dim=-1)

        # sample for terminal time
        t = self.t_hi * torch.ones(self.nb_states, device=self.device)
        unif = torch.rand(self.nb_states * self.dim_in, device=self.device).reshape(
            self.dim_in, -1
        )
        x = self.x_lo + (self.x_hi - self.x_lo) * unif
        tx_term = torch.cat((t.unsqueeze(0), x), dim=0)

        # fix all dimensions (except the first) to be the middle value
        if self.dim_in > 1 and self.fix_all_dim_except_first:
            x_mid = (self.x_hi + self.x_lo) / 2
            tx[2:, :] = x_mid
            tx_term[2:, :] = x_mid

        return tx, tx_term

    @staticmethod
    def latex_print(tensor):
        mess = ""
        for i in tensor[:-1]:
            mess += f"& {i.item():.2E} "
        mess += "& --- \\\\"
        logging.info(mess)

    def error_calculation(self, exact_u_fun, exact_p_fun, nb_pts_time=11, nb_pts_spatial=2*126+1, error_multiplier=1):
        x = np.linspace(self.ori_x_lo, self.ori_x_hi, nb_pts_spatial)
        t = np.linspace(self.t_lo, self.t_hi, nb_pts_time)
        arr = np.array(np.meshgrid(*([x]*self.dim_in + [t]))).T.reshape(-1, self.dim_in + 1)
        arr[:, [-1, 0]] = arr[:, [0, -1]]
        arr = torch.tensor(arr, device=self.device, dtype=torch.get_default_dtype())
        error = []
        nn = []
        cur, batch_size, last = 0, 100000, arr.shape[0]
        while cur < last:
            nn.append(self(arr[cur:min(cur+batch_size, last)], coordinate=0, all_u=True).detach())
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
                        self(xx, coordinate=i).sum(),
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
            ((self.ori_x_hi - self.ori_x_lo)**self.dim_in * numerator.mean(dim=-1)).sqrt()
        )

        del grad, xx
        torch.cuda.empty_cache()
        arr = arr.reshape(nb_pts_time, -1, self.dim_in + 1)[-1].detach()
        nn = []
        cur, batch_size, last = 0, 100000, arr.shape[0]
        while cur < last:
            nn.append(
                self(
                    arr[cur:min(cur+batch_size, last), :],
                    coordinate=-1,
                ).detach()
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

    def compare_with_exact(
        self,
        exact_fun,
        return_error=False,
        nb_points=100,
        show_plot=True,
        exclude_terminal=False,
        ylim=None,
    ):
        grid = np.linspace(self.ori_x_lo, self.ori_x_hi, nb_points)
        x_mid = (self.ori_x_lo + self.ori_x_hi) / 2
        grid_d_dim = np.concatenate((
            np.expand_dims(grid, axis=0),
            x_mid * np.ones((self.dim_in - 1, nb_points))
        ), axis=0)
        grid_d_dim_with_t = np.concatenate((self.t_lo * np.ones((1, nb_points)), grid_d_dim), axis=0)

        nn_input = grid_d_dim_with_t
        error = []
        for_range = self.dim_out
        for i in range(for_range):
            f = plt.figure()
            true = exact_fun(
                self.t_lo,
                grid_d_dim,
                self.t_hi,
                i
            )
            terminal = exact_fun(self.t_hi, grid_d_dim, self.t_hi, i)
            nn = (
                self(torch.tensor(
                    nn_input.T, device=self.device, dtype=torch.get_default_dtype()
                ), coordinate=i)
                .detach()
                .cpu()
                .numpy()
            )
            error.append(np.abs(true - nn).mean())
            plt.plot(grid, nn, label="NN")
            plt.plot(grid, true, label="True solution")
            if not exclude_terminal:
                plt.plot(grid, terminal, label="Terminal solution")
            plt.xlabel("$x_1$")
            plt.ylabel(f"$u_{i+1}$")
            plt.legend()
            if ylim is not None and i == 0:
                # only change ylim for u0
                plt.ylim(*ylim)
            f.savefig(
                f"{self.working_dir_full_path}/plot/dgm_u{i}_comparison_with_exact.png", bbox_inches="tight"
            )
            if show_plot:
                plt.show()
            plt.close()

            data = np.stack((grid, true, terminal, nn)).T
            np.savetxt(
                f"{self.working_dir_full_path}/data/dgm_u{i}_comparison_with_exact.csv",
                data,
                delimiter=",",
                header="x,true,terminal,branch",
                comments=""
            )
        if return_error:
            return np.array(error)

    def train_and_eval(self, debug_mode=False):
        """
        generate sample and evaluate (plot) NN approximation when debug_mode=True
        """
        # initialize optimizer
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        start = time.time()
        self.train()  # training mode

        # loop through epochs
        for epoch in range(self.epochs):
            tx, tx_term = self.gen_sample()

            # clear gradients and evaluate training loss
            optimizer.zero_grad()

            loss = 0
            for idx in self.coordinate:
                # terminal loss + pde loss
                loss = self.loss(self(tx_term.T, coordinate=idx), self.phi_fun(tx_term[1:, :], coordinate=idx))
                loss = loss + self.pde_loss(tx, coordinate=idx)

            # divergence free condition
            if self.deriv_condition_zeta_map is not None:
                grad = 0
                xx = tx.detach().clone().requires_grad_(True)
                for (idx, c) in zip(self.deriv_condition_zeta_map, self.deriv_condition_deriv_map):
                    # additional t coordinate
                    grad += self.nth_derivatives(
                        np.insert(c, 0, 0), self(xx.T, coordinate=idx), xx
                    )

            # update model weights
            loss.backward()
            optimizer.step()

            # print loss information every 500 epochs
            if epoch % 500 == 0 or epoch + 1 == self.epochs:
                if debug_mode:
                    grid = np.linspace(self.x_lo, self.x_hi, 100).astype(np.float32)
                    x_mid = (self.x_lo + self.x_hi) / 2
                    grid_nd = np.concatenate(
                        (
                            self.t_lo * np.ones((1, 100)),
                            np.expand_dims(grid, axis=0),
                            x_mid * np.ones((self.dim_in - 1, 100)),
                        ),
                        axis=0,
                    ).astype(np.float32)
                    self.eval()
                    for idx in self.coordinate:
                        nn = (
                            self(torch.tensor(grid_nd.T, device=self.device), coordinate=idx)
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        plt.plot(grid, nn)
                        plt.title(f"DGM approximation of coordinate {idx} at epoch {epoch}.")
                        plt.show()
                    self.train()
                if self.verbose:
                    logging.info(f"Epoch {epoch} with loss {loss.detach()}")

        torch.save(
            self.state_dict(), f"{self.working_dir_full_path}/checkpoint.pt"
        )
        if self.verbose:
            logging.info(
                f"Training of neural network with {self.epochs} epochs take {time.time() - start} seconds."
            )
        self.eval()


if __name__ == "__main__":
    # configurations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T, x_lo, x_hi, nu = .25, 0, 2 * math.pi, 2
    # deriv_map is n x d array defining lambda_1, ..., lambda_n
    deriv_map = np.array(
        [
            [1, 0],  # for nabla p
            [0, 1],
            [0, 0],  # for u
            [0, 0],
            [1, 0],  # for nabla u1
            [0, 1],
            [1, 0],  # for nabla u2
            [0, 1],
            [2, 0],  # for Laplacian
            [0, 2],
        ]
    )
    zeta_map = np.array([-1, -1, 0, 1, 0, 0, 1, 1, 0, 1])
    deriv_condition_deriv_map = np.array(
        [
            [1, 0],
            [0, 1],
        ]
    )
    deriv_condition_zeta_map = np.array([0, 1])
    _, dim = deriv_map.shape

    def f_example(y, coordinate):
        """
        idx 0 -> no deriv
        """
        f = -y[coordinate]
        for j in range(dim):
            f += -y[dim + j] * y[2 * dim + dim * coordinate + j]
            # Laplacian
            f += nu / 2 * (y[2 * dim + dim * dim + j])
        return f

    def phi_example(x, coordinate):
        if coordinate == 0:
            return -torch.cos(x[0]) * torch.sin(x[1])
        else:
            return torch.sin(x[0]) * torch.cos(x[1])


    # initialize model and training
    model = DGMNet(
        dgm_f_fun=f_example,
        phi_fun=phi_example,
        dgm_deriv_map=deriv_map,
        dgm_zeta_map=zeta_map,
        deriv_condition_deriv_map=deriv_condition_deriv_map,
        deriv_condition_zeta_map=deriv_condition_zeta_map,
        t_hi=T,
        x_lo=x_lo,
        x_hi=x_hi,
        device=device,
        verbose=True,
    )
    model.train_and_eval(debug_mode=True)

    # define exact solution and plot the graph
    def exact_fun(t, x, T):
        return np.log(1 + (x.sum(axis=0) + dim * (T - t)) ** 2)

    grid = torch.linspace(x_lo, x_hi, 100).unsqueeze(dim=-1)
    nn_input = torch.cat((torch.zeros((100, 1)), grid, torch.zeros((100, 2))), dim=-1)
    plt.plot(grid, model(nn_input).detach(), label="Deep Galerkin")
    plt.plot(grid, exact_fun(0, nn_input[:, 1:].numpy().T, T), label="True solution")
    plt.legend()
    plt.show()
