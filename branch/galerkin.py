import math
import time
import torch
import torch.nn.functional as F
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
        **kwargs,
    ):
        super(DGMNet, self).__init__()
        self.f_fun = dgm_f_fun
        self.n, self.dim = dgm_deriv_map.shape
        # add one more dimension of time to the left of deriv_map
        self.deriv_map = np.append(np.zeros((self.n, 1)), dgm_deriv_map, axis=-1)
        self.zeta_map = dgm_zeta_map if dgm_zeta_map is not None else np.zeros(self.n, dtype=int)
        self.deriv_condition_deriv_map = deriv_condition_deriv_map
        self.deriv_condition_zeta_map = deriv_condition_zeta_map
        self.dim_out = dim_out if dim_out is not None else self.zeta_map.max() + 1
        self.coordinate = np.array(range(self.dim_out))
        # add dt to the top of deriv_map
        self.deriv_map = np.append(
            np.array([[1] + [0] * self.dim]), self.deriv_map, axis=0
        )
        # the final deriv_map has the shape of (n + 1) x (dim + 1)

        self.phi_fun = phi_fun
        self.u_layer = torch.nn.ModuleList(
            [torch.nn.Linear(self.dim + 1, neurons, device=device)]
            + [torch.nn.Linear(neurons, neurons, device=device) for _ in range(layers)]
            + [torch.nn.Linear(neurons, self.dim_out, device=device)]
        )
        self.u_bn_layer = torch.nn.ModuleList(
            [torch.nn.BatchNorm1d(neurons, device=device) for _ in range(layers + 2)]
        )
        self.p_layer = torch.nn.ModuleList(
            [torch.nn.Linear(self.dim + 1, neurons, device=device)]
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

    def forward(self, x, coordinate):
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
        return y[:, coordinate]

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
        unif = torch.rand(self.nb_states * self.dim, device=self.device).reshape(
            self.dim, -1
        )
        x = self.x_lo + (self.x_hi - self.x_lo) * unif
        tx = torch.cat((t.unsqueeze(0), x), dim=0)

        # sample for initial time, to be merged with intermediate value
        t = self.t_lo * torch.ones(self.nb_states, device=self.device)
        unif = torch.rand(self.nb_states * self.dim, device=self.device).reshape(
            self.dim, -1
        )
        x = self.x_lo + (self.x_hi - self.x_lo) * unif
        tx = torch.cat([tx, torch.cat((t.unsqueeze(0), x), dim=0)], dim=-1)

        # sample for terminal time
        t = self.t_hi * torch.ones(self.nb_states, device=self.device)
        unif = torch.rand(self.nb_states * self.dim, device=self.device).reshape(
            self.dim, -1
        )
        x = self.x_lo + (self.x_hi - self.x_lo) * unif
        tx_term = torch.cat((t.unsqueeze(0), x), dim=0)

        # fix all dimensions (except the first) to be the middle value
        if self.dim > 1 and self.fix_all_dim_except_first:
            x_mid = (self.x_hi + self.x_lo) / 2
            tx[2:, :] = x_mid
            tx_term[2:, :] = x_mid

        return tx, tx_term

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
                            x_mid * np.ones((self.dim - 1, 100)),
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
                    print(f"Epoch {epoch} with loss {loss.detach()}")
        if self.verbose:
            print(
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
