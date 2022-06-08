# Deep branching solver for PDE system
Authors: Jiang Yu Nguwi and Nicolas Privault.

If this code is used for research purposes, please cite as \
J.Y. Nguwi, G. Penent, and N. Privault.
A deep branching solver for fully nonlinear partial differential equations.
*arXiv preprint arXiv:2203.03234*, 2022.
<br/><br/>

Deep branching solver based on [[NPP22]](#nguwi2022deepbranching)
aims to solve system of fully nonlinear PDE system of the form\
<img src="https://latex.codecogs.com/svg.image?&space;&space;&space;&space;\partial_t&space;u_i(t,&space;x)&space;&plus;&space;\frac{\nu}{2}&space;\Delta&space;u_i(t,&space;x)&space;&space;&space;&space;&plus;&space;f_i((\partial_{\alpha^j}&space;u_{\zeta^j}(t,&space;x))_{0&space;\le&space;j&space;<&space;n})&space;=&space;0,&space;&space;&space;&space;\quad&space;0&space;\le&space;i&space;<&space;d_{out}," />\
with\
<img src="https://latex.codecogs.com/svg.image?&space;&space;&space;u_i(T,&space;x)&space;=&space;\phi(x)," />\
the Poisson equation \
<img src="https://latex.codecogs.com/svg.image?&space;&space;&space;&space;\Delta&space;u_{-1}(t,&space;x)&space;=&space;&space;&space;&space;-\sum\limits_{i=0}^{d_{out}-1}&space;&space;&space;&space;\partial_{1_i}&space;u_j(t,&space;x)&space;&space;&space;&space;\partial_{1_j}&space;u_i(t,&space;x)," />\
and the derivatives condition \
<img src="https://latex.codecogs.com/svg.image?&space;&space;&space;&space;\sum\limits_{i&space;=&space;0}^{m&space;-&space;1}&space;&space;&space;&space;\partial_{\alpha_{deriv}^i}&space;u_{\zeta_{deriv}^i}(t,&space;x)&space;=&space;0,&space;&space;&space;&space;\quad&space;(t,&space;x)&space;\in&space;[t_{lo},&space;T]&space;\times&space;\Omega,&space;&space;&space;&space;\quad&space;\Omega&space;\subset&space;\mathbb{R}^{d_{in}}." />

## Quick Start
* <a href="https://nbviewer.org/github/nguwijy/deep_branching_with_domain/blob/main/notebooks/index.ipynb"><img src="https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg" alt="Render nbviewer" /></a>

* [github.com's notebook viewer](https://github.com/nguwijy/deep_branching_with_domain/blob/main/notebooks/index.ipynb) also works but it's not ideal: it's slower, the math equations are not always displayed correctly, and large notebooks often fail to open.

## Using deep branching solver
There are two ways to utilize the deep branching solver:
1. Edit the templates of `main.py`, then run `python main.py` from your terminal.
2. Write your own code and import the solver to your code
    via `from branch.branch import Net`,
    see the notebooks for more details.
    This requires the parent directory to be in your system path.

It is highly recommended to read the documentation
via `help(Net)` or [this html page](https://rawcdn.githack.com/nguwijy/deep_branching_with_domain/main/doc/branch.html).

## Notebooks
There are three python notebooks in this repo:
1. `deep_branching.ipynb` presents 6 PDE examples
       with `d_out=1` and without boundary condition.
       In addition, deep branching solver is compared to
       the deep BSDE method [[HJE18]](#han2018solving) and
       the deep Galerkin method [[SS18]](#sirignano2018dgm).
2. `deep_navier_stokes.ipynb` presents 2 Navier-Stokes examples
       without boundary condition.
3. `deep_branching_with_domain.ipynb` presents PDE examples
       with boundary condition.

## References
<a id="han2018solving">[HJE18]</a>
J. Han, A. Jentzen, and W. E.
Solving high-dimensional partial differential equations using deep
learning.
*Proceedings of the National Academy of Sciences*,
115(34):8505--8510, 2018.

<a id="nguwi2022deepbranching">[NPP22]</a>
J.Y. Nguwi, G. Penent, and N. Privault.
A deep branching solver for fully nonlinear partial differential equations.
*arXiv preprint arXiv:2203.03234*, 2022.

<a id="sirignano2018dgm">[SS18]</a>
J. Sirignano and K. Spiliopoulos.
DGM: A deep learning algorithm for solving partial differential
equations.
*Journal of computational physics*,
375:1339--1364, 2018.
