# BayesianBasinTracking

Code and figures for

> *Bayesian Basin Tracking: Efficient Global Continuation of Multistable Dynamical Systems*
> P. Haerter, A. Wagemakers, A. Daza, G. Datseris, M. A. F. Sanjuán — [arXiv:2607.14762](https://arxiv.org/abs/2607.14762)

## Summary

Mapping the basins of a multistable system at every parameter value is expensive, because
each parameter is usually sampled from scratch. Basin boundaries, however, move slowly:
what was learned at one parameter is a good prior for the next. Bayesian Basin Tracking
exploits this by tiling the region of interest into boxes and giving each box a
Dirichlet–multinomial posterior over which attractors it contains. Along a continuation the
posteriors are carried forward with a forgetting factor λ and updated with a sparse sample
of initial conditions. A log Bayes factor η per box compares the carried-over prior against
the fresh data; a negative η means the box's history no longer explains what it sees, and
triggers a dense re-sample there. Bifurcations are thus detected automatically and paid for
only where and when they happen, giving the full basin fractions and basin entropy — with
posterior error bars — from a fraction of the usual simulations (an almost sixfold speed-up
on a 300-dimensional Rössler network).

The sampler itself (`BayesianUpdateSampler`, tiling, sampling, η, panic logic) lives
upstream in [Attractors.jl](https://github.com/JuliaDynamics/Attractors.jl). This repository
holds what is specific to the paper.

Results land in `plots/`, cached simulations in `data/`.

## Reproducing

This code base is using the [Julia Language](https://julialang.org/) and
[DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/)
to make a reproducible scientific project named
> BayesianBasinTracking

To (locally) reproduce this project, do the following:

0. Download this code base. Notice that raw data are typically not included in the
   git-history and may need to be downloaded independently.
1. Open a Julia console and do:
   ```
   julia> using Pkg
   julia> Pkg.add("DrWatson") # install globally, for using `quickactivate`
   julia> Pkg.activate("path/to/this/project")
   julia> Pkg.instantiate()
   ```

This will install all necessary packages for you to be able to run the scripts and
everything should work out of the box, including correctly finding local paths.

You may notice that most scripts start with the commands:
```julia
using DrWatson
@quickactivate "BayesianBasinTracking"
```
which auto-activate the project and enable local path handling from DrWatson.
