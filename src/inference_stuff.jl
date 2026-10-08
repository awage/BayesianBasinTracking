

"""
Posterior mean of the Shannon entropy of a Dirichlet(α) distribution,
    E[S] = ψ(α₀ + 1) − Σₖ (αₖ/α₀) ψ(αₖ + 1),    α₀ = Σₖ αₖ
"""
function bayes_entropy(alpha::AbstractDict{Int, Float64})
    a0 = sum(values(alpha))
    a0 > 0 || return 0.0
    term2 = 0.0
    for val in values(alpha)
        term2 += (val / a0) * digamma(val + 1)
    end
    return digamma(a0 + 1) - term2
end

"""
Exact posterior variance of the Shannon entropy under Dirichlet(α), following
Wolpert & Wolf (1995), Theorem 16 (Eqs. 16.1–16.2). The `i ≠ j` cross terms are
summed with the `(Σ)² − Σ(·²)` trick, so the cost is O(K) rather than O(K²).
"""
function bayes_entropy_variance(alpha::AbstractDict{Int, Float64})
    vals = values(alpha)
    a0 = sum(vals)
    a0 > 0 || return 0.0

    # Polygamma terms of the total, needed by every summand
    psi_a0_1 = digamma(a0 + 1)
    psi_a0_2 = digamma(a0 + 2)
    tri_a0_2 = trigamma(a0 + 2)

    E_S = 0.0      # E[S], Eq. 16.1
    sum_A = 0.0    # Σ Aᵢ, for the cross-term trick
    sum_A2 = 0.0   # Σ Aᵢ²
    sum_a_sq = 0.0 # Σ αᵢ²
    D = 0.0        # diagonal (i == j) terms of Eq. 16.2

    for a_i in vals
        a_i > 0 || continue
        E_S -= (a_i / a0) * (digamma(a_i + 1) - psi_a0_1)

        A_i = a_i * (digamma(a_i + 1) - psi_a0_2)
        sum_A += A_i
        sum_A2 += A_i^2
        sum_a_sq += a_i^2

        D += (a_i * (a_i + 1)) / (a0 * (a0 + 1)) *
             ((digamma(a_i + 2) - psi_a0_2)^2 + trigamma(a_i + 2) - tri_a0_2)
    end

    C = ((sum_A^2 - sum_A2) - tri_a0_2 * (a0^2 - sum_a_sq)) / (a0 * (a0 + 1))
    # Var = E[S²] − E[S]²; clamp away float noise around 0
    return max(0.0, (C + D) - E_S^2)
end


"""
Basin entropy of the whole region: the mean over boxes of their posterior mean
entropy. 
"""
mean_entropy(alphas::AbstractVector{<:AbstractDict{Int, Float64}}) =
    isempty(alphas) ? 0.0 : sum(bayes_entropy, alphas) / length(alphas)

"""
Variance of [`mean_entropy`](@ref). The boxes are treated as independent, so the
variance of their mean is `(1/N²) Σᵢ Var[Sᵢ]`.
"""
function mean_entropy_variance(alphas::AbstractVector{<:AbstractDict{Int, Float64}})
    n = length(alphas)
    n > 0 || return 0.0
    return sum(bayes_entropy_variance, alphas) / n^2
end


"""
Relative volume of each basin, as believed by the priors:

    V_k ≈ (1/N) Σᵢ αᵢₖ / αᵢ₀

Values sum to 1. This is the *prior-based* estimate, which carries the memory of
previous parameters through the sampler's forgetting factor λ. It is not the same
quantity as the `fractions_cont` returned by a global continuation, which
`Attractors.weighted_fractions` computes from the labels of the current parameter
only; 
"""
function basin_volumes(alphas::AbstractVector{<:AbstractDict{Int, Float64}})
    vol = Dict{Int, Float64}()
    n = length(alphas)
    n > 0 || return vol
    for alpha in alphas
        a0 = sum(values(alpha))
        a0 > 0 || continue
        for (k, a) in alpha
            vol[k] = get(vol, k, 0.0) + a / (a0 * n)
        end
    end
    return vol
end

"""
Posterior variance of each entry of [`basin_volumes`](@ref). Within one box,
Dirichlet(α) gives `Var[pₖ] = αₖ(α₀ − αₖ) / (α₀²(α₀ + 1))`; the boxes are averaged
as independent contributions, so `Var[V_k] = (1/N²) Σᵢ Var[pᵢₖ]`.
"""
function basin_volume_variance(alphas::AbstractVector{<:AbstractDict{Int, Float64}})
    var_vol = Dict{Int, Float64}()
    n = length(alphas)
    n > 0 || return var_vol
    for alpha in alphas
        a0 = sum(values(alpha))
        a0 > 0 || continue
        for (k, ak) in alpha
            var_vol[k] = get(var_vol, k, 0.0) + ak * (a0 - ak) / (a0^2 * (a0 + 1) * n^2)
        end
    end
    return var_vol
end

"""
    panic_boxes(etas) → Vector{Int}

The boxes that asked for a dense re-sample at a parameter, given their log Bayes
factors. An alarm is a *negative* η: the box's history explains its data worse than
no history at all.
"""
panic_boxes(etas::AbstractVector{<:Real}) = findall(<(0), etas)


"""
Every quantity the figures need, computed from the per-parameter record the sampler
kept during a [`global_continuation`](@ref). The sampler must have been built with
`history = true`; `history` may also be given directly as the
`(; alphas, etas)` of `Attractors.bayesian_sampler_history`.
"""
function bayes_estimates(history::NamedTuple)
    alphas, etas = history.alphas, history.etas
    isempty(alphas) && throw(ArgumentError(
        "the sampler kept no history; build it with `BayesianUpdateSampler(...; history = true)`"
    ))
    n_p, n_b = length(alphas), length(first(alphas))
    resets = hasproperty(history, :resets) ? history.resets : falses(n_p)
    return (;
        mean_S = [mean_entropy(a) for a in alphas],
        var_S = [mean_entropy_variance(a) for a in alphas],
        min_eta = [minimum(e) for e in etas],
        n_panics = [count(<(0), e) for e in etas],
        panic_boxes = [panic_boxes(e) for e in etas],
        global_resets = collect(resets),
        volumes = [basin_volumes(a) for a in alphas],
        vol_var = [basin_volume_variance(a) for a in alphas],
        full_S = [bayes_entropy(alphas[i][j]) for i in 1:n_p, j in 1:n_b],
        full_eta = [etas[i][j] for i in 1:n_p, j in 1:n_b],
    )
end

bayes_estimates(sampler::BayesianUpdateSampler) = bayes_estimates(bayesian_sampler_history(sampler))

"""
Helper function to get the band diagram correctly.
"""
function volume_series(volumes::AbstractVector{<:AbstractDict},
                       labels = sort!(collect(reduce(union, keys.(volumes)))))
    return Dict{Int, Vector{Float64}}(
        Int(k) => [Float64(get(v, k, 0.0)) for v in volumes] for k in labels
    )
end
