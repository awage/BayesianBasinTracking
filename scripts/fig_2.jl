"""
Bayesian basin monitoring of the Hénon map

    x_{n+1} = a - x_n² + b y_n
    y_{n+1} = x_n
"""

using DrWatson
@quickactivate "BayesianBasinTracking"
using LinearAlgebra
using Statistics
using StaticArrays
using SpecialFunctions
using Attractors

include(srcdir("inference_stuff.jl"))

@inline @inbounds function henon_rule(u, p, n)
    a = p[1]; b = p[2]
    return SVector{2}(a - u[1]^2 + b * u[2], u[1])
end

function henon_bayes_continuation(params)
    @unpack a_range, b, sparse_n, dense_n, n_tiles, global_bounds, λ, β, global_reset = params

    grid_rec = (range(-4, 4; length = 1000), range(-4, 4; length = 1000))

    ds = DeterministicIteratedMap(henon_rule, [0.0, 0.0], [first(a_range), b])
    bmap = BasinMapRecurrences(ds, grid_rec;
        consecutive_recurrences = 50000, Ttr = 5000, show_progress = false)

    sampler = BayesianUpdateSampler(global_bounds, n_tiles;
        sparse_n, dense_n, λ, β, global_reset, seed = 20260802, history = true,
    )

    pcurve = [Dict(1 => a, 2 => b) for a in a_range]

    algo = RecurrencesFindAndMatch(bmap; distance = Hausdorff(), threshold = Inf)
    fractions, attractors = global_continuation(algo, pcurve, sampler)

    est = bayes_estimates(sampler)
    return @strdict(
        fractions,
        mean_S = est.mean_S, var_S = est.var_S,
        min_eta = est.min_eta, n_panics = est.n_panics,
        global_resets = est.global_resets,
        volumes = est.volumes, vol_var = est.vol_var,
        full_S = est.full_S, full_eta = est.full_eta,
    )
end

λ = 0.7
β = 0.5
sparse_n = 20
dense_n = sparse_n^2

n_tiles = 15
global_bounds = ((-2.0, 2.0), (-2.0, 2.0))

b = -0.3

ai = 1.0
af = 2.0
len = 100
a_range = range(ai, af, length = len)

global_reset = false

params = @strdict a_range b sparse_n dense_n n_tiles global_bounds λ β global_reset

data, file = produce_or_load(
    datadir("data"),
    params,
    henon_bayes_continuation;
    prefix = "henon_bayes", storepatch = false,
    suffix = "jld2", force = false,
    filename = hash
)

@unpack mean_S, var_S, min_eta, n_panics, global_resets, full_S, volumes, fractions = data

println("Done. Mean entropy range: ", extrema(mean_S))
println("Minimum η range: ", extrema(min_eta))
println("Global resets at a = ", [round(a, digits = 3) for a in a_range[global_resets]])

using CairoMakie

lab_args = (; yticklabelsize = 20, xticklabelsize = 20, ylabelsize = 25, xlabelsize = 25)

all_labels = sort!(Int.(reduce(union, keys.(fractions))))
vol_series = volume_series(fractions, all_labels)

fig = Figure(size = (800, 1000))

upper_band = mean_S .+ (3.0 .* sqrt.(var_S))
lower_band = mean_S .- (3.0 .* sqrt.(var_S))

ax1 = Axis(fig[1, 1]; ylabel = L"S_b", lab_args...)
lines!(ax1, a_range, mean_S, color = :black)
xlims!(ax1, ai, af)
band!(ax1, a_range, lower_band, upper_band,
        color = (:black, 0.2),
        label = "Confidence (±3σ)")
Label(fig[1, 1, TopLeft()], "(a)",
        fontsize = 25,
        padding = (0, 50, -10, 0),
        halign = :right)

ax2 = Axis(fig[2, 1]; ylabel = "# alarms", lab_args...)
# vlines!(ax2, a_range[global_resets], color = (:dodgerblue, 0.9), linewidth = 2,
        # label = "global reset")
stairs!(ax2, a_range, n_panics, color = :red)
# axislegend(ax2, position = :lt)
xlims!(ax2, ai, af)
Label(fig[2, 1, TopLeft()], "(b)",
        fontsize = 25,
        padding = (0, 50, -10, 0),
        halign = :right)

colors = Makie.wong_colors()
ax3 = Axis(fig[3, 1]; ylabel = "Basin fraction", xlabel = L"a", lab_args...)
let lower = zeros(length(a_range))
    for (i, k) in enumerate(all_labels)
        upper = lower .+ vol_series[k]
        band!(ax3, a_range, lower, upper,
              color = (colors[mod1(i, length(colors))], 0.8),
              label = "Basin $k")
        lines!(ax3, a_range, upper, color = colors[mod1(i, length(colors))], linewidth = 0.8)
        lower = copy(upper)
    end
end
axislegend(ax3, position = :rt)
xlims!(ax3, ai, af)
ylims!(ax3, 0, 1)
Label(fig[3, 1, TopLeft()], "(c)",
        fontsize = 25,
        padding = (0, 50, -10, 0),
        halign = :right)

save(plotsdir("fig2.png"), fig)
