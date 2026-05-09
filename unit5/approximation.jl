# Plots for approximate inference for a 1D Gaussian
#
# 2026 by Ralf Herbrich
# Hasso-Plattner Institute

module Approximation

using Random
using LinearAlgebra
using Distributions
using LaTeXStrings
using Plots

struct DoublyTrunactedGaussian
    lower::Float64
    upper::Float64
    normal::Normal
end

"""
    plot_approximation(μ = 0, σ = 1)

Generates a plot of the true and approximate posterior (marginal) and likelihood (message) for a 1D truncated (at zero) Gaussian. `μ` and `σ` are the mean and variance of the prior, respectively.
"""
function plot_approximation(μ=0, σ=1)
    # adds a plot for a function
    function plot_function(f; color=:blue, style=:solid)
        plot!(xs, f, linewidth=3, color=color, style=style)
    end

    # compute the mean and variance of the best approximation
    t = μ / σ
    v = pdf(Normal(), t) / cdf(Normal(), t)
    μ_approx_posterior = μ + σ * v
    σ_approx_posterior = sqrt(σ * σ * (1 - (v * (v + t))))

    τ_prior = μ / (σ^2)
    ρ_prior = 1 / (σ^2)
    τ_approx_posterior = μ_approx_posterior / (σ_approx_posterior^2)
    ρ_approx_posterior = 1 / (σ_approx_posterior^2)
    τ_approx_likelihood = τ_approx_posterior - τ_prior
    ρ_approx_likelihood = ρ_approx_posterior - ρ_prior
    μ_approx_likelihood = τ_approx_likelihood / ρ_approx_likelihood
    σ_approx_likelihood = sqrt(1 / ρ_approx_likelihood)

    println(
        "σ2 (prior) = ",
        σ^2,
        "   σ2 (likel) = ",
        σ_approx_likelihood^2,
        "    σ2 (posterior) = ",
        σ_approx_posterior^2,
    )

    # draw all the plots
    d = Normal(μ, σ)
    xs = range(start=μ - 3 * σ, stop=μ + 3 * σ, length=1000)
    # generic plot parameter
    p = plot(
        legend=false,
        color=:blue,
        xtickfontsize=14,
        ytickfontsize=14,
        xguidefontsize=16,
        yguidefontsize=16,
    )
    xlabel!(L"x")
    ylabel!(L"p(x)")

    # plot the prior PDF
    plot_function(x -> pdf(d, x); color=:green)
    # plot the factor function (true likelihood)
    plot_function(x -> (x > 0) ? 1 : 0; color=:red)
    # plot the true posterior PDF
    Z = 1 - cdf(d, 0)
    plot_function(x -> (x > 0) ? (1 / Z * pdf(d, x)) : 0; color=:black)

    # plot the approximate PDF of the posterior
    plot_function(
        x -> pdf(Normal(μ_approx_posterior, σ_approx_posterior), x);
        color=:black,
        style=:dash,
    )
    # plot the approximate PDF of the posterior
    plot_function(
        x -> pdf(Normal(μ_approx_likelihood, σ_approx_likelihood), x);
        color=:red,
        style=:dash,
    )

    display(p)
end

"""
    plot_double_truncated(dt::DoublyTrunactedGaussian)

Generates a plot of the density of a double truncated Gaussian
"""
function plot_double_truncated(dt::DoublyTrunactedGaussian)
    function double_truncated_pdf(x)
        if x < dt.lower || x > dt.upper
            return 0.0
        else
            return pdf(dt.normal, x) / (cdf(dt.normal, dt.upper) - cdf(dt.normal, dt.lower))
        end
    end

    x_min, x_max = dt.lower - 0.5 * (dt.upper - dt.lower), dt.upper + 0.5 * (dt.upper - dt.lower)
    xs = range(start = x_min, stop = x_max, length = 1000)

    # create integral shape for case 1
    pts = [(dt.lower, 0.0)]
    for x in range(start = dt.lower, stop = dt.upper, length = 500)
        push!(pts, (x, pdf(dt.normal, x)))
    end
    for x in range(start = dt.upper, stop = dt.lower, length = 500)
        push!(pts, (x, 0.0))
    end
    push!(pts, (0.0, 0.0))

    p = plot(
        legend = false,
        xlabel = L"x",
        ylabel = L"p_X(x)",
        xtickfontsize = 14,
        ytickfontsize = 14,
        xguidefontsize = 16,
        yguidefontsize = 16,
    )
    plot!(Shape(pts), fillcolor = :red, fillalpha = 0.2, linewidth = 0.5)
    plot!(xs, x -> pdf(dt.normal, x), linewidth = 1, color = :black, style=:dash)
    plot!(xs, x -> double_truncated_pdf(x), linewidth = 3, color = :blue)
    display(p)
end

"""
    plot_v(; lower=0, upper=100)

Plots the additive correction `v(z)` for the mean of a doubly-truncated Gaussian as a
function of the normalized mean `z = μ/σ`. The correction is defined as
`v(z) = (φ(l - z) - φ(u - z)) / (Φ(u - z) - Φ(l - z))` where `l` and `u` are the
lower and upper truncation bounds and `φ`/`Φ` are the standard normal PDF/CDF.
"""
function plot_v(; lower=0, upper=100)
    function v(t)
        d = Normal(0,1)
        return (pdf(d, lower - t) - pdf(d, upper - t)) / (cdf(d, upper - t) - cdf(d, lower - t))
    end

    ts = range(start = -6, stop = 6, length = 1000)
    vs = v.(ts)
    l = max(lower, minimum(ts))
    u = min(upper, maximum(ts))

    p = plot(
        legend = false,
        # xlabel = L"\frac{\mu}{\sigma}",
        # ylabel = L"\frac{\mu_{\mathrm{new}} - \mu}{\sigma}",
        xlabel = L"z",
        ylabel = L"v(z)",
        xtickfontsize = 14,
        ytickfontsize = 14,
        xguidefontsize = 16,
        yguidefontsize = 16,
    )
    plot!([l, l], [minimum(vs), maximum(vs)], linewidth = 1, color = :red, style = :dash)
    plot!([u, u], [minimum(vs), maximum(vs)], linewidth = 1, color = :red, style = :dash)
    plot!(Shape([(l, minimum(vs)), (l, maximum(vs)), (u, maximum(vs)), (u, minimum(vs))]), fillcolor = :red, fillalpha = 0.2, linewidth = 0)
    plot!(ts, vs, linewidth = 5, color = :blue)
    display(p)
end

"""
    plot_w(; lower=0, upper=100)

Plots the multiplicative correction `w(z)` for the variance of a doubly-truncated Gaussian
as a function of the normalized mean `z = μ/σ`. The correction captures how much the
variance shrinks due to the truncation at `lower` and `upper`.
"""
function plot_w(; lower=0, upper=100)
    function v(t)
        d = Normal(0,1)
        return (pdf(d, lower - t) - pdf(d, upper - t)) / (cdf(d, upper - t) - cdf(d, lower - t))
    end

    function w(t)
        d = Normal(0,1)
        return ((t + upper) * pdf(d, upper - t) - (t + lower) * pdf(d, lower - t)) / (cdf(d, upper - t) - cdf(d, lower - t)) + v(t) * (2*t + v(t))
    end

    ts = range(start = -6, stop = 6, length = 1000)
    ws = w.(ts)
    l = max(lower, minimum(ts))
    u = min(upper, maximum(ts))

    p = plot(
        legend = false,
        # xlabel = L"\frac{\mu}{\sigma}",
        # ylabel = L"\frac{\sigma_{\mathrm{new}}^2}{\sigma^2}",
        xlabel = L"z",
        ylabel = L"w(z)",
        xtickfontsize = 14,
        ytickfontsize = 14,
        xguidefontsize = 16,
        yguidefontsize = 16,
    )
    plot!([l, l], [minimum(ws), maximum(ws)], linewidth = 1, color = :red, style = :dash)
    plot!([u, u], [minimum(ws), maximum(ws)], linewidth = 1, color = :red, style = :dash)
    plot!(Shape([(l, minimum(ws),), (l, maximum(ws)), (u, maximum(ws)), (u, minimum(ws),)]), fillcolor = :red, fillalpha = 0.2, linewidth = 0)
    plot!(ts, ws, linewidth = 5, color = :blue)
    display(p)
end

function main(; base="~/Downloads/")
    plot_approximation(0, 0.8)
    savefig(base * "approximation.svg")

    plot_double_truncated(DoublyTrunactedGaussian(-1.5, 1.0, Normal(0,1)))
    savefig(base * "double_truncated.svg")

    plot_v(lower = -100, upper=0)
    savefig(base * "v_100_0.svg")
    plot_v(lower = -5, upper=0)
    savefig(base * "v_5_0.svg")
    plot_v(lower = -3, upper=3)
    savefig(base * "v_3_3.svg")
    plot_v(lower = 0, upper=100)
    savefig(base * "v_0_100.svg")

    plot_w(lower = -100, upper=0)
    savefig(base * "w_100_0.svg")
    plot_w(lower = -5, upper=0)
    savefig(base * "w_5_0.svg")
    plot_w(lower = -3, upper=3)
    savefig(base * "w_3_3.svg")
    plot_w(lower = 0, upper=100)
    savefig(base * "w_0_100.svg")
end

end

Approximation.main()
