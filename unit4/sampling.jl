# Plots for sampling with weighted empirical distributions
#
# 2026 by Ralf Herbrich
# Hasso-Plattner Institute

module ImportanceSampling

using Random
using LinearAlgebra
using Distributions
using LaTeXStrings
using Plots

"""
    plot_importance_sampling(dn, proposal; n=100000, bins=100, xlim=(-4, 4))

Plot a weighted histogram of `n` samples drawn from `proposal` and reweighted to
approximate the density `dn` via importance sampling. Overlays the true density.
"""
function plot_importance_sampling(dn, proposal; n=100000, bins = 100, xlim=(-4, 4))
    xs = rand(proposal, n)
    ws = pdf(dn, xs) ./ pdf(proposal, xs)
    p = histogram(
        xs,
        weights=ws,
        color=:blue,
        normalize = :pdf,
        legend=false,
        bins = bins,
        xlim = xlim,
        xtickfontsize=14,
        ytickfontsize=14,
        xguidefontsize=16,
        yguidefontsize=16
    )
    xs = range(xlim[1], xlim[2], length=1000)
    plot!(xs, pdf(dn, xs), color=:red, linewidth=2)
    xlabel!(L"x")
    ylabel!(L"\hat{p}(x)")
    display(p)
end

"""
    plot_importance_weights(dn, proposal; xlim=(-4, 4))

Plot the importance weight function `p(x)/q(x)` over the range `xlim`, where
`dn` is the target distribution and `proposal` is the proposal distribution.
"""
function plot_importance_weights(dn, proposal; xlim=(-4, 4))
    xs = range(xlim[1], xlim[2], length=1000)
    p = plot(
        xs,
        pdf(dn, xs) ./ pdf(proposal, xs),
        color=:red,
        linewidth=3,
        legend=false,
        xlim = xlim,
        xtickfontsize=14,
        ytickfontsize=14,
        xguidefontsize=16,
        yguidefontsize=16
    )
    xlabel!(L"x")
    ylabel!(L"\frac{p(x)}{q(x)}")
    display(p)
end


"""
    sample(; n=1000000, μ1=0.0, σ1=1.0, μ2=0.0, σ2=1.0, β=1.0, proposal_s1, proposal_s2, proposal_p1, proposal_p2, proposal_y)

Draw `n` importance-weighted samples from the TrueSkill factor graph model using the
given proposal distributions. Returns `(samples, weights)` where each sample is
`[s1, s2, p1, p2, y]`.
"""
function sample(; n = 1000000, μ1=0.0, σ1=1.0, μ2=0.0, σ2=1.0, β=1.0,
    proposal_s1 = Normal(μ1, σ1), proposal_s2 = Normal(μ2, σ2),
    proposal_p1 = Normal(μ1, sqrt(σ1^2 + β^2)), proposal_p2 = Normal(μ2, sqrt(σ2^2 + β^2)),
    proposal_y = Bernoulli(0.5))

    samples = Vector{Vector{Float64}}(undef, n)
    weights = Vector{Float64}(undef, n)
    s1 = rand(proposal_s1, n)
    s2 = rand(proposal_s2, n)
    p1 = rand(proposal_p1, n)
    p2 = rand(proposal_p2, n)
    y = rand(proposal_y, n)

    for i in 1:n
        f1 = pdf(Normal(μ1, σ1), s1[i])
        f2 = pdf(Normal(μ2, σ2), s2[i])
        f3 = pdf(Normal(s1[i], β), p1[i])
        f4 = pdf(Normal(s2[i], β), p2[i])
        f5 = if (y[i] == 1 && p1[i] > p2[i]) || (y[i] == 0 && p1[i] < p2[i]) 1.0 else 0.0 end

        g1 = pdf(proposal_s1, s1[i])
        g2 = pdf(proposal_s2, s2[i])
        g3 = pdf(proposal_p1, p1[i])
        g4 = pdf(proposal_p2, p2[i])
        g5 = pdf(proposal_y, y[i])

        weights[i] = f1 * f2 * f3 * f4 * f5 / (g1 * g2 * g3 * g4 * g5)
        samples[i] = [s1[i], s2[i], p1[i], p2[i], y[i]]
    end
    return samples, weights
end

"""
    sample_with_outcome(; n=1000000, μ1=0.0, σ1=1.0, μ2=0.0, σ2=1.0, β=1.0, proposal_s1, proposal_s2, proposal_p1, proposal_p2)

Draw `n` importance-weighted samples from the TrueSkill factor graph model
conditioned on outcome `y=1`. Returns `(samples, weights)` where each sample is
`[s1, s2, p1, p2]`.
"""
function sample_with_outcome(; n = 1000000, μ1=0.0, σ1=1.0, μ2=0.0, σ2=1.0, β=1.0,
    proposal_s1 = Normal(μ1, σ1), proposal_s2 = Normal(μ2, σ2),
    proposal_p1 = Normal(μ1, sqrt(σ1^2 + β^2)), proposal_p2 = Normal(μ2, sqrt(σ2^2 + β^2)))

    samples = Vector{Vector{Float64}}(undef, n)
    weights = Vector{Float64}(undef, n)
    s1 = rand(proposal_s1, n)
    s2 = rand(proposal_s2, n)
    p1 = rand(proposal_p1, n)
    p2 = rand(proposal_p2, n)

    for i in 1:n
        f1 = pdf(Normal(μ1, σ1), s1[i])
        f2 = pdf(Normal(μ2, σ2), s2[i])
        f3 = pdf(Normal(s1[i], β), p1[i])
        f4 = pdf(Normal(s2[i], β), p2[i])
        f5 = (p1[i] > p2[i]) ? 1.0 : 0.0

        g1 = pdf(proposal_s1, s1[i])
        g2 = pdf(proposal_s2, s2[i])
        g3 = pdf(proposal_p1, p1[i])
        g4 = pdf(proposal_p2, p2[i])

        weights[i] = f1 * f2 * f3 * f4 * f5 / (g1 * g2 * g3 * g4)
        samples[i] = [s1[i], s2[i], p1[i], p2[i]]
    end
    return samples, weights
end

"""
    plot_histogram(xss, wss; ylabel="Frequency", xlabel="x", xlim=(-5, 5), bins=100)

Plot overlaid weighted histograms (normalized as PDFs) for each pair of sample vector
and weight vector in `xss` and `wss`.
"""
function plot_histogram(xss, wss; ylabel = "Frequency", xlabel = "x", xlim = (-5, 5), bins = 100)
    p = plot(
        legend=false,
        label=false,
        color=:blue,
        xlim=xlim,
        xtickfontsize=18,
        ytickfontsize=18,
        xguidefontsize=20,
        yguidefontsize=20,
        legendfontsize=20,
    )
    for i in eachindex(xss)
        histogram!(xss[i], weights=wss[i], label=false, bins=bins, normalize=:pdf, alpha=0.5)
    end
    ylabel!(ylabel)
    xlabel!(xlabel)
    display(p)
end

"""
    plot_bars(xs, ws; ylabel="Frequency", xlabel="x")

Plot a bar chart showing the weighted empirical fractions of outcomes 0 and 1 in `xs`,
using importance weights `ws`.
"""
function plot_bars(xs, ws; ylabel = "Frequency", xlabel = "x")
    Z = sum(ws)
    y_minus_1_frac = sum(ws[xs .== 0]) / Z
    y_plus_1_frac = sum(ws[xs .== 1]) / Z
    p = plot(
        bar([-1, 1], [y_minus_1_frac, y_plus_1_frac], alpha=0.5, bar_width = 0.75),
        legend=false,
        label=false,
        color=:blue,
        xtickfontsize=18,
        ytickfontsize=18,
        xguidefontsize=20,
        yguidefontsize=20,
        legendfontsize=20,
    )
    ylabel!(ylabel)
    xlabel!(xlabel)
    display(p)
end

function main()
    Random.seed!(2025)

    # plots a Normal distribution with the same distribution as proposal
    plot_importance_sampling(Normal(0, 1), Normal(0, 1), n=1000000)
    savefig("~/Downloads/importance_sampling.svg")
    plot_importance_weights(Normal(0, 1), Normal(0, 1))
    savefig("~/Downloads/importance_weights.svg")
    plot_importance_sampling(Normal(0, 1), Normal(1.5, 1), n=1000000)
    savefig("~/Downloads/importance_sampling2.svg")
    plot_importance_weights(Normal(0, 1), Normal(1.5, 1))
    savefig("~/Downloads/importance_weights2.svg")
    plot_importance_sampling(Normal(0, 1), Uniform(-2.5, 2.5), n=1000000)
    savefig("~/Downloads/importance_sampling3.svg")
    plot_importance_weights(Normal(0, 1), Uniform(-2.5, 2.5))
    savefig("~/Downloads/importance_weights3.svg")


    samples, weights = sample()
    samples_with_outcome, weights_with_outcome = sample_with_outcome()
    plot_histogram([[x[1] for x in samples]], [weights], ylabel=L"\hat{p}\left(s_1\right)", xlabel=L"s_1")
    savefig("~/Downloads/s1_importance.svg")
    plot_histogram([[x[2] for x in samples]], [weights], ylabel=L"\hat{p}\left(s_2\right)", xlabel=L"s_2")
    savefig("~/Downloads/s2_importance.svg")
    plot_histogram([[x[3] for x in samples]], [weights], ylabel=L"\hat{p}\left(p_1\right)", xlabel=L"p_1")
    savefig("~/Downloads/p1_importance.svg")
    plot_histogram([[x[4] for x in samples]], [weights], ylabel=L"\hat{p}\left(p_2\right)", xlabel=L"p_2")
    savefig("~/Downloads/p2_importance.svg")
    plot_bars([x[5] for x in samples], weights, ylabel=L"\hat{p}\left(y\right)", xlabel=L"y")
    savefig("~/Downloads/y_importance.svg")

    plot_histogram([[x[1] for x in samples], [x[1] for x in samples_with_outcome]], [weights, weights_with_outcome], ylabel=L"\hat{p}\left(s_1\right)", xlabel=L"s_1")
    savefig("~/Downloads/s1_importance_with_outcome.svg")
    plot_histogram([[x[2] for x in samples], [x[2] for x in samples_with_outcome]], [weights, weights_with_outcome], ylabel=L"\hat{p}\left(s_2\right)", xlabel=L"s_2")
    savefig("~/Downloads/s2_importance_with_outcome.svg")
    plot_histogram([[x[3] for x in samples], [x[3] for x in samples_with_outcome]], [weights, weights_with_outcome], ylabel=L"\hat{p}\left(p_1\right)", xlabel=L"p_1")
    savefig("~/Downloads/p1_importance_with_outcome.svg")
    plot_histogram([[x[4] for x in samples], [x[4] for x in samples_with_outcome]], [weights, weights_with_outcome], ylabel=L"\hat{p}\left(p_2\right)", xlabel=L"p_2")
    savefig("~/Downloads/p2_importance_with_outcome.svg")
end

main()

end
