# Demonstrates sampling on the TrueSkill graphical model
#
# 2026 by Ralf Herbrich
# Hasso-Plattner Institute

module TrueSkillSampling

using Random
using LinearAlgebra
using Distributions
using LaTeXStrings
using Plots

"""
    sample(; n=100000, μ1=0.0, σ1=1.0, μ2=0.0, σ2=1.0, β=1.0)

Draw `n` samples from the TrueSkill graphical model with prior means `μ1`, `μ2`,
prior standard deviations `σ1`, `σ2`, and performance noise `β`. Returns a vector
of vectors `[s1, s2, p1, p2, y]` where `y` is +1 or -1 indicating the outcome.
"""
function sample(; n = 100000, μ1=0.0, σ1=1.0, μ2=0.0, σ2=1.0, β=1.0)
    samples = Vector{Vector{Float64}}(undef, n)
    for i in 1:n
        s1 = rand(Normal(μ1, σ1))
        s2 = rand(Normal(μ2, σ2))
        p1 = rand(Normal(s1, β))
        p2 = rand(Normal(s2, β))
        y = p1 > p2 ? 1.0 : -1.0
        samples[i] = [s1, s2, p1, p2, y]
    end
    return samples
end

"""
    plot_histogram(xss; ylabel="Frequency", xlabel="x", xlim=(-5, 5), bins=100)

Plot overlaid histograms (normalized as PDFs) for each vector of samples in `xss`.
"""
function plot_histogram(xss; ylabel = "Frequency", xlabel = "x", xlim = (-5, 5), bins = 100)
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
    for xs in xss
        histogram!(xs, label=false, bins=bins, normalize=:pdf, alpha=0.5)
    end
    ylabel!(ylabel)
    xlabel!(xlabel)
    display(p)
end


"""
    plot_bars(xs; ylabel="Frequency", xlabel="x")

Plot a bar chart showing the empirical fractions of outcomes -1 and +1 in `xs`.
"""
function plot_bars(xs; ylabel = "Frequency", xlabel = "x")
    y_minus_1_frac = length(filter(x -> x == -1.0, xs)) / length(xs)
    y_plus_1_frac = length(filter(x -> x == +1.0, xs)) / length(xs)
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

    # plot the histograms of the marginal distributions (with and without evidence of y=1)
    data = sample(n = 1000000)
    plot_histogram([[x[1] for x in data]], ylabel=L"\hat{p}\left(s_1\right)", xlabel=L"s_1")
    savefig("~/Downloads/s1.svg")
    plot_histogram([[x[2] for x in data]], ylabel=L"\hat{p}\left(s_2\right)", xlabel=L"s_2")
    savefig("~/Downloads/s2.svg")
    plot_histogram([[x[3] for x in data]], ylabel=L"\hat{p}\left(p_1\right)", xlabel=L"p_1")
    savefig("~/Downloads/p1.svg")
    plot_histogram([[x[4] for x in data]], ylabel=L"\hat{p}\left(p_2\right)", xlabel=L"p_2")
    savefig("~/Downloads/p2.svg")
    plot_bars([x[5] for x in data], ylabel=L"\hat{p}\left(y\right)", xlabel=L"y")
    savefig("~/Downloads/y.svg")

    plot_histogram([[x[1] for x in data], [x[1] for x in filter(x -> x[5] == 1.0, data)]], ylabel=L"\hat{p}\left(s_1\right)", xlabel=L"s_1")
    savefig("~/Downloads/s1_y1=1.svg")
    plot_histogram([[x[2] for x in data], [x[2] for x in filter(x -> x[5] == 1.0, data)]], ylabel=L"\hat{p}\left(s_2\right)", xlabel=L"s_2")
    savefig("~/Downloads/s2_y1=1.svg")
    plot_histogram([[x[3] for x in data], [x[3] for x in filter(x -> x[5] == 1.0, data)]], ylabel=L"\hat{p}\left(p_1\right)", xlabel=L"p_1")
    savefig("~/Downloads/p1_y1=1.svg")
    plot_histogram([[x[4] for x in data], [x[4] for x in filter(x -> x[5] == 1.0, data)]], ylabel=L"\hat{p}\left(p_2\right)", xlabel=L"p_2")
    savefig("~/Downloads/p2_y1=1.svg")

    println("Fraction of samples kept after evidence: ", length(filter(x -> x[5] == 1.0, data)) / length(data))

    # plot the histograms of the marginal distributions in asymmetric case (with and without evidence of y=1)
    data = sample(n = 1000000, μ1=-1.5, σ1=1.0, μ2=1.5, σ2=1.0, β=1.0)
    plot_histogram([[x[1] for x in data]], ylabel=L"\hat{p}\left(s_1\right)", xlabel=L"s_1")
    savefig("~/Downloads/s1_asymmetric.svg")
    plot_histogram([[x[2] for x in data]], ylabel=L"\hat{p}\left(s_2\right)", xlabel=L"s_2")
    savefig("~/Downloads/s2_asymmetric.svg")
    plot_histogram([[x[3] for x in data]], ylabel=L"\hat{p}\left(p_1\right)", xlabel=L"p_1")
    savefig("~/Downloads/p1_asymmetric.svg")
    plot_histogram([[x[4] for x in data]], ylabel=L"\hat{p}\left(p_2\right)", xlabel=L"p_2")
    savefig("~/Downloads/p2_asymmetric.svg")
    plot_bars([x[5] for x in data], ylabel=L"\hat{p}\left(y\right)", xlabel=L"y")
    savefig("~/Downloads/y_asymmetric.svg")

    plot_histogram([[x[1] for x in data], [x[1] for x in filter(x -> x[5] == 1.0, data)]], ylabel=L"\hat{p}\left(s_1\right)", xlabel=L"s_1")
    savefig("~/Downloads/s1_y1=1_asymmetric.svg")
    plot_histogram([[x[2] for x in data], [x[2] for x in filter(x -> x[5] == 1.0, data)]], ylabel=L"\hat{p}\left(s_2\right)", xlabel=L"s_2")
    savefig("~/Downloads/s2_y1=1_asymmetric.svg")
    plot_histogram([[x[3] for x in data], [x[3] for x in filter(x -> x[5] == 1.0, data)]], ylabel=L"\hat{p}\left(p_1\right)", xlabel=L"p_1")
    savefig("~/Downloads/p1_y1=1_asymmetric.svg")
    plot_histogram([[x[4] for x in data], [x[4] for x in filter(x -> x[5] == 1.0, data)]], ylabel=L"\hat{p}\left(p_2\right)", xlabel=L"p_2")
    savefig("~/Downloads/p2_y1=1_asymmetric.svg")

    println("Fraction of samples kept after evidence: ", length(filter(x -> x[5] == 1.0, data)) / length(data))
end

main()

end
