# Plots for Bayesian regression with linear basis function and predictive distributions
#
# 2026 by Ralf Herbrich
# Hasso-Plattner Institute

module PredictiveDistribution

using Random
using LinearAlgebra
using Distributions
using LaTeXStrings
using Plots

include("../shared/PMLCourse.jl")
using .PMLCourse

"""
    plot_Bayesian_fit(train_data; σ=0.1, τ=0.5, ϕ=x -> map(j -> polynomial_basis(x,j), 1:2), color=:blue)

Generates a plot of the Bayesian posterior for the training data `train_data` using the feature map `ϕ`. For the prior, it's assumed to be zero mean with standard deviation of `τ`; the likelihood is assumed to have standard deviation of `σ`. Using `color` for the line and ribbon.
"""
function plot_Bayesian_fit(
    train_data;
    σ = 0.1,
    τ = 0.5,
    ϕ = x -> map(j -> polynomial_basis(x, j), 1:2),
    color = :blue,
)
    # generate the feature representation of the input data
    Φ = transpose(hcat([ϕ(x) for x in train_data.x]...))

    # compute the parameters of the Bayesian posterior over w
    Σinv = 1.0 / τ^2 * Diagonal(ones(size(Φ, 2)))
    y = train_data.y
    C = cholesky(Σinv + 1.0 / σ^2 * Φ' * Φ)
    μ = C.U \ (C.L \ (1.0 / σ^2 * Φ' * y))

    # compute the mean and standard deviation of the predictive distribution over all test points 
    xs = collect(range(-0.3, 1.3, 100))
    Φ_test = transpose(hcat([ϕ(x) for x in xs]...))
    pred = map(ϕ -> (vec(μ)' * ϕ, sqrt(σ^2 + ϕ' * (C.U \ (C.L \ ϕ)))), eachrow(Φ_test))
    plot!(
        xs,
        map(x -> x[1], pred),
        ribbon = map(x -> x[2], pred),
        fillalpha = 0.2,
        linewidth = 3,
        color = color,
    )
end

function main()
    # generates training data
    Random.seed!(41)
    # train_data = generate_data(5, x -> sin(x*π), σ=0.15, from=0, to=1)
    train_data = generate_data(11, x -> sin(x * π), σ = 0.15, from = 0, to = 1)
    # train_data = generate_data(11, x -> 1/(1+x^2), σ=0.05, from=-5, to=5)


    p = plot(
        train_data.x,
        train_data.y,
        seriestype = :scatter,
        legend = false,
        color = :orange,
        xtickfontsize = 14,
        ytickfontsize = 14,
        xguidefontsize = 16,
        yguidefontsize = 16,
    )
    xlabel!(L"x")
    ylabel!(L"y")
    plot_Bayesian_fit(
        train_data,
        τ = 1,
        color = :blue,
        ϕ = x -> map(j -> polynomial_basis(x, j), 0:6),
    )
    plot_Bayesian_fit(
        train_data,
        τ = 10,
        color = :red,
        ϕ = x -> map(j -> polynomial_basis(x, j), 0:6),
    )
    display(p)
    savefig("~/Downloads/poly_fit.svg")

    # p = plot(train_data.x, train_data.y, seriestype=:scatter, legend=false, color = :orange,
    #          xtickfontsize=14, ytickfontsize=14, xguidefontsize=16, yguidefontsize=16)
    # xlabel!(L"x")
    # ylabel!(L"y")
    # plot_Bayesian_fit(train_data, τ=1, color=:blue, ϕ=x -> map(j -> fourier_basis(x,j), 0:15))
    # plot_Bayesian_fit(train_data, τ=10, color=:red, ϕ=x -> map(j -> fourier_basis(x,j), 0:15))
    # display(p)

    p = plot(
        train_data.x,
        train_data.y,
        seriestype = :scatter,
        legend = false,
        color = :orange,
        xtickfontsize = 14,
        ytickfontsize = 14,
        xguidefontsize = 16,
        yguidefontsize = 16,
    )
    xlabel!(L"x")
    ylabel!(L"y")
    plot_Bayesian_fit(
        train_data,
        τ = 1,
        color = :blue,
        ϕ = x -> map(j -> gauss_basis(x, j / 20, σ = 0.15), 0:20),
    )
    plot_Bayesian_fit(
        train_data,
        τ = 10,
        color = :red,
        ϕ = x -> map(j -> gauss_basis(x, j / 20, σ = 0.15), 0:20),
    )
    display(p)
    savefig("~/Downloads/gauss_fit.svg")
end

main()

end # module PredictiveDistribution
