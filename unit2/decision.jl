# Plots for decision boundary
#
# 2026 by Ralf Herbrich
# Hasso-Plattner Institute

module Decision

using Random
using LinearAlgebra
using Distributions
using LaTeXStrings
using Plots

include("../shared/PMLCourse.jl")
using .PMLCourse

# plot the sigmoid function
function plot_sigmoid(; xs=-4:0.1:4.0)
    p = plot(
        xs,
        map(sigmoid, xs),
        legend=:top,
        label=L"p(1|x)",
        linewidth=3,
        color=:blue,
        xtickfontsize=14,
        ytickfontsize=14,
        xguidefontsize=16,
        yguidefontsize=16,
        legendfontsize=16,
    )
    plot!(xs, map(x -> 1 - sigmoid(x), xs), label=L"p(0|x)", linewidth=3, color=:red)
    ylabel!(L"p(y|x)")
    xlabel!(L"x")
    display(p)
end

"""
    plot_squared_loss(; xs=range(-4,4,1000))

Plots the squared loss function `l(y, ŷ) = (y - ŷ)²` over the given range `xs`.
Displays the resulting plot.
"""
function plot_squared_loss(; xs=range(-4,4,1000))
    p = plot(
        xs,
        map(x -> x^2, xs),
        legend=false,
        linewidth=3,
        color=:blue,
        xtickfontsize=14,
        ytickfontsize=14,
        xguidefontsize=16,
        yguidefontsize=16,
        legendfontsize=16,
    )
    ylabel!(L"l\left(y,\hat{y}\right)")
    xlabel!(L"y - \hat{y}")
    display(p)
end

function main()
    plot_sigmoid()
    savefig("~/Downloads/sigmoid.svg")

    plot_squared_loss()
    savefig("~/Downloads/squared_loss.svg")
end

end # module Decision

Decision.main()
