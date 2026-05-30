# Plots for linear basis function models
#
# 2026 by Ralf Herbrich
# Hasso-Plattner Institute

module BasisFunctions

using Random
using LinearAlgebra
using Distributions
using LaTeXStrings
using Plots

include("../shared/PMLCourse.jl")
using .PMLCourse

"""
    plot_function_sample(basis,n=99,d=5;min_x=0,max_x=5, color1=:blue, color2=:red)

Plots a sample of `n`+1 linear basis functions with the `d` basis function `basis`
"""
function plot_function_sample(
    basis,
    n = 99,
    d = 5;
    min_x = 0,
    max_x = 5,
    color1 = :blue,
    color2 = :red,
)
    xs = range(min_x, max_x, 1000)
    v = randn(d, 1)
    p = plot(
        xs,
        map(x -> linear_basis_function(x, v, basis), xs),
        legend = false,
        linewidth = 2,
        color = color1,
        xtickfontsize = 14,
        ytickfontsize = 14,
        xguidefontsize = 16,
        yguidefontsize = 16,
    )
    for i = 1:n
        v = randn(d, 1)
        plot!(
            xs,
            map(x -> linear_basis_function(x, v, basis), xs),
            legend = false,
            linewidth = 0.3,
            color = color2,
        )
    end
    xlabel!(L"x")
    ylabel!(L"f(x)")

    return p
end

function main()
    # plot a sample of basis functions
    Random.seed!(42)
    p = plot_function_sample(polynomial_basis, 99, 5, min_x = -5, max_x = 5)
    savefig(p, "~/Downloads/poly.png")
    display(p)

    p = plot_function_sample(fourier_basis, 99, 5, min_x = -2, max_x = 2)
    savefig(p, "~/Downloads/fourier.png")
    display(p)

    p = plot_function_sample(gauss_basis, 99, 5, min_x = -5, max_x = 10)
    savefig(p, "~/Downloads/gauss.png")
    display(p)

    p = plot_function_sample(sigmoid_basis, 99, 5, min_x = -5, max_x = 10)
    savefig(p, "~/Downloads/sigmoid.png")
    display(p)
end

main()

end # module BasisFunctions
