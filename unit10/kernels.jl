# Plots for Kernels slides
#
# 2026 by Ralf Herbrich
# Hasso-Plattner Institute

module KernelPlots

using Random
using LinearAlgebra
using Distributions
using LaTeXStrings
using Plots

include("../shared/PMLCourse.jl")
using .PMLCourse

"""
    plot_kernels(ϕ, x; xlim=(-1,1), base_name="~/Downloads/kernel")

Plots the basis functions of `ϕ` and the kernel evaluations at `x`
"""
function plot_kernels(ϕ, x; xlim = (-1, 1), base_name = "~/Downloads/kernel")
    N = length(ϕ(0))
    xs = range(start = xlim[1], stop = xlim[2], length = 100)
    yss = map(x -> ϕ(x), xs)
    ys = map(ys -> ys' * ϕ(x), yss)
    p = plot(
        legend = false,
        xtickfontsize = 14,
        ytickfontsize = 14,
        xguidefontsize = 16,
        yguidefontsize = 16,
    )
    xlabel!(L"x")
    ylabel!(L"\phi(x)")
    for i = 1:N
        plot!(xs, map(ys -> ys[i], yss), color = i, linewidth = 3)
    end
    display(p)
    savefig(p, base_name * "_basis.svg")

    p = plot(
        legend = false,
        xtickfontsize = 14,
        ytickfontsize = 14,
        xguidefontsize = 16,
        yguidefontsize = 16,
    )
    xlabel!(L"x")
    ylabel!(L"k(x,x^\prime)")
    plot!(xs, ys, color = :blue, linewidth = 3)
    plot!([x, x], [minimum(ys), maximum(ys)], linewdith = 0.5, color = :red)
    display(p)
    savefig(p, base_name * "_kernel.svg")
end

function main()
    plot_kernels(
        x -> map(j -> polynomial_basis(x, j), 1:11),
        -0.5,
        base_name = "~/Downloads/poly",
    )
    plot_kernels(
        x -> map(j -> gauss_basis(x, j / 5 - 1, σ = 0.15), 0:10),
        0,
        base_name = "~/Downloads/gauss",
    )
    plot_kernels(
        x -> map(j -> sigmoid_basis(x, j / 5 - 1, σ = 0.15), 0:10),
        0,
        base_name = "~/Downloads/sigmoid",
    )
end

main()

end # module KernelPlots
