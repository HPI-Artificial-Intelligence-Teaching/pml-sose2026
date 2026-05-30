# Plots for linear algebra
#
# 2026 by Ralf Herbrich
# Hasso-Plattner Institute

module LinearAlgebraPlots

using Random
using LinearAlgebra
using Distributions
using LaTeXStrings
using Plots

"""
    plot_mapping(A)

Plots the effect of a 2×2 matrix `A` as an R² → R² linear mapping. Visualizes the
transformation by showing a unit circle mapped through each stage of the SVD decomposition
(V', Σ·V', U·Σ·V', and A·X), highlighting two selected basis vectors in blue and red.

Returns a tuple of five plots `(p1, p2, p3, p4, p5)` corresponding to the original unit
circle, the rotation by V', the scaling by Σ, the rotation by U, and the full mapping by A.
"""
function plot_mapping(A)
    function plot_points(pts, i1, i2)
        p = plot(
            pts[1, :],
            pts[2, :],
            legend = false,
            linewidth = 3,
            color = :black,
            xtickfontsize = 14,
            ytickfontsize = 14,
            xguidefontsize = 16,
            yguidefontsize = 16,
            aspect_ratio = :equal,
        )
        plot!([0, pts[1, i1]], [0, pts[2, i1]], arrow = true, linewidth = 5, color = :blue)
        plot!([0, pts[1, i2]], [0, pts[2, i2]], arrow = true, linewidth = 5, color = :red)
        xlabel!(L"x_1")
        ylabel!(L"x_2")
        return p
    end

    U, S, V = svd(A, alg = LinearAlgebra.QRIteration())

    i1, i2 = 1, 26
    θs = range(0, stop = 2π, length = 100)
    X = hcat(sin.(θs), cos.(θs))'
    p1 = plot_points(X, i1, i2)
    p2 = plot_points(V' * X, i1, i2)
    p3 = plot_points(Diagonal(S) * V' * X, i1, i2)
    p4 = plot_points(U * Diagonal(S) * V' * X, i1, i2)
    p5 = plot_points(A * X, i1, i2)

    return (p1, p2, p3, p4, p5)
end



function main()
    # plot a sample of basis functions
    (p1, p2, p3, p4, p5) = plot_mapping([[1 1]; [0 1]])
    savefig(p1, "~/Downloads/circle_X.svg")
    display(p1)
    savefig(p2, "~/Downloads/circle_V'*X.svg")
    display(p2)
    savefig(p3, "~/Downloads/circle_S*V'*X.svg")
    display(p3)
    savefig(p4, "~/Downloads/circle_U*S*V'*X.svg")
    display(p4)
    savefig(p5, "~/Downloads/circle_A*X.svg")
    display(p5)
end

main()

end # module LinearAlgebraPlots
