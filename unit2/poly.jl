# Plots for polynomial regression
#
# 2026 by Ralf Herbrich
# Hasso-Plattner Institute

module PolynomialRegression

using Random
using LinearAlgebra
using Distributions
using LaTeXStrings
using Plots

include("../shared/PMLCourse.jl")
using .PMLCourse

"""
    generate_features(data; max_degree=2, min_degree=1)

Generates the feature matrix from the raw data
"""
function generate_features(data; max_degree=2, min_degree=1)
    return hcat([data .^ i for i = min_degree:max_degree]...)
end

"""
    fit_polynomial(train_data; degree=2, min_degree=1, color=:blue, threed=false)

Generates a least-square fit of a polynomial of degree and plots the fit
"""
function fit_polynomial(
    train_data;
    degree=2,
    min_degree=1,
    color=:blue,
    threed=false,
)
    train_X = generate_features(train_data.x, min_degree=min_degree, max_degree=degree)
    w = inv(train_X' * train_X) * train_X' * train_data.y
    xs = collect(range(minimum(train_data.x), maximum(train_data.x), 100))
    test_X = generate_features(xs, min_degree=min_degree, max_degree=degree)
    ys = test_X * w
    if (threed)
        plot!(xs, ys, zeros(100), linewidth=3, color=color)
    else
        plot!(xs, ys, linewidth=3, color=color)
    end
    return w
end

"""
    likelihood(w, X, y; σ = 0.15)

Computes the likelihood of N(y|Xw,σ^2 ⋅ I)
"""
function likelihood(w, X, y; σ=0.15)
    return pdf(MvNormal(X * w, I * σ^2), y)[1]
end

"""
    plot_training_data()

Plots the training data as a scatter plot with orange markers. Uses the module-level
`train_data` named tuple. Returns the plot object for further composition.
"""
function plot_training_data()
    p = plot(
        train_data.x,
        train_data.y,
        seriestype=:scatter,
        legend=false,
        color=:orange,
        xtickfontsize=14,
        ytickfontsize=14,
        ztickfontsize=14,
        xguidefontsize=16,
        yguidefontsize=16,
        zguidefontsize=16,
    )
    xlabel!(L"x")
    ylabel!(L"y")
    return p
end

# plot of the training data with best fit of n-th order polynomial
function plot_polynomial_fit(min_degree=1, degree=2)
    p = plot_training_data()
    w_best =
        fit_polynomial(train_data, min_degree=min_degree, degree=degree, color=:blue)
    display(p)
    return w_best

end

"""
    plot_2D_surface(f; w1, w2, w1_coarse, w2_coarse, colors=:blues)

Plots a 3D surface of the function `f(w)` over a grid defined by `w1` and `w2` ranges.
A wireframe is overlaid using the coarser grids `w1_coarse` and `w2_coarse`. The `colors`
argument controls the surface color scheme.
"""
function plot_2D_surface(
    f;
    w1=range(-5, stop=5, length=100),
    w2=range(-5, stop=5, length=100),
    w1_coarse=range(-5, stop=5, length=40),
    w2_coarse=range(-5, stop=5, length=40),
    colors=:blues,
)
    p = plot(
        w1,
        w2,
        (w1, w2) -> f([w1, w2]),
        seriestype = :surface,
        line_z = 0.9,
        color = colors,
        fillalpha = 0.5,
        legend = false,
        xtickfontsize = 14,
        ytickfontsize = 14,
        ztickfontsize = 14,
        xguidefontsize = 16,
        yguidefontsize = 16,
        zguidefontsize = 16,
    )
    for w1 in w1_coarse
        plot!(
            [w1 for w2 in w2_coarse],
            w2_coarse,
            [f([w1; w2]) for w2 in w2_coarse],
            line_z=[f([w1; w2]) for w2 in w2_coarse],
            color=:black,
            linewidth=0.1
        )
    end

    for w2 in w2_coarse
        plot!(
            w1_coarse,
            [w2 for w1 in w1_coarse],
            [f([w1; w2]) for w1 in w1_coarse],
            line_z=[f([w1; w2]) for w1 in w1_coarse],
            color=:black,
            linewidth=0.1
        )
    end
    xlabel!(L"w_1")
    ylabel!(L"w_2")
    display(p)
end

"""
    plot_predictive_distribution(train_data, w_best)

Plots the predictive distribution `p(y|x, w)` as a 3D surface over the input-output space,
overlaid with the training data points and the polynomial fit. The predictive distribution
is `N(y; w₁x + w₂x², 0.15)` evaluated over a grid of `(x, y)` values.
"""
function plot_predictive_distribution(train_data, w_best)
    # plot the predictive distribution
    x = range(0, stop=maximum(train_data.x), length=100)
    y = range(minimum(train_data.y), stop=maximum(train_data.y), length=100)
    x_coarse = range(minimum(train_data.x), stop=maximum(train_data.x), length=30)
    y_coarse = range(minimum(train_data.y), stop=maximum(train_data.y), length=30)
    p = plot(train_data.x, train_data.y, zeros(11), seriestype=:scatter, color=:orange)
    fit_polynomial(train_data, min_degree=1, degree=2, color=:blue, threed=true)

    surface!(
        x,
        y,
        (x, y) -> pdf(Normal(w_best[1] * x + w_best[2] * x * x, 0.15), y),
        line_z = 0.9,
        color=:greys,
        fillalpha=0.5,
        legend=false,
        xtickfontsize=14,
        ytickfontsize=14,
        ztickfontsize=14,
        xguidefontsize=16,
        yguidefontsize=16,
        zguidefontsize=16,
        colorbar=false,
    )
    for x in x_coarse
        plot!(
            [x for y in y_coarse],
            y_coarse,
            [pdf(Normal(w_best[1] * x + w_best[2] * x * x, 0.15), y) for y in y_coarse],
            line_z=[pdf(Normal(w_best[1] * x + w_best[2] * x * x, 0.15), y) for y in y_coarse],
            color=:black,
            linewidth=0.1
        )
    end

    for y in y_coarse
        plot!(
            x_coarse,
            [y for x in x_coarse],
            [pdf(Normal(w_best[1] * x + w_best[2] * x * x, 0.15), y) for x in x_coarse],
            line_z=[pdf(Normal(w_best[1] * x + w_best[2] * x * x, 0.15), y) for x in x_coarse],
            color=:black,
            linewidth=0.1
        )
    end

    xlabel!(L"x")
    ylabel!(L"y")
    display(p)
end

# module-level training data (set in main)
train_data = nothing

function main()
    # generates training data
    Random.seed!(41)
    global train_data = generate_data(11, x -> sin(x * π), σ=0.15, from=0, to=1)
    # train_data = generate_data(11, x -> 1/(1+x^2), σ=0.05, from=-5, to=5)
    # train_data = generate_data(11, x -> sin(x * π) * cos(x * π), σ = 0.15, from = 0, to = 1)

    # do all the plots
    p = plot_training_data()
    display(p)
    savefig("~/Downloads/training_data.svg")

    w_best = plot_polynomial_fit(1, 2)
    savefig("~/Downloads/polynomial_fit_1_2.svg")
    plot_polynomial_fit(0, 6)
    savefig("~/Downloads/polynomial_fit_0_6.svg")
    plot_polynomial_fit(0, 10)
    savefig("~/Downloads/polynomial_fit_0_10.svg")

    # plot the prior over the weight space
    mv = MvNormal([0, 0], I * 2)
    plot_2D_surface(w -> pdf(mv, w)[1])
    savefig("~/Downloads/prior.png")

    # plot the likelihood over the weight space
    train_X = generate_features(train_data.x, min_degree=1, max_degree=2)
    plot_2D_surface(
        w -> likelihood(w, train_X, train_data.y),
        w1=range(3, stop=5, length=100),
        w2=range(-5, stop=-3, length=100),
        w1_coarse=range(3, stop=5, length=30),
        w2_coarse=range(-5, stop=-3, length=30),
        colors=:reds,
    )
    savefig("~/Downloads/likelihood.png")

    plot_2D_surface(
        w -> likelihood(w, train_X, train_data.y) * pdf(mv, w),
        w1=range(3, stop=5, length=100),
        w2=range(-5, stop=-3, length=100),
        w1_coarse=range(3, stop=5, length=30),
        w2_coarse=range(-5, stop=-3, length=30),
        colors=:greens,
    )
    savefig("~/Downloads/posterior.png")

    plot_predictive_distribution(train_data, w_best)
    savefig("~/Downloads/predictive_distribution.png")
end

end # module PolynomialRegression

PolynomialRegression.main()
