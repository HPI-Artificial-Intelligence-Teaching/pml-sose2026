# Plots for Bayesian probit regression
#
# 2026 by Ralf Herbrich
# Hasso-Plattner Institute

module BayesProbit

using Random
using LinearAlgebra
using Distributions
using LaTeXStrings
using Plots
using MLDatasets
using Statistics

include("../shared/PMLCourse.jl")
using .PMLCourse

"""
    BayesProbitRegression

Structure storing the parameters for the Bayesian probit regression model, including
the posterior mean `μ`, covariance `Σ`, latent variance `β2`, and the site parameters
`a` (precision-mean) and `b` (precision) of data messages.
"""
struct BayesProbitRegression
    μ::Vector{Float64}      # mean of the posterior distribution
    Σ::Matrix{Float64}      # covariance of the posterior distribution
    β2::Float64             # latent variance
    a::Vector{Float64}      # precision-mean of te data messages from the Gaussian projection-noise to w
    b::Vector{Float64}      # precision of the data messages from the Gaussian projection-noise to w
end

"""
    BayesProbitRegression(; μ=vec(zeros(2,1)), Σ=Diagonal(ones(2)), β=1.0)

Constructs a `BayesProbitRegression` with the given prior mean `μ`, prior covariance `Σ`,
and latent noise standard deviation `β`. The site parameters are initialized to zero.
"""
function BayesProbitRegression(; μ = vec(zeros(2, 1)), Σ = Diagonal(ones(2)), β = 1.0)
    return BayesProbitRegression(μ, Σ, β^2, zeros(size(Σ, 1)), zeros(size(Σ, 1)))
end


"""
    bpr_new = learn(X, y, bpr::BayesProbitRegression; ϵ = 1e-4, features = identity)

Learns a Bayesian probit regression model from the examples in the row of `X` and the labels `y` and returns the new model. 
"""
function learn(
    X,
    y,
    bpr::BayesProbitRegression;
    ϵ = 1e-4, 
    x_min = -1.0,
    x_max = 1.0,
    y_min = -1.0,
    y_max = 1.0,
    class0_color = :red, 
    class1_color = :blue, 
    mark_coefficient = false,
    iteration_plots = false,
    filename_base = "~/Downloads/bayes_probit_decision_surface",
    features = identity,
)
    norm_dist = Normal()

    # computes the additive correction of a single-sided truncated Gaussian with unit variance
    function v(t, ϵ)
        denom = cdf(norm_dist, t - ϵ)
        if (denom < floatmin(Float64))
            return (ϵ - t)
        else
            return (pdf(norm_dist, t - ϵ) / denom)
        end
    end

    # computes the multiplicative correction of a single-sided truncated Gaussian with unit variance
    function w(t, ϵ)
        denom = cdf(norm_dist, t - ϵ)
        if (denom < floatmin(Float64))
            return ((t - ϵ < 0.0) ? 1.0 : 0.0)
        else
            vt = v(t, ϵ)
            return (vt * (vt + t - ϵ))
        end
    end

    X_train = hcat(map(i -> features(X[i, :]), 1:size(X, 1))...)'

    # extract dimensions and number of examples
    (n, _) = size(X_train)

    # initialize the parameters of the message passing algorithm
    μ = bpr.μ
    Σ = bpr.Σ
    a = zeros(n, 1)
    b = zeros(n, 1)

    # defines the message passing update for a single example at index `i`
    function update(i)
        ϕ = vec(X_train[i, :]')
        Σϕ = Σ * ϕ
        ϕΣϕ = ϕ' * Σϕ
        ϕμ = ϕ' * μ

        # compute the mean and variance of the message from the Gaussian projection-noise to the latent variable
        c = 1 - b[i] * ϕΣϕ
        s2 = ϕΣϕ / c + bpr.β2
        s = sqrt(s2)
        m = (y[i] * ϕμ - a[i] * ϕΣϕ) / c

        # compute the precision and precision-mean of the message from the latent variable to the Gaussian projection-noise
        V = v(m / s, 0)
        W = w(m / s, 0)
        d = s2 * (1 - W)
        τ = (s * V + m * W) / d
        ρ = W / d

        # compute the site parameters a[i] and b[i] and update the posterior mean and covariance        
        a_new = τ / (1 + ρ * bpr.β2)
        b_new = ρ / (1 + ρ * bpr.β2)

        Δa = a_new - a[i]
        Δb = b_new - b[i]

        c2 = 1 + Δb * ϕΣϕ
        Σ -= Δb / c2 * Σϕ * Σϕ'
        μ += Σϕ * (Δa * y[i] - Δb * ϕμ) / c2

        a[i] = a_new
        b[i] = b_new
        return
    end

    # run message passing for `no_iterations` many iterations for each example 
    Δ = ϵ
    iteration = 1
    while (Δ ≥ ϵ)
        atmp = copy(a)
        btmp = copy(b)
        for idx = 1:n
            update(idx)
        end
        Δ = max(maximum(abs.(a - atmp)), maximum(abs.(b - btmp)))
        println("Iteration ", iteration, ": Δ = ", Δ)
        if (iteration_plots)
            plot_decision_surface(
                X, y, BayesProbitRegression(μ, Σ, bpr.β2, a[:], b[:]);
                x_min = x_min,
                x_max = x_max,
                y_min = y_min,
                y_max = y_max,
                class0_color = class0_color,
                class1_color = class1_color,
                mark_coefficient = mark_coefficient,
                filename = filename_base * "_iter_$(iteration).png",
                features = features,
            )
        end
        iteration += 1
    end
    println("Converged after ", iteration - 1, " iterations with Δ = ", Δ)
    println()

    return (BayesProbitRegression(μ, Σ, bpr.β2, a[:], b[:]))
end

"""
    posterior(ϕ, bpr)

Computes the posterior probability P(y = -1 | ϕ) for a given feature vector `ϕ` under
the Bayesian probit regression model `bpr`, by integrating over the Gaussian posterior
on the weights.
"""
function posterior(ϕ, bpr)
    N = Normal(bpr.μ' * ϕ, sqrt(ϕ' * bpr.Σ * ϕ + bpr.β2))
    return (cdf(N, 0))
end


"""
    plot_decision_surface(X, y, bpr; x_min = -1.0, x_max = 1.0, y_min = -1.0, y_max = 1.0, class0_color = :red, class1_color = :blue,  mark_coefficient = false, filename = "~/Downloads/bayes_probit_decision_surface.png", features = identity)

Plots the decision surface for the model learned from `X` and `y` in the model probit-regression model `bpr` when applying the feature vector mapping `features` to each row of `X`. The learned weight vector is also plotted as a white line. If `mark_coefficient` is set to `true`, the points are marked with an alpha-channel that indicates the standard deviation (and a size that indicates the mean) of the predictive distribution at that point. The resulting plot is saved as an image in `filename`.
"""
function plot_decision_surface(
    X,
    y,
    bpr::BayesProbitRegression;
    x_min = -1.0,
    x_max = 1.0,
    y_min = -1.0,
    y_max = 1.0,
    class0_color = :red, 
    class1_color = :blue, 
    mark_coefficient = false,
    filename = "~/Downloads/bayes_probit_decision_surface.png",
    features = identity,
)
    # plot the posterior probability of each class prediction 
    X_draw = range(start = x_min, stop = x_max, length = 100)
    Y_draw = range(start = y_min, stop = y_max, length = 100)
    p = heatmap(
        X_draw,
        Y_draw,
        (x1, x2) -> posterior(vec(features([x1, x2])), bpr),
        legend = false,
        color = :berlin,
    )

    idx_class0 = (y .== -1)[:]
    idx_class1 = (y .== +1)[:]
    if mark_coefficient
        class0_μ = bpr.a[idx_class0] ./ bpr.b[idx_class0]
        class1_μ = bpr.a[idx_class1] ./ bpr.b[idx_class1]
        μ_max = maximum(bpr.a ./ bpr.b)

        class0_σ = sqrt.(1 ./ bpr.b[idx_class0])
        class1_σ = sqrt.(1 ./ bpr.b[idx_class1])
        σ_max = maximum(sqrt.(1 ./ bpr.b))

        scatter!(
            X[idx_class0, 1],
            X[idx_class0, 2],
            legend = false,
            color = class0_color,
            # markercolor = [RGBA(x, 0, 0, 1) for x in class0_μ ./ class0_μ_max],
            markercolor = class0_color,
            markeralpha = [x for x in class0_σ ./ σ_max],
            markersize = 10 .* (class0_μ ./ μ_max),
            aspect_ratio = :equal,
        )
        scatter!(
            X[idx_class1, 1], 
            X[idx_class1, 2], 
            color = class1_color,
            # markercolor = [RGBA(0, 0, x, 1) for x in class1_μ ./ class1_μ_max],
            markercolor = class1_color,
            markeralpha = [x for x in class1_σ ./ σ_max],
            markersize = 10 .* (class1_μ ./ μ_max),
        )
    else
        scatter!(
            X[idx_class0, 1],
            X[idx_class0, 2],
            legend = false,
            color = class0_color,
            aspect_ratio = :equal,
        )
        scatter!(
            X[idx_class1, 1], 
            X[idx_class1, 2], 
            color = class1_color,
        )
    end

    contour!(
        X_draw,
        Y_draw,
        (x1, x2) -> posterior(vec(features([x1, x2])), bpr),
        levels = [0.5],
        linewidth = 2,
        color = :white,
    )

    # decorate the plot
    xlims!(x_min, x_max)
    ylims!(y_min, y_max)
    xlabel!(L"x_1")
    ylabel!(L"x_2")
    display(p)
    savefig(filename)
end

"""
    plot_bayes_probit_regression(n=250; class0_color=:red, class1_color=:blue, iteration_plots=false, mark_coefficient=false, class0=1, class1=9, ϵ=1e-4, τ=0.1, β=1.0, filename_base = "~/Downloads/bayes_probit")

Plots `n` examples of MNIST image feature vectors (top- and bottom-half average pixel intensity) where the one class is the digits with the label `class0` and the other class is the digits with the label `class1` (using the colors `class0_color` and `class1_color`, respectively). Then performs the probit regression learning and plots the decision surface as the algorithm converges. The learned weight vector is also plotted as a white line. If `mark_coefficient` is set to `true`, the points are marked with an alpha-channel that indicates the standard deviation (and a size that indicates the mean) of the predictive distribution at that point. The resulting plot is saved as an image in `filename`.
"""
function plot_bayes_probit_regression(
    n = 250;
    class0_color = :red,
    class1_color = :blue,
    mark_coefficient=false,
    iteration_plots = false,
    class0 = 1,
    class1 = 9,
    ϵ = 1e-4,
    τ = 0.1,
    β = 1.0,
    filename_base = "~/Downloads/bayes_probit",
)
    # creates the data set
    y = MNIST(split = :train).targets
    X = MNIST(split = :train).features
    idx0 = range(1, length(y))[y.==class0]
    idx1 = range(1, length(y))[y.==class1]
    X0 = hcat(map(i -> PMLCourse.average_intensity_features(X[:, :, idx0[i]]), 1:n)...)'
    X1 = hcat(map(i -> PMLCourse.average_intensity_features(X[:, :, idx1[i]]), 1:n)...)'
    mn = (sum(X1, dims = 1) + sum(X0, dims = 1)) / (2 * n)
    X0 -= repeat(mn, n)
    X1 -= repeat(mn, n)

    # augment the data with a one column for the bias and concatenate the two datasets
    X = vcat(X0, X1)
    x_max, y_max = 1.2 * maximum(X, dims = 1)
    x_min, y_min = 1.2 * minimum(X, dims = 1)
    y = vcat(-ones(n, 1), 1 * ones(n, 1))

    # the RBF features for this problem
    function rbf_features(x)
        local p = [
            (x1, x2, (x_max - x_min) / 2^s) for x1 in range(x_min, x_max, 5) for
            x2 in range(y_min, y_max, 5) for s = 0:8
        ]
        return map(v -> exp(-(v[1] - x[1])^2 / v[3]^2 - (v[2] - x[2])^2 / v[3]^2), p)
    end

    X_train = hcat(map(i -> rbf_features(X[i, :]), 1:size(X, 1))...)'
    bpr_prior = BayesProbitRegression(
        Σ = τ^2 * Diagonal(ones(size(X_train, 2))),
        μ = vec(zeros(size(X_train, 2), 1)),
        β = β,
    )
    bpr = learn(X, y, bpr_prior, ϵ = ϵ, 
        x_min = x_min, x_max = x_max, y_min = y_min, y_max = y_max, 
        iteration_plots = iteration_plots,
        mark_coefficient = mark_coefficient, 
        class0_color = class0_color, class1_color = class1_color, 
        filename_base = filename_base, features = rbf_features)

    plot_decision_surface(X, y, bpr, 
        x_min = x_min, x_max = x_max, y_min = y_min, y_max = y_max, mark_coefficient = mark_coefficient, 
        class0_color = class0_color, class1_color = class1_color, 
        filename = filename_base * ".png", features = rbf_features)
end

"""
    plot_MNSIT_bayes_probit_regression(n=250; class0=1, class1=9, τ = 0.01, β = 10.0, ϵ = 1e-4, base = "~/Downloads/bayes_probit.png")

Learns a Bayesian probit regression model for `n` examples of the MNIST examples of the digits with the label `class0` and `class1` and outputs the learned weight vector as an image (using `no_iterations` many iterations)
"""
function plot_MNIST_bayes_probit_regression(
    n = 250;
    class0 = 1,
    class1 = 9,
    τ = 0.01,
    β = 10,
    ϵ = 1e-4,
    base = "~/Downloads/bayes_probit",
)
    # transformation on a weight vector to plot it as an image
    function plot_transform(x)
        return (hcat(map(r -> r[28:-1:1], eachrow(reshape(x[(28*28):-1:1], 28, 28)'))...)')
    end

    # plot the raw data 
    y = MNIST(split = :train).targets
    X = MNIST(split = :train).features
    idx0 = range(1, length(y))[y.==class0]
    idx1 = range(1, length(y))[y.==class1]
    X0 = hcat(map(i -> vec(X[:, :, idx0[i]]), 1:n)...)'
    X1 = hcat(map(i -> vec(X[:, :, idx1[i]]), 1:n)...)'
    mn = (sum(X1, dims = 1) + sum(X0, dims = 1)) / (2 * n)
    X0 -= repeat(mn, n)
    X1 -= repeat(mn, n)

    # augment the data with a one column for the bias and concatenate the two datasets
    X = vcat(X0, X1)
    y = vcat(zeros(n, 1), 1 * ones(n, 1))

    μ = vec(zeros(size(X, 2), 1))
    Σ = τ^2 * Diagonal(ones(size(X, 2)))
    bpr = learn(X, y, BayesProbitRegression(μ = μ, Σ = Σ, β = β), ϵ = ϵ)

    # plot the mean of the weight vector distribution
    p = heatmap(
        plot_transform(bpr.μ),
        colormap = :grays,
        legend = false,
        aspect_ratio = :equal,
        xaxis = nothing,
        yaxis = nothing,
        bordercolor = :white,
    )
    display(p)
    savefig(base * "_mean.png")

    # plot the variance of the weight vector distribution
    p = heatmap(
        plot_transform(diag(bpr.Σ)),
        colormap = :grays,
        legend = false,
        aspect_ratio = :equal,
        xaxis = nothing,
        yaxis = nothing,
        bordercolor = :white,
    )
    display(p)
    savefig(base * "_variance.png")
end


function main()
    plot_MNIST_bayes_probit_regression(
        1000,
        class0 = 1,
        class1 = 8,
        base = "~/Downloads/bayes_probit_1_vs_8",
    )
    plot_MNIST_bayes_probit_regression(
        1000,
        class0 = 1,
        class1 = 9,
        base = "~/Downloads/bayes_probit_1_vs_9",
    )
    plot_MNIST_bayes_probit_regression(
        1000,
        class0 = 3,
        class1 = 4,
        base = "~/Downloads/bayes_probit_3_vs_4",
    )

    plot_bayes_probit_regression(
        50,
        τ = 1,
        β = 5,
        filename_base = "~/Downloads/bayes_probit_100_τ=1",
    )
    plot_bayes_probit_regression(
        50,
        τ = 1,
        β = 5,
        mark_coefficient = true,
        filename_base = "~/Downloads/bayes_probit_100_τ=1_marked",
    )
    plot_bayes_probit_regression(
        50,
        τ = 10,
        β = 5,
        filename_base = "~/Downloads/bayes_probit_100_τ=10",
    )
    plot_bayes_probit_regression(
        50,
        τ = 10,
        β = 5,
        mark_coefficient = true,
        filename_base = "~/Downloads/bayes_probit_100_τ=10_marked",
    )
    plot_bayes_probit_regression(
        250,
        τ = 1,
        β = 5,
        filename_base = "~/Downloads/bayes_probit_500_τ=1",
    )
    plot_bayes_probit_regression(
        250,
        τ = 1,
        β = 5,
        mark_coefficient = true,
        filename_base = "~/Downloads/bayes_probit_500_τ=1_marked",
    )
    plot_bayes_probit_regression(
        250,
        τ = 10,
        β = 5,
        filename_base = "~/Downloads/bayes_probit_500_τ=10",
    )
    plot_bayes_probit_regression(
        250,
        τ = 10,
        β = 5,
        mark_coefficient = true,
        iteration_plots = true,
        filename_base = "~/Downloads/bayes_probit_500_τ=10_marked",
    )
end

end # module BayesProbit

using .BayesProbit
BayesProbit.main()