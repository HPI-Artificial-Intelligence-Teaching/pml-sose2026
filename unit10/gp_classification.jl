# Plots for Gaussian Processes classification
#
# 2026 by Ralf Herbrich
# Hasso-Plattner Institute

module GPClassification

using Random
using LinearAlgebra
using Distributions
using LaTeXStrings
using Plots
using MLDatasets

include("../shared/PMLCourse.jl")
using .PMLCourse

# structure storing the parameters for the Gaussian process probit regression
struct GPProbitRegression
    C::Cholesky{Float64, Matrix{Float64}}   # Cholesky decomposition of the kernel matrix augmented with the per-example noise variance
    α::Matrix{Float64}                      # inverse of the augmented kernel matrix multiplied with the per-example adjusted targets
    train_X::Matrix{Float64}                # training examples
    β2::Float64                             # latent variance
end


"""
    gppr_new = learn(X, y; kernel = RBF_kernel, β2 = 1.0, ϵ = 1e-4)

Learns a Gaussian process probit regression model from the examples in the row of `X` and the labels `y` and returns the new model (using the kernel function `kernel` and the latent noise variance `β2`).
"""
function learn(X, y; kernel = RBF_kernel, β2 = 1.0, ϵ = 1e-4)
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

    # extract the number of examples
    n = size(X, 1)

    # initialize the parameters of the message passing algorithm
    μ = zeros(n, 1)
    Σ0 =[kernel(x1, x2) for x1 in eachrow(X), x2 in eachrow(X)]
    Σ = copy(Σ0)
    a = zeros(n, 1)
    b = zeros(n, 1)

    # defines the message passing update for a single example at index `i`
    function update(i)
        Σϕ = Σ[:, i]

        # compute the mean and variance of the message from the Gaussian projection-noise to the latent variable
        c = 1 - b[i] * Σ[i, i]
        s2 = Σ[i, i] / c + β2
        s = sqrt(s2)
        m = (y[i] * μ[i] - a[i] * Σ[i, i]) / c

        # compute the precision and precision-mean of the message from the latent variable to the Gaussian projection-noise
        V = v(m / s, 0)
        W = w(m / s, 0)
        d = s2 * (1 - W)
        τ = (s * V + m * W) / d
        ρ = W / d

        # compute the site parameters a[i] and b[i] and update the posterior mean and covariance
        a_new = τ / (1 + ρ * β2)
        b_new = ρ / (1 + ρ * β2)

        Δa = a_new - a[i]
        Δb = b_new - b[i]

        c2 = 1 + Δb * Σ[i, i]
        Σ -= Δb / c2 * Σϕ * Σϕ'
        μ += Σϕ * (Δa * y[i] - Δb * μ[i]) / c2

        a[i] = a_new
        b[i] = b_new
        return
    end

    # run message passing for `no_iterations` many iterations for each example
    Δ = ϵ
    while (Δ ≥ ϵ)
        a_tmp = copy(a)
        b_tmp = copy(b)
        for idx = 1:n
            update(idx)
        end
        Δ = max(maximum(abs.(a - a_tmp)), maximum(abs.(b - b_tmp)))
        println("Δ = ", Δ)
    end

    # K_inv = inv(Σ0)
    # K_inv_y = zeros(n, 1)

    C = cholesky(Σ0 + Diagonal(1 ./ b[:]))
    α = C.U \ (C.L \ ((a[:] .* y) ./ b[:]))

    return (GPProbitRegression(C, α, X, β2))
end

# compute the posterior probability at the example (`x1`,`x2`)
function posterior(x, gppr; kernel = RBF_kernel)
    k = [kernel(x, x_train) for x_train in eachrow(gppr.train_X)]
    μ = k' * gppr.α
    σ2 = kernel(x, x) + gppr.β2 - k' * (gppr.C.U \ (gppr.C.L \ k))
    N = Normal(μ[1], sqrt(σ2[1]))
    return (cdf(N, 0))
end

"""
    plot_gp_logistic(n=250; class0_color = :red, class1_color = :blue, class0=1, class1=9, no_iterations=3)

Plots `n` examples of MNIST image feature vectors (top- and bottom-half average pixel intensity) where the one class is the digits with the label `class0` and the other class is the digits with the label `class1` (using the colors `class0_color` and `class1_color`, respectively). Then performs the Gaussian Process (logit) regression learning with Laplace approximation and plots the decision surface.
"""
function plot_gp_logistic(
    n = 250;
    class0_color = :red,
    class1_color = :blue,
    class0 = 1,
    class1 = 9,
    no_iterations = 10,
    kernel = RBF_kernel,
)
    # extract the data and centralizes by subtracting the mean for all dimensions
    y = MNIST(split = :train).targets
    X = MNIST(split = :train).features
    idx0 = range(1, length(y))[y.==class0]
    idx1 = range(1, length(y))[y.==class1]
    X0 = hcat(map(i -> average_intensity_features(X[:, :, idx0[i]]), 1:n)...)'
    X1 = hcat(map(i -> average_intensity_features(X[:, :, idx1[i]]), 1:n)...)'
    mn = (sum(X1, dims = 1) + sum(X0, dims = 1)) / (2 * n)
    X0 -= repeat(mn, n)
    X1 -= repeat(mn, n)
    trainX = vcat(X0, X1)
    trainY = vcat(zeros(n, 1), 1 * ones(n, 1))

    # compute the kernel matrix
    C = [kernel(x1, x2) for x1 in eachrow(trainX), x2 in eachrow(trainX)]
    Cinv = inv(C)
    f = hcat(zeros(size(C)[1]))

    # # run Newton-Raphson to compute the mode of the posterior
    for idx = 1:no_iterations
        g = exp.(f) ./ (1 .+ exp.(f))
        d = (trainY - g) - Cinv * f
        H = Diagonal(vec(-g .* (1 .- g))) - Cinv
        Δ = inv(H) * d
        println("Iteration ", idx, ": e = ", norm(Δ), ": d = ", norm(d))
        f -= Δ
    end
    m = f

    # compute the inverse of the Hessian for the covariance
    g = exp.(m) ./ (1 .+ exp.(m))
    d = (trainY - g) - Cinv * m
    H = Cinv - Diagonal(vec(-g .* (1 .- g)))
    A = inv(H)
    α = Cinv * m
    B = Cinv - Cinv * A * Cinv
    function posterior(x1, x2)
        k = [kernel(x, [x1, x2]) for x in eachrow(trainX)]
        μ = (k'*α)[1]
        σ2 = kernel([x1, x2], [x1, x2]) - k' * B * k + π / 8
        N = Normal(μ, sqrt(σ2))
        return (cdf(N, 0))
    end

    # plot the posterior probability of each class prediction
    x_max, y_max = 1.2 * maximum(trainX, dims = 1)
    x_min, y_min = 1.2 * minimum(trainX, dims = 1)
    X = range(start = x_min, stop = x_max, length = 100)
    Y = range(start = y_min, stop = y_max, length = 100)
    p = heatmap(X, Y, (x1, x2) -> posterior(x1, x2), legend = false, color = :berlin)
    contour!(
        X,
        Y,
        (x1, x2) -> posterior(x1, x2),
        levels = [0.5],
        linewidth = 1,
        color = :white,
    )
    plot!(
        X0[:, 1],
        X0[:, 2],
        seriestype = :scatter,
        color = class0_color,
        alpha = 0.5,
        legend = false,
        aspect_ratio = :equal,
    )
    plot!(
        X1[:, 1],
        X1[:, 2],
        seriestype = :scatter,
        color = class1_color,
        alpha = 0.5
    )

    # decorate the plot
    xlims!(x_min, x_max)
    ylims!(y_min, y_max)
    xlabel!(L"x_1")
    ylabel!(L"x_2")
    display(p)
end

"""
    plot_gp_probit(n=250; class0_color = :red, class1_color = :blue, class0=1, class1=9)

Plots `n` examples of MNIST image feature vectors (top- and bottom-half average pixel intensity) where the one class is the digits with the label `class0` and the other class is the digits with the label `class1` (using the colors `class0_color` and `class1_color`, respectively). Then performs the Gaussian process probit regression learning via message passing (mp) and plots the decision surface.
"""
function plot_gp_probit(
    n = 250;
    class0_color = :red,
    class1_color = :blue,
    class0 = 1,
    class1 = 9,
    kernel = RBF_kernel,
)
    # extract the data and centralizes by subtracting the mean for all dimensions
    y = MNIST(split = :train).targets
    X = MNIST(split = :train).features
    idx0 = range(1, length(y))[y.==class0]
    idx1 = range(1, length(y))[y.==class1]
    X0 = hcat(map(i -> average_intensity_features(X[:, :, idx0[i]]), 1:n)...)'
    X1 = hcat(map(i -> average_intensity_features(X[:, :, idx1[i]]), 1:n)...)'
    mn = (sum(X1, dims = 1) + sum(X0, dims = 1)) / (2 * n)
    X0 -= repeat(mn, n)
    X1 -= repeat(mn, n)
    trainX = vcat(X0, X1)
    trainY = vcat(-1 * ones(n, 1), 1 * ones(n, 1))

    # computes the GP probit regression model
    gppr = learn(trainX, trainY; kernel = kernel, β2 = 1, ϵ = 1e-4)

    # plot the posterior probability of each class prediction
    x_max, y_max = 1.2 * maximum(trainX, dims = 1)
    x_min, y_min = 1.2 * minimum(trainX, dims = 1)
    X = range(start = x_min, stop = x_max, length = 100)
    Y = range(start = y_min, stop = y_max, length = 100)
    p = heatmap(X, Y, (x1, x2) -> posterior([x1, x2], gppr, kernel = kernel), legend = false, color = :berlin)
    contour!(
        X,
        Y,
        (x1, x2) -> posterior([x1, x2], gppr, kernel = kernel),
        levels = [0.5],
        linewidth = 1,
        color = :white,
    )
    plot!(
        X0[:, 1],
        X0[:, 2],
        seriestype = :scatter,
        color = class0_color,
        alpha = 0.5,
        legend = false,
        aspect_ratio = :equal,
    )
    plot!(
        X1[:, 1],
        X1[:, 2],
        seriestype = :scatter,
        color = class1_color,
        alpha = 0.5
    )

    # decorate the plot
    xlims!(x_min, x_max)
    ylims!(y_min, y_max)
    xlabel!(L"x_1")
    ylabel!(L"x_2")
    display(p)
end

function main()
    plot_gp_logistic(50, kernel = (x1, x2) -> RBF_kernel(x1, x2, λ = 0.1))
    savefig("~/Downloads/gp_logistic_50_lambda=0.1.png")
    plot_gp_logistic(50, kernel = (x1, x2) -> RBF_kernel(x1, x2, λ = 0.05))
    savefig("~/Downloads/gp_logistic_50_lambda=0.03.png")
    plot_gp_logistic(250, kernel = (x1, x2) -> RBF_kernel(x1, x2, λ = 0.1))
    savefig("~/Downloads/gp_logistic_250_lambda=0.1.png")
    plot_gp_logistic(250, kernel = (x1, x2) -> RBF_kernel(x1, x2, λ = 0.05))
    savefig("~/Downloads/gp_logistic_250_lambda=0.03.png")

    plot_gp_probit(50, kernel = (x1, x2) -> RBF_kernel(x1, x2, λ = 0.1))
    savefig("~/Downloads/gp_probit_50_lambda=0.1.png")
    plot_gp_probit(50, kernel = (x1, x2) -> RBF_kernel(x1, x2, λ = 0.05))
    savefig("~/Downloads/gp_probit_50_lambda=0.03.png")
    plot_gp_probit(250, kernel = (x1, x2) -> RBF_kernel(x1, x2, λ = 0.1))
    savefig("~/Downloads/gp_probit_250_lambda=0.1.png")
    plot_gp_probit(250, kernel = (x1, x2) -> RBF_kernel(x1, x2, λ = 0.05))
    savefig("~/Downloads/gp_probit_250_lambda=0.03.png")
end

main()

end # module GPClassification
