# Plots for Fisher Discriminants classification
#
# 2026 by Ralf Herbrich
# Hasso-Plattner Institute

module FisherDiscriminant

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
    plot_MNIST(n=500; class0_color=:red, class1_color=:blue, class0=1, class1=8, w=[1; 1; -0.25], compute_FDA=false)

Plots `n` examples of MNIST image feature vectors (top- and bottom-half average pixel intensity)
where one class has label `class0` and the other `class1`. If `compute_FDA` is false, uses the
vector `w` to plot a decision surface; otherwise, computes the FDA solution first and then plots.
Also plots histograms of the projected data along with fitted Gaussians.
"""
function plot_MNIST(
    n = 500;
    class0_color = :red,
    class1_color = :blue,
    class0 = 1,
    class1 = 8,
    w = [1; 1; -0.25],
    compute_FDA = false,
)
    # plot the raw data and one separating hyperplane
    y = MNIST(split = :train).targets
    X = MNIST(split = :train).features
    idx0 = range(1, length(y))[y.==class0]
    idx1 = range(1, length(y))[y.==class1]
    X0 = hcat(map(i -> average_intensity_features(X[:, :, idx0[i]]), 1:n)...)'
    X1 = hcat(map(i -> average_intensity_features(X[:, :, idx1[i]]), 1:n)...)'
    p = plot(
        X0[:, 1],
        X0[:, 2],
        legend = false,
        seriestype = :scatter,
        color = class0_color,
        alpha = 0.5,
    )
    plot!(X1[:, 1], X1[:, 2], seriestype = :scatter, color = class1_color, alpha = 0.5)

    if (compute_FDA)
        m0 = mean(X0, dims = 1)'
        m1 = mean(X1, dims = 1)'
        S = zeros(size(X0, 2), size(X0, 2))
        for x in eachrow(X0)
            S += (x - m0) * (x - m0)'
        end
        for x in eachrow(X1)
            S += (x - m1) * (x - m1)'
        end

        w = inv(S) * (m1 - m0)
        w = [w[1]; w[2]; -w' * (1 / 2 * m0 + 1 / 2 * m1)]
        println(w)
    end

    # augment the data with a one column for the bias
    X0 = hcat(X0, ones(size(X0, 1), 1))
    X1 = hcat(X1, ones(size(X1, 1), 1))

    x_min = -(0.35 * w[2] + w[3]) / w[1]
    x_max = -w[3] / w[1]
    xs = range(start = x_min, stop = x_max, length = 100)
    plot!(xs, x -> -(x * w[1] + w[3]) / w[2], linewidth = 2, color = :black)
    display(p)

    # plot the distribution of the projections
    f0, f1 = X0 * w, X1 * w
    P0, P1 = Normal(mean(f0), sqrt(var(f0))), Normal(mean(f1), sqrt(var(f1)))
    mn, mx = min(f0..., f1...), max(f0..., f1...)
    xs, bins = range(mn, mx, length = 100), range(mn, mx, length = 30)

    p = histogram(
        f0,
        label = L"f_0",
        normalize = :pdf,
        bins = bins,
        color = class0_color,
        alpha = 0.5,
    )
    histogram!(
        f1,
        label = L"f_1",
        normalize = :pdf,
        bins = bins,
        color = class1_color,
        alpha = 0.5,
    )
    plot!(xs, x -> pdf(P0, x), label = L"N(f_0)", lw = 3, color = class0_color)
    plot!(xs, x -> pdf(P1, x), label = L"N(f_1)", lw = 3, color = class1_color)
    display(p)
end

"""
    plot_MNIST_FDA(n=250; class0_color=:red, class1_color=:blue, class0=1, class1=8, no_scales=8, filename="~/Downloads/FDA.png")

Plots `n` examples of MNIST image feature vectors (top- and bottom-half average pixel intensity)
where one class has label `class0` and the other `class1`. Applies the FDA algorithm with RBF
features at `no_scales` different length scales and plots the resulting decision surface.
"""
function plot_MNIST_FDA(
    n = 250;
    class0_color = :red,
    class1_color = :blue,
    class0 = 1,
    class1 = 8,
    no_scales = 8,
    filename = "~/Downloads/FDA.png",
)
    # plot the raw data
    y = MNIST(split = :train).targets
    X = MNIST(split = :train).features
    idx0 = range(1, length(y))[y.==class0]
    idx1 = range(1, length(y))[y.==class1]
    X0 = hcat(map(i -> average_intensity_features(X[:, :, idx0[i]]), 1:n)...)'
    X1 = hcat(map(i -> average_intensity_features(X[:, :, idx1[i]]), 1:n)...)'

    # augment the data with a one column for the bias and concatenate the two datasets
    X = vcat(X0, X1)
    x_max, y_max = 1.2 * maximum(X, dims = 1)
    x_min, y_min = 1.2 * minimum(X, dims = 1)

    # the RBF features for this problem
    function rbf_features(x)
        local p = [
            (x1, x2, (x_max - x_min) / 2^s) for x1 in range(x_min, x_max, 5) for
            x2 in range(y_min, y_max, 5) for s = 0:no_scales
        ]
        return map(v -> exp(-(v[1] - x[1])^2 / v[3]^2 - (v[2] - x[2])^2 / v[3]^2), p)
    end

    X0_train = hcat(map(i -> rbf_features(X0[i, :]), 1:size(X0, 1))...)'
    X1_train = hcat(map(i -> rbf_features(X1[i, :]), 1:size(X1, 1))...)'

    m0 = vec(mean(X0_train, dims = 1))
    m1 = vec(mean(X1_train, dims = 1))
    S = zeros(size(X0_train, 2), size(X0_train, 2))
    for x in eachrow(X0_train)
        S += (vec(x) - m0) * (vec(x) - m0)'
    end
    for x in eachrow(X1_train)
        S += (vec(x) - m1) * (vec(x) - m1)'
    end

    w = pinv(S) * (m1 - m0)
    w0 = (-w'*(1/2*m0+1/2*m1))[1]

    # plot the posterior probability of each class prediction
    X_draw = range(start = x_min, stop = x_max, length = 100)
    Y_draw = range(start = y_min, stop = y_max, length = 100)
    p = contour(
        X_draw,
        Y_draw,
        (x1, x2) -> -vec(rbf_features([x1, x2]))'w - w0,
        levels = [0],
        linewidth = 3,
        color = :black,
        legend = false,
    )
    plot!(
        X0[:, 1],
        X0[:, 2],
        legend = false,
        seriestype = :scatter,
        color = class0_color,
        alpha = 0.5,
        aspect_ratio = :equal,
    )
    plot!(X1[:, 1], X1[:, 2], seriestype = :scatter, color = class1_color, alpha = 0.5)


    # decorate the plot
    xlims!(x_min, x_max)
    ylims!(y_min, y_max)
    xlabel!(L"x_1")
    ylabel!(L"x_2")
    display(p)
    savefig(filename)
end


"""
    learn_FDA(n=250; class0=1, class1=9, base="~/Downloads/bayes_log")

Learns an FDA model for `n` examples of the MNIST digits with labels `class0` and `class1`
using raw pixel features. Plots the class means, mean difference, and FDA weight vector as images.
"""
function learn_FDA(n = 250; class0 = 1, class1 = 9, base = "~/Downloads/bayes_log")
    # plot the raw data
    y = MNIST(split = :train).targets
    X = MNIST(split = :train).features
    idx0 = range(1, length(y))[y.==class0]
    idx1 = range(1, length(y))[y.==class1]
    X0 = hcat(map(i -> vec(X[:, :, idx0[i]]), 1:n)...)'
    X1 = hcat(map(i -> vec(X[:, :, idx1[i]]), 1:n)...)'

    # augment the data with a one column for the bias and concatenate the two datasets
    m0 = mean(X0, dims = 1)'
    m1 = mean(X1, dims = 1)'
    S = zeros(size(X0, 2), size(X0, 2))
    for x in eachrow(X0)
        S += (x - m0) * (x - m0)'
    end
    for x in eachrow(X1)
        S += (x - m1) * (x - m1)'
    end

    w = pinv(S) * (m1 - m0)

    # plot the mean of class0
    p = heatmap(
        plot_transform(m0),
        colormap = :grays,
        legend = false,
        aspect_ratio = :equal,
        xaxis = nothing,
        yaxis = nothing,
        bordercolor = :white,
    )
    display(p)

    # plot the mean of class1
    p = heatmap(
        plot_transform(m1),
        colormap = :grays,
        legend = false,
        aspect_ratio = :equal,
        xaxis = nothing,
        yaxis = nothing,
        bordercolor = :white,
    )
    display(p)

    # plot the mean difference
    p = heatmap(
        plot_transform(m0 - m1),
        colormap = :grays,
        legend = false,
        aspect_ratio = :equal,
        xaxis = nothing,
        yaxis = nothing,
        bordercolor = :white,
    )
    display(p)

    # plot the mean of FDA solution
    p = heatmap(
        plot_transform(w),
        colormap = :grays,
        legend = false,
        aspect_ratio = :equal,
        xaxis = nothing,
        yaxis = nothing,
        bordercolor = :white,
    )
    display(p)
end

"""
    main()

Runs all Fisher Discriminant Analysis demonstrations: plots MNIST data with various weight
vectors, computes FDA solutions, and generates decision surface plots.
"""
function main()
    plot_MNIST(w = [1; 1; -0.25])
    plot_MNIST(w = [1; -1; 0.05])
    plot_MNIST(compute_FDA = true)

    plot_MNIST_FDA(no_scales = 8, filename = "~/Downloads/FDA_9scales.png")
    plot_MNIST_FDA(no_scales = 4, filename = "~/Downloads/FDA_5scales.png")
    plot_MNIST_FDA(no_scales = 2, filename = "~/Downloads/FDA_3scales.png")

    # learn_FDA(1000)
end

end # module FisherDiscriminant

FisherDiscriminant.main()
