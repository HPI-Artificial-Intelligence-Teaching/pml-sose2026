# Plots for Bayesian classification
#
# 2026 by Ralf Herbrich
# Hasso-Plattner Institute

module BayesClassification

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
    plot_bayesian_inference(X, y; τ=sqrt(1/2), no_samples=200, base_name="fig")

Generates the plots of data, likelihood, posterior and `no_samples` function samples for a given 2D dataset and stores them in a folder with the base name `base_name`
"""
function plot_bayesian_inference(
    X,
    y;
    τ = sqrt(1 / 2),
    base_name = "fig",
)
    # Computes the logistic likelihood of the 2D linear function for the dataset given by `X` and `y` 
    function likelihood(w)
        t = X * w
        z = exp.(t .* 20) ./ (1 .+ exp.(t .* 20))
        return (prod(y .* z .+ (1 .- y) .* (1 .- z)))
    end

    # generates a 3D surface plot with a mesh for a given 2D->1D function and color scheme
    function plot_surface(f; colors = :blues, alpha = 0.1, start=-4, stop=4)
        w1 = range(start, stop = stop, length = 100)
        w2 = range(start, stop = stop, length = 100)
        w1_coarse = range(start, stop = stop, length = 30)
        w2_coarse = range(start, stop = stop, length = 30)

        Z = sum([f([x; y]) for x in w1, y in w2])
        Z_coarse = sum([f([x; y]) for x in w1_coarse, y in w2_coarse])

        p = plot(
            w1,
            w2,
            (w1, w2) -> f([w1, w2]) / Z,
            seriestype = :surface,
            line_z = 0.5,
            color = colors,
            fillalpha = alpha,
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
                [f([w1; w2]) / Z_coarse for w2 in w2_coarse],
                line_z=[f([w1; w2]) for w2 in w2_coarse],
                color=:black,
                linewidth=0.5
            )
        end
    
        for w2 in w2_coarse
            plot!(
                w1_coarse,
                [w2 for w1 in w1_coarse],
                [f([w1; w2]) / Z_coarse for w1 in w1_coarse],
                line_z=[f([w1; w2]) for w1 in w1_coarse],
                color=:black,
                linewidth=0.5
            )
        end
    
        xlabel!(L"w_1")
        ylabel!(L"w_2")
        display(p)
    end

    # compute the Bayesian prior
    Σ = τ^2 * Diagonal(ones(2))
    μ = zeros(2)

    # plot the prior over the weight space
    plot_surface(
        W -> pdf(MvNormal(μ, Σ), W)[1],
        colors = :blues,
        alpha = 0.1,
    )
    savefig(p, base_name * "_prior.png")

    # plot the likelihood over the weight space
    plot_surface(
        likelihood,
        colors = :reds,
        alpha = 0.1,
    )
    savefig(p, base_name * "_likel.png")

    # plot the posterior over the weight space
    plot_surface(
        W -> pdf(MvNormal(μ, Σ), W)[1] * likelihood(W),
        colors = :greens,
        alpha = 0.1,
    )
    savefig(p, base_name * "_post.png")

    # plot the input space
    x_max, y_max = 1.1 * maximum(X, dims = 1)
    x_min, y_min = 1.1 * minimum(X, dims = 1)
    idx0 = range(1, length(y))[y.==0]
    idx1 = range(1, length(y))[y.==1]
    p = scatter(
        X[idx0, 1],
        X[idx0, 2],
        legend = false,
        color = :red,
        aspect_ratio = :equal,
        xtickfontsize = 14,
        ytickfontsize = 14,
        ztickfontsize = 14,
        xguidefontsize = 16,
        yguidefontsize = 16,
        zguidefontsize = 16,
    )
    scatter!(
        X[idx1, 1],
        X[idx1, 2], 
        color = :blue,
    )
    # decorate the plot
    xlims!(x_min, x_max)
    ylims!(y_min, y_max)
    xlabel!(L"x")
    ylabel!(L"y")
    display(p)
    savefig(p, base_name * "_data.png")    


    # sample from the posterior over w (with importance sampling) and compute the 
    # weighted average of predictive probabilities at each point in the input space
    no_samples = 100
    W_samples = rand(MvNormal(μ, Σ), no_samples)
    weights = [likelihood(w) for w in eachcol(W_samples)]
    weights /= sum(weights)
    function predictive_probabilities(x)
        return sum(weights .* [1 ./ (1 .+ exp.(x'*w .* 20)) for w in eachcol(W_samples)])
    end

    # plots the input space with the predictive probabilities
    X_coarse = range(start = x_min, stop = x_max, length = 100)
    Y_coarse = range(start = y_min, stop = y_max, length = 100)
    p = heatmap(
        X_coarse,
        Y_coarse,
        (x, y) -> predictive_probabilities([x; y]),
        color = :berlin,
        legend = false,
        xtickfontsize = 14,
        ytickfontsize = 14,
        ztickfontsize = 14,
        xguidefontsize = 16,
        yguidefontsize = 16,
        zguidefontsize = 16,
    )
    scatter!(
        X[idx0, 1],
        X[idx0, 2],
        legend = false,
        color = :red,
        aspect_ratio = :equal,
    )
    scatter!(
        X[idx1, 1],
        X[idx1, 2], 
        color = :blue,
    )
    contour!(
        X_coarse,
        Y_coarse,
        (x, y) -> predictive_probabilities([x; y]),
        levels = [0.5],
        linewidth = 2,
        color = :white,
    )
    # decorate the plot
    xlims!(x_min, x_max)
    ylims!(y_min, y_max)
    xlabel!(L"x")
    ylabel!(L"y")
    display(p)
    savefig(p, base_name * "_data_with_predictions.png")    
end

function main()
    X, y = extract_mnist_data(50)

    plot_bayesian_inference(
        X[[1, 2, 51, 52], :],
        y[[1, 2, 51, 52]],
        base_name = "~/Downloads/bayes2",
    )
    plot_bayesian_inference(
        X[[1, 2, 3, 4, 51, 52, 53, 54], :],
        y[[1, 2, 3, 4, 51, 52, 53, 54]],
        base_name = "~/Downloads/bayes4",
    )
    plot_bayesian_inference(
        X[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60], :],
        y[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60]],
        base_name = "~/Downloads/bayes10",
    )

    plot_bayesian_inference(
        X[1:100, :],
        y[1:100],
        base_name = "~/Downloads/bayes50",
    )
end

end # module BayesClassification

using .BayesClassification
BayesClassification.main()