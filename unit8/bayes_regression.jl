# Plots for polynomial regression
#
# 2026 by Ralf Herbrich
# Hasso-Plattner Institute

module BayesianRegression

using Random
using LinearAlgebra
using Distributions
using LaTeXStrings
using Plots


"""
    X, y = generate_data_points(n; a0 = 0.5, a1 = -0.3, σ=0.2)

Generates `n` data points for a fixed 2D linear function with zero-mean Gaussian noise of standard deviation `σ`

# Examples
```jldoctest
julia> X, y = generate_data_points(3)
([0.0001399347625403724 1.0; 0.995233586895877 1.0; 0.3592783640142643 1.0], [-0.09607863062267713; 0.44637719230488665; -0.08091636576342287;;])
```
"""
function generate_data_points(n; a0 = 0.5, a1 = -0.3, σ = 0.2)
    # generates training data 
    Random.seed!(40)
    X = hcat(2 * rand(n, 1) - ones(n, 1), ones(n, 1))
    y = X * [a0; a1] + randn(n, 1) * σ

    return X, y
end

"""
    plot_bayesian_inference(X, y; σ=0.2, τ=sqrt(1/2), no_samples=200, base_name="fig")

Generates the plots of data, likelihood, posterior and `no_samples` function samples for a given 2D dataset and stores them in a folder with the base name `base_name`

# Examples
```jldoctest
julia> plot_bayesian_inference(X, y; σ=0.2, τ=sqrt(1/2), no_samples=200, base_name="~/Downloads/bayes3")
"/Users/rherbrich/Downloads/bayes3_functions.png"
```
"""
function plot_bayesian_inference(
    X,
    y;
    σ = 0.2,
    τ = sqrt(1 / 2),
    no_samples = 200,
    base_name = "fig",
)
    # Computes the likelihood of the 2D linear function for the dataset given by `X` and `y` using the noise standard deviation `σ`
    function likelihood(w; σ = sqrt(1 / 2))
        return pdf(MvNormal(X * w, I * σ^2), y)
    end

    # generates a 3D surface plot with a mesh for a given 2D->1D function and color scheme
    function plot_surface(f; colors = :blues, alpha = 0.1, start=-1, stop=1)
        w1 = range(start, stop = stop, length = 100)
        w2 = range(start, stop = stop, length = 100)
        w1_coarse = range(start, stop = stop, length = 30)
        w2_coarse = range(start, stop = stop, length = 30)
        p = plot(
            w1,
            w2,
            (w1, w2) -> f([w1, w2]),
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
                [f([w1; w2]) for w2 in w2_coarse],
                line_z=[f([w1; w2]) for w2 in w2_coarse],
                color=:black,
                linewidth=0.5
            )
        end
    
        for w2 in w2_coarse
            plot!(
                w1_coarse,
                [w2 for w1 in w1_coarse],
                [f([w1; w2]) for w1 in w1_coarse],
                line_z=[f([w1; w2]) for w1 in w1_coarse],
                color=:black,
                linewidth=0.5
            )
        end
    
        xlabel!(L"w_1")
        ylabel!(L"w_2")
        display(p)
    end

    # compute the Bayesian posterior
    Σ = Hermitian(inv(1 / σ^2 * X' * X + 1 / τ^2 * Diagonal(ones(2))))
    μ = vec(1 / σ^2 * Σ * X' * y)

    # plot the likelihood over the weight space
    plot_surface(
        W -> likelihood(W)[1],
        colors = :reds,
        alpha = 0.1,
        start=-1,
        stop=1
    )
    savefig(p, base_name * "_likel.png")

    # plot the posterior over the weight space
    plot_surface(
        W -> pdf(MvNormal(μ, Σ), W)[1],
        colors = :greens,
        alpha = 0.1,
        start=-1,
        stop=1
    )
    savefig(p, base_name * "_post.png")

    # plot the input space
    p = plot(
        X[:, 1],
        y,
        color = :blue,
        seriestype = :scatter,
        legend = false,
        xtickfontsize = 14,
        ytickfontsize = 14,
        xguidefontsize = 16,
        yguidefontsize = 16,
    )
    xlabel!(L"x")
    ylabel!(L"y")
    xlims!(-1, 1)
    ylims!(-1, 1)
    display(p)
    savefig(p, base_name * "_data.png")

    xs = range(-1, stop = 1, length = 100)
    w = rand(MvNormal(μ, Σ), no_samples)
    for j = 1:no_samples
        width = if (j % no_samples ÷ 3 == 0)
            0.1
        else
            0.1
        end
        plot!(xs, map(x -> w[1, j] * x + w[2, j], xs), linewidth = width, color = :red)
    end
    plot!(X[:, 1], y, color = :blue, seriestype = :scatter)
    display(p)
    savefig(p, base_name * "_functions.png")
end

function main()
    X, y = generate_data_points(20)

    plot_bayesian_inference(X[1:2, :], y[1:2], base_name = "~/Downloads/bayes2")
    plot_bayesian_inference(X[1:3, :], y[1:3], base_name = "~/Downloads/bayes3")
    plot_bayesian_inference(X[1:5, :], y[1:5], base_name = "~/Downloads/bayes5")
    plot_bayesian_inference(X[1:20, :], y[1:20], base_name = "~/Downloads/bayes20")
end

main()

end # module BayesianRegression