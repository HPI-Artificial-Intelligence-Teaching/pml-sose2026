# Plots for visualization of inference with two different parameterizations of Gaussian distributions
#
# 2026 by Ralf Herbrich
# Hasso-Plattner Institute

module GaussianInference

using Plots
using Distributions
using LaTeXStrings
using Random

"""
    Gaussian(τ, ρ)

A Gaussian distribution in natural (information) parameterization with precision-weighted
mean `τ` and precision `ρ`. The precision `ρ` must be non-negative.
"""
struct Gaussian
    τ::Float64
    ρ::Float64

    # default constructor
    Gaussian(τ, ρ) =
        (ρ < 0) ? error("precision of a Gaussian must be non-negative") :
        new(promote(τ, ρ)...)
end

"""
    Gaussian()

Initializes a standard Gaussian distribution with `τ=0` and `ρ=1` (i.e., mean 0 and variance 1).
"""
Gaussian() = Gaussian(0, 1)

"""
    GaussianFromMeanVariance(μ, σ²)

Creates a `Gaussian` from the given mean `μ` and variance `σ²` by converting to natural parameters.
"""
function GaussianFromMeanVariance(μ, σ²)
    return Gaussian(μ / σ², 1 / σ²)
end

"""
    μ(g::Gaussian)

Returns the mean of the Gaussian distribution `g`.
"""
function μ(g::Gaussian)
    return g.τ / g.ρ
end

"""
    σ²(g::Gaussian)

Returns the variance of the Gaussian distribution `g`.
"""
function σ²(g::Gaussian)
    return 1 / g.ρ
end

"""
    Base.:*(g1::Gaussian, g2::Gaussian)

Multiplies two Gaussian distributions `g1` and `g2` by adding their natural parameters,
which corresponds to multiplying the density functions (up to normalization).
"""
function Base.:*(g1::Gaussian, g2::Gaussian)
    return Gaussian(g1.τ + g2.τ, g1.ρ + g2.ρ)
end

"""
    generate_sample(n; μ=0.0, σ²=1.0, β²=1.0)

Generates `n` noisy observations from a Gaussian model: first draws a true mean `m ~ N(μ, σ²)`,
then draws `n` samples from `N(m, β²)`. Prints the true mean and returns the sample vector.
"""
function generate_sample(n::Int64; μ=0.0, σ²=1.0, β²=1.0)
    m = rand(Normal(μ, sqrt(σ²)), 1)[1]
    println("True mean: $m")
    return rand(Normal(m, sqrt(β²)), n)
end

"""
    plot_Gaussian_inference_μσ²(sample, prior; β²=1.0)

Plots the trajectory of Bayesian inference updates in the mean-variance (μ, σ²) space.
Starting from `prior`, each observation in `sample` is incorporated via multiplication
with a likelihood `N(x, β²)`. The trajectory is plotted on a log-scale for σ².
"""
function plot_Gaussian_inference_μσ²(sample::Vector{Float64}, prior::Gaussian; β²=1.0)
    posteriors = Vector{Gaussian}()
    for (i, x) in enumerate(sample)
        if i == 1
            push!(posteriors, prior)
        end
        posterior = prior * GaussianFromMeanVariance(x, β²)
        push!(posteriors, posterior)
        prior = posterior
    end

    μs = map(d -> μ(d), posteriors)
    σ²s = map(d -> σ²(d), posteriors)
    p = plot(
        μs,
        σ²s,
        legend=false,
        yscale=:log10,
        linewidth=3,
        color=:blue,
        xtickfontsize=14,
        ytickfontsize=14,
        xguidefontsize=16,
        yguidefontsize=16,
    )
    scatter!(μs, σ²s)
    xlabel!(L"\mu")
    ylabel!(L"\sigma^2")
    display(p)
end

"""
    plot_Gaussian_inference_τρ(sample, prior; β²=1.0)

Plots the trajectory of Bayesian inference updates in the natural (τ, ρ) parameter space.
Starting from `prior`, each observation in `sample` is incorporated via multiplication
with a likelihood `N(x, β²)`. The trajectory is plotted in the τ-ρ plane.
"""
function plot_Gaussian_inference_τρ(sample::Vector{Float64}, prior::Gaussian; β²=1.0)
    posteriors = Vector{Gaussian}()
    for (i, x) in enumerate(sample)
        if i == 1
            push!(posteriors, prior)
        end
        posterior = prior * GaussianFromMeanVariance(x, β²)
        push!(posteriors, posterior)
        prior = posterior
    end

    τs = map(d -> d.τ, posteriors)
    ρs = map(d -> d.ρ, posteriors)
    p = plot(
        τs,
        ρs,
        legend=false,
        linewidth=3,
        color=:blue,
        xtickfontsize=14,
        ytickfontsize=14,
        xguidefontsize=16,
        yguidefontsize=16,
    )
    scatter!(τs, ρs)
    xlabel!(L"\tau")
    ylabel!(L"\rho")
    display(p)
end

"""
    plot_dirac_animation()

Creates an animation showing a sequence of Gaussian distributions `N(0, 1/i²)` for
`i = 1, ..., 50`, illustrating convergence to a Dirac delta distribution as the
variance shrinks to zero. Returns the animation object.
"""
function plot_dirac_animation()

    x = range(-3, 3, length=1000)
    anim = @animate for i in 1:50
        plot(
            legend=false,
            xlim=(-3, 3),
            xtickfontsize=14,
            ytickfontsize=14,
            xguidefontsize=16,
            yguidefontsize=16,
        )
        plot!(x, pdf(Normal(0, 1/i), x), color=:blue, linewidth=3)
        ylabel!(L"p(x)")
        xlabel!(L"x")
    end

    return anim
end

"""
    plot_uniform_animation()

Creates an animation showing a sequence of Gaussian distributions `N(0, i²)` for
`i = 1, ..., 50`, illustrating convergence towards a uniform distribution as the
variance grows without bound. Returns the animation object.
"""
function plot_uniform_animation()

    x = range(-3, 3, length=1000)
    anim = @animate for i in 1:50
        plot(
            legend=false,
            xlim=(-3, 3),
            ylim=(0, 1),
            xtickfontsize=14,
            ytickfontsize=14,
            xguidefontsize=16,
            yguidefontsize=16,
        )
        plot!(x, pdf(Normal(0, i), x), color=:blue, linewidth=3)
        ylabel!(L"p(x)")
        xlabel!(L"x")
    end

    return anim
end

function main()
    # set the random seed to 42
    Random.seed!(42)
    data = generate_sample(20, β²=0.1)

    plot_Gaussian_inference_μσ²(data, Gaussian(0, 1), β²=0.1)
    savefig("~/Downloads/gaussian_inference_μσ².svg")
    plot_Gaussian_inference_τρ(data, Gaussian(0, 1), β²=0.1)
    savefig("~/Downloads/gaussian_inference_τρ.svg")

    anim = plot_dirac_animation()
    gif(anim, "~/Downloads/dirac.gif", fps=10)
    anim = plot_uniform_animation()
    gif(anim, "~/Downloads/uniform.gif", fps=10)
end

end # module GaussianInference

GaussianInference.main()
