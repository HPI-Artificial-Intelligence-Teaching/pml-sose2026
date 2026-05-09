# Computes the closest Gaussian approximation for a mixture of Gaussian
#
# 2026 by Ralf Herbrich
# Hasso-Plattner Institute

module Distance

using LaTeXStrings
using Plots
using Distributions
using Optim

"""
    MixtureOfGaussian(weights, gaussians)

A mixture of Gaussian distributions defined by a vector of non-negative `weights`
(which should sum to 1) and a corresponding vector of `Normal` distributions.
"""
struct MixtureOfGaussian
    weights::Vector{Float64}
    gaussians::Vector{Normal}
end

"""
    ScaledNormal(normal, scale)

A non-normalized Gaussian: a `Normal` distribution multiplied by a positive `scale`.
Its "density" at `x` is `scale * pdf(normal, x)` and its integral is `scale`.
"""
struct ScaledNormal
    normal::Normal
    scale::Float64
end

Distributions.pdf(s::ScaledNormal, x::Real) = s.scale * pdf(s.normal, x)

"""
    get_range(mog::MixtureOfGaussian)

Computes the effective range `(min_x, max_x)` of a mixture of Gaussians by taking the
minimum and maximum of `μ ∓ 6σ` across all component distributions.
"""
function get_range(mog::MixtureOfGaussian)
    min_x = minimum(map(x -> x.μ, mog.gaussians) - 6 * map(x -> x.σ, mog.gaussians))
    max_x = maximum(map(x -> x.μ, mog.gaussians) + 6 * map(x -> x.σ, mog.gaussians))
    return min_x, max_x
end

"""
    mog_pdf(mog::MixtureOfGaussian, x::Float64)

Returns the probability density of the mixture of Gaussians `mog` evaluated at `x`,
computed as the weighted sum of the component Gaussian PDFs.
"""
function mog_pdf(mog::MixtureOfGaussian, x::Float64)
    y = 0.0
    for i in 1:length(mog.weights)
        y += mog.weights[i] * pdf(mog.gaussians[i], x)
    end
    return y
end

"""
    plot_mog(mog::MixtureOfGaussian; approx=nothing, title=nothing, ylim=nothing)

Plots the PDF of a mixture of Gaussians. Optionally overlays an approximate `Normal`
distribution in red, sets a plot `title`, and constrains the y-axis with `ylim`.
"""
function plot_mog(mog::MixtureOfGaussian; approx = nothing, title=nothing, ylim=nothing)
    min_x, max_x = get_range(mog)
    xs = range(min_x, max_x, length=1000)

    plt = plot(xs,
        x -> mog_pdf(mog, x),
        label=false,
        xlabel=L"x",
        ylabel=L"p(x)",
        linewidth=8,
        xtickfontsize=14,
        ytickfontsize=14,
        xguidefontsize=16,
        yguidefontsize=16,
        legendfontsize=16,
    )
    if !isnothing(approx)
        plot!(xs,
            x -> pdf(approx, x),
            linewidth=8,
            label = false,
            color=:red,
        )
    end
    if !isnothing(title)
        title!(title)
    end
    if !isnothing(ylim)
        ylims!(ylim)
    end
    return (plt)
end

"""
    KL(p::Vector{Float64}, q::Vector{Float64})

Computes the generalized forward Kullback-Leibler divergence `KL(p || q)` between two
discretized (possibly non-normalized) densities `p` and `q` (each given as
density-times-`dx` on a uniform grid). Defined as
`∫ p log(p/q) dx + ∫ (q - p) dx`, i.e. `sum(p .* log.(p./q) .+ (q .- p))`. For
normalized `p` and `q` the correction term vanishes.
"""
function KL(p::Vector{Float64}, q::Vector{Float64})
    return sum(p .* log.(p ./ q) .+ (q .- p))
end

"""
    KL_reverse(p::Vector{Float64}, q::Vector{Float64})

Computes the generalized reverse Kullback-Leibler divergence `KL(q || p)` between two
discretized (possibly non-normalized) densities `p` and `q`, defined as
`∫ q log(q/p) dx + ∫ (p - q) dx`.
"""
function KL_reverse(p::Vector{Float64}, q::Vector{Float64})
    return sum(q .* log.(q ./ p) .+ (p .- q))
end

"""
    α_divergence(p::Vector{Float64}, q::Vector{Float64}; α=0.5)

Computes the generalized alpha-divergence between two discretized (possibly
non-normalized) densities `p` and `q`. For `α=1` this reduces to `KL(p || q)`, for
`α=0` to `KL(q || p)`, and otherwise to
`∫ (α p + (1-α) q - p^α q^(1-α)) dx / (α (1-α))`.
"""
function α_divergence(p::Vector{Float64}, q::Vector{Float64}; α=0.5)
    if α == 1
        return KL(p, q)
    elseif α == 0
        return KL_reverse(p, q)
    else
        return sum(α .* p .+ (1 - α) .* q .- (p.^α) .* (q.^(1 - α))) / (α * (1 - α))
    end
end

"""
    mean_and_variance(mog::MixtureOfGaussian)

Computes the mean and variance of a mixture of Gaussians `mog` in closed form:
`μ = Σ wᵢ μᵢ` and `σ² = Σ wᵢ (σᵢ² + μᵢ²) − μ²`.
"""
function mean_and_variance(mog::MixtureOfGaussian)
    μ = sum(w * g.μ for (w, g) in zip(mog.weights, mog.gaussians))
    σ2 = sum(w * (g.σ^2 + g.μ^2) for (w, g) in zip(mog.weights, mog.gaussians)) - μ^2
    return μ, σ2
end

"""
    closest_gaussian(mog::MixtureOfGaussian; n=10000, distance=KL)

Finds the closest non-normalized Gaussian approximation `Z * N(μ, σ)` to a mixture
of Gaussians `mog` by Nelder-Mead minimization over `(μ, log σ, log Z)`, using the
generalized `distance` (which permits unequal mass between `p` and `q`). Uses `n`
discretization points; integrals are approximated via the rectangle rule on a
uniform grid with spacing `dx`, and densities are passed in as `density * dx`.
Returns a `ScaledNormal`.
"""
function closest_gaussian(mog::MixtureOfGaussian; n = 10000, distance = KL)
    μ0, σ2_0 = mean_and_variance(mog)
    σ0 = sqrt(σ2_0)
    xs = collect(range(μ0 - 10*σ0, μ0 + 10*σ0, length=n))
    dx = xs[2] - xs[1]
    p_mog = [mog_pdf(mog, x) for x in xs] .* dx

    # Optimize over (μ, log σ, log Z) so σ and Z stay positive without box constraints.
    function objective(params)
        μ, logσ, logZ = params[1], params[2], params[3]
        σ = exp(logσ)
        Z = exp(logZ)
        q = Z .* [pdf(Normal(μ, σ), x) for x in xs] .* dx
        return distance(p_mog, q)
    end

    result = optimize(objective, [μ0, log(σ0), 0.0], NelderMead())
    best_μ, best_logσ, best_logZ = Optim.minimizer(result)
    best_σ = exp(best_logσ)
    best_Z = exp(best_logZ)

    println("Best: μ=$best_μ, σ²=$(best_σ^2), Z=$best_Z")

    return ScaledNormal(Normal(best_μ, best_σ), best_Z)
end

"""
    plot_α_anim(mog; αs=range(-1.5, 1.5, length=30), anim_filename=..., μ_match_filename=..., σ2_match_filename=...)

Generates an animation of optimal Gaussian approximations of the alpha-divergence over all
`α` values in `αs`, and produces plots comparing the mean and variance of the optimal
Gaussian approximation against the true mean and variance of the mixture of Gaussians `mog`.
"""
function plot_α_anim(mog; αs=range(-1.5, 1.5, length=30),
                     anim_filename="~/Downloads/mog_anim.mp4",
                     μ_match_filename="~/Downloads/mog_μ_match.png",
                     σ2_match_filename="~/Downloads/mog_σ2_match.png")
    mog_μ, mog_σ2 = mean_and_variance(mog)

    μs = Vector{Float64}(undef, length(αs))
    σ2s = Vector{Float64}(undef, length(αs))

    anim = Animation()
    for (i, α) in enumerate(αs)
        best_approx = closest_gaussian(mog, distance = (p,q) -> α_divergence(p, q, α=α))
        μs[i], σ2s[i] = best_approx.normal.μ, best_approx.normal.σ^2
        plt = plot_mog(
            mog,
            approx = best_approx,
            title="α = $(round(α, digits=1))",
            ylim=(0, 0.32)
        )
        frame(anim, plt)
    end
    mp4(anim, anim_filename, fps=10)

    # plots the μs
    plt = plot(
            αs,
            μs,
            linewidth=3,
            color = :red,
            label = L"\mu_{\mathrm{approximation}}",
            xlabel=L"\alpha",
            ylabel=L"\mu_{\mathrm{approximation}}",
            xtickfontsize=14,
            ytickfontsize=14,
            xguidefontsize=16,
            yguidefontsize=16,
            legendfontsize=16,
        )
    plot!(
        αs,
        mog_μ * ones(length(αs)),
        linewidth=3,
        label = L"\mu_p",
        color = :blue,
    )
    display(plt)
    savefig(μ_match_filename)

    plt = plot(
            αs,
            σ2s,
            linewidth=3,
            color = :red,
            label=L"\sigma^2_{\mathrm{approximation}}",
            xlabel=L"\alpha",
            ylabel=L"\sigma^2_{\mathrm{approximation}}",
            xtickfontsize=14,
            ytickfontsize=14,
            xguidefontsize=16,
            yguidefontsize=16,
            legendfontsize=16,
        )
    plot!(
        αs,
        mog_σ2 * ones(length(αs)),
        linewidth=3,
        label = L"\sigma^2_p",
        color = :blue,
    )
    display(plt)
    savefig(σ2_match_filename)
end

function main()
    mog = MixtureOfGaussian([0.25, 0.45, 0.3], [Normal(-3, 0.7), Normal(0, 1), Normal(3, 1.5)])

    plot_α_anim(mog,
                αs = range(-2, 10, length=100),
                anim_filename="~/Downloads/mog_anim.mp4",
                μ_match_filename="~/Downloads/mog_μ_match.svg",
                σ2_match_filename="~/Downloads/mog_σ2_match.svg"
    )

    display(plot_mog(mog, approx = closest_gaussian(mog, distance = (p,q) -> α_divergence(p, q, α=-2.5))))
    savefig("~/Downloads/mog_α_-100.svg")
    display(plot_mog(mog, approx = closest_gaussian(mog, distance = (p,q) -> α_divergence(p, q, α=-1))))
    savefig("~/Downloads/mog_α_-1.svg")
    display(plot_mog(mog, approx = closest_gaussian(mog, distance = (p,q) -> α_divergence(p, q, α=0))))
    savefig("~/Downloads/mog_α_0.svg")
    display(plot_mog(mog, approx = closest_gaussian(mog, distance = (p,q) -> α_divergence(p, q, α=1))))
    savefig("~/Downloads/mog_α_1.svg")
    display(plot_mog(mog, approx = closest_gaussian(mog, distance = (p,q) -> α_divergence(p, q, α=10))))
    savefig("~/Downloads/mog_α_100.svg")
end

end

Distance.main()
