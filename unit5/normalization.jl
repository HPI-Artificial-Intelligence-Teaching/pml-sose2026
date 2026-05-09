# Plots for normalization constant algorithms
#
# 2026 by Ralf Herbrich
# Hasso-Plattner Institute

module Normalization

using Random
using LinearAlgebra
using Distributions
using LaTeXStrings
using Plots

"""
    Discretization(threshold)

A discretization of the real line into `n+1` intervals where `n = length(threshold)`.
The first interval starts at `-Inf` and the last interval ends at `+Inf`.
"""
struct Discretization
    threshold::Vector{Float64}
end

"""
    Base.iterate(d::Discretization, state=0)

Returns an iterator over all the intervals of the discretization. Each interval is
represented as a tuple `(a, b)` where `a` is the lower bound and `b` is the upper bound.
"""
function Base.iterate(d::Discretization, state=0)
    i = state
    if i == 0
        return ((-Inf, d.threshold[i + 1]), i + 1)
    elseif i < length(d.threshold)
        return ((d.threshold[i], d.threshold[i + 1]), i + 1)
    elseif i == length(d.threshold)
        return ((d.threshold[i], Inf), i + 1)
    else
        return nothing
    end
end

"""
    Base.length(d::Discretization)

Returns the number of intervals in the discretization, which is `length(threshold) + 1`.
"""
function Base.length(d::Discretization)
    return length(d.threshold) + 1
end

"""
    Base.size(d::Discretization)

Returns the size of the discretization as a 1-tuple `(length(threshold) + 1,)`.
"""
function Base.size(d::Discretization)
    return (length(d.threshold) + 1, )
end

"""
    Base.eachindex(d::Discretization)

Returns a range `1:length(d)` that can be used to iterate over all interval indices of the discretization.
"""
function Base.eachindex(d::Discretization)
    return 1:length(d)
end

"""
    Base.getindex(d::Discretization, i::Int)

Returns the `i`-th interval of the discretization as a tuple `(a, b)`. The first interval
starts at `-Inf` and the last interval ends at `+Inf`.
"""
function Base.getindex(d::Discretization, i::Int)
    if i == 1
        return (-Inf, d.threshold[i])
    elseif i == length(d.threshold) + 1
        return (d.threshold[i - 1], Inf)
    else
        return (d.threshold[i - 1], d.threshold[i])
    end
end

"""
    mid_point(d::Discretization, i::Int)

Returns the mid-point of the `i`-th interval of the discretization. For the first and last
intervals (which extend to `-Inf` and `+Inf` respectively), an approximation is used based
on the spacing of neighboring thresholds.
"""
function mid_point(d::Discretization, i::Int)
    if i == 1
        return d.threshold[1] - (d.threshold[2] - d.threshold[1]) / 2
    elseif i == length(d.threshold) + 1
        return d.threshold[i - 1] + (d.threshold[i - 1] - d.threshold[i - 2]) / 2
    else
        return (d.threshold[i - 1] + d.threshold[i]) / 2
    end
end

"""
    find_interval(d::Discretization, x::Float64)

Finds the index of the interval in the discretization that contains the value `x`,
using binary search over the thresholds.
"""
function find_interval(d::Discretization, x::Float64)
    l = 1
    r = length(d)
    while l < r
        m = div(l + r, 2)
        if x < d.threshold[m]
            r = m
        else
            l = m + 1
        end
    end
    return l
end

"""
    plot_factor_function(discrete::Discretization, f; x_title="x", y_title="f(x)")

Plots a single-variable factor function as a bar chart over the discretization intervals.
`f` is a vector of function values (one per interval) and `x_title`/`y_title` control the
axis labels.
"""
function plot_factor_function(discrete::Discretization, f; x_title="x", y_title="f(x)")
    p = bar(
        [mid_point(discrete, i) for i in 1:length(discrete)],
        f,
        legend=false,
        color=:blue,
        xtickfontsize=14,
        ytickfontsize=14,
        xguidefontsize=16,
        yguidefontsize=16,
    )
    xlabel!(x_title)
    ylabel!(y_title)
    display(p)
end


"""
    compute_messages(discrete::Discretization, μ1, σ1, μ2, σ2, β)

Computes all messages for the two-player TrueSkill factor graph/tree using the given
discretization. The factor graph models two players with Gaussian skill priors
`N(μ1, σ1)` and `N(μ2, σ2)`, Gaussian performance likelihoods with noise `β`, and an
indicator factor enforcing that player 1 outperforms player 2. Returns a tuple of
factor functions and computed messages.
"""
function compute_messages(discrete::Discretization, μ1::Float64, σ1::Float64, μ2::Float64, σ2::Float64, β::Float64)
    # setup the factor functions; the functions are:
    #   f_1(s_1) = N(s_1; μ1, σ1)
    #   f_2(s_2) = N(s_2; μ2, σ2)
    #   g_1(s_1, p_1) = N(p_1; s_1, β)
    #   g_2(s_2, p_2) = N(p_2; s_2, β)
    #   h(p_1, p_2) = I(p_1 > p_2)

    f1 = [cdf(Normal(μ1, σ1), b) - cdf(Normal(μ1, σ1), a) for (a, b) in discrete]
    f2 = [cdf(Normal(μ2, σ2), b) - cdf(Normal(μ2, σ2), a) for (a, b) in discrete]
    g1 = zeros(length(discrete), length(discrete))
    g2 = zeros(length(discrete), length(discrete))
    h = zeros(length(discrete), length(discrete))
    for i in eachindex(discrete)
        for (j, (c, d)) in enumerate(discrete)
            h[i, j] = (i > j) ? 1.0 : 0.0
            m = mid_point(discrete, i)
            g1[i, j] = g2[i, j] = cdf(Normal(m, β), d) - cdf(Normal(m, β), c)
        end
    end

    # compute the message from f_1 to s_1 and from f_2 to s_2
    msg_f1_s1 = deepcopy(f1)
    msg_f2_s2 = deepcopy(f2)

    # compute the message from g_1 to p_1 and from g_2 to p_2
    msg_g1_p1 = zeros(length(discrete))
    for p1 in eachindex(discrete)
        msg_g1_p1[p1] = sum(g1[:, p1] .* msg_f1_s1)
    end
    msg_g2_p2 = zeros(length(discrete))
    for p2 in eachindex(discrete)
        msg_g2_p2[p2] = sum(g2[:, p2] .* msg_f2_s2)
    end

    # compute the message from h to p_1 and p_2
    msg_h_p1 = zeros(length(discrete))
    for p1 in eachindex(discrete)
        msg_h_p1[p1] = sum(h[p1, :] .* msg_g2_p2)
    end
    msg_h_p2 = zeros(length(discrete))
    for p2 in eachindex(discrete)
        msg_h_p2[p2] = sum(h[:, p2] .* msg_g1_p1)
    end

    # compute the message from g1 to s1 and from g2 to s2
    msg_g1_s1 = zeros(length(discrete))
    for s1 in eachindex(discrete)
        msg_g1_s1[s1] = sum(g1[s1, :] .* msg_h_p1)
    end
    msg_g2_s2 = zeros(length(discrete))
    for s2 in eachindex(discrete)
        msg_g2_s2[s2] = sum(g2[s2, :] .* msg_h_p2)
    end

    return (f1, f2, g1, g2, h), (msg_f1_s1, msg_f2_s2, msg_g1_p1, msg_g2_p2, msg_h_p1, msg_h_p2, msg_g1_s1, msg_g2_s2)
end

function main(n = 100, μ1=0.0, σ1=1.0, μ2=0.0, σ2=0.0, β=0.5; base="~/Downloads/norm_")
    thresholds = collect(range(start=-6.0, stop=6.0, length=n))
    dt = Discretization(thresholds)
    (_, _, g1, g2, h), (msg_f1_s1, msg_f2_s2, msg_g1_p1, msg_g2_p2, msg_h_p1, msg_h_p2, msg_g1_s1, msg_g2_s2) = compute_messages(dt, μ1, σ1, μ2, σ2, β)

    plot_factor_function(dt, msg_f1_s1, x_title = L"s_1", y_title = L"m_{f_1 \to s_1}")
    savefig(base * "f1_s1.svg")
    plot_factor_function(dt, msg_f2_s2, x_title = L"s_2", y_title = L"m_{f_2 \to s_2}")
    savefig(base * "f2_s2.svg")
    plot_factor_function(dt, msg_g1_p1, x_title = L"p_1", y_title = L"m_{g_1 \to p_1}")
    savefig(base * "g1_p1.svg")
    plot_factor_function(dt, msg_g2_p2, x_title = L"p_2", y_title = L"m_{g_2 \to p_2}")
    savefig(base * "g2_p2.svg")

    plot_factor_function(dt, msg_h_p1, x_title = L"p_1", y_title = L"m_{h \to p_1}")
    savefig(base * "h_p1.svg")
    plot_factor_function(dt, msg_h_p2, x_title = L"p_2", y_title = L"m_{h \to p_2}")
    savefig(base * "h_p2.svg")


    plot_factor_function(dt, msg_g1_s1, x_title = L"s_1", y_title = L"m_{g_1 \to s_1}")
    savefig(base * "g1_s1.svg")
    plot_factor_function(dt, msg_g2_s2, x_title = L"s_2", y_title = L"m_{g_2 \to s_2}")
    savefig(base * "g2_s2.svg")


    s1 = msg_g1_s1 .* msg_f1_s1
    s2 = msg_g2_s2 .* msg_f2_s2
    p1 = msg_h_p1 .* msg_g1_p1
    p2 = msg_h_p2 .* msg_g2_p2

    plot_factor_function(dt, s1, x_title = L"s_1", y_title = L"p(s_1)")
    savefig(base * "s1.svg")
    plot_factor_function(dt, s2, x_title = L"s_2", y_title = L"p(s_2)")
    savefig(base * "s2.svg")
    plot_factor_function(dt, p1, x_title = L"p_1", y_title = L"p(p_1)")
    savefig(base * "p1.svg")
    plot_factor_function(dt, p2, x_title = L"p_2", y_title = L"p(p_2)")
    savefig(base * "p2.svg")

    println("The normalization constant over s1 is: ", sum(s1))
    println("The normalization constant over s2 is: ", sum(s2))
    println("The normalization constant over p1 is: ", sum(p1))
    println("The normalization constant over p2 is: ", sum(p2))

    # Z_h = 0.0
    # for p1 in eachindex(dt)
    #     for p2 in eachindex(dt)
    #         if p1 > p2
    #             Z_h += msg_g1_p1[p1] * msg_g2_p2[p2]
    #         end
    #     end
    # end
    # println("The normalization constant over  h is: ", Z_h)

    # Let's try to normalize
    msg_f1_s1 /= sum(msg_f1_s1)
    msg_f2_s2 /= sum(msg_f2_s2)
    msg_g1_p1 /= sum(msg_g1_p1)
    msg_g2_p2 /= sum(msg_g2_p2)
    msg_h_p1 /= sum(msg_h_p1)
    msg_h_p2 /= sum(msg_h_p2)
    msg_g1_s1 /= sum(msg_g1_s1)
    msg_g2_s2 /= sum(msg_g2_s2)

    # now compute the normalization per variable
    Z_s1 = sum(msg_g1_s1 .* msg_f1_s1)
    Z_s2 = sum(msg_g2_s2 .* msg_f2_s2)
    Z_p1 = sum(msg_h_p1 .* msg_g1_p1)
    Z_p2 = sum(msg_h_p2 .* msg_g2_p2)

    # now compute the normalization per factor
    Z_f1 = sum(msg_f1_s1)
    Z_f2 = sum(msg_f2_s2)
    Z_g1 = 0.0
    for s1 in eachindex(dt)
        for p1 in eachindex(dt)
            Z_g1 += g1[s1, p1] * msg_f1_s1[s1] * msg_h_p1[p1]
        end
    end
    Z_g1 /= (Z_s1 * Z_p1)

    Z_g2 = 0.0
    for s2 in eachindex(dt)
        for p2 in eachindex(dt)
            Z_g2 += g2[s2, p2] * msg_f2_s2[s2] * msg_h_p2[p2]
        end
    end
    Z_g2 /= (Z_s2 * Z_p2)

    Z_h = 0.0
    for p1 in eachindex(dt)
        for p2 in eachindex(dt)
            Z_h += h[p1, p2] * msg_g1_p1[p1] * msg_g2_p2[p2]
        end
    end
    Z_h /= (Z_p1 * Z_p2)

    println("Z_s1: ", Z_s1)
    println("Z_s2: ", Z_s2)
    println("Z_p1: ", Z_p1)
    println("Z_p2: ", Z_p2)
    println("Z_f1: ", Z_f1)
    println("Z_f2: ", Z_f2)
    println("Z_g1: ", Z_g1)
    println("Z_g2: ", Z_g2)
    println("Z_h: ", Z_h)

    println("The normalization constant overall is: ", Z_s1 * Z_s2 * Z_p1 * Z_p2 * Z_f1 * Z_f2 * Z_g1 * Z_g2 * Z_h)
end

end

Normalization.main(100, 0.0, 1.0, 0.0, 1.0, 0.5)
