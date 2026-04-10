# Shared library for the Probabilistic Machine Learning Course
#
# This module consolidates common utility functions used across multiple units
# to avoid code duplication.
#
# 2026 by Ralf Herbrich
# Hasso-Plattner Institute

module PMLCourse

using Distributions
using MLDatasets
using Statistics

export sigmoid, normal_density
export polynomial_basis, fourier_basis, gauss_basis, sigmoid_basis, linear_basis_function
export generate_data
export average_intensity_features, extract_mnist_data, extract_mnist_raw
export RBF_kernel, OU_kernel
export plot_transform

# ─────────────────────────────────────────────
# Basic functions
# ─────────────────────────────────────────────

"""
    sigmoid(x=0)

Returns the value of the logistic sigmoid at `x`, i.e., σ(x) = exp(x) / (1 + exp(x)).

# Examples
```jldoctest
julia> sigmoid(0)
0.5

julia> sigmoid(1)
0.7310585786300049
```
"""
function sigmoid(x=0)
    return exp(x) / (1 + exp(x))
end

"""
    normal_density(x, μ=0, σ=1)

Returns the value of the normal density at `x` with mean `μ` and standard deviation `σ`.

# Examples
```jldoctest
julia> normal_density(0)
0.3989422804014327
```
"""
function normal_density(x, μ=0, σ=1)
    return 1 / (sqrt(2 * π) * σ) * exp(-(x - μ)^2 / (2 * σ^2))
end

# ─────────────────────────────────────────────
# Basis functions for linear models
# ─────────────────────────────────────────────

"""
    polynomial_basis(x, j)

Computes the `j`th basis function value at `x` for polynomial basis functions, i.e., ϕⱼ(x) = xʲ.

# Examples
```jldoctest
julia> polynomial_basis(2.0, 3)
8.0

julia> polynomial_basis(2.0, 0)
1.0
```
"""
function polynomial_basis(x, j)
    return x^j
end

"""
    fourier_basis(x, j)

Computes the `j`th basis function value at `x` for the Fourier basis functions, i.e., ϕⱼ(x) = cos(πjx).

# Examples
```jldoctest
julia> fourier_basis(2.0, 3)
1.0

julia> fourier_basis(2.0, 0)
1.0
```
"""
function fourier_basis(x, j)
    return cos(π * j * x)
end

"""
    gauss_basis(x, j; σ=1)

Computes the `j`th basis function value at `x` for the Gaussian basis functions with standard deviation `σ`, i.e., ϕⱼ(x) = 𝒩(x; j, σ²).

# Examples
```jldoctest
julia> gauss_basis(2.0, 3)
0.24197072451914337

julia> gauss_basis(2.0, 0)
0.05399096651318806
```
"""
function gauss_basis(x, j; σ=1)
    return pdf(Normal(j, σ), x)
end

"""
    sigmoid_basis(x, j; σ=1)

Computes the `j`th basis function value at `x` for the sigmoid basis functions at length scale `σ`, i.e., ϕⱼ(x) = σ((x-j)/σ).

# Examples
```jldoctest
julia> sigmoid_basis(2.0, 3)
0.2689414213699951

julia> sigmoid_basis(2.0, 0)
0.8807970779778824
```
"""
function sigmoid_basis(x, j; σ=1)
    return exp((x - j) / σ) / (1 + exp((x - j) / σ))
end

"""
    linear_basis_function(x, w, basis)

Computes the value of a linear basis function model at `x` with parameter vector `w`
using the first `length(w)` basis functions from `basis`, i.e., f(x) = Σⱼ wⱼ ϕⱼ(x).

# Examples
```jldoctest
julia> linear_basis_function(1.0, [1.0, 2.0], polynomial_basis)
3.0
```
"""
function linear_basis_function(x, w, basis)
    y = 0
    for j in eachindex(w)
        y += w[j] * basis(x, j - 1)
    end
    return y
end

# ─────────────────────────────────────────────
# Data generation
# ─────────────────────────────────────────────

"""
    generate_data(n, f; σ=0.1, from=0, to=1)

Generates a dataset of `n` observations using a given function `f` with additive,
zero-mean Gaussian noise with standard deviation `σ` at fixed intervals in the range
`from` to `to`. Returns a named tuple `(x=..., y=...)`.

# Examples
```jldoctest
julia> d = generate_data(3, x -> x^2; σ=0.0, from=0, to=1)
(x = [0.0, 0.5, 1.0], y = [0.0; 0.25; 1.0;;])
```
"""
function generate_data(n::Int64, f; σ=0.1, from=0, to=1)
    xs = collect(range(from, to, n))
    ys = map(f, xs) + randn((n, 1)) * σ
    return (x=xs, y=ys)
end

# ─────────────────────────────────────────────
# MNIST feature extraction
# ─────────────────────────────────────────────

"""
    average_intensity_features(img)

Computes the average pixel intensity in the top and bottom halves of a 28×28 MNIST image
and returns them as a 2-element vector `[mean(top), mean(bottom)]`.

# Examples
```jldoctest
julia> average_intensity_features(ones(28, 28))
2-element Vector{Float64}:
 1.0
 1.0
```
"""
function average_intensity_features(img)
    return [mean(img[1:14, :]), mean(img[15:28, :])]
end

"""
    extract_mnist_data(n; class0=1, class1=8, features=average_intensity_features)

Extracts a dataset of `2n` MNIST image feature vectors where the first `n` examples are
from `class0` and the next `n` from `class1`. Features are computed using the `features`
function. Returns centered data `(X, y)` where `y ∈ {0, 1}`.
"""
function extract_mnist_data(n=9; class0=1, class1=8, features=average_intensity_features)
    y_all = MNIST(split=:train).targets
    X_all = MNIST(split=:train).features
    idx0 = range(1, length(y_all))[y_all.==class0]
    idx1 = range(1, length(y_all))[y_all.==class1]
    X0 = hcat(map(i -> features(X_all[:, :, idx0[i]]), 1:n)...)'
    X1 = hcat(map(i -> features(X_all[:, :, idx1[i]]), 1:n)...)'
    X = vcat(X0, X1)
    mn = mean(X, dims=1)
    X -= repeat(mn, size(X, 1))
    y = vcat(zeros(n, 1), 1 * ones(n, 1))
    return X, y
end

"""
    extract_mnist_raw(n; class0=1, class1=8)

Extracts raw (vectorized) MNIST images for two classes. Returns centered `(X, y)` where
each row of `X` is a 784-dimensional vector and `y ∈ {0, 1}`.
"""
function extract_mnist_raw(n=250; class0=1, class1=8)
    y_all = MNIST(split=:train).targets
    X_all = MNIST(split=:train).features
    idx0 = range(1, length(y_all))[y_all.==class0]
    idx1 = range(1, length(y_all))[y_all.==class1]
    X0 = hcat(map(i -> vec(X_all[:, :, idx0[i]]), 1:n)...)'
    X1 = hcat(map(i -> vec(X_all[:, :, idx1[i]]), 1:n)...)'
    mn = (sum(X1, dims=1) + sum(X0, dims=1)) / (2 * n)
    X0 -= repeat(mn, n)
    X1 -= repeat(mn, n)
    X = vcat(X0, X1)
    y = vcat(zeros(n, 1), 1 * ones(n, 1))
    return X, y
end

# ─────────────────────────────────────────────
# Kernel functions
# ─────────────────────────────────────────────

"""
    RBF_kernel(x1, x2; λ=1)

Computes the radial basis function (squared exponential) kernel value at `x1` and `x2`
with length scale `λ`, i.e., k(x₁,x₂) = exp(-‖x₁-x₂‖²/λ²).

Works for both scalar and vector inputs.

# Examples
```jldoctest
julia> RBF_kernel(0, 1; λ=1)
0.36787944117144233

julia> RBF_kernel([0,0], [1,1]; λ=1)
0.1353352832366127
```
"""
function RBF_kernel(x1, x2; λ=1)
    return exp(-sum((x1 .- x2).^2) / λ^2)
end

"""
    OU_kernel(x1, x2; λ=1)

Computes the Ornstein-Uhlenbeck (Matérn-1/2) kernel value at `x1` and `x2` with
length scale `λ`, i.e., k(x₁,x₂) = exp(-|x₁-x₂|/λ).

# Examples
```jldoctest
julia> OU_kernel(0, 1; λ=1)
0.36787944117144233
```
"""
function OU_kernel(x1, x2; λ=1)
    return exp(-abs(x1 - x2) / λ)
end

# ─────────────────────────────────────────────
# MNIST visualization helper
# ─────────────────────────────────────────────

"""
    plot_transform(x)

Transforms a 784-dimensional weight vector `x` into a 28×28 image matrix suitable for
display with `heatmap`, accounting for the row-major to column-major layout transformation.
"""
function plot_transform(x)
    return (hcat(map(r -> r[28:-1:1], eachrow(reshape(x[(28*28):-1:1], 28, 28)'))...)')
end

end # module PMLCourse
