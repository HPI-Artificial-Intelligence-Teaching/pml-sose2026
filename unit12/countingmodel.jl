# Library for Counting Model with Dirichlet Smoothing for Prediction
#
# 2026 by Ralf Herbrich
# Hasso-Plattner Institute

module CountingModel

export CountModel, update!, predict

"""
    CountModel{T}

A counting model for characters with a context. Stores token mappings and
context-conditioned counts for predicting the next token in a sequence.

# Fields
- `no_tokens::Int`: number of distinct tokens
- `idx_to_token::Dict{Int,T}`: mapping from integer ids to their respective tokens
- `token_to_idx::Dict{T,Int}`: mapping from tokens to their respective integer ids
- `counts::Dict{Int64,Vector{Int}}`: maps a context key to a vector of counts per token
"""
struct CountModel{T}
    # number of tokens
    no_tokens::Int
    # mapping from a set of integer ids to their respective tokens
    idx_to_token::Dict{Int,T}
    # mapping from a set of tokens to their respective integer ids
    token_to_idx::Dict{T,Int}
    # maps a context string to a sparse list of next characters and their count
    counts::Dict{Int64,Vector{Int}}
end

"""
    CountModel(tokens)

Construct a `CountModel` from an iterable collection of tokens. Each token is
assigned a unique integer index, and an empty counts dictionary is initialized.
"""
function CountModel(tokens)
    # number the tokens one-by-one
    no_tokens = length(tokens)
    idx_to_token = Dict{Int,typeof(collect(tokens)[1])}()
    token_to_idx = Dict{typeof(collect(tokens)[1]),Int}()
    i = 1
    for c in tokens
        idx_to_token[i] = c
        token_to_idx[c] = i
        i += 1
    end
    counts = Dict{Int64,Vector{Int}}()
    return CountModel(no_tokens, idx_to_token, token_to_idx, counts)
end

"""
    update!(model::CountModel, context::Vector, symbol)

Update the counting model by incrementing the count for `symbol` given the
specified `context`. The context is encoded as a unique integer key derived
from the sequence of context symbols.
"""
function update!(model::CountModel, context::Vector, symbol)
    # convert the list of symbols in the context into a unique Int64
    con = 0
    for i in eachindex(context)
        con *= (model.no_tokens + 1)
        con += model.token_to_idx[context[end-i+1]]
    end

    if haskey(model.counts, con)
        model.counts[con][model.token_to_idx[symbol]] += 1
    else
        d = zeros(model.no_tokens)
        d[model.token_to_idx[symbol]] = 1
        model.counts[con] = d
    end
    return
end

"""
    predict(model::CountModel, context::Vector; α=1)

Compute a Laplace-smoothed mixture probability distribution over tokens given
the `context`. The prediction averages over all sub-contexts (from the empty
context up to the full context), each smoothed with parameter `α`.

Returns a vector of probabilities over all tokens.
"""
function predict(model::CountModel, context::Vector; α = 1)
    # initialize the probabilities with zero
    P = zeros(model.no_tokens)

    # iterate over all context starting with the empty context
    K = length(context) + 1
    con = 0

    for i = 1:K
        counts = get(model.counts, con, nothing)
        smoothed_counts = (counts !== nothing) ? (counts .+ α) : (α * ones(model.no_tokens))
        total = sum(smoothed_counts)
        P += (smoothed_counts / (K * total))

        # add one more context item
        if (i < K)
            con *= (model.no_tokens + 1)
            con += model.token_to_idx[context[end-i+1]]
        end
    end

    # return the distribution
    return (P)
end

end # module CountingModel
