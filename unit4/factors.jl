# A set of types and functions for factors over discrete variables
#
# 2026 by Ralf Herbrich
# Hasso-Plattner Institute

module Factors

export PriorDiscreteFactor, CouplingDiscreteFactor, update_msg_to_x!, update_msg_to_y!

using ..DistributionCollections
using ..DiscreteDistribution

struct PriorDiscreteFactor{T}
    db::DistributionBag{Discrete{T}}
    x::Int64
    prior::Discrete{T}
    msg_to_x::Int64
end

"""
    PriorDiscreteFactor(db::DistributionBag{Discrete{T}}, x::Int64, prior)

Creates a prior factor for a discrete variable `x` with a given prior distribution `prior`

# Example

```julia-repl

julia> db = DistributionBag(Discrete(3))
0-element DistributionBag{Discrete{3}}

julia> y1 = add!(db)
1

julia> f1 = PriorDiscreteFactor(db, y1, Discrete([0.0, 100.0, 0.0]))
PriorDiscreteFactor{3}(Uniform:  P = [0.3333, 0.3333, 0.3333]
  [1]:  P = [0.3333, 0.3333, 0.3333]
  [2]:  P = [0.3333, 0.3333, 0.3333]
, 1,  P = [0.0, 1.0, 0.0], 2)

```
"""
PriorDiscreteFactor(db::DistributionBag{Discrete{T}}, x::Int64, prior) where {T} = PriorDiscreteFactor{T}(db, x, prior, add!(db))


"""
    update_msg_to_x!(f::PriorDiscreteFactor{T})

Updates the message from the factor `f` to the associated variable x

# Example

```julia-repl

julia> db = DistributionBag(Discrete(3))
0-element DistributionBag{Discrete{3}}

julia> y1 = add!(db)
1

julia> f1 = PriorDiscreteFactor(db, y1, Discrete([0.0, 100.0, 0.0]))
PriorDiscreteFactor{3}(Uniform:  P = [0.3333, 0.3333, 0.3333]
  [1]:  P = [0.3333, 0.3333, 0.3333]
  [2]:  P = [0.3333, 0.3333, 0.3333]
, 1,  P = [0.0, 1.0, 0.0], 2)

julia> update_msg_to_x!(f1)
 P = [0.0, 1.0, 0.0]

```
"""
function update_msg_to_x!(f::PriorDiscreteFactor{T}) where {T}
    incoming_msg = f.db[f.x] / f.db[f.msg_to_x]
    f.db[f.x] = incoming_msg * f.prior
    f.db[f.msg_to_x] = f.prior
end

struct CouplingDiscreteFactor{T}
    db::DistributionBag{Discrete{T}}
    x::Int64
    y::Int64
    P::Matrix{Float64}
    msg_to_x::Int64
    msg_to_y::Int64
end

"""
    CouplingDiscreteFactor(db::DistributionBag{Discrete{T}}, x::Int64, y::Int64, P::Matrix{Float64})

Creates a coupling factor for two discrete variables `x` and `y` with a given couling matrix `P`

# Example

```julia-repl

julia> db = DistributionBag(Discrete(3))
0-element DistributionBag{Discrete{3}}

julia> y1 = add!(db)
1

julia> y2 = add!(db)
2

julia> f1 = PriorDiscreteFactor(db, y1, Discrete([0.0, 100.0, 0.0]))
PriorDiscreteFactor{3}(Uniform:  P = [0.3333, 0.3333, 0.3333]
  [1]:  P = [0.3333, 0.3333, 0.3333]
  [2]:  P = [0.3333, 0.3333, 0.3333]
  [3]:  P = [0.3333, 0.3333, 0.3333]
, 1,  P = [0.0, 1.0, 0.0], 3)

julia> f2 = CouplingDiscreteFactor(db, y1, y2, Matrix([0.5 0.25 0.25; 0.25 0.5 0.25; 0.25 0.25 0.5]))
CouplingDiscreteFactor{3}(Uniform:  P = [0.3333, 0.3333, 0.3333]
  [1]:  P = [0.3333, 0.3333, 0.3333]
  [2]:  P = [0.3333, 0.3333, 0.3333]
  [3]:  P = [0.3333, 0.3333, 0.3333]
  [4]:  P = [0.3333, 0.3333, 0.3333]
  [5]:  P = [0.3333, 0.3333, 0.3333]
, 1, 2, [0.5 0.25 0.25; 0.25 0.5 0.25; 0.25 0.25 0.5], 4, 5)

```
"""
CouplingDiscreteFactor(db::DistributionBag{Discrete{T}}, x::Int64, y::Int64, P::Matrix{Float64}) where {T} = CouplingDiscreteFactor{T}(db, x, y, P, add!(db), add!(db))


"""
    update_msg_to_x!(f::CouplingDiscreteFactor{T})

Updates the message from the factor `f` to the associated variable x

# Example

```julia-repl

julia> db = DistributionBag(Discrete(3))
0-element DistributionBag{Discrete{3}}

julia> y1 = add!(db)
1

julia> y2 = add!(db)
2

julia> f1 = PriorDiscreteFactor(db, y1, Discrete([0.0, 100.0, 0.0]))
PriorDiscreteFactor{3}(Uniform:  P = [0.3333, 0.3333, 0.3333]
  [1]:  P = [0.3333, 0.3333, 0.3333]
  [2]:  P = [0.3333, 0.3333, 0.3333]
  [3]:  P = [0.3333, 0.3333, 0.3333]
, 1,  P = [0.0, 1.0, 0.0], 3)

julia> f2 = CouplingDiscreteFactor(db, y2, y1, Matrix([0.5 0.25 0.25; 0.25 0.5 0.25; 0.25 0.25 0.5]))
CouplingDiscreteFactor{3}(Uniform:  P = [0.3333, 0.3333, 0.3333]
  [1]:  P = [0.3333, 0.3333, 0.3333]
  [2]:  P = [0.3333, 0.3333, 0.3333]
  [3]:  P = [0.3333, 0.3333, 0.3333]
  [4]:  P = [0.3333, 0.3333, 0.3333]
  [5]:  P = [0.3333, 0.3333, 0.3333]
, 1, 2, [0.5 0.25 0.25; 0.25 0.5 0.25; 0.25 0.25 0.5], 4, 5)

julia> update_msg_to_x!(f2)
 P = [0.25, 0.5, 0.25]

```
"""
function update_msg_to_x!(f::CouplingDiscreteFactor{T}) where {T}
    # get the incoming message from y
    incoming_msg_probs = ℙ(f.db[f.y] / f.db[f.msg_to_y])
    # compute the new message probabilities by using the conditional probability matrix
    new_msg_to_x_probs = Vector{Float64}(undef, T)
    for i in 1:T
        new_msg_to_x_probs[i] = sum(incoming_msg_probs .* f.P[:, i])
    end
    # convert the new message probabilities to a discrete distribution
    new_msg_to_x = Discrete(log.(new_msg_to_x_probs))
    # compute the new marginal by divinding out the old message from the factor and multiplying in the new message
    f.db[f.x] = f.db[f.x] / f.db[f.msg_to_x] * new_msg_to_x
    # update the message
    f.db[f.msg_to_x] = new_msg_to_x
end

"""
    update_msg_to_y!(f::CouplingDiscreteFactor{T})

Updates the message from the factor `f` to the associated variable x

# Example

```julia-repl

julia> db = DistributionBag(Discrete(3))
0-element DistributionBag{Discrete{3}}

julia> y1 = add!(db)
1

julia> y2 = add!(db)
2

julia> f1 = PriorDiscreteFactor(db, y1, Discrete([0.0, 100.0, 0.0]))
PriorDiscreteFactor{3}(Uniform:  P = [0.3333, 0.3333, 0.3333]
  [1]:  P = [0.3333, 0.3333, 0.3333]
  [2]:  P = [0.3333, 0.3333, 0.3333]
  [3]:  P = [0.3333, 0.3333, 0.3333]
, 1,  P = [0.0, 1.0, 0.0], 3)

julia> f2 = CouplingDiscreteFactor(db, y1, y2, Matrix([0.5 0.25 0.25; 0.25 0.5 0.25; 0.25 0.25 0.5]))
CouplingDiscreteFactor{3}(Uniform:  P = [0.3333, 0.3333, 0.3333]
  [1]:  P = [0.3333, 0.3333, 0.3333]
  [2]:  P = [0.3333, 0.3333, 0.3333]
  [3]:  P = [0.3333, 0.3333, 0.3333]
  [4]:  P = [0.3333, 0.3333, 0.3333]
  [5]:  P = [0.3333, 0.3333, 0.3333]
, 1, 2, [0.5 0.25 0.25; 0.25 0.5 0.25; 0.25 0.25 0.5], 4, 5)

julia> update_msg_to_y!(f2)
 P = [0.25, 0.5, 0.25]

```
"""
function update_msg_to_y!(f::CouplingDiscreteFactor{T}) where {T}
    # get the incoming message from x
    incoming_msg_probs = ℙ(f.db[f.x] / f.db[f.msg_to_x])
    # compute the new message probabilities by using the conditional probability matrix
    new_msg_to_y_probs = Vector{Float64}(undef, T)
    for i in 1:T
        new_msg_to_y_probs[i] = sum(incoming_msg_probs .* f.P[i, :])
    end
    # convert the new message probabilities to a discrete distribution
    new_msg_to_y = Discrete(log.(new_msg_to_y_probs))
    # compute the new marginal by divinding out the old message from the factor and multiplying in the new message
    f.db[f.y] = f.db[f.y] / f.db[f.msg_to_y] * new_msg_to_y
    # update the message
    f.db[f.msg_to_y] = new_msg_to_y
end

end