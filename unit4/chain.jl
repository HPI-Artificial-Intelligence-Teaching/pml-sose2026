# Demonstrates message passing in a chain of discrete nodes
#
# 2026 by Ralf Herbrich
# Hasso-Plattner Institute

module ChainMessagePassing

include("discrete.jl")
include("distributionbag.jl")
include("factors.jl")

using .DistributionCollections
using .DiscreteDistribution
using .Factors

"""
    example_3_nodes()

Run a step-by-step example of belief-propagation message passing on a chain of
3 discrete nodes, printing marginals after each message update.
"""
function example_3_nodes()
    db = DistributionBag(Discrete(3))
    y1 = add!(db)
    y2 = add!(db)
    y3 = add!(db)

    # print the marginals on the screen
    function print_marginals()
        println("   Y1: ", db[y1])
        println("   Y2: ", db[y2])
        println("   Y3: ", db[y3])
    end

    f1 = PriorDiscreteFactor(db, y1, Discrete([0.0, 100.0, 0.0]))
    f2 = CouplingDiscreteFactor(db, y1, y2, Matrix([0.5 0.25 0.25; 0.25 0.5 0.25; 0.25 0.25 0.5]'))
    f3 = CouplingDiscreteFactor(db, y2, y3, Matrix([0.5 0.25 0.25; 0.25 0.5 0.25; 0.25 0.25 0.5]'))
    f4 = PriorDiscreteFactor(db, y3, Discrete([0.0, 0.0, 100.0]))

    println("Before all message updates")
    print_marginals()
    update_msg_to_x!(f1)
    println("After f1->y1 message update")
    print_marginals()

    update_msg_to_x!(f4)
    println("After f4->y3 message update")
    print_marginals()

    update_msg_to_y!(f2)
    println("After f2->y2 message update")
    print_marginals()

    update_msg_to_y!(f3)
    println("After f3->y3 message update")
    print_marginals()

    update_msg_to_x!(f3)
    println("After f3->y2 message update")
    print_marginals()

    update_msg_to_x!(f2)
    println("After f2->y1 message update")
    print_marginals()
end

"""
    example_n_nodes(n=3)

Run belief-propagation message passing on a chain of `n` discrete nodes with
prior factors on the first and last node. Prints marginals before and after
all message updates.
"""
function example_n_nodes(n = 3)
    db = DistributionBag(Discrete(3))
    y = [add!(db) for i in 1:n]

    # print the marginals on the screen
    function print_marginals()
        for i in 1:n
            println("   Y", i, ": ", db[y[i]])
        end
    end

    f1 = PriorDiscreteFactor(db, y[begin], Discrete([100.0, 0.0, 0.0]))
    f2 = PriorDiscreteFactor(db, y[end], Discrete([0.0, 0.0, 100.0]))
    f = [CouplingDiscreteFactor(db, y[i], y[i+1], Matrix([0.5 0.25 0.25; 0.25 0.5 0.25; 0.25 0.25 0.5]')) for i in 1:n-1]

    println("Before all message updates")
    print_marginals()

    update_msg_to_x!(f1)
    update_msg_to_x!(f2)

    # update forward messages
    for i in 1:n-1
        update_msg_to_y!(f[i])
    end

    # update backward messages
    for i in n-1:-1:1
        update_msg_to_x!(f[i])
    end

    println("After all message updates")
    print_marginals()
end

function main()
    println("\nStep-by-step example of message passing in a chain of 3 nodes")
    println("-------------------------------------------------------------")
    example_3_nodes()

    println("\nExample of a chain with 9 nodes")
    println("---------------------------------")
    example_n_nodes(9)
end

main()

end
