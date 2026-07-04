# Code for Arithmetic Coding
#
# 2026 by Ralf Herbrich
# Hasso-Plattner Institute

module ArithmeticDemo

using Plots
using ProgressMeter: @showprogress, Progress, next!, finish!

include("arithmetic.jl")
include("countingmodel.jl")

using .ArithmeticCoding
using .CountingModel

"""
    learn_text(file_name; bits=32)

Read a file, compress it using arithmetic coding with a 4th-order counting model
(Dirichlet-smoothed), then decompress the result to verify correctness. Prints
the compressed size in bits and the theoretical target size based on the
cumulative log-probability.
"""
function learn_text(file_name; bits = 32)
    # initialize the tokens that we want to predict (including -1 for the EOF)
    tokens = Set{Int16}(collect(0:255))
    push!(tokens, -1)

    # initialize the counting model
    model = CountModel(tokens)

    # initialize the compressor
    compressor = Compressor()

    # count the total log probability (will be the final file size)
    logP = 0.0

    c1 = c2 = c3 = c4 = 0x0
    open(file_name, "r") do file
        @showprogress desc = "Compressing: " for c in collect(readeach(file, UInt8))
            # get the pdf for the next symbol (important for updating lower and upper)
            cum, p =
                round_prob(predict(model, [c1, c2, c3, c4], α = 1 / 256), bits = bits)[model.token_to_idx[c]]
            logP += (log2(p) - log2(1 << bits))

            # compress the character read
            compress!(compressor, cum, cum + p, bits = bits)

            # update the probabilty model of the data
            update!(model, [], c)
            update!(model, [c4], c)
            update!(model, [c3, c4], c)
            update!(model, [c2, c3, c4], c)
            update!(model, [c1, c2, c3, c4], c)

            c1, c2, c3, c4 = c2, c3, c4, c
        end
    end
    # push the EOF character
    cum, p =
        round_prob(predict(model, [c1, c2, c3, c4], α = 1 / 256), bits = bits)[model.token_to_idx[-1]]
    logP += (log2(p) - log2(1 << bits))
    compress!(compressor, cum, cum + p, bits = bits)
    flush!(compressor)

    # print the final stats
    println("[", length(compressor.output_bits), "] ")
    println("\nTarget size of file = ", -logP / 8)

    # initialize the de-compressor
    decompressor = Decompressor(compressor.output_bits)

    # initialize the counting model
    model = CountModel(tokens)

    c1 = c2 = c3 = c4 = 0x0
    c = 0
    progress = Progress(filesize(file_name), desc = "Decompressing: ")
    open(file_name * ".reconstructed", "w") do out
        while (c != -1)
            P = round_prob(predict(model, [c1, c2, c3, c4], α = 1 / 256), bits = bits)
            c = model.idx_to_token[decompress!(decompressor, P, bits = bits)]
            if (c != -1)
                # Base.print(Char(c))
                next!(progress)
                write(out, UInt8(c))

                # update the probabilty model of the data
                update!(model, [], c)
                update!(model, [c4], c)
                update!(model, [c3, c4], c)
                update!(model, [c2, c3, c4], c)
                update!(model, [c1, c2, c3, c4], c)
                c1, c2, c3, c4 = c2, c3, c4, c
            else
                finish!(progress)
                println()
            end
        end
    end
end

"""
    test_compress(line="CAD", prob=[1/2, 1/4, 1/8, 1/16, 1/64, 1/64, 1/64, 1/64]; bits=8)

Test arithmetic coding on a short string `line` with a fixed probability
distribution `prob` over letters A-H. Compresses the string followed by an
EOF marker ('E'), prints the resulting bit pattern and target size, then
decompresses and prints the recovered text to verify round-trip correctness.
"""
function test_compress(
    line = "CAD",
    prob = [1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 64, 1 / 64, 1 / 64, 1 / 64];
    bits = 8,
)
    # initialize the compressor
    compressor = Compressor()

    # compute the rounded probabilities
    P = round_prob(prob, bits = bits)

    # count the total log probability (will be the final file size)
    logP = 0.0

    for c in Vector{Char}(line)
        cum, p = P[Int(c - 'A' + 1)]
        logP += (log2(p) - log2(1 << bits))

        compress!(compressor, cum, cum + p, bits = bits)
    end
    # push the EOF character
    cum, p = P[Int('E' - 'A' + 1)]
    logP += (log2(p) - log2(1 << bits))
    compress!(compressor, cum, cum + p, bits = bits)
    flush!(compressor)

    # print the final stats
    println(
        "[",
        length(compressor.output_bits),
        "] ",
        map(x -> if x
            '1'
        else
            '0'
        end, compressor.output_bits)...,
    )
    println("\nTarget size of file = ", -logP)

    # initialize the de-compressor
    decompressor = Decompressor(compressor.output_bits)

    # compute the rounded probabilities
    P = round_prob(prob, bits = bits)

    c = 'A'
    while (c != 'E')
        c = Char(decompress!(decompressor, P, bits = bits) + Int('A') - 1)
        if (c != 'E')
            Base.print(c)
        else
            println()
        end
    end
end

"""
    arithmetic_base10_demo(txt; pdf=..., scaled_up_plots=true)

Demonstrate arithmetic coding in base 10 on the string `txt` using the given
discrete probability distribution `pdf`. Visualizes each encoding step as a
stacked bar chart showing how the interval narrows. When `scaled_up_plots` is
true, each step is plotted at unit scale; otherwise the actual interval bounds
are shown. Prints the resulting arithmetic code and the base-10 entropy.
"""
function arithmetic_base10_demo(
    txt;
    pdf = Dict{Char,Rational{Int64}}( 'E' => 3//10, 'H' => 1//10, 'L' => 2//10, 'O' => 3//10, '!' => 1//10),
    scaled_up_plots = true
)
    # compute the cumulative distribution function
    cdf = Dict{Char,Rational{Int64}}()
    sum = 0 // 1
    for (c, p) in pdf
        cdf[c] = sum
        sum += p
    end
    if (sum != 1 // 1)
        error("The sum of the probabilities must be 1.0")
    end

    # plots a stacked bar chart of the pdf
    function plot_cdf(x, lower, upper)
        if (scaled_up_plots)
            range = upper - lower
            for (c, p) in pdf
                plot!(
                    [x - 0.5, x + 0.5, x + 0.5, x - 0.5, x - 0.5],
                    [cdf[c], cdf[c], cdf[c] + pdf[c], cdf[c] + pdf[c], cdf[c]],
                    legend = false,
                    color = :black
                )
                annotate!(
                    [x],
                    [cdf[c] + pdf[c] / 2],
                    text(string(c), :center),
                    color = :black,
                )
                annotate!(
                    [x + 0.53],
                    [cdf[c] + pdf[c]],
                    text(string(float(lower + (cdf[c] + pdf[c]) * range)), 6, :left),
                    color = :black,
                )
            end
            annotate!(
                [x + 0.53],
                [0],
                text(string(float(lower)), 6, :left),
                color = :black,
            )
        else
            range = upper - lower
            for (c, p) in pdf
                plot!(
                    [x - 0.5, x + 0.5, x + 0.5, x - 0.5, x - 0.5],
                    [lower + cdf[c] * range, lower + cdf[c] * range, lower + (cdf[c] + pdf[c]) * range, lower + (cdf[c] + pdf[c]) * range, lower + cdf[c] * range],
                    legend = false,
                    color = :black
                )
                annotate!(
                    [x],
                    [lower + (cdf[c] + pdf[c] / 2) * range],
                    text(string(c), :center),
                    color = :black,
                )
                annotate!(
                    [x + 0.53],
                    [lower + (cdf[c] + pdf[c]) * range],
                    text(string(float(lower + (cdf[c] + pdf[c]) * range)), 6, :left),
                    color = :black,
                )
            end
            annotate!(
                [x + 0.53],
                [lower],
                text(string(float(lower)), 6, :left),
                color = :black,
            )
        end
    end

    # print the pdf and cdf
    for (c, _) in pdf
        println(c, "\t", float(pdf[c]), "\t", float(cdf[c]))
    end
    println()

    # prepare the plot
    p =
        if scaled_up_plots
            plot(
                legend = false,
                xaxis = false,
                yaxis = false,
                grid = false,
            )
        else
            plot(
                legend = false,
                xaxis = false,
                grid = false,
                ytickfontsize = 16,
                yguidefontsize = 18,
            )
        end

    # carry out the arithmetic coding
    lower = 0 // 1
    upper = 1 // 1
    entropy = 0.0
    x = 0
    println("Initial Range:  [", float(lower), ", ", float(upper), "]")
    for c in txt
        # plots the current cdf
        plot_cdf(x, lower, upper)
        x += 2

        # compute the range of the next character
        range = upper - lower
        upper = lower + range * (cdf[c] + pdf[c])
        lower = lower + range * cdf[c]

        # compute the entropy
        entropy -= log10(pdf[c])

        # print the current range
        println("After seeing ", c, ": [", float(lower), ", ", float(upper), "]")
    end

    # convert the lower and upper bounds to a string
    lower_str, upper_str = string(float(lower)), string(float(upper))

    # extract the first leading digits of the lower and upper bounds that are the same
    code = ""
    for (l, u) in zip(lower_str, upper_str)
        if l == u
            code *= l
        else
            # pick the character in the middle of l and u
            code *= string(Char((UInt8(l) + UInt8(u)) ÷ 2 + 1))
            println("\nArithmetic code of '", txt ,"': ", code)
            break
        end
    end

    println("Entropy (base 10): ", entropy)
    xlims!(-1, 2*(length(txt) + 0.5))
    display(p)
end

function main()
    arithmetic_base10_demo("HELLO", scaled_up_plots = true)
    savefig("~/Downloads/artihmetic_scaled.svg")

    arithmetic_base10_demo("AAB", pdf = Dict{Char,Rational{Int64}}( 'A' => 5//10, 'B' => 3//10, 'C' => 2//10), scaled_up_plots = false)
    savefig("~/Downloads/artihmetic_non_scaled.svg")
end

end # module ArithmeticDemo
