# Plots for the information theory section
#
# 2026 by Ralf Herbrich
# Hasso-Plattner Institute

module InformationTheory

using Plots
using LaTeXStrings

"""
    average_ascii_frequency(file_name)

Read a text file and return a `Dict{Char,Int}` mapping each character found in
the file to its total occurrence count.
"""
function average_ascii_frequency(file_name)
    letter_count = Dict{Char,Int}()
    open(file_name, "r") do file
        for line in eachline(file)
            for c in line
                # c = Char(rand(UInt8))
                if (haskey(letter_count, c))
                    letter_count[c] += 1
                else
                    letter_count[c] = 1
                end
            end
        end
    end

    return letter_count
end

"""
    entropy(d)

Compute the Shannon entropy (in bits) of a dictionary-based distribution `d`,
where keys are symbols and values are counts. The counts are normalized
internally to form a probability distribution.
"""
function entropy(d)
    n = sum(values(d))
    return sum(map(k -> -k / n * log(2, k / n), values(d)))
end

"""
    plot_frequent_letters(d; n=15)

Plot a bar chart of the `n` most frequent characters from a dictionary-based
character count `d`. Frequencies are shown as proportions of the total count.
"""
function plot_frequent_letters(d; n=15)
    total = sum(values(d))
    sorted_keys = sort(collect(keys(d)), by = key -> -d[key])
    p = bar(
        sorted_keys[1:n],
        map(k -> d[k] / total, sorted_keys[1:15]),
        legend = false,
        xtickfontsize = 16,
        ytickfontsize = 16,
        xguidefontsize = 18,
        yguidefontsize = 18,
    )
    ylabel!("Frequency")
    xlabel!("ASCII codes")
    display(p)
end

"""
    plot_binary_entropy()

Plot the binary entropy function H_2(p) = -p log2(p) - (1-p) log2(1-p) for
p in [0, 1].
"""
function plot_binary_entropy()
    # computes the binary entropy
    function binary_entropy(p = 0.5)
        if (p == 0)
            return 0
        elseif (p == 1)
            return 0
        end

        return -p * log(2, p) - (1 - p) * log(2, 1 - p)
    end

    ps = range(0, stop = 1, length = 100)
    p = plot(
        ps,
        binary_entropy,
        linewidth = 3,
        legend = false,
        xtickfontsize = 16,
        ytickfontsize = 16,
        xguidefontsize = 18,
        yguidefontsize = 18,
    )
    ylabel!(L"H_2[{\mathrm{Ber}}(\cdot;p)]")
    xlabel!(L"p")
    display(p)
end

"""
    plot_Gaussian_differential_entropy(; σ2_min=0.01, σ2_max=1)

Plot the differential entropy of a Gaussian distribution,
h(X) = 0.5 * (log(2*pi*sigma^2) + 1), for variance values in
[`σ2_min`, `σ2_max`].
"""
function plot_Gaussian_differential_entropy(; σ2_min = 0.01, σ2_max = 1)
    σ2s = range(start = σ2_min, stop = σ2_max, length = 300)
    p = plot(
        σ2s,
        σ2s -> 0.5 * (log(2 * π * σ2s) + 1),
        linewidth = 3,
        legend = false,
        xtickfontsize = 16,
        ytickfontsize = 16,
        xguidefontsize = 18,
        yguidefontsize = 18,
    )
    ylabel!(L"H_e[X]")
    xlabel!(L"\sigma^2")
    display(p)
end

function main()
    # plot the binary entropy
    plot_binary_entropy()
    savefig("~/Downloads/binary_entropy.svg")

    # plots the entropy of letters in an actual file
    d = average_ascii_frequency(
        "/Users/rherbrich/src/HPI/Probabilistic ML Course/SS2026/unit12/bible.txt",
    )
    plot_frequent_letters(d)
    savefig("~/Downloads/frequent_letters.svg")
    println("H = ", entropy(d))

    plot_Gaussian_differential_entropy()
    savefig("~/Downloads/gaussian_entropy.svg")
end

end # module InformationTheory
