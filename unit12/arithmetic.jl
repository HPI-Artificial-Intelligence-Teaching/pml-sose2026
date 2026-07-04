# Library for Arithmetic Coding
#
# 2026 by Ralf Herbrich
# Hasso-Plattner Institute

module ArithmeticCoding

export Compressor, Decompressor, compress!, flush!, decompress!, round_prob

"""
    Compressor

Mutable data structure capturing the state of an arithmetic compressor.
Maintains a lower bound and range in 32-bit fixed-point arithmetic, and
accumulates compressed output bits.

# Fields
- `lower::UInt32`: bit pattern of the lower bound
- `range::UInt64`: probability range still in play (in 32-bit fixed-point arithmetic)
- `output_bits::BitArray`: bit array of the compressed sequence
- `pending::Int`: number of deferred complementary bits (E3 underflow bookkeeping)
"""
mutable struct Compressor
    # bit pattern of the lower bound
    lower::UInt32
    # value of the probability range still in play (in 32-bit fixed point arithmetic)
    range::UInt64
    # bit array of the compressed sequence
    output_bits::BitArray
    # number of complementary bits still owed once the next bit resolves (see emit_bit!)
    pending::Int
end
# Default constructor for compression
Compressor() = Compressor(0x00000000, 0x100000000, BitVector(), 0)

"""
    emit_bit!(compressor::Compressor, bit::Bool)

Append `bit` to the output, followed by any bits that were deferred while the
interval straddled the midpoint (E3 underflow). See `compress!` for context.
"""
function emit_bit!(compressor::Compressor, bit::Bool)
    push!(compressor.output_bits, bit)
    for _ = 1:compressor.pending
        push!(compressor.output_bits, !bit)
    end
    compressor.pending = 0
    return nothing
end

"""
    compress!(compressor::Compressor, p_lower, p_upper; bits=32)

Compress a new symbol specified by its lower (`p_lower`) and upper (`p_upper`)
cumulative probability bounds in fixed-point representation. Extracts matching
leading bits from the narrowed interval and appends them to the output.

Returns the updated `compressor`.
"""
function compress!(compressor::Compressor, p_lower, p_upper; bits = 32)
    # compute the new lower and upper boundary
    lower = compressor.lower + UInt32((compressor.range * p_lower) >> bits)
    upper =
        compressor.lower + UInt32(
            (p_upper == (1 << bits)) ? (compressor.range * p_upper - 1) >> bits :
            (compressor.range * p_upper) >> bits,
        )

    # Renormalize. Two cases can make progress:
    #  E1/E2: lower and upper agree on their leading bit -> that bit is
    #         resolved, output it (plus any bits deferred by E3 below).
    #  E3:    the interval is squeezed into the middle half (lower just below
    #         0.5, upper just above, e.g. lower=0100...  upper=1011...). No
    #         bit can be resolved yet, but we know it will be the complement
    #         of whatever the *next* resolved bit turns out to be, so we
    #         widen back out around the midpoint and remember to flush a
    #         complementary bit later via `pending`.
    # Probabilities close to but not exactly 0.5 (e.g. 0.4999999999999999)
    # routinely land in the E3 case for several symbols in a row. Without
    # this branch the interval only ever narrows and never renormalizes,
    # so `range` shrinks to 0 and `flush!` spins forever looking for a set
    # bit that no longer exists.
    for _ = 1:bits
        if (lower & 0x80000000) == (upper & 0x80000000)
            emit_bit!(compressor, (lower & 0x80000000) != 0)
            lower <<= 1
            upper <<= 1
        elseif (lower & 0x40000000) != 0 && (upper & 0x40000000) == 0
            compressor.pending += 1
            lower -= 0x40000000
            upper -= 0x40000000
            lower <<= 1
            upper <<= 1
        else
            break
        end
    end

    # `round_prob` guarantees every positive-probability symbol gets a width
    # of at least 1, but that alone isn't sufficient: if `compressor.range`
    # has already been narrowed by preceding symbols, a width-1 sub-interval
    # can still round down to nothing here. That's a hard precision floor of
    # `bits`-bit fixed point, not something `round_prob` can fix in
    # isolation -- fail loudly right here instead of silently continuing
    # with a zero-width interval that corrupts every symbol encoded after it.
    upper > lower || error(
        "compress!: probability interval collapsed to zero width -- this " *
        "symbol's probability is too small to represent given the current " *
        "range at $bits bits of precision",
    )

    # update compressor state
    compressor.lower = lower
    compressor.range = upper - lower
    return compressor
end

"""
    flush!(compressor::Compressor)

Flush the remaining bits from the compressor state after all symbols have been
compressed. This ensures the final interval is fully encoded in the output bit
stream.

Returns the updated `compressor`.
"""
function flush!(compressor::Compressor)
    # after the last compress! call, lower and upper (implicitly
    # lower + range) can no longer be renormalized (else compress! would
    # already have done it), which guarantees lower < 0.5. Emitting one more
    # bit -- 0 if lower is in the lower quarter, 1 otherwise -- together with
    # any bits still owed from E3 underflow, is enough to pin the decoder to
    # a value inside [lower, lower+range).
    compressor.pending += 1
    emit_bit!(compressor, (compressor.lower & 0x40000000) != 0)

    return (compressor)
end

"""
    Decompressor

Mutable data structure capturing the state of an arithmetic decompressor.
Tracks the current interval, the buffered value from the compressed bit stream,
and read positions.

# Fields
- `lower::UInt32`: bit pattern of the lower bound
- `range::UInt64`: probability range still in play
- `value::UInt32`: current value from the next 32 bits of compressed input
- `bit_processed::Int`: number of bits processed so far
- `bits_copied::Int`: number of bits copied from input
- `input_bits::BitArray`: the compressed bit array to decompress
"""
mutable struct Decompressor
    # bit pattern of the lower bound
    lower::UInt32
    # value of the probability range still in play (in 32-bit fixed point arithmetic)
    range::UInt64
    # current value of the next 32 bits from the compressed sequence
    value::UInt32
    # number of bits processed
    bit_processed::Int
    # number of bits copied
    bits_copied::Int
    # bit array of the compressed sequence
    const input_bits::BitArray
end

"""
    Decompressor(compressed_bits)

Construct a `Decompressor` from a `BitArray` of compressed bits. Initializes the
decompressor state by loading the first 32 bits (or fewer if the stream is shorter)
into the value buffer.
"""
function Decompressor(compressed_bits)
    value = 0
    bits_copied = min(length(compressed_bits), 32)
    for i = 1:32
        value <<= 1
        if (i <= bits_copied)
            value += compressed_bits[i]
        end
    end
    Decompressor(0x00000000, 0x100000000, value, 0, bits_copied, compressed_bits)
end

"""
    decompress!(decompressor::Decompressor, P; bits=32)

Decompress the next symbol from the compressed bit stream. `P` is a vector of
`(cumulative_lower, width)` tuples representing the rounded probability intervals
for each symbol. Returns the 1-based index of the decoded symbol.
"""
function decompress!(decompressor::Decompressor, P; bits = 32)
    # linearly search for the interval that contains the small number we have
    for i in eachindex(P)
        p_lower = P[i][1]
        p_upper = p_lower + P[i][2]
        lower = decompressor.lower + UInt32((decompressor.range * p_lower) >> bits)
        upper =
            decompressor.lower + UInt32(
                (p_upper == (1 << bits)) ? (decompressor.range * p_upper - 1) >> bits :
                (decompressor.range * p_upper) >> bits,
            )

        if (decompressor.value >= lower && decompressor.value < upper)
            # remove all the bits that are the same from lower and upper (E1/E2),
            # and undo the E3 midpoint offset when the interval straddles it --
            # mirrors the renormalization performed in compress!, including the
            # bits deferred there via `pending` (see compress! for why E3 is
            # needed: near-0.5 probabilities land here for runs of symbols).
            for _ = 1:bits
                if (lower & 0x80000000) == (upper & 0x80000000)
                    # E1/E2: leading bit resolved
                elseif (lower & 0x40000000) != 0 && (upper & 0x40000000) == 0
                    # E3: undo the midpoint offset before shifting
                    lower -= 0x40000000
                    upper -= 0x40000000
                    decompressor.value -= 0x40000000
                else
                    break
                end
                lower <<= 1
                upper <<= 1
                decompressor.value <<= 1
                decompressor.bit_processed += 1
                if (decompressor.bits_copied < length(decompressor.input_bits))
                    decompressor.bits_copied += 1
                    decompressor.value +=
                        decompressor.input_bits[decompressor.bits_copied]
                end
            end

            decompressor.lower = lower
            decompressor.range = upper - lower
            return (i)
        end
    end
end

"""
    round_prob(P; bits=32)

Round a probability vector `P` to fixed-point representation with the given
number of `bits`. Returns a vector of `(cumulative_lower, width)` tuples where
each value is an unsigned integer in `[0, 2^bits]`.

Every symbol with `P[j] > 0` is guaranteed a width of at least 1: rounding
cumulative probabilities independently for each symbol can otherwise squeeze
a genuinely positive probability down to a width of 0 (e.g. because floating-
point summation of the earlier entries already reached 1.0), which makes that
symbol permanently unencodable and corrupts `compress!`/`decompress!` state
for everything that follows. The (tiny) rounding slack this introduces is
absorbed into the largest bin so widths still sum to exactly `2^bits`.
"""
function round_prob(P; bits = 32)
    # base equals 2^bits
    base = UInt(1) << bits
    n = length(P)
    count(>(0), P) <= base ||
        error("round_prob: more distinct symbols than representable at $bits bits")

    widths = Vector{UInt}(undef, n)
    for j = 1:n
        widths[j] = P[j] > 0 ? max(UInt(1), UInt(round(P[j] * base))) : UInt(0)
    end

    # independent rounding (and the minimum-width bump above) can push the
    # total slightly above or below `base`; absorb the difference in the
    # largest bin so the cumulative widths sum to exactly `base`.
    total = sum(widths)
    if total != base
        j = argmax(widths)
        widths[j] = UInt(Int(widths[j]) + Int(base) - Int(total))
    end

    P_out = Vector{Tuple{UInt,UInt}}(undef, n)
    cumD = UInt(0)
    for j = 1:n
        P_out[j] = (cumD, widths[j])
        cumD += widths[j]
    end

    return (P_out)
end

end # module ArithmeticCoding
