export swap_1_L,
       swap_spin1_sites,
       swaps,
       symmetric_projector


# spin-1 swap operators 
function swap_1_L(L::Int)
    d = 3
    id = speye(d)
    # computational basis projectors |i⟩⟨j|
    basis = [sparse([i == k ? 1.0 : 0.0 for i in 1:d, j in 1:d]) for k in 1:d]

    swap = spzeros(d^L, d^L)

    for i in 1:d, j in 1:d
        op1 = basis[i] * basis[j]'  # |i⟩⟨j|
        opL = basis[j] * basis[i]'  # |j⟩⟨i|

        ops = Any[]
        push!(ops, op1)
        for _ in 2:L-1
            push!(ops, id)
        end
        push!(ops, opL)

        swap += kron(ops...)
    end

    return swap./9.
end

function swap_spin1_sites()
    S = zeros(9, 9)
    for i in 1:3, j in 1:3
        # Basis index in tensor product: |i⟩⊗|j⟩ → |j⟩⊗|i⟩
        bra = 3*(i-1) + j         # |i⟩⊗|j⟩ → row index
        ket = 3*(j-1) + i         # |j⟩⊗|i⟩ → column index
        S[bra, ket] = 1.0
    end
    return S
end

function swaps(L::Int)
    sw = swap_spin1_sites()
    swap = [speye(3^(i-1)) ⊗ sw ⊗ speye(3^(L-i-1)) for i in 1:L-1]
    return push!(swap, swap_1_L(L))
end

# documentation, that L is the system size, and d the local dimension of the Hilbert space
"""
    symmetric_projector(L::Int, d::Int)

Constructs a symmetric projector for a system of size `L` with local dimension `d`.
"""
function symmetric_projector(L::Int, d::Int)
    @assert d>0 "Local dimension `d` must be greater than 0"
    ds = generalized_dicke(L, d)
    ds = proj.(ds)
    return sum(ds)
end