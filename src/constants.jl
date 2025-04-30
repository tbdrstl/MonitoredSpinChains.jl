export  X, 
        Y, 
        Z, 
        up, 
        down, 
        plus, 
        minus, 
        hadamard,
        SWAP,
        CNOT,
        Projector,
        singlet,
        fredkin,
        motzkin,
        sz,
        upm,
        flatm,
        downm,
        Xm


⊗(A,B) = kron(A,B)
speye(n::Int64) = IMatrix(n)

const X              ::SparseMatrixCSC{Float64, Int64}         = sparse([0. 1.; 1. 0.])
const Y              ::SparseMatrixCSC{ComplexF64, Int64}         = sparse([0. im; -im 0.])
const Z              ::SparseMatrixCSC{ComplexF64, Int64}         = sparse([1. 0.; 0. -1.])
const up             ::SparseVector{Float64, Int64}               = sparse([1., 0.])
const down           ::SparseVector{Float64, Int64}               = sparse([0., 1.])
const plus           ::SparseVector{Float64, Int64}               = sparse([1., 1.])/sqrt(2.)
const minus          ::SparseVector{Float64, Int64}               = sparse([1., -1.])/sqrt(2.)
const hadamard       ::AbstractMatrix{Float64}                    = [1. 1.; 1. -1.] / sqrt(2.)
const SWAP           ::AbstractMatrix{Float64}                    = sparse([1. 0. 0. 0.; 0. 0. 1. 0.; 0. 1. 0. 0.; 0. 0. 0. 1.])
const CNOT           ::AbstractMatrix{Float64}                    = sparse([1. 0. 0. 0.; 0. 1. 0. 0.; 0. 0. 0. 1.; 0. 0. 1. 0.])
const PauliMatricies ::Vector{SparseMatrixCSC{ComplexF64, Int64}} = [X,Y,Z]
const Projector      ::SparseMatrixCSC{ComplexF64, Int64}         = 0.25 * (speye(4) - X⊗X - Y⊗Y - Z⊗Z)
const singlet        ::SparseVector{Float64, Int64}               = sparse([0.,1.,-1.,0.]) ./sqrt(2.)
# implementation of Fredkin gate
# Fredkin gate is a controlled swap gate
const fredkin       ::SparseMatrixCSC{ComplexF64, Int64}         = (up*up') ⊗ Projector + Projector ⊗ (down*down')

# implementation of Motzkin spin gate
upm = [1,0,0]
flatm = [0,1,0]
downm = [0,0,1]

Xm = [0 0 1; 0 1 0; 0 0 1]

function proj(vec::AbstractVector{N}) where N <: Number
    return vec * vec'
end

const U = 0.5 * proj(upm⊗flatm - flatm⊗upm)
const D = 0.5 * proj(downm⊗flatm - flatm⊗downm)
const F = 0.5 * proj(upm⊗downm - flatm⊗flatm)
const motzkin = U + D + F |> SparseMatrixCSC{ComplexF64, Int64}

# const sz = Diagonal([1,-1,1]) |> SparseMatrixCSC{ComplexF64, Int64}
const sz = [0 0 1; 0 1 0; 1 0 0 ] |> SparseMatrixCSC{ComplexF64, Int64}

const legal_observables = [:OP]
const legal_feedbacks = [:Id, :Z]
const required_params = [   "name", 
                            "systemSize", 
                            "meas_steps", 
                            "average", 
                            "bc", 
                            "initialState", 
                            "measurement", 
                            "feedback", 
                            "result_folder", 
                            "observables", 
                            "trajectories_averaged", 
                            "thermalizationSteps", 
                            "meas_every",
                            "local_spin"]
