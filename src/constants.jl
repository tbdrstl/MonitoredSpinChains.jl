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
        up1,
        flat1,
        down1,
        Xm


⊗(A,B) = kron(A,B)
speye(n::Int64) = IMatrix(n)

const X              ::SparseMatrixCSC{Float64, Int64}         = sparse([0. 1.; 1. 0.])
const Y              ::SparseMatrixCSC{ComplexF64, Int64}         = sparse([0. im; -im 0.])
const Z              ::SparseMatrixCSC{Float64, Int64}         = sparse([1. 0.; 0. -1.])
const up             ::SparseVector{Float64, Int64}               = sparse([1., 0.])
const down           ::SparseVector{Float64, Int64}               = sparse([0., 1.])
const plus           ::SparseVector{Float64, Int64}               = sparse([1., 1.])/sqrt(2.)
const minus          ::SparseVector{Float64, Int64}               = sparse([1., -1.])/sqrt(2.)
const hadamard       ::AbstractMatrix{Float64}                    = [1. 1.; 1. -1.] / sqrt(2.)
const SWAP           ::AbstractMatrix{Float64}                    = sparse([1. 0. 0. 0.; 0. 0. 1. 0.; 0. 1. 0. 0.; 0. 0. 0. 1.])
const CNOT           ::AbstractMatrix{Float64}                    = sparse([1. 0. 0. 0.; 0. 1. 0. 0.; 0. 0. 0. 1.; 0. 0. 1. 0.])
const PauliMatricies ::Vector{SparseMatrixCSC{ComplexF64, Int64}} = [X,Y,Z]
const Projector      ::SparseMatrixCSC{Float64, Int64}         = 0.25 * (speye(4) - X⊗X - Y⊗Y - Z⊗Z) |> SparseMatrixCSC{Float64, Int64}
const singlet        ::SparseVector{Float64, Int64}               = sparse([0.,1.,-1.,0.]) ./sqrt(2.)
# implementation of Fredkin gate
# Fredkin gate is a controlled swap gate
const fredkin       ::SparseMatrixCSC{Float64, Int64}         = (up*up') ⊗ Projector + Projector ⊗ (down*down') |> SparseMatrixCSC{Float64, Int64}


# implementation of Motzkin/spin-1 gate
const up1 = [1.,0.,0.]
const flat1 = [0.,1.,0.]
const down1 = [0.,0.,1.]

const Z1 = sparse(Diagonal([1.,0.,-1.]))
const exp_z1 = sparse(Diagonal([im, 1, -im]))
const Px = [0. 0. 1.; 1. 0. 0.; 0. 1. 0.]
const X1 = [0 1 0; 1 0 1; 0 1 0] ./ sqrt(2)
const PauliX1 = [0. 0. 1.; 1. 0. 0. ; 0. 1. 0.]
const exp_x1 = [0.5 im/sqrt(2)  -0.5; im/sqrt(2) 0 im/sqrt(2); -0.5 im/sqrt(2) 0.5]

function proj(vec::AbstractVector{N}) where N <: Number
    return vec * vec'
end

function spin1_max_angular_mom_proj() 
    p = (
          proj(up1 ⊗ up1 )
        + proj(down1 ⊗ down1 )
        + proj(1.0/sqrt(2)*(up1⊗flat1 + flat1 ⊗ up1) )
        + proj(1.0/sqrt(2)*(down1 ⊗ flat1 + flat1 ⊗ down1))
        + proj(1.0/sqrt(6)*(up1 ⊗ down1 + 2*flat1⊗flat1 + down1 ⊗ up1))
    )
    return p |> SparseMatrixCSC
end

const Proj1 = speye(3^2) - spin1_max_angular_mom_proj()

const U = 0.5 * proj(up1⊗flat1 - flat1⊗up1)
const D = 0.5 * proj(down1⊗flat1 - flat1⊗down1)
const F = 0.5 * proj(flat1⊗flat1 - up1⊗down1)
const motzkin = U + D + F |> SparseMatrixCSC{Float64, Int64}

# const sz = Diagonal([1,-1,1]) |> SparseMatrixCSC{ComplexF64, Int64}
const sz = [0 0 1; 0 1 0; 1 0 0 ] |> SparseMatrixCSC{ComplexF64, Int64}

const legal_observables = [:OP, :EE, :M, :MX, :EEfin, :OPH, :OPQ]
const legal_feedbacks = [:Id, :Z]
const legal_models = ["fredkin", "motzkin", "su2", "aklt"]
const legal_unitarySetups = [:twoSpinHaar, :singleSpinHaar, :noProj, :su2symmetric]
const required_params = [   "systemSize", 
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
                            "model"]

# Hashing helpers: fields that should not affect circuit identity (run logistics)
const VOLATILE_FIELDS = (:average, :trajectories_averaged, :result_folder)

# Hashing helpers: optional defaults that, when matched, are omitted from the hash
# Extend this as you add new optional physics parameters with a true "no-op" default
const DEFAULT_FIELD_VALUES = Dict{Symbol,Any}()

const anomalousstatepath = joinpath(homedir(),".julia/data/fredkin_pbc_anomalous_ground_states")
const dickestatepath = joinpath(homedir(),".julia/data/dickeStates")