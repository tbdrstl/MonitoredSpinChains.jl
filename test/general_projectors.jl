using Test
using MonitoredSpinChains
using SparseArrays
using LinearAlgebra
import MonitoredSpinChains: get_proj_su2, X, Y, Z, speye, Projector, ⊗

@testset "SU2 projectors r=1 obc compatibility" begin
    L = 6
    proj_obc = get_proj_su2(L; r=1, pbc=false)
    @test length(proj_obc) == L-1
    for site in 1:L-1
        expected = speye(2^(site-1)) ⊗ Projector ⊗ speye(2^(L-site-1)) |> SparseMatrixCSC{ComplexF64,Int64}
        P = proj_obc[site]
        @test P ≈ expected
        @test P ≈ P'                      # Hermitian
        @test P * P ≈ P                   # Idempotent
    end
end

@testset "SU2 projectors r=1 pbc extra wrap projector" begin
    L = 6
    proj_pbc = get_proj_su2(L; r=1, pbc=true)
    proj_obc = get_proj_su2(L; r=1, pbc=false)
    @test length(proj_pbc) == L          # one additional projector
    for k in 1:L-1
        @test proj_pbc[k] ≈ proj_obc[k]  # first L-1 identical
    end
    # Construct expected wrap-around projector (1,L)
    IdMid = speye(2^(L-2))
    expected_wrap = (speye(2^L) - X ⊗ (IdMid ⊗ X) - Y ⊗ (IdMid ⊗ Y) - Z ⊗ (IdMid ⊗ Z)) * 0.25
    Pw = proj_pbc[end]
    @test Pw ≈ expected_wrap
    @test Pw ≈ Pw'
    @test Pw * Pw ≈ Pw
end

@testset "SU2 projectors general r pbc properties" begin
    L = 10
    for r in 1:div(L,2)
        projs = get_proj_su2(L; r=r, pbc=true)
        expected_len = (iseven(L) && r == div(L,2)) ? div(L,2) : L
        @test length(projs) == expected_len
        for P in projs
            @test P ≈ P'
            @test P * P ≈ P
        end
    end
end

@testset "SU2 projectors general r obc properties" begin
    L = 10
    for r in 1:div(L,2)
        projs = get_proj_su2(L; r=r, pbc=false)
        @test length(projs) == L - r
        for P in projs
            @test P ≈ P'
            @test P * P ≈ P
        end
    end
end