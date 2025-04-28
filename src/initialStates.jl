
function rand_spinhalf_im(L::Int)
    psi = randn(ComplexF64,2^L) 
    return psi ./ norm(psi)
end

function rand_spinone_im(L::Int)
    psi = randn(ComplexF64,3^L)
    return psi ./ norm(psi)
end

function rand_spinhalf_real(L::Int)
    psi = randn(ComplexF64,2^L) 
    return psi ./ norm(psi)
end

function rand_spinone_real(L::Int)
    psi = randn(ComplexF64,3^L)
    return psi ./ norm(psi)
end