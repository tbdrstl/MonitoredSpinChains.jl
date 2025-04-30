export rand_spinhalf_im,
         rand_spinone_im,
         rand_spinhalf_real,
         rand_spinone_real


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