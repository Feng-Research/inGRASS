
using SparseArrays
using LinearAlgebra
using Random
using Laplacians
using Arpack

function Setup(TCND, AP)
    ## Graph Decomposition
    println("-------- Graph decomposition --------")
    idx_mat = decompositionW(AP)

    ## node indices
    Nid_mat = NID(idx_mat)

    ## Find the filtering level according to the given condition number
    NL = ceil.(Int, TCND/2)
    FL = FindFL(Nid_mat, NL)

    ## dictionary of edges in the initial sparsifier at filtering level
    println("----- Edge bucketing of initial sparsifier ------")
    dictSM = FLD(Nid_mat[FL], AP)

    println("+++++++++++++++ The setup phase has been successfully completed +++++++++++++")

    return Nid_mat, dictSM, FL

end # end of function


function Update(Nid_mat, ext, dictSM, FL, WP, W_add)

    ## similarity check; bucketing of newly added edges using node embedding
    # vectors computed at setup phase
    println("------- Edge bucketing of newly introduced edges -------")
    dictF = EBsel(Nid_mat, ext)

    ## Exclude the edges if there is an existing inter-cluster edges
    println("------- Filtering based on edge distortion and edge similarity ------")
    ESA, WP, W_esa = DelE(dictF, FL, ext, Nid_mat, dictSM, WP, W_add)

    println("+++++++++++++++ The update phase has been successfully completed +++++++++++++")

    return ESA, WP, W_esa

end # end of function


## Array to adjacency matrix conversion
# Inputs:
# ar: the array of arrays corresponding to the graph
# W: the edge weights

# Output: the adjacency matrix
function CliqueW(ar, W)
    mx = mx_func(ar)
    rr = zeros(Int, 0)
    cc = zeros(Int, 0)
    vv = zeros(Float32, 0)
    for i =1:length(ar)
        nd = sort(ar[i])
        append!(rr, nd[1])
        append!(cc, nd[2])
        append!(vv, W[i])
    end
    mat1 = sparse(rr,cc,vv, mx, mx)
    return mat2 = mat1 + sparse(mat1')
end

## Maximum value in the array of arrays
# Input: array of arrays
# Output: maximum value
function mx_func(ar)
    mx2 = Int(0)
    aa = Int(0)
    for i =1:length(ar)
    	mx2 = max(aa, maximum(ar[i]))
    	aa = mx2
    end
    return mx2
end

#=
function mtx2ar(Inp)

    io = open(Inp, "r")
    rr = zeros(Int, 0)
    cc = zeros(Int, 0)
    while !eof(io)
        ln = readline(io)
        sp = split(ln)
        r = parse(Int, sp[1])
        c = parse(Int, sp[2])
        append!(rr, r)
        append!(cc, c)
    end

    ar = Any[]

    for ii = 1:length(rr)

        push!(ar, sort([rr[ii], cc[ii]]))

    end # for ii

    A = Clique_sm(ar)

    return ar, A

end #end of function
=#

function mtx2arW(Inp)
    io = open(Inp, "r")
    rr = zeros(Int, 0)
    cc = zeros(Int, 0)
    w1 = zeros(Float64, 0)
    ff = zeros(Float64, 0)
    while !eof(io)
        ln = readline(io)
        sp = split(ln)
        r = parse(Int, sp[1])
        c = parse(Int, sp[2])
        v = parse(Float64, sp[3])
        f = parse(Float64, sp[4])
        append!(rr, r)
        append!(cc, c)
        append!(w1, v)
        append!(ff, f)
    end

    ## Creating the adjacency matrix
    L = sparse(rr, cc, w1./ff)
    L1 = tril(L,-1).*-1
    A = L1 + sparse(L1')
    ## Creating ar
    fdnz = findnz(triu(A,1))
    rr = fdnz[1]
    cc = fdnz[2]
    W = fdnz[3]
    ar = Any[]
    for ii = 1:length(rr)
        push!(ar, sort([rr[ii], cc[ii]]))
    end # for ii

    return ar, W, A
end #end of function


function Rext(Inp)
    io = open(Inp, "r")
    rr = zeros(Int, 0)
    cc = zeros(Int, 0)
    W = zeros(Float64, 0)
    while !eof(io)
        ln = readline(io)
        sp = split(ln)
        r = parse(Int, sp[1])
        c = parse(Int, sp[2])
        v = parse(Float64, sp[3])
        append!(rr, r)
        append!(cc, c)
        append!(W, v)
    end
    ar = Any[]
    for ii = 1:length(rr)
        push!(ar, sort([rr[ii], cc[ii]]))
    end # for ii
    return ar, W
end


function mtx2arWG(Inp)
    io = open(Inp, "r")
    rr = zeros(Int, 0)
    cc = zeros(Int, 0)
    w1 = zeros(Float64, 0)
    ff = zeros(Float64, 0)
    while !eof(io)
        ln = readline(io)
        sp = split(ln)
        r = parse(Int, sp[1])
        c = parse(Int, sp[2])
        v = parse(Float64, sp[3])
        append!(rr, r)
        append!(cc, c)
        append!(w1, v)
    end

    ## Creating the adjacency matrix
    L = sparse(rr, cc, w1)
    L1 = tril(L,-1).*-1
    A = L1 + sparse(L1')
    ## Creating ar
    fdnz = findnz(triu(A,1))
    rr = fdnz[1]
    cc = fdnz[2]
    W = fdnz[3]
    ar = Any[]
    for ii = 1:length(rr)
        push!(ar, sort([rr[ii], cc[ii]]))
    end # for ii

    return ar, W, A
end #end of function

## Computing the smooth vectors using Krylov subspace
# Inputs:
# rv: initial random vector
# rho: order of Krylov subspace
# AD: adjacency matrix
# initial: starting point of smoothing operation; default = 0
# interval: interval in order of krylov subspace
# Ntot: total number of vectors

# Output:
# V: the node embedding vectors
function smV(rv, rho, AD, mx, initial, interval, Ntot)

    sz = size(AD, 1)
    V = zeros(mx, Ntot);
    AD = AD .* 1.0
    AD[diagind(AD, 0)] = AD[diagind(AD, 0)] .+ 0.1
    dg = sum(AD, dims = 1) .^ (-.5)
    I2 = 1:sz
    D = sparse(I2, I2, sparsevec(dg))
    on = ones(Int, length(rv))
    sm = (rv - ((dot(rv, on) / dot(on, on)) * on)) ./ norm(rv - ((dot(rv, on) / dot(on, on)) * on))
    temp = similar(sm)  # Create temporary matrix for intermediate results
    count = 1
    for loop in 1:rho
        mul!(temp, D, sm)
        copyto!(sm, temp)  # Update sm with the intermediate result
        mul!(temp, AD, sm)
        copyto!(sm, temp)
        mul!(temp, D, sm)
        copyto!(sm, temp)
        if rem(loop, interval) == 0
            @views V[:, count] = (sm .- ((dot(rv, on) / dot(on, on)) .* on)) ./ norm(sm .- ((dot(rv, on) / dot(on, on)) .* on));
            count +=1
        end
    end # for loop
    return V
end #end of function

## decompose the graph into multiple cluster with low-resistance-diameters (LRDs)
# Input:
# A: adjacency matrix

# Output:
# idx_mat: mapping informations
function decompositionW(A)

    idx_mat = Any[]
    Neff = zeros(Float64, size(A,1))
    ## Tmx specifies the number of nodes in the coarsest level
    Tmx = round(Int, 0.01 * size(A,1))

    while size(A, 1) > Tmx
        mx = size(A, 1)
        fdnz = findnz(triu(A, 1))
        rr = fdnz[1]
        cc = fdnz[2]
        W = fdnz[3]
        ar1 = Any[]
        for ii = 1:length(rr)
            push!(ar1, [rr[ii], cc[ii]])
        end
        MM = length(rr)

        ## computing the smoothed vectors
        initial = 0
        rho = 300
        interval = 20
        Nrv = 1
        Nsm = floor.(Int, (rho - initial) / interval)
        Ntot = Nrv * Nsm
        Eratio = zeros(Float64, MM, Ntot)
        SV = zeros(Float64, mx, Ntot)

        for ii = 1:Nrv
            sm = zeros(mx, Nsm)
            Random.seed!(1234); randstring()
            rv = (rand(Float64, size(A, 1), 1) .- 0.5).*2
            sm = smV(rv, rho, A, mx, initial, interval, Nsm)
            SV[:, (ii-1)*Nsm+1 : ii*Nsm] = sm
        end

        ## Make all the smoothed vectors orthogonal to each other
        QR1 = qr(SV)
        SV = Matrix(QR1.Q)

        ## Computing the ratios using all the smoothed vectors
        Evec = zeros(Float64, MM)
        for jj = 1:size(SV, 2)
            #include("h_score3_fast.jl")
            E1 = scoreSPF(rr, cc, W, SV[:, jj])
            Evec += E1

        end #for jj


        # Adding the effective resistance of super nodes from previous levels
        @inbounds for kk = 1:length(rr)
            Evec[kk] = Evec[kk] + Neff[rr[kk]] + Neff[cc[kk]]
        end
        ## Choosing a ratio of the hyperedges for contraction
        Nsample = ceil(Int, MM)
        Pos = sortperm(Evec[:,1])

        ## low-ER diameter clustering which starts by contracting
        # the hyperedges with low ER diameter
        flag = falses(mx)
        val = 1
        idx = zeros(Int, mx)
        Neff_new = zeros(Float64, 0)
        @inbounds for ii = 1:Nsample
            nd = [rr[Pos[ii]], cc[Pos[ii]]]
            fg = flag[nd]
            fd1 = findall(x->x==0, fg)
            if length(fd1) > 1
                nd = nd[fd1]
                idx[nd] .= val
                flag[nd] .= 1
                val +=1
                ## creating the super node weights
                append!(Neff_new, Evec[Pos[ii]] + sum(Neff[nd]))
            end # endof if
        end #end of for ii

        ## indexing the isolated nodes
        fdz = findall(x-> x==0, flag)
        V = vec(val:val+length(fdz)-1)
        idx[fdz] = V
        ## Adding the weight of isolated nodes
        append!(Neff_new, Neff[fdz])
        #push!(Nmat, Neff)
        push!(idx_mat, idx)
        ## generating the coarse hypergraph
        rr_new = zeros(Int,0)
        cc_new = zeros(Int,0)
        @inbounds for ii = 1:MM
            append!(rr_new, min(idx[rr[ii]], idx[cc[ii]]))
            append!(cc_new, max(idx[rr[ii]], idx[cc[ii]]))
        end #end of for ii

        R1 = vcat(rr_new, cc_new)
        R2 = vcat(cc_new, rr_new)
        W = vcat(W,W)
        A = sparse(R1, R2, W)
        Neff = Neff_new

    end #end for loop

    return idx_mat
end

## Effective resistance estimation using node embeddings computed from
# utilizing Krylov-subspace

# Inputs:
# rrH & ccH: node indices
# WH: edge weights
# SV: node embedding vectors

# Output:
# R: the estimated effective resistance
function scoreSPF(rrH, ccH, WH, SV)
    DST = zeros(eltype(SV), length(rrH))
    Qval = 0
    @inbounds for i in eachindex(rrH)
            DST[i] = (SV[rrH[i]] - SV[ccH[i]])^2
            Qval = Qval + (DST[i] * WH[i])
    end
    R = DST ./ Qval

    return R
end

## generates a dictionary indicating the edges that are
# intersecting between the same clusters in the initial graph sparsifier

# Inputs:
# NDM: node embedding vectors
# AP: initial graph sparsifier

# Output:
# dictSM: edges that are intersecting between the same clusters, for example:
# [10, 22]-> [6, 12, 18] means that the edges 6, 12, and 18 are crossing
# between the cluster index 10 and cluster index 22
function FLD(NDM, AP)
    rr, cc, ww = findnz(triu(AP,1))
    dictSM = Dict{Array{Int64,1}, Array{Int64,1}}()
    @inbounds for ii = 1:length(rr)
        N1 = NDM[rr[ii]]
        N2 = NDM[cc[ii]]
        key = sort([N1,N2])
        vals = get!(Array{Int64,1}, dictSM, key)
        push!(vals, ii)
    end # for ii

    return dictSM

end

## Map node clusters from coarse levels to the original level
# Input: node mapper
# Output: node indices
function NID(idx_mat)
    Nid_mat = Any[]
    for jj = 1:length(idx_mat)
        id1 = idx_mat[end-jj+1]
        for ii = jj:length(idx_mat)-1
            id1 = id1[idx_mat[end-ii]]
        end # for ii
        push!(Nid_mat, id1)
    end # for jj
    push!(Nid_mat, vec(1:length(idx_mat[1])))

    return Nid_mat
end # end of function


## Find the level of filtering for similarity according to maximum cluster size
# Input:
# CLsz: maximum cluster size which is maximum number of nodes in one cluster

# Output:
# FL: the filtering level
function FindFL(Nid_mat, CLsz)
    clS = zeros(Int, 0)
    for ii = 1:length(Nid_mat)
        dict1 = Dict{Int,Array{Int64,1}}()
        V = Nid_mat[ii]
        for jj = 1:length(V)
            key = V[jj]
            val = get!(Array{Int64,1},dict1, key)
            push!(val, jj)
        end # for jj
        VL = collect(values(dict1))
        mxS = 0
        for jj = 1:length(VL)
            mxS = max(mxS, length(VL[jj]))
        end # for jj
        append!(clS, mxS)
    end # for ii
    ## Finding the filter level
    fdN = findall(x-> x < CLsz,  clS)
    FL = minimum(fdN)
    return FL
end # function


## Computes a dictionary for the newly introduced edges that identifies
# the coarsening level at which each edge is contracted; Also, it
# indicates the cluster indices that edges are intersecting in the prior
# level of being merged.

# Inputs:
# ext: the newly introduced edges
# Nid_mat: the node embedding vectors

# Output:
# dictF: similarity dictionary, for example:
# (2, 1, 3)->3 means that the edge number 3 of newly introduced edges
# is merged at level 2; it also means that in level 3, which is prior
# coarsening level for contraction, the edge 3, is intersecting between
# cluster index 1, and cluster index 3
function EBsel(Nid_mat, ext)
    tempV = collect(1:length(ext))
    dictF = Dict{Tuple{Int,Int, Int}, Array{Int64,1}}()
    @inbounds for ii = 1:length(tempV)
        nd1 = ext[tempV[ii]]
        N1 = 1
        N2 = 2
        CC = length(Nid_mat)-1
        while N1 != N2 && CC>0
            N1 = Nid_mat[CC][nd1[1]]
            N2 = Nid_mat[CC][nd1[2]]
            if N1==N2
                Nb1 = Nid_mat[CC+1][nd1[1]]
                Nb2 = Nid_mat[CC+1][nd1[2]]
                key = (CC, Nb1, Nb2)
                vals = get!(Array{Int64,1}, dictF, key)
                push!(vals, tempV[ii])
            end
            CC = CC-1
            if CC ==0 && N1!= N2
                Nb1 = Nid_mat[CC+1][nd1[1]]
                Nb2 = Nid_mat[CC+1][nd1[2]]
                key = (CC, Nb1, Nb2)
                vals = get!(Array{Int64,1}, dictF, key)
                push!(vals, tempV[ii])
            end
        end # while
    end # for ii
    return dictF
end # function

## Deleting similar edges according to the filter level; Update the edge weights
# If two edges cross between the same clusters, we filter out
# the one with the smaller edge weight.
# Additionally, if an edge has nodes that belong to the same cluster,
# we do not add this edge; instead, we only update
# the edge weights of the edges within that cluster.

# Inputs:
# dictF: similarity dictionary of newly introduced edges
# FL: the filtering level
# ext: newly introduced edges
# Nid_mat: the node embedding vectors
# WP: the edge weights of initial sparsifier
# W_add: the edge weights of newly added edges

# Output:
# ESA: the indices of newly introduced edges that are included in the initial sparsifier.
# WP: the updated edge weights of initial sparsifier
# W_add: the updated edge weight of newly added edges
function DelE(dictF, FL, ext, Nid_mat, dictSM, WP, W_add)
    # sort the dictionary leys to start with global edges;
    # A global edge means an edge that is coarsened at the latest levels
    # The edge that is coarsened at level 0, is the most global edge
    KSF = sort(collect(keys(dictF)), by=key -> key[1])
    ESD = zeros(Int, 0)
    for ii = 1:length(KSF)
        CT = KSF[ii]
        LVL = CT[1]
        if LVL>-1
            Egs = dictF[CT]
            for jj = 1:length(Egs)
                nds = ext[Egs[jj]]
                N1 = Nid_mat[FL][nds[1]]
                N2 = Nid_mat[FL][nds[2]]
                TF = haskey(dictSM, sort([N1,N2]))
                if TF == true
                    IE = dictSM[sort([N1,N2])]
                    if IE[1] == -1
                        W_add[IE[2]] = W_add[IE[2]] + W_add[Egs[jj]]
                    else
                        WP[IE] .= WP[IE] .+ (W_add[Egs[jj]] ./ length(IE))
                    end
                    append!(ESD, Egs[jj])
                else
                    dictSM[sort([N1, N2])] = [-1, Egs[jj]]
                end # if
            end # for jj
        end #if
    end # for ii

    ## filter out the similar edges
    ESA = collect(1:length(ext))
    fd1 = findall(x->in(x, ESD), ESA)
    deleteat!(ESA, fd1)

    return ESA, WP, W_add
end # function
