include("Functions.jl")

cd("../data/")
## Importing the original matrix
ar_org, W, A = mtx2arWG("G2_circuit.mtx")

## Importing the initial graph sparsifier
arP, WP, AP = mtx2arW("G2_initial_sparsifier.mtx")

## Importing the newly added edges; the edges are weighted
ext, W_add = Rext("extEG.mtx")
cd("../src/")

## Target condition number
TCND = 87
## Setup phase; this phase is executed only once prior to receiving
# a stream of newly introduced edges
println("+++++++++++++++ Setup phase +++++++++++++")
Nid_mat, dictSM, FL = Setup(TCND, AP)


## Update phase
println("+++++++++++++++ Update phase +++++++++++++")
ESA, WP, W_esa = Update(Nid_mat, ext, dictSM, FL, WP, W_add)

## Adding selected edges after similarity check to the sparse graph
append!(arP, ext[ESA])
append!(WP, W_esa[ESA])

## off-tree edge percentage
println("+++++++++++++++ Density of updated graph sparsifier (percentage %) +++++++++++++")
offT = ((length(arP) - mx_func(arP)) / mx_func(arP)) * 100
println("D = ", offT)

## Adding the newly introduced edges to the original graph
append!(ar_org, ext)
append!(W, W_add)
M_add = CliqueW(ar_org, W)
L_add = lap(M_add)
for ii = 1:size(A,1)
    L_add[ii,ii] = L_add[ii,ii]+1e-6
end

## Compute the Laplacian matrix of graph sparsifier
# after adding the selected edges using inGRASS
M = CliqueW(arP, WP)
L = lap(M)
for ii = 1:size(AP,1)
    L[ii,ii] = L[ii,ii]+1e-6
end

## Computing condition number
println("+++++++++++++++ Condition number +++++++++++++")
CND = eigs(L_add, L)
println("The recovered condition number = ", maximum(CND[1]))

## Output
WmtxW("Output.mtx", L)
