# inGRASS

https://arxiv.org/abs/2402.16990

Ali Aghdaei and Zhuo Feng

inGRASS is a novel algorithm designed for incremental spectral sparsification of large undirected graphs. 
The proposed inGRASS algorithm is highly scalable and parallel-friendly, having a nearly-linear time complexity for the setup phase and the ability to update the spectral sparsifier
in $O(\log N)$ time for each incremental change made to the original graph with $N$ nodes. A key component in the setup phase of inGRASS  is a multilevel resistance
embedding framework introduced for efficiently identifying spectrally-critical edges and effectively detecting redundant ones, which is achieved by decomposing the
initial sparsifier into many node clusters with bounded effective-resistance diameters leveraging a low-resistance-diameter decomposition (LRD) scheme.
The update phase of inGRASS exploits low-dimensional node embedding vectors for efficiently estimating the importance and uniqueness of each newly added edge.
As demonstrated through extensive experiments, inGRASS achieves up to over $200 \times$ speedups while retaining comparable solution quality in incremental
spectral sparsification of graphs obtained from various datasets, such as circuit simulations, finite element analysis, and social networks.

# Requirements
You need the following Julia packages to run the code:
Arpack v0.4.0
LinearAlgebra
SparseArrays
RandomV06 v0.0.2
Laplacians v1.2.0 `https://github.com/danspielman/Laplacians.jl.git#master`

You can use the following commands to add a package in Julia:
using Pkg
Pkg.add("Package_Name")

# Usage
Run the file "Run_inGRASS.jl" under the src directory for computing the incremental spectral sparsification of the given graph

# Dataset
The Laplacian matrix of the original graph and the initial graph sparsifier are utilized.

The extra edges (weighted) are provided.

# Output
The Laplacian matrix of the updated graph sparsifier is generated and saved in a file named "Output.mtx".
