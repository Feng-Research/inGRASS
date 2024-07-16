This work presents inGRASS, a novel algorithm designed for incremental spectral sparsification of large undirected graphs. 
The proposed inGRASS algorithm is highly scalable and parallel-friendly, having a nearly-linear time complexity for the setup phase and the ability to update the spectral sparsifier
in $O(\log N)$ time for each incremental change made to the original graph with $N$ nodes. A key component in the setup phase of inGRASS  is a multilevel resistance
embedding framework introduced for efficiently identifying spectrally-critical edges and effectively detecting redundant ones, which is achieved by decomposing the
initial sparsifier into many node clusters with bounded effective-resistance diameters leveraging a low-resistance-diameter decomposition (LRD) scheme.
The update phase of inGRASS exploits low-dimensional node embedding vectors for efficiently estimating the importance and uniqueness of each newly added edge.
As demonstrated through extensive experiments, inGRASS achieves up to over $200 \times$ speedups while retaining comparable solution quality in incremental
spectral sparsification of graphs obtained from various datasets, such as circuit simulations, finite element analysis, and social networks.
