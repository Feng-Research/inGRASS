function Setup(CND, AP)
    ## Graph Decomposition
    println("--------(setup): Graph decomposition --------")
    idx_mat = decompositionW(AP)

    ## node indices
    Nid_mat = NID(idx_mat)

    ## Find the filtering level according to the given condition number
    NL = CND/2
    FL = FindFL(Nid_mat, NL)

    ## dictionary of edges in the initial sparsifier at filtering level
    println("-----(setup): filter dictionary ------")
    dictSM = FLD(Nid_mat[FL], AP)

    return Nid_mat, dictSM

end # end of function
