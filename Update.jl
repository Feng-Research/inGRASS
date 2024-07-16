function Update(Nid_mat, ext, dictSM, FL, AP, W_add)

    ## similarity check; bucketing of newly added edges using node embedding
    # vectors computed at setup phase
    println("------- Edge bucketing of initial sparsifier -------")
    dictF = EBsel(Nid_mat, ext)

    ## Exclude the edges if there is an existing inter-cluster edges
    println("------- Filtering based on edge distortion and edge similarity ------")
    ESA, WP, W_esa = DelE(dictF, FL, ext, Nid_mat, dictSM, AP, W_add)

    return ESA

end # end of function
