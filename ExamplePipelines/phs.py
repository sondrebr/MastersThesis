from gudhi import SimplexTree


def st_to_pd(sts: SimplexTree, seed):
    """Convert Gudhi SimplexTrees to H_1 persistence intervals."""
    pds = []

    for st in sts:
        new_st = st.copy()
        new_st.compute_persistence()
        pds.append(new_st.persistence_intervals_in_dimension(1))

    return pds
