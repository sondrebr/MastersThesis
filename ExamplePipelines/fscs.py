from gudhi import AlphaComplex


def pc_to_alpha(data, seed):
    """Convert point clouds to Alpha complex SimplexTrees."""
    sts = []
    for pc in data:
        ac = AlphaComplex(points=pc)
        st = ac.create_simplex_tree()
        sts.append(st)
    return sts
