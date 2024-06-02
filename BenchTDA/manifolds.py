"""Module for sampling point clouds based on manifolds or the Power Spherical distribution."""

import numpy as np
# import scipy as sp  # For alternate genus g torus generation


# Helper function
def _get_rng(random_state: np.random.RandomState | int | None):
    if (type(random_state) is np.random.RandomState):
        rng = random_state
    elif (type(random_state) is int):
        rng = np.random.RandomState(seed=random_state)
    elif (random_state is None):
        rng = np.random.RandomState()
    else:
        raise TypeError("random_state is not a RandomState instance or an int.")
    return rng


""" Spheres """


def nSphere(
    n: int, size: int,
    scale=1.0, noise_scale=0.0,
    distribution=lambda gen, n, size: gen.multivariate_normal(np.zeros(n), np.eye(n), size),
    noise_distribution=lambda gen, size: gen.normal(0, 1, size),
    random_state: int | np.random.RandomState | None = None,
):
    """Generate a (by default) uniform n-sphere based on normalized gaussian sampling for each dimension.

    Args:
        n (int): The n-sphere to sample.
        size (int): The number of points to sample.
        scale (float, optional): The radius of the n-sphere.
        noise_scale (float, optional): Scales the noise applied to the points. Defaults to 0.0.
        distribution (Callable[[np.random.RandomState, int, int], np.ndarray], optional): Function which calls RandomState distribution from which the points are sampled, with correct n and size. Defaults to lambda gen, n, size: gen.multivariate_normal(np.zeros(n), np.eye(n), size).
        noise_distribution (Callable[[np.random.RandomState, int], np.ndarray], optional): Function which calls the RandomState distribution from which the noise is sampled, with correct size. Defaults to lambda gen, size: gen.normal(0, 1, size).
        random_state (int | np.random.RandomState | None, optional): The (seed of the) RandomState instance to be used. If None, it uses an unseeded instance. Defaults to None.
    """
    rng = _get_rng(random_state)

    points = distribution(rng, n+1, size)
    noised_radii = scale * (1 + (noise_scale * noise_distribution(rng, size)))
    norms = np.divide(np.linalg.norm(points, axis=-1), noised_radii)
    final = points / np.expand_dims(norms, axis=-1)
    return final


# https://arxiv.org/pdf/2006.04437.pdf
def power_spherical(
        dim: int, size: int, concentration: float, direction: np.ndarray,
        noise_scale=0.0,
        noise_distribution=lambda gen, size: gen.normal(0, 1, size),
        random_state: int | np.random.RandomState | None = None,
):
    """ Generate a sphere based on the Power Sphercial distribution, as described in https://arxiv.org/pdf/2006.04437.pdf.

    Args:
        dim (int): The dimension of the sphere, e.g. 3 for a 2-sphere.
        size (int): The number of points to sample.
        concentration (float): The concentration of the points. Must be positive.
        direction (np.ndarray): The direction in which the points are concentrated. Must have shape (dim,), and have non-zero magnitude.
        noise_scale (float, optional): Scales the noise applied to the points. Defaults to 0.0.
        noise_distribution (Callable[[np.random.RandomState, int], np.ndarray], optional): Function which calls the RandomState distribution from which the noise is sampled, with correct size. Defaults to lambda gen, size: gen.normal(0, 1, size).
        random_state (int | np.random.RandomState | None, optional): The (seed of the) RandomState instance to be used. If None, it uses an unseeded instance. Defaults to None.
    """
    # Verify inputs
    if (concentration < 0):
        raise ValueError("Concentration cannot be negative.")
    if (direction.shape[0] != dim and direction.T.shape[0] != dim):
        raise ValueError(f'Direction has invalid shape: {direction.shape}')
    if ((dir_norm := np.linalg.norm(direction)) == 0):
        raise ValueError("Direction cannot be zero.")

    rng = _get_rng(random_state)

    # Scale direction to S^(d-1)
    dir = direction.reshape(dim) / dir_norm

    # Sample z ∼ Beta (Z; (d − 1)/2 + κ,(d − 1)/2)
    z = rng.beta(
        a=((dim-1) / 2) + concentration,
        b=(dim-1) / 2,
        size=(size, 1),
    )

    # Sample v ∼ U(S^(d−2))
    v = nSphere(
        n=dim-2,
        size=size,
        random_state=rng,
    )

    t = (2 * z) - 1

    y = np.concatenate(
        (t, np.multiply(np.sqrt(1-(t**2)), v)),
        axis=-1,
    )

    e1 = np.zeros(dim, dtype=float)
    e1[0] = 1.0
    u_roof = e1 - dir

    if ((u_norm := np.linalg.norm(u_roof)) == 0.0):
        u = u_roof.reshape((dim, 1))
    else:
        u = (u_roof / u_norm).reshape((dim, 1))

    x = y @ (np.identity(dim) - (2*(u @ u.T)))

    # Noise
    x *= ((noise_scale * noise_distribution(rng, size=x.shape)) + 1)
    return x


""" Toruses """


def uniform_torus(
    size: int, R=1.0, r=0.5, noise_scale=0.0,
    rev_distribution=lambda gen: gen.uniform(),
    noise_distribution=lambda gen: gen.normal(),
    random_state: int | np.random.RandomState | None = None,
):
    """Generate a uniform torus using rejection sampling. Code ported from R code in https://arxiv.org/abs/1206.6913.

    Args:
        size (int): The total number of points.
        R (float, optional): Big radius. Defaults to 1.0.
        r (float, optional): Small radius. Defaults to 0.5.
        noise_scale (float, optional): Scales the noise applied to the points. Defaults to 0.0.
        rev_distribution (Callable[[np.random.RandomState], float], optional): Function calling the RandomState distribution from which the angle for the axis of revolution is sampled. Defaults to lambda gen: gen.uniform().
        noise_distribution (Callable[[np.random.RandomState], float], optional): A function calling the RandomState distribution from which the noise is sampled. Defaults to lambda gen: gen.normal().
        random_state (int | np.random.RandomState | None, optional): The (seed of the) RandomState instance to be used. If None, it uses an unseeded instance. Defaults to None.
    """
    rng = _get_rng(random_state)

    points = np.zeros((size, 3))

    rs_angles = np.zeros(size)
    i = 0
    while (i < size):
        x = rng.uniform() * 2*np.pi
        y = rng.uniform() * (1/np.pi)
        fx = (1 + ((r/R) * np.cos(x))) / (2*np.pi)
        if (y < fx):
            rs_angles[i] = x
            i += 1

    for i in range(size):
        phi = rev_distribution(rng) * 2*np.pi
        noise = 1 + (noise_scale * noise_distribution(rng))
        points[i, 0] = (R + ((noise * r) * np.cos(rs_angles[i]))) * np.cos(phi)
        points[i, 1] = (R + ((noise * r) * np.cos(rs_angles[i]))) * np.sin(phi)
        points[i, 2] = (noise * r) * np.sin(rs_angles[i])

    return points


def genus_g_torus(
    g: int, size: int,
    R=1.0, r=0.5,
    cutoff_mod=0.0,
    noise_scale=0.0,
    noise_distribution=lambda gen: gen.normal(),
    random_state: int | np.random.RandomState | None = None,
):
    """Generate a torus of genus g.

    Args:
        size (int): The total number of points
        g (int): Genus
        R (float, optional): Big radius. Defaults to 1.0.
        r (float, optional): Small radius. Defaults to 1.
        cutoff_mod (float, optional): Must be in the range [-1.0, 1.0]. Used to calculate cutoff point (R + (cutoff_mod * r)).
        noise_scale (float, optional): Scales the noise applied to the points. Defaults to 0.0.
        noise_distribution (Callable[[np.random.RandomState], float], optional): A function calling the RandomState distribution from which the noise is sampled. Defaults to lambda gen: gen.normal().
        random_state (int | np.random.RandomState | None, optional): The (seed of the) RandomState instance to be used. If None, it uses an unseeded instance. Defaults to None.
    """
    rng = _get_rng(random_state)

    torii = np.zeros((size, 3))
    drop_list = [(i > 0, i < g-1) for i in range(g)]

    cutoff = R + (cutoff_mod * r)
    translation = {t: 2*cutoff*(t-((g-1)/2)) for t in range(g)}

    """ Rejection sampling the cut-off means the genus g torus generation is less efficient.
        The commented-out lines below contain a more efficient, but less straight-forward
        alternative using the surface areas of the toruses as probabilites for the torus
        selection to eliminate rejection steps 2-3
    """
    # total_area = (2 * np.pi * r) * (2 * np.pi * R)
    # cutoff_loss = 4 * (sp.integrate.quad(
    #     lambda x: (R+(r*np.cos(x)))*np.arccos((R + (cutoff_mod * r)) / (R + (r * np.cos(x)))),
    #     0, np.arccos(cutoff_mod)))
    # p = [total_area - (cutoff_loss * sum(drop_list[i])) for i in range(g)]
    # t = rng.choice(g, p=p) - Replaces t = rng.choice(g)

    i = 0
    while (i < size):
        t = rng.choice(g)
        drop_left, drop_right = drop_list[t]

        # Modified rejection sampling
        theta = rng.uniform() * 2*np.pi
        y = rng.uniform() * (1/np.pi)
        fx = (1 + ((r/R) * np.cos(theta))) / (2*np.pi)
        if (y >= fx):
            continue  # Rejection step 1

        phi = rng.uniform() * 2*np.pi
        x_coord = (R + (r * np.cos(theta))) * np.cos(phi)  # Check x coordinate pre-noise
        if (drop_left and x_coord < -cutoff):
            continue  # Rejection step 2
        if (drop_right and x_coord > cutoff):
            continue  # Rejection step 3

        noise = 1 + (noise_scale * noise_distribution(rng))
        # X adjusted for torus no.
        torii[i, 0] = (R + ((noise * r) * np.cos(theta))) * np.cos(phi) + translation[t]
        # Y
        torii[i, 1] = (R + ((noise * r) * np.cos(theta))) * np.sin(phi)
        # Z
        torii[i, 2] = (noise * r) * np.sin(theta)

        i += 1

    return torii
