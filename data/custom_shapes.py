import numpy as np
'''
source: https://github.com/danchern97/RTD_AE/blob/main/src/custom_shapes.py
'''

__all__ = ["torus", "dsphere", "sphere", "swiss_roll", "infty_sign"]


def embed(data, ambient=50):
    """ Embed `data` in `ambient` dimensions, regardless of dimensionality of data.

    Inputs
    ------
    data : array-like

    ambient : int
        Dimension of embedding space. Must be greater than dimensionality of data.
    """

    n, d = data.shape
    assert (
        ambient > d
    ), "Dimensionality of ambient space ({}) must be greater than dimensionality of data ({}).".format(
        ambient, d
    )

    base = np.zeros((n, ambient))
    base[:, :d] = data

    # construct a rotation matrix of dimension `ambient`.
    random_rotation = np.random.random((ambient, ambient))
    q, r = np.linalg.qr(random_rotation)

    base = np.dot(base, q)

    return base

def get_phi(row):
    """Helper function for rotating data.

    row: 2D-entry in matrix with shape (?, 2)
    """
    return np.arctan2(row[0], row[1])


def rotate_2D(d, angle):
    """Rotate a 2-dimensional figure.

    Parameters
    ============

    d: the 2-dimensional data to rotate
    angle: the angle (in radians) to rotate the data around 0
    centered_around_zero: Boolean specifying whether the data is centered around zero.
    """
    try:
        assert d.shape[1] == 2
    except AssertionError:
        raise ValueError(
            "Error: data has {} dimensions, but should only be 2. ".format(d.shape[1])
        )

    rot = angle - np.pi / 2
    phis = np.apply_along_axis(get_phi, 1, d)
    phi_new = phis + rot
    r = np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2)

    x = r * np.cos(phi_new)
    y = r * np.sin(phi_new)
    return np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), axis=1)


class Shape:
    def __init__(self):
        pass


def dsphere(n=100, d=2, r=1, noise=None, ambient=None):
    """
    Sample `n` data points on a d-sphere.

    Parameters
    -----------
    n : int
        Number of data points in shape.
    r : float
        Radius of sphere.
    ambient : int, default=None
        Embed the sphere into a space with ambient dimension equal to `ambient`. The sphere is randomly rotated in this high dimensional space.
    """
    original = np.random.randn(n, d + 1)

    # Normalize points to the sphere
    original = r * original / np.sqrt(np.sum(original ** 2, 1, keepdims=True))
    data = original.copy()

    if noise:
        data += noise * np.random.randn(*data.shape)

    if ambient:
        assert ambient > d, "Must embed in higher dimensions"
        data = embed(data, ambient)

    return original, data


def sphere(n=100, r=1, noise=None, ambient=None):
    """
        Sample `n` data points on a sphere.

    Parameters
    -----------
    n : int
        Number of data points in shape.
    r : float
        Radius of sphere.
    ambient : int, default=None
        Embed the sphere into a space with ambient dimension equal to `ambient`. The sphere is randomly rotated in this high dimensional space.
    """

    theta = np.random.random((n,)) * 2.0 * np.pi
    phi = np.random.random((n,)) * np.pi
    rad = np.ones((n,)) * r

    data = np.zeros((n, 3))

    data[:, 0] = rad * np.cos(theta) * np.cos(phi)
    data[:, 1] = rad * np.cos(theta) * np.sin(phi)
    data[:, 2] = rad * np.sin(theta)

    if noise:
        data += noise * np.random.randn(*data.shape)

    if ambient:
        data = embed(data, ambient)

    return data, theta


def torus(n=100, c=2, a=1, noise=None, ambient=None):
    """
    Sample `n` data points on a torus.

    Parameters
    -----------
    n : int
        Number of data points in shape.
    c : float
        Distance from center to center of tube.
    a : float
        Radius of tube.
    ambient : int, default=None
        Embed the torus into a space with ambient dimension equal to `ambient`. The torus is randomly rotated in this high dimensional space.
    """

    assert a <= c, "That's not a torus"

    theta = np.random.random((n,)) * 2.0 * np.pi
    phi = np.random.random((n,)) * 2.0 * np.pi

    data = np.zeros((n, 3))
    data[:, 0] = (c + a * np.cos(theta)) * np.cos(phi)
    data[:, 1] = (c + a * np.cos(theta)) * np.sin(phi)
    data[:, 2] = a * np.sin(theta)

    if noise:
        data += noise * np.random.randn(*data.shape)

    if ambient:
        embedded = embed(data, ambient)

    # return embedded, data
    return data


def swiss_roll(n=100, r=10, noise=None, ambient=None):
    """Swiss roll implementation

    Parameters
    ----------
    n : int
        Number of data points in shape.
    r : float
        Length of roll
    ambient : int, default=None
        Embed the swiss roll into a space with ambient dimension equal to `ambient`. The swiss roll is randomly rotated in this high dimensional space.

    References
    ----------
    Equations mimic [Swiss Roll and SNE by jlmelville](https://jlmelville.github.io/smallvis/swisssne.html)
    """

    phi = (np.random.random((n,)) * 3 + 1.5) * np.pi
    psi = np.random.random((n,)) * r

    data = np.zeros((n, 3))
    data[:, 0] = phi * np.cos(phi)
    data[:, 1] = phi * np.sin(phi)
    data[:, 2] = psi

    if noise:
        data += noise * np.random.randn(*data.shape)

    if ambient:
        embedded = embed(data, ambient)

    return embedded, data


def infty_sign(n=100, noise=None, ambient=None):
    """Construct a figure 8 or infinity sign with :code:`n` points and noise level with :code:`noise` standard deviation.

    Parameters
    ============

    n: int
        number of points in returned data set.
    noise: float
        standard deviation of normally distributed noise added to data.

    """

    t = np.linspace(0, 2 * np.pi, n + 1)[0:n]
    data = np.zeros((n, 2))
    data[:, 0] = np.cos(t)
    data[:, 1] = np.sin(2 * t)

    if noise:
        data += noise * np.random.randn(n, 2)

    if ambient:
        embedded = embed(data, ambient)

    return embedded, data

def circle_embedded(n=3000,ambient=3):
    r = 1.0
    angles = np.linspace(0, 2 * np.pi, num=1000)
    angles = np.random.choice(angles, size=n)
    data = np.stack([r * np.cos(angles), r * np.sin(angles)], axis=1)

    if ambient:
        embedded = embed(data, ambient)
    return embedded, angles