import numpy as np
import quflow as qf
import pytest
import quflow.laplacian.sparse as qusparse
import quflow.laplacian.direct as qudirect


def get_random_omega_real(N=5):
    return np.random.randn(N**2)


def get_random_mat(N=5):
    W = np.random.randn(N, N) + 1j*np.random.randn(N, N)
    W -= W.conj().T
    W -= np.eye(N)*np.trace(W)/N
    return W


def get_smooth_mat(N=5):
    omegar = np.array([-1.94289029e-16,  9.34389008e-15,  1.68615122e-15, -1.59309186e-15,
                        4.18294664e-01,  5.04879629e-01,  2.05858273e-01, -6.60776951e-01,
                        6.78180820e-01, -1.01935097e-01,  2.47658404e-01,  4.13107476e-01,
                        7.95673082e-02, -1.03883724e+00, -4.76852974e-01, -6.03026919e-01,
                       -2.68486178e-01, -1.47245426e-01, -5.41379946e-01, -7.27999391e-01,
                       -3.75207725e-02, -1.44058680e+00, -1.16117652e+00,  5.68201184e-01,
                        5.08163712e-02,  1.11902515e-01,  8.86920379e-01,  4.99418111e-01,
                       -1.48839722e-01, -3.61772325e-02, -2.97263898e-01,  4.94654879e-01,
                       -6.32478266e-01,  5.63515676e-02,  1.91048517e-01,  7.25260028e-01,
                        3.75714435e-01,  5.82612449e-01, -6.40261374e-01, -1.32589452e+00,
                        5.23797607e-01, -3.55600726e-01,  4.57633735e-01, -1.30008266e-01,
                        4.92011373e-01,  6.49726166e-01, -4.12497032e-01,  1.66305782e+00,
                       -7.84263691e-01, -7.50191831e-02, -2.82536547e-01,  9.77878954e-02,
                       -8.76951809e-01,  9.06721330e-02, -3.66159831e-02,  1.29758889e+00,
                       -2.23747252e-01, -3.43399134e-01,  1.09170204e-01,  1.40505556e-02,
                        1.14392871e+00,  1.39062889e-01,  8.28874065e-04, -3.40094316e-01,
                        1.16377075e+00,  1.60273703e-01,  1.06793866e+00,  1.36358591e-01,
                       -1.21668436e-01, -9.24004606e-02, -6.96029709e-01,  7.63455927e-01,
                        5.38242429e-01,  1.15031413e+00, -1.63548693e-01, -5.89147388e-02,
                       -3.23008105e-01, -6.97439871e-01, -6.12100120e-01, -1.68078253e+00,
                       -4.98627578e-01,  1.01844798e+00,  6.28096588e-02,  9.68062582e-01,
                       -1.22622918e-01,  1.14519931e+00,  5.98794888e-01, -1.02654169e+00,
                       -1.15832511e+00, -4.94097705e-02,  5.05476698e-01, -2.16757310e-01,
                        2.64943343e-01, -3.53073342e-01, -2.51501415e-01,  2.50421740e-01,
                        3.27569622e-01,  5.93631160e-01,  1.07535561e+00, -1.77619828e-01,
                       -5.27919934e-01, -4.94415346e-01, -1.80494298e-01, -4.55706656e-01,
                        1.09265186e-01, -7.30372640e-01, -3.74684646e-01, -4.07975658e-01,
                        8.52516531e-01, -6.31882404e-01,  1.30327119e-01,  6.88272649e-01,
                       -4.39038315e-01,  1.62940079e-01,  1.95443276e-01, -2.21053706e-01,
                        5.26453285e-01,  6.00394049e-01,  3.36134319e-01, -7.46320605e-01,
                       -9.77896464e-01])

    # Convert to matrix representation
    return qf.shr2mat(omegar, N=N)


@pytest.mark.parametrize("N", [33, 65, 128, 513])
def test_laplace(N):

    P = get_random_mat(N)

    W_sparse = qusparse.laplace(P)
    W_direct = qudirect.laplace(P)

    assert np.abs(W_sparse-W_direct).max() < 1e-10


@pytest.mark.parametrize("N", [33, 65, 128, 513])
def test_laplace_tridiagonal(N):

    P = get_random_mat(N)

    W_tri = qf.laplacian.tridiagonal.laplace(P)
    W_direct = qudirect.laplace(P)

    np.testing.assert_allclose(W_tri, W_direct)
    # assert np.abs(W_tri-W_direct).max() < 1e-10


@pytest.mark.parametrize("N", [33, 65, 128, 513])
def test_solve_poisson(N):

    W = get_random_mat(N)

    P_sparse = qusparse.solve_poisson(W)
    P_direct = qudirect.solve_poisson(W)

    assert np.abs(P_sparse-P_direct).max() < 1e-10


@pytest.mark.parametrize("N", [33, 65, 128, 513])
def test_solve_poisson_tridiagonal(N):

    W = get_random_mat(N)

    P_tri = qf.laplacian.tridiagonal.solve_poisson(W)
    P_direct = qudirect.solve_poisson(W)

    # P_tri -= np.eye(W.shape[0])*np.trace(P_tri)/W.shape[0]
    # diag_P = qf.mat2diagh(P_tri)
    # diag_P[0, :] -= diag_P[0, :].sum()/W.shape[0]
    # P_tri = qf.diagh2mat(diag_P)

    print(np.trace(W))
    print(qf.mat2diagh(W)[0, :].sum())
    print(np.trace(P_direct))
    print(np.trace(P_tri))
    np.testing.assert_allclose(P_tri, P_direct)
    # assert np.abs(P_tri-P_direct).max() < 1e-10


@pytest.mark.parametrize("N", [4, 33, 64])
def test_solve_poisson_nonskewh(N):

    np.random.seed(42)  # For reproducability
    omegaW = np.random.randn(16) + 1.0j*np.random.randn(16)
    ells = qf.ind2elm(np.arange(16))[0][1:]
    omegaP = omegaW.copy()
    omegaP[1:] /= -ells*(ells+1)
    omegaW[0] = 0.0
    omegaP[0] = 0.0

    W = qf.shc2mat(omegaW, N=N)
    Pexact = qf.shc2mat(omegaP, N=N)

    current_solve_direct_ = qf.laplacian.direct.solve_direct_

    qf.laplacian.direct.solve_direct_ = qf.laplacian.direct.solve_direct_nonskewh_
    P = qf.laplacian.direct.solve_poisson(W)

    qf.laplacian.direct.solve_direct_ = current_solve_direct_

    np.testing.assert_allclose(P, Pexact)


@pytest.mark.parametrize("N", [4, 33, 64])
def test_solve_poisson_skewh(N):

    np.random.seed(42)  # For reproducability
    omegaW = np.random.randn(16).copy()
    ells = qf.ind2elm(np.arange(16))[0][1:]
    omegaP = omegaW.copy()
    omegaP[1:] /= -ells*(ells+1)
    omegaW[0] = 0.0
    omegaP[0] = 0.0

    W = qf.shr2mat(omegaW, N=N)
    Pexact = qf.shr2mat(omegaP, N=N)

    current_solve_direct_ = qf.laplacian.direct.solve_direct_

    qf.laplacian.direct.solve_direct_ = qf.laplacian.direct.solve_direct_skewh_
    P = qf.laplacian.direct.solve_poisson(W)

    qf.laplacian.direct.solve_direct_ = current_solve_direct_

    np.testing.assert_allclose(P, Pexact)


@pytest.mark.parametrize("N", [4, 33, 64])
def test_laplace_skewh(N):

    np.random.seed(22)  # For reproducability
    omegaP = np.random.randn(16).copy()
    ells = qf.ind2elm(np.arange(16))[0][1:]
    omegaW = omegaP.copy()
    omegaW[1:] *= -ells*(ells+1)
    omegaW[0] = 0.0
    omegaP[0] = 0.0

    Wexact = qf.shr2mat(omegaW, N=N)
    P = qf.shr2mat(omegaP, N=N)

    current_dot_direct_ = qf.laplacian.direct.dot_direct_

    qf.laplacian.direct.dot_direct_ = qf.laplacian.direct.dot_direct_skewh_
    W = qf.laplacian.direct.laplace(P)

    qf.laplacian.direct.dot_direct_ = current_dot_direct_

    np.testing.assert_allclose(W, Wexact)


@pytest.mark.parametrize("N", [4, 33, 64])
def test_laplace_nonskewh(N):

    np.random.seed(22)  # For reproducability
    omegaP = np.random.randn(16) + 1.0j*np.random.randn(16)
    ells = qf.ind2elm(np.arange(16))[0][1:]
    omegaW = omegaP.copy()
    omegaW[1:] *= -ells*(ells+1)
    omegaW[0] = 0.0
    omegaP[0] = 0.0

    Wexact = qf.shc2mat(omegaW, N=N)
    P = qf.shc2mat(omegaP, N=N)

    current_dot_direct_ = qf.laplacian.direct.dot_direct_

    qf.laplacian.direct.dot_direct_ = qf.laplacian.direct.dot_direct_nonskewh_
    W = qf.laplacian.direct.laplace(P)

    qf.laplacian.direct.dot_direct_ = current_dot_direct_

    np.testing.assert_allclose(W, Wexact)


@pytest.mark.parametrize("N", [33, 65, 128, 513])
def test_solve_poisson_sparse(N):

    W = get_random_mat(N)

    P = qusparse.solve_poisson(W)
    W2 = qusparse.laplace(P)

    assert np.abs(W-W2).max() < 1e-10


@pytest.mark.parametrize("N", [33, 65, 128, 513])
def test_solve_poisson_direct(N):

    W = get_random_mat(N)

    P = qudirect.solve_poisson(W)
    W2 = qudirect.laplace(P)

    assert np.abs(W-W2).max() < 1e-10


@pytest.mark.parametrize("N", [9, 32])
def test_solve_heat_vs_viscdamp_direct(N):
    W0 = get_smooth_mat(N)

    Wheat = W0.copy()
    Wviscdamp = W0.copy()
    for k in range(100):
        Wheat = qudirect.solve_heat(1e-2*0.1, Wheat)
        Wviscdamp = qudirect.solve_viscdamp(0.1, Wviscdamp, nu=1e-2, alpha=0, theta=1)

    assert np.abs(Wheat-Wviscdamp).max() < 1e-10


def test_solve_viscdamp_direct():
    N = 9
    W0 = get_smooth_mat(N)

    omegatref = np.array([ 0.00000000e+00,  2.33443923e-17,  2.20906193e-18,  5.28986229e-18,
                        6.18700651e-04,  7.46768683e-04,  3.04485470e-04, -9.77356789e-04,
                        1.00309889e-03, -8.40420883e-05,  2.04186096e-04,  3.40593338e-04,
                        6.56005923e-05, -8.56486663e-04, -3.93149376e-04, -4.97175586e-04,
                       -1.01744720e-04, -5.57996866e-05, -2.05159726e-04, -2.75880474e-04,
                       -1.42187598e-05, -5.45920469e-04, -4.40035984e-04,  2.15323823e-04,
                        1.92572202e-05,  1.60989452e-05,  1.27597513e-04,  7.18491876e-05,
                       -2.14129461e-05, -5.20466661e-06, -4.27661094e-05,  7.11639214e-05,
                       -9.09919937e-05,  8.10706353e-06,  2.74853483e-05,  1.04340116e-04,
                        1.69822495e-05,  2.63340160e-05, -2.89397408e-05, -5.99302805e-05,
                        2.36755919e-05, -1.60731121e-05,  2.06849924e-05, -5.87635874e-06,
                        2.22388577e-05,  2.93675483e-05, -1.86448186e-05,  7.51700231e-05,
                       -3.54486291e-05, -8.83690023e-07, -3.32814511e-06,  1.15189454e-06,
                       -1.03300720e-05,  1.06807427e-06, -4.31318732e-07,  1.52849752e-05,
                       -2.63563539e-06, -4.04507722e-06,  1.28597268e-06,  1.65508811e-07,
                        1.34749320e-05,  1.63809418e-06,  9.76373924e-09, -4.00614805e-06,
                        2.97107373e-06,  4.09174220e-07,  2.72641712e-06,  3.48119617e-07,
                       -3.10616068e-07, -2.35895756e-07, -1.77694411e-06,  1.94908134e-06,
                        1.37411766e-06,  2.93671936e-06, -4.17535176e-07, -1.50407658e-07,
                       -8.24630534e-07, -1.78054422e-06, -1.56267425e-06, -4.29099015e-06,
                       -1.27298207e-06])

    Wt = W0.copy()
    for k in range(100):
        Wt = qudirect.solve_viscdamp(0.1, Wt, nu=1e-2, alpha=0.6, theta=0.7)

    assert np.abs(qf.mat2shr(Wt)-omegatref).max() < 1e-10
