cimport cython
from libc.stdlib cimport malloc, free
from libc.math cimport sqrt, isnan, floor

cdef:
    double COEFF_LOG2E = 1.442695040888963
    double COEFF_P4_A = -3.70239070165e-06
    double COEFF_P4_B = 0.307033839653000
    double COEFF_P4_C = -0.241638340487000
    double COEFF_P4_D = -0.051690438287400
    double COEFF_P4_E = -0.013697656096900
    double * POW2_INTS = <double *> malloc(63 * sizeof(double))
    Py_ssize_t i

with cython.boundscheck(False), cython.wraparound(False), cython.cdivision(True):
    for i in range(31):
        POW2_INTS[i] = 1. / <double> (1 << (31 - i))
    for i in range(31, 63):
        POW2_INTS[i] = <double> (1 << (i - 31))


@cython.wraparound(False)
@cython.boundscheck(False)
cdef double _expFast(double x):
    """
    Fast approximation of exp(x) using a 4th degree polynomial.

    References:
        Malossi ACI, Ineichen Y, Bekas C, Curioni A. Fast Exponential
        Computation on SIMD Architectures. HiPEAC-1st Workshop On
        Approximate Computing (WAPCO)At: Amsterdam (NL). 2015.

    """
    cdef:
        double xf, kf, lx
        ssize_t xi

    x *= COEFF_LOG2E
    lx = floor(x)
    xi = <ssize_t> (lx + 31.)
    xf = x - lx
    kf = (COEFF_P4_A + xf * (COEFF_P4_B + xf * (COEFF_P4_C + xf *
          (COEFF_P4_D + xf * COEFF_P4_E))))

    return POW2_INTS[xi] * (1. + xf - kf)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def logistic_regression(float[::1] x, int[::1] y, double a0, double b0):
    """
    Logistic regression using iterative re-weighted least squares. This
    is only for univariate case:
        p(y=0) = 1 / (1 + exp(a + b * x))

    To reduce the bias in MLE due to separation of samples, Firth's
    procedure is applied.

    Args:
        x: x
        y: y
        a0: initial guess of the slop, defaulted to 0.
        b0: initial guess of the intercept, defaulted to 0.

    Returns:
        a and b

    References:
    [1] Firth D. Bias Reduction of Maximum Likelihood Estimates.
        Biometrika. 1993, 80(1), 27-38.
    [2] Heinze G., Schemper M. A solution to the problem of separation
        in logistic regression. Statist Med. 2002, 21, 2409-2419.

    """

    cdef:
        Py_ssize_t n = x.shape[0]
        Py_ssize_t i
        int max_iter = 100
        int it = 0
        double * x_sq = <double *> malloc(n * sizeof(double))
        double a = a0
        double b = b0
        double tol = 0.000001
        double d = 1.
        double n1 = 0.
        double sx1 = 0.
        double t, q, w, p, ta, tb, aa, ab, ac, ad, da, db, f
        double wx1, wx2, cr1, cr2, cr3, cr4

    for i in range(n):
        if y[i] == 1:
            n1 += 1.
            sx1 += x[i]
        x_sq[i] = x[i] * x[i]

    while it < max_iter and d > tol:
        ta, tb = 0., 0.
        aa, ab, ad = 0., 0., 0.
        cr1, cr2, cr3, cr4 = 0., 0., 0., 0.
        for i in range(n):
            q = - a - b * x[i]
            if q <= 25.:
                if q <= -25.:
                    ta += 1.
                    tb += x[i]
                else:
                    p = 1. / (1. + _expFast(q))
                    w = p * (1. - p)
                    wx1 = w * x[i]
                    wx2 = w * x_sq[i]
                    ta += p
                    tb += p * x[i]
                    aa += w
                    ab += wx1
                    ad += wx2
                    # for bias reduction
                    f = 0.5 - p
                    cr1 += f * x[i] * wx2
                    cr2 += f * wx2
                    cr3 += f * wx1
                    cr4 += f * w
        ac = ab
        t = aa * ad - ab * ac
        ta = n1 - ta + (aa * cr2 - 2 * ac * cr3 + ad * cr4) / t
        tb = sx1 - tb + (aa * cr1 - 2 * ac * cr2 + ad * cr3) / t

        da = (ad * ta - ab * tb) / t
        db = (aa * tb - ac * ta) / t

        a += da
        b += db

        d = sqrt(da * da + db * db) / sqrt(a * a + b * b)

        if isnan(a):
            a, b, d = 1., 1., 1.
            it = 0

        it += 1

    free(x_sq)

    return a, b
