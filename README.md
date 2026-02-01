# Univariate Logistic Regression
`Cython` codes for univariate logistic regression. The slope $b$ and  intercept $a$ of the log odd function:

$$ ln\Big({p(y=0| \mathbf{x} ) \over p(y=1| \mathbf{x} )}\Big) = a + b \mathbf{x} $$

are estimated using iterative reweighted least squares (IRLS). To reduce the bias generated during the maximum likelihood estimation (MLE) of the parameters for separation data, Firth's approach<sup>[1, 2]</sup> is used.

Since only a single variable is used this logistic regression, the implementation is largely simplified as the inverse of Hessian matrix (size of 2 by 2) can be solved numerically.
## Usage
```a, b = logistic_regression(x, y, a0, b0)```

* `x`: A vector **$\mathbf{x}$** of length `n`.
* `y`: A vector **$\mathbf{y}$** of length `n`, $y_i\in {\\{0, 1}\\} \text{ for } i = 1, 2, ..., n$.
* `a0`: Initial guess of the slop $a$, could be `0` or `1`.
* `b0`: Initial guess of the intercept $b$, could be `0` or `1`.

The MLE of $a$ and $b$ would be returned in `a` and `b`.

## Note
Since this is implemented in Cython, it requires a setup procedure to compile the `.pyx` to make it importable as normal Python function. The setup procedure can be easily found online.

## References
[1] Firth D. Bias Reduction of Maximum Likelihood Estimates. Biometrika. 1993, 80(1), 27-38.

[2] Heinze G., Schemper M. A solution to the problem of separation in logistic regression. Statist Med. 2002, 21, 2409-2419.
