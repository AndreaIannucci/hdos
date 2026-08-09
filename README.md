# HDOS

HDOS is a small, dependency-free C++20 library for batch and online
numerical statistics. It provides focused implementations of common
estimators that can be fitted from a complete dataset and, where supported,
updated as new observations arrive.

The project exists to make the numerical behaviour of these algorithms
explicit while keeping the public API and build system compact. HDOS is not
intended to replace a full linear-algebra or machine-learning framework.

> **Project status:** HDOS is an early-stage project. Version `0.1.0` is the
> first public release and its API may evolve before `1.0`.

## Features

| Component | Capabilities |
| --- | --- |
| `LinearRegression` | Batch fitting, rank-one and batch updates, optional intercept, ridge regularisation, Cholesky and SVD solvers, prediction, and numerically stable training R-squared |
| `RunningMean` | Numerically stable element-wise means for vector observations, with single-observation and batch updates |
| `RunningVariance` | Numerically stable element-wise means and unbiased sample variances, with single-observation and batch updates |
| `PCA` | Centred batch PCA, fixed-rank incremental updates, batch updates, singular values, and explained variances |

HDOS also contains the Cholesky, QR, Householder, Jacobi rotation, and
one-sided Jacobi SVD routines used by these algorithms. They are implementation
details rather than part of the public API.

## Requirements

- A C++20 compiler.
- CMake 3.20 or newer.
- Git and an internet connection when configuring the tests for the first
  time, because CMake fetches GoogleTest 1.17.0.

The library itself has no third-party runtime dependencies.

## Quick start

All public facilities are available through the umbrella header:

```cpp
#include <hdos/hdos.hpp>

#include <array>
#include <iostream>
#include <vector>

int main()
{
    // Five observations and two features, stored column-major.
    const std::vector<double> X{
        0.0, 1.0, 2.0, 3.0, 4.0,   // first feature
        1.0, 0.0, 2.0, -1.0, 3.0   // second feature
    };

    // y = 1 + 2*x1 - 3*x2
    const std::vector<double> y{
        -2.0, 3.0, -1.0, 10.0, 0.0
    };

    hdos::LinearRegressionOptions options;
    options.solver = hdos::LinearRegressionSolver::svd;

    hdos::LinearRegression model(2, options);
    model.fit(X, y);

    const auto& coefficients = model.coefficients();
    const std::array<double, 2> observation{5.0, 2.0};

    std::cout << "intercept = " << model.intercept() << '\n'
              << "coefficients = [" << coefficients[0] << ", "
              << coefficients[1] << "]\n"
              << "R-squared = " << model.r_squared() << '\n'
              << "prediction = " << model.predict(observation) << '\n';
}
```

This example recovers an intercept of `1`, coefficients `[2, -3]`, an
R-squared of `1`, and a prediction of `5`.

## Build and test

Configure the project with tests and examples enabled:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DHDOS_BUILD_TESTS=ON -DHDOS_BUILD_EXAMPLES=ON
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

`CMAKE_BUILD_TYPE` selects the configuration for single-configuration
generators such as Makefiles and Ninja. The `--config Release` and `-C Release`
arguments select it for multi-configuration generators such as Visual Studio.

The available HDOS options are:

| Option | Default | Purpose |
| --- | --- | --- |
| `HDOS_BUILD_TESTS` | `ON` for a top-level build; otherwise `OFF` | Build the GoogleTest test suite |
| `HDOS_BUILD_EXAMPLES` | `OFF` | Build the public examples |
| `HDOS_ENABLE_WARNINGS` | `ON` for a top-level build; otherwise `OFF` | Enable strict compiler warnings for HDOS |

The example targets are:

| Target | Source |
| --- | --- |
| `hdos_linear_regression_example` | `examples/linear_regression.cpp` |
| `hdos_online_moments_example` | `examples/online_moments.cpp` |
| `hdos_pca_example` | `examples/pca.cpp` |

## Installation

Configure and install HDOS into a clean prefix:

```sh
cmake -S . -B build-install -DCMAKE_BUILD_TYPE=Release -DHDOS_BUILD_TESTS=OFF -DHDOS_BUILD_EXAMPLES=OFF
cmake --build build-install --config Release
cmake --install build-install --config Release --prefix path/to/hdos-install
```

This installs the public headers, the HDOS library, and the CMake package files.

### Using `find_package(hdos)`

After installation, a consumer can import the namespaced target:

```cmake
cmake_minimum_required(VERSION 3.20)

project(hdos_consumer LANGUAGES CXX)

find_package(hdos 0.1 CONFIG REQUIRED)

add_executable(hdos_consumer main.cpp)
target_link_libraries(hdos_consumer PRIVATE hdos::hdos)
```

If HDOS was installed outside a standard system prefix, provide its location
when configuring the consumer:

```sh
cmake -S . -B build -DCMAKE_PREFIX_PATH=path/to/hdos-install
cmake --build build --config Release
```

### Using `add_subdirectory`

HDOS can also be included directly in another CMake project:

```cmake
add_subdirectory(path/to/hdos)
target_link_libraries(my_target PRIVATE hdos::hdos)
```

When HDOS is a subproject, its tests and strict warning flags are disabled by
default so that they do not alter the parent project's normal build.

## Additional examples

### Online mean and sample variance

```cpp
#include <hdos/hdos.hpp>

#include <array>

hdos::RunningVariance moments(2);

moments.update(std::array<double, 2>{1.0, 10.0});
moments.update(std::array<double, 2>{2.0, 20.0});
moments.update(std::array<double, 2>{3.0, 30.0});

const auto mean = moments.mean();
const auto variance = moments.variance();
```

`mean` is `[2, 20]` and `variance` is `[1, 100]`.

### Batch and incremental PCA

```cpp
#include <hdos/hdos.hpp>

#include <array>
#include <vector>

// Six observations and two features, stored column-major.
const std::vector<double> X{
    -3.0, -2.0, -1.0, 1.0, 2.0, 3.0,
    -3.1, -1.9, -1.1, 0.9, 2.1, 3.0
};

hdos::PCA pca(2, 1);
pca.fit(X);

const auto component = pca.components();
const auto explained_variance = pca.explained_variance();

pca.update(std::array<double, 2>{4.0, 4.1});
```

`fit()` replaces the current state. `PCA::update()` and
`PCA::batch_update()` require an initial fit, whereas the online-moment and
linear-regression accumulators can be updated from an empty state. The initial
PCA fit requires at least two observations and
`n_components <= min(n_samples, n_features)`.

## Matrix layout

Every public batch matrix is a flat, column-major array. If `X` has
`n_samples` rows and `n_features` columns, then

```text
X(i, j) = X[i + j * n_samples].
```

For example, the conceptual matrix

```text
[ 1  10 ]
[ 2  20 ]
[ 3  30 ]
```

is passed as

```cpp
const std::vector<double> X{
    1.0, 2.0, 3.0,
    10.0, 20.0, 30.0
};
```

Single observations remain ordinary feature vectors. Public APIs accept
contiguous data through `std::span<const double>`, so `std::vector<double>` and
`std::array<double, N>` can be passed directly.

## Numerical conventions

### Sample variance

`RunningVariance` reports the unbiased sample variance independently for each
feature:

```text
variance = M2 / (n - 1).
```

The reported variance is zero until at least two observations have been
accumulated.

### Ridge regularisation and the intercept

For `l2_penalty = lambda`, linear regression minimises

```text
sum_i (y_i - intercept - x_i^T beta)^2 + lambda * ||beta||^2.
```

Only the feature coefficients are penalised. The intercept is never
penalised. The penalty must be finite and non-negative.

### Cholesky and SVD regression backends

The default Cholesky backend forms the regularised normal equations and
maintains their factorisation under rank-one updates. It has lower storage and
computational overhead, but an unregularised rank-deficient system cannot be
solved and forming normal equations is less suitable for ill-conditioned
problems.

The SVD backend maintains a compact QR representation under online updates and
solves the resulting system using a one-sided Jacobi SVD. It supports
rank-deficient and underdetermined problems and computes a minimum-norm
least-squares solution after numerically zero singular directions are removed.

Select a backend when constructing the model:

```cpp
hdos::LinearRegressionOptions options;
options.solver = hdos::LinearRegressionSolver::cholesky; // default
// options.solver = hdos::LinearRegressionSolver::svd;
```

### SVD cutoff

With the SVD backend, a singular value is treated as zero when

```text
sigma <= svd_rcond * sigma_max.
```

The default `svd_rcond = 0` selects the automatic relative cutoff

```text
machine_epsilon * max(n_observations, n_parameters),
```

where `n_parameters` includes the intercept when one is fitted. A positive
`svd_rcond` uses the supplied relative cutoff directly.

### PCA output layout

`PCA::components()` returns a column-major `n_features` by `n_components`
matrix. Component `j`, feature `i` is therefore stored at

```text
components[i + j * n_features].
```

Each column is one principal direction. Components and singular values are
ordered from largest to smallest singular value, and explained variance is
computed as

```text
singular_value[j]^2 / (n_observations - 1).
```

As with every SVD-based PCA implementation, the sign of each principal
direction is arbitrary.

### R-squared

Training R-squared is

```text
R^2 = 1 - RSS / TSS,
```

with `TSS` centred around the response mean even when `fit_intercept` is
`false`. The ridge penalty is not included in `RSS`, and R-squared may be
negative.

For a constant response, HDOS returns `1` for a numerically perfect fit and
`0` otherwise. Centred sufficient statistics are maintained to reduce
catastrophic cancellation when features or responses have large offsets.

## Current limitations

- The public API currently supports dense `double` data only.
- Matrices must be supplied as contiguous column-major buffers.
- Missing values, NaNs, and infinities are rejected rather than imputed.
- The implementations are single-threaded and do not explicitly use SIMD, a
  BLAS, a GPU, or another external acceleration backend.
- No performance claims are made yet; comparative benchmarks are planned
  after `v0.1.0`.
- Incremental PCA retains only the requested number of components. When
  `n_components < n_features`, discarded directions are not recoverable, so
  sequential updates need not equal a complete batch refit.
- PCA does not yet provide `transform()` or `inverse_transform()` helpers.
- Regression currently has no observation weights, multi-output response,
  inference statistics, or model serialisation.
- HDOS does not yet promise a stable ABI or shared-library compatibility.
- The API is not designed for concurrent mutation of the same estimator.

## License

HDOS is available under the [MIT License](LICENSE).
