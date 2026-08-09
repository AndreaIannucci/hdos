
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <limits>

#include "hdos/pca.hpp"
#include "hdos/online_moments.hpp"
#include "detail/one_sided_jacobi.hpp"
#include "detail/stable_norm.hpp"


namespace hdos {
        PCA::PCA(
        std::size_t n_features,
        std::size_t n_components
    )
        : n_features_(n_features),
        n_components_(n_components),
        mean_(n_features)
    {
        if (n_features == 0) {
            throw std::invalid_argument("Number of features cannot be 0");
        }

        if (n_components == 0) {
            throw std::invalid_argument("Number of components cannot be 0");
        }

        if (n_components > n_features) {
            throw std::invalid_argument(
                "Number of components cannot exceed the number of features"
            );
        }
    }


    void PCA::fit(std::span<const double> X)
    {
        if (X.size() % n_features_ != 0) {
            throw std::invalid_argument("Invalid matrix dimensions");
        }

        const std::size_t n_samples = X.size() / n_features_;

        if (n_samples < 2) {
            throw std::invalid_argument("PCA requires at least two samples");
        }

        if (n_components_ > std::min(n_samples, n_features_)) {
            throw std::invalid_argument("Too many principal components");
        }

        // Compute the mean from scratch.
        RunningMean new_mean(n_features_);
        new_mean.batch_update(X);

        std::span<const double> mean = new_mean.mean();

        // Center X. X is column-major: n_samples x n_features.
        std::vector<double> X_centered(X.begin(), X.end());

        for (std::size_t j = 0; j < n_features_; ++j) {
            for (std::size_t i = 0; i < n_samples; ++i) {
                X_centered[i + j * n_samples] -= mean[j];
            }
        }

        // X_centered = U Sigma V^T
        const auto svd =
            detail::SVD(X_centered, n_samples, n_features_);

        // V is n_features x n_features (or thin as appropriate).
        // Its first n_components columns are the principal directions.
        std::vector<double> new_components(
            svd.V.begin(),
            svd.V.begin() + n_features_ * n_components_
        );

        std::vector<double> new_singular_values(
            svd.singular_values.begin(),
            svd.singular_values.begin() + n_components_
        );

        std::vector<double> new_explained_variance(n_components_);

        for (std::size_t j = 0; j < n_components_; ++j) {
            const double sigma = new_singular_values[j];

            new_explained_variance[j] =
                sigma * sigma / static_cast<double>(n_samples - 1);
        }

        // Commit only after everything above succeeded.
        mean_ = std::move(new_mean);
        components_ = std::move(new_components);
        singular_values_ = std::move(new_singular_values);
        explained_variance_ = std::move(new_explained_variance);
    }





    void PCA::update(std::span<const double> x)
    {
        if (x.size() != n_features_) {
            throw std::invalid_argument("Invalid observation dimension");
        }

        const std::size_t n = mean_.n_observations();

        if (n == 0) {
        throw std::logic_error("The model has not been fitted");
        }

        std::span<const double> old_mean = mean_.mean();

        // Mean-corrected new direction:
        //
        // z = sqrt(n / (n + 1)) * (x - old_mean)
        std::vector<double> z(n_features_);

        const double scale = std::sqrt(
            static_cast<double>(n) /
            static_cast<double>(n + 1)
        );

        for (std::size_t i = 0; i < n_features_; ++i) {
            z[i] = scale * (x[i] - old_mean[i]);
        }

        // project onto the current PCA subspace
        std::vector<double> a(n_components_);
        for (std::size_t j = 0; j < n_components_; ++j) {
        
            double dot_prod = 0.0;
            for (std::size_t i = 0; i < n_features_; ++i) {
                dot_prod += components_[i + j * n_features_] * z[i];
            }

            a[j] = dot_prod; 
        }

       std::vector<double> r(z.begin(), z.end());

        for (std::size_t i = 0; i < n_features_; ++i) {
            for (std::size_t j = 0; j < n_components_; ++j) {
                r[i] -= components_[i + j * n_features_] * a[j];
            }
        }

        const double rho = detail::stable_norm(r);
        constexpr double tol_const = 10.0;
        const double tol = tol_const * std::numeric_limits<double>::epsilon()
                            * std::max(1.0, detail::stable_norm(z));
        
        std::vector<double> new_components(n_features_ * n_components_, 0.0);
        std::vector<double> new_singular_values(n_components_, 0.0);
        
        if (rho > tol && n_components_ < n_features_){
            std::vector<double> q(n_features_);

            for (std::size_t i = 0; i < n_features_; ++i) {
                q[i] = r[i] / rho;
            }
            std::vector<double> W(n_features_ * (1+ n_components_));
            std::copy(components_.begin(), components_.end(), W.begin());
            std::copy(q.begin(), q.end(), W.begin() + n_features_ * n_components_);
        
        std::vector<double> K((n_components_ + 1) * (n_components_ + 1), 0.0);
        for (std::size_t j = 0; j < n_components_; ++j) {
            // Diagonal Sigma.
            K[j + j * (n_components_ + 1)] = singular_values_[j];

            // Last row a^T.
            K[n_components_ + j * (n_components_ + 1)] = a[j];
        }

        // Bottom-right rho.
        K[n_components_ + n_components_ * (n_components_ + 1)] = rho;

        detail::SVDResult K_decomp = detail::SVD(K, n_components_+ 1, n_components_+1);

        // Compute only the first k columns of V_new = W * R.
        for (std::size_t j = 0; j < n_components_; ++j) {
            for (std::size_t l = 0; l < n_components_ + 1; ++l) {
                const double r_lj = K_decomp.V[l + j * (n_components_ +1)];

                for (std::size_t i = 0; i < n_features_; ++i) {
                    new_components[i + j * n_features_] +=
                        W[i + l * n_features_] * r_lj;
                }
            }
        }
        std::copy(K_decomp.singular_values.begin(),
          K_decomp.singular_values.begin() + n_components_,
          new_singular_values.begin());

    }
        else{
            std::vector<double> K((n_components_ + 1) * n_components_, 0.0);
            for (std::size_t j = 0; j < n_components_; ++j) {
                // Diagonal Sigma.
                K[j + j * (n_components_ + 1)] = singular_values_[j];

                // Last row a^T.
                K[n_components_ + j * (n_components_ + 1)] = a[j];
            }
            detail::SVDResult K_decomp = detail::SVD(K, n_components_ + 1, n_components_);
            
            for (std::size_t j = 0; j < n_components_; ++j) {
                for (std::size_t l = 0; l < n_components_; ++l) {
                    const double r_lj = K_decomp.V[l + j * n_components_];

                    for (std::size_t i = 0; i < n_features_; ++i) {
                        new_components[i + j * n_features_] +=
                            components_[i + l * n_features_] * r_lj;
                }
            }
        }
        std::copy(K_decomp.singular_values.begin(),
          K_decomp.singular_values.begin() + n_components_,
          new_singular_values.begin());
        }


        components_ = std::move(new_components);
        singular_values_ = std::move(new_singular_values);

        // Only update the mean after the PCA update has succeeded.
        mean_.update(x);

        // explained_variance[j] = sigma[j]^2 / n
        // because there are now n + 1 observations.
        for (std::size_t j = 0; j < n_components_; ++j) {
            explained_variance_[j] =
                singular_values_[j] * singular_values_[j]
                / static_cast<double>(n);
        }
    }


    void PCA::batch_update(std::span<const double> X)
    {
        if (X.size() % n_features_ != 0) {
            throw std::invalid_argument("Invalid batch dimensions");
        }

        if (mean_.n_observations() == 0) {
            throw std::logic_error("The model has not been fitted");
        }

        const std::size_t n_samples =
            X.size() / n_features_;

        std::vector<double> x(n_features_);

        for (std::size_t i = 0; i < n_samples; ++i) {

            // Extract observation i from column-major X.
            for (std::size_t j = 0; j < n_features_; ++j) {
                x[j] = X[i + j * n_samples];
            }

            update(x);
        }
    }


    std::span<const double> PCA::components() const noexcept{
        return components_;
    }

    std::span<const double> PCA::singular_values() const noexcept{
        return singular_values_;
    }

    std::span<const double> PCA::explained_variance() const noexcept{
        return explained_variance_;
    }

    std::span<const double> PCA::mean() const noexcept{
        return mean_.mean();
    }

    std::size_t PCA::n_features() const noexcept{
        return n_features_;
    }

    std::size_t PCA::n_components() const noexcept{
        return n_components_;
    }

    std::size_t PCA::n_observations() const noexcept{
        return mean_.n_observations();
    }

    void PCA::reset()
{
    mean_.reset();

    std::fill(
        components_.begin(),
        components_.end(),
        0.0);

    std::fill(
        singular_values_.begin(),
        singular_values_.end(),
        0.0);

    std::fill(
        explained_variance_.begin(),
        explained_variance_.end(),
        0.0);
}
}
