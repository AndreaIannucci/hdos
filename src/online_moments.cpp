#include <vector>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <algorithm>
#include <cmath>


#include "hdos/online_moments.hpp"

namespace hdos {

RunningMean::RunningMean(std::size_t n_features)
    : n_features_(n_features),
      n_observations_(0),
      mean_(n_features, 0.0)
{
    if (n_features == 0) {
        throw std::invalid_argument(
            "Number of features must be positive");
    }
}


void RunningMean::update(std::span<const double> x)
{
    if (x.size() != n_features_) {
        throw std::invalid_argument("Invalid observation dimension");
    }

    for (double value : x) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "The matrix must contain only finite entries");
        }
    }

    ++n_observations_;

    const double weight =
        1.0 / static_cast<double>(n_observations_);

    for (std::size_t k = 0; k < n_features_; ++k) {
        mean_[k] +=
            weight * (x[k] - mean_[k]);
    }
}


void RunningMean::batch_update(std::span<const double> X)
{
    if (X.size() % n_features_ != 0) {
        throw std::invalid_argument(
            "Invalid batch dimensions");
    }

    const std::size_t n_samples =
        X.size() / n_features_;

    if (n_samples == 0) {
        return;
    }

    for (double value : X) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "The matrix must contain only finite entries");
        }
    }

    const std::size_t new_n =
        n_observations_ + n_samples;

    const double weight =
        static_cast<double>(n_samples) /
        static_cast<double>(new_n);

    for (std::size_t j = 0; j < n_features_; ++j) {

        double batch_mean = 0.0;

        for (std::size_t i = 0; i < n_samples; ++i) {
            batch_mean += X[i + j * n_samples];
        }

        batch_mean /= static_cast<double>(n_samples);

        mean_[j] +=
            weight * (batch_mean - mean_[j]);
    }

    n_observations_ = new_n;
}


std::span<const double> RunningMean::mean() const noexcept
{
    return mean_;
}


std::size_t RunningMean::n_features() const noexcept
{
    return n_features_;
}


std::size_t RunningMean::n_observations() const noexcept
{
    return n_observations_;
}


void RunningMean::reset()
{
    n_observations_ = 0;
    std::fill(mean_.begin(), mean_.end(), 0.0);
}


RunningVariance::RunningVariance(std::size_t n_features)
    : n_features_(n_features),
      n_observations_(0),
      mean_(n_features, 0.0),
      M2_(n_features, 0.0),
      variance_(n_features, 0.0)
{
    if (n_features == 0) {
        throw std::invalid_argument(
            "Number of features must be positive");
    }
}


void RunningVariance::update(std::span<const double> x)
{
    if (x.size() != n_features_) {
        throw std::invalid_argument(
            "Invalid observation dimension");
    }

    for (double value : x) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "The matrix must contain only finite entries");
        }
    }

    ++n_observations_;

    const double n =
        static_cast<double>(n_observations_);

    for (std::size_t j = 0; j < n_features_; ++j) {

        const double delta =
            x[j] - mean_[j];

        mean_[j] += delta / n;

        const double delta2 =
            x[j] - mean_[j];

        M2_[j] += delta * delta2;
        }
    
    is_current = false;
    }


void RunningVariance::batch_update(
    std::span<const double> X)
{
    if (X.size() % n_features_ != 0) {
        throw std::invalid_argument(
            "Invalid batch dimensions");
    }

    for (double value : X) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "The matrix must contain only finite entries");
        }
    }

    const std::size_t n_samples =
        X.size() / n_features_;

    if (n_samples == 0) {
        return;
    }

    std::vector<double> batch_mean(
        n_features_, 0.0);

    std::vector<double> batch_M2(
        n_features_, 0.0);

    // Compute batch moments using Welford.
    for (std::size_t j = 0; j < n_features_; ++j) {

        double local_mean = 0.0;
        double local_M2 = 0.0;

        for (std::size_t i = 0; i < n_samples; ++i) {

            const double value =
                X[i + j * n_samples];

            const double count = static_cast<double>(i + 1);

            const double delta = value - local_mean;

            local_mean += delta / count;

            const double delta2 =
                value - local_mean;

            local_M2 += delta * delta2;
        }

        batch_mean[j] = local_mean;
        batch_M2[j] = local_M2;
    }

    const std::size_t old_n = n_observations_;
    const std::size_t new_n = old_n + n_samples;
    const double old_n_d = static_cast<double>(old_n);
    const double batch_n_d = static_cast<double>(n_samples);
    const double new_n_d = static_cast<double>(new_n);

    // Merge the existing moments with the batch moments.
    for (std::size_t j = 0; j < n_features_; ++j) {

        const double delta =
            batch_mean[j] - mean_[j];

        M2_[j] +=
            batch_M2[j]
            + delta * delta
                * old_n_d
                * batch_n_d
                / new_n_d;

        mean_[j] +=
            delta * batch_n_d / new_n_d;
    }

    n_observations_ = new_n;
    is_current = false;
}

std::span<const double>
RunningVariance::mean() const noexcept
{
    return mean_;
}


std::span<const double>
RunningVariance::variance() noexcept
{   
    if (n_observations_ <= 1){
        return variance_;
    }

    if (is_current){
        return variance_;
    }

    for (std::size_t j = 0; j < n_features_; ++j) {
            variance_[j] =
                M2_[j] /
                static_cast<double>(
                    n_observations_ - 1);
        }
    
    return variance_;
    }


std::size_t
RunningVariance::n_features() const noexcept
{
    return n_features_;
}


std::size_t
RunningVariance::n_observations() const noexcept
{
    return n_observations_;
}


void RunningVariance::reset()
{
    n_observations_ = 0;
    is_current = false;

    std::fill(
        mean_.begin(),
        mean_.end(),
        0.0);

    std::fill(
        M2_.begin(),
        M2_.end(),
        0.0);
    
    std::fill(
        variance_.begin(),
        variance_.end(),
        0.0);
}

RunningCovariance::RunningCovariance(std::size_t n_features)
    : n_features_(n_features),
      n_observations_(0),
      mean_(n_features, 0.0),
      M2_(n_features*n_features, 0.0),
      covariance_(n_features * n_features, 0.0)
{
    if (n_features == 0) {
        throw std::invalid_argument(
            "Number of features must be positive");
    }
}


void RunningCovariance::update(std::span<const double> x)
{
    if (x.size() != n_features_) {
        throw std::invalid_argument(
            "Invalid observation dimension");
    }

    for (double value : x) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "The matrix must contain only finite entries");
        }
    }

    
    const double denominator =   1 / static_cast<double>(n_observations_ + 1);
    std::vector<double> delta(n_features_);
    
    for (std::size_t j = 0; j < n_features_; ++j) {
        delta[j] = x[j] - mean_[j];
        mean_[j] += denominator * delta[j];
        for (std::size_t i = 0; i <= j ; ++i ){
            M2_[i + j*n_features_] += (1 - denominator) * delta[i] * delta[j];
            }
        }
    
    ++n_observations_;
    is_current = false;
    }


void RunningCovariance::batch_update(
    std::span<const double> X)
{
    if (X.size() % n_features_ != 0) {
        throw std::invalid_argument(
            "Invalid batch dimensions");
    }

    for (double value : X) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "The matrix must contain only finite entries");
        }
    }

    const std::size_t n_samples = X.size() / n_features_;

    if (n_samples == 0) {
        return;
    }

    std::vector<double> batch_mean(n_features_, 0.0);
    std::vector<double> batch_M2(
        n_features_ * n_features_, 0.0);
    std::vector<double> delta(n_features_);

    // Compute the moments of the batch.
    for (std::size_t n = 0; n < n_samples; ++n) {
        const std::size_t batch_count = n + 1;
        const double inverse_count =
            1.0 / static_cast<double>(batch_count);
        const double scale = 1.0 - inverse_count;

        for (std::size_t j = 0; j < n_features_; ++j) {
            // Column-major n_samples x n_features matrix.
            const double value = X[n + j * n_samples];
            delta[j] = value - batch_mean[j];
        }

        for (std::size_t j = 0; j < n_features_; ++j) {
            batch_mean[j] += inverse_count * delta[j];
        }

        for (std::size_t j = 0; j < n_features_; ++j) {
            for (std::size_t i = 0; i <= j; ++i) {
                batch_M2[i + j * n_features_] +=
                    scale * delta[i] * delta[j];
            }
        }
    }

    const std::size_t old_n = n_observations_;
    const std::size_t new_n = old_n + n_samples;

    const double old_n_d = static_cast<double>(old_n);
    const double batch_n_d = static_cast<double>(n_samples);
    const double new_n_d = static_cast<double>(new_n);

    // This must be computed before changing mean_.
    std::vector<double> mean_delta(n_features_);

    for (std::size_t j = 0; j < n_features_; ++j) {
        mean_delta[j] = batch_mean[j] - mean_[j];
    }

    const double merge_scale =
        old_n_d * batch_n_d / new_n_d;

    for (std::size_t j = 0; j < n_features_; ++j) {
        for (std::size_t i = 0; i <= j; ++i) {
            M2_[i + j * n_features_] +=
                batch_M2[i + j * n_features_]
                + merge_scale
                    * mean_delta[i]
                    * mean_delta[j];
        }
    }

    const double batch_weight = batch_n_d / new_n_d;

    for (std::size_t j = 0; j < n_features_; ++j) {
        mean_[j] += batch_weight * mean_delta[j];
    }

    n_observations_ = new_n;
    is_current = false;
}


std::span<const double>
RunningCovariance::mean() const noexcept
{
    return mean_;
}


std::span<const double>
RunningCovariance::covariance() noexcept
{   
    if (n_observations_ <=1){
        return covariance_;
    }

    if (is_current){
        return covariance_;
    }

    const double denominator =
        static_cast<double>(n_observations_ - 1);

    for (std::size_t j = 0; j < n_features_; ++j) {
        for (std::size_t i = 0; i <= j; ++i) {
            const double value =
                M2_[i + j * n_features_] / denominator;
            covariance_[i + j * n_features_] = value;
            covariance_[j + i * n_features_] = value;
        }
    }
    is_current = true;
    return covariance_;
}

std::size_t
RunningCovariance::n_features() const noexcept
{
    return n_features_;
}


std::size_t
RunningCovariance::n_observations() const noexcept
{
    return n_observations_;
}


void RunningCovariance::reset()
{
    n_observations_ = 0;

    std::fill(
        mean_.begin(),
        mean_.end(),
        0.0);

    std::fill(
        M2_.begin(),
        M2_.end(),
        0.0);
    
    std::fill(
        covariance_.begin(),
        covariance_.end(),
        0.0);
    
    is_current = false;
}

}