#pragma once

#include <vector>
#include <cstddef>
#include <span>

namespace hdos {

class RunningMean {
public:

    RunningMean(std::size_t n_features);

    void update(std::span<const double> x);
    void batch_update(std::span<const double> X);
    std::span<const double> mean() const noexcept;
    std::size_t n_features() const noexcept;
    std::size_t n_observations() const noexcept;
    void reset();

private:
    std::size_t n_features_;
    std::size_t n_observations_;

    std::vector<double> mean_;
};


class RunningVariance {
public:

    explicit RunningVariance(std::size_t n_features);

    void update(std::span<const double> x);

    // X is column-major: n_samples x n_features.
    void batch_update(std::span<const double> X);

    std::span<const double> mean() const noexcept;

    std::span<const double> variance() const noexcept;

    std::size_t n_features() const noexcept;

    std::size_t n_observations() const noexcept;

    void reset();

private:

    std::size_t n_features_;
    std::size_t n_observations_;

    std::vector<double> mean_;
    std::vector<double> M2_;
    std::vector<double> variance_;
};

}