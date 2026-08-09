#pragma once

#include <vector>
#include <span>
#include <cstddef>

#include "online_moments.hpp"

namespace hdos {

class PCA {
public:

    PCA(
        std::size_t n_features,
        std::size_t n_components);

    // Fit from scratch.
    void fit(std::span<const double> X);

    // Incorporate one new observation.
    void update(std::span<const double> x);

    // Incorporate a batch of new observations.
    void batch_update(std::span<const double> X);

    std::span<const double> components() const noexcept;

    std::span<const double> singular_values() const noexcept;

    std::span<const double> explained_variance() const noexcept;

    std::span<const double> mean() const noexcept;

    std::size_t n_features() const noexcept;

    std::size_t n_components() const noexcept;

    std::size_t n_observations() const noexcept;

    void reset();

private:

    std::size_t n_features_;
    std::size_t n_components_;

    RunningMean mean_;

    std::vector<double> components_;
    std::vector<double> singular_values_;
    std::vector<double> explained_variance_;
};

}