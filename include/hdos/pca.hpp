#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "hdos/online_moments.hpp"

namespace hdos {

/// Batch and incremental principal component analysis.
///
/// Input matrices use column-major storage. For a matrix with
/// n_samples rows and n_features columns, element (i, j) is stored at
///
///     X[i + j * n_samples].
///
/// Principal components are stored as a column-major matrix with
/// n_features rows and n_components columns. Each column is one
/// principal direction.
class PCA {
public:
    /// Construct an unfitted PCA model.
    ///
    /// @param n_features Observation dimension.
    /// @param n_components Number of principal components retained.
    ///
    /// @throws std::invalid_argument if either dimension is zero or
    /// n_components exceeds n_features.
    PCA(
        std::size_t n_features,
        std::size_t n_components
    );

    /// Fit the PCA model from scratch.
    ///
    /// @param X Column-major n_samples by n_features matrix.
    ///
    /// At least two observations are required.
    ///
    /// @throws std::invalid_argument for incompatible, insufficient,
    /// or non-finite input.
    void fit(std::span<const double> X);

    /// Incorporate one new observation.
    ///
    /// @throws std::logic_error if the model has not been fitted.
    /// @throws std::invalid_argument for incompatible or non-finite
    /// input.
    void update(std::span<const double> x);

    /// Incorporate a column-major batch of new observations.
    ///
    /// An empty batch leaves the fitted model unchanged.
    ///
    /// @throws std::logic_error if the model has not been fitted.
    /// @throws std::invalid_argument for incompatible or non-finite
    /// input.
    void batch_update(std::span<const double> X);

    /// Return the retained principal directions.
    ///
    /// The result represents a column-major n_features by
    /// n_components matrix.
    [[nodiscard]]
    std::span<const double> components() const noexcept;

    /// Return the retained singular values in descending order.
    [[nodiscard]]
    std::span<const double> singular_values() const noexcept;

    /// Return the explained variances of the retained components.
    ///
    /// For n observations, each entry is sigma^2 / (n - 1).
    [[nodiscard]]
    std::span<const double> explained_variance() const noexcept;

    /// Return the current feature means.
    [[nodiscard]]
    std::span<const double> mean() const noexcept;

    /// Return the observation dimension.
    [[nodiscard]]
    std::size_t n_features() const noexcept;

    /// Return the number of retained components.
    [[nodiscard]]
    std::size_t n_components() const noexcept;

    /// Return the number of represented observations.
    [[nodiscard]]
    std::size_t n_observations() const noexcept;

    /// Clear the fitted model and accumulated observations.
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