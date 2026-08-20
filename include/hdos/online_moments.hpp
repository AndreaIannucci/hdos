#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace hdos {

/// Numerically stable online mean for vector-valued observations.
///
/// Batch matrices use column-major storage. For a matrix with
/// n_samples rows and n_features columns, element (i, j) is stored at
///
///     X[i + j * n_samples].
class RunningMean {
public:
    /// Construct an empty accumulator.
    ///
    /// @throws std::invalid_argument if n_features is zero.
    explicit RunningMean(std::size_t n_features);

    /// Incorporate one observation.
    ///
    /// @throws std::invalid_argument for incompatible or non-finite
    /// input.
    void update(std::span<const double> x);

    /// Incorporate a column-major batch of observations.
    ///
    /// An empty batch leaves the accumulator unchanged.
    ///
    /// @throws std::invalid_argument for incompatible or non-finite
    /// input.
    void batch_update(std::span<const double> X);

    /// Return the current feature means.
    ///
    /// Before the first observation and after reset(), all entries
    /// are zero.
    [[nodiscard]]
    std::span<const double> mean() const noexcept;

    /// Return the observation dimension.
    [[nodiscard]]
    std::size_t n_features() const noexcept;

    /// Return the number of accumulated observations.
    [[nodiscard]]
    std::size_t n_observations() const noexcept;

    /// Clear all accumulated observations.
    void reset();

private:
    std::size_t n_features_;
    std::size_t n_observations_;

    std::vector<double> mean_;
};

/// Numerically stable online mean and sample variance.
///
/// The variance is the unbiased sample variance
///
///     M2 / (n - 1).
///
/// It is reported as zero when fewer than two observations have been
/// accumulated.
class RunningVariance {
public:
    /// Construct an empty accumulator.
    ///
    /// @throws std::invalid_argument if n_features is zero.
    explicit RunningVariance(std::size_t n_features);

    /// Incorporate one observation.
    ///
    /// @throws std::invalid_argument for incompatible or non-finite
    /// input.
    void update(std::span<const double> x);

    /// Incorporate a column-major batch of observations.
    ///
    /// An empty batch leaves the accumulator unchanged.
    ///
    /// @throws std::invalid_argument for incompatible or non-finite
    /// input.
    void batch_update(std::span<const double> X);

    /// Return the current feature means.
    [[nodiscard]]
    std::span<const double> mean() const noexcept;

    /// Return the current unbiased sample variances.
    // ddof = 1: sample covariance
    // ddof = 0: population covariance
    [[nodiscard]]
    std::span<const double> variance() noexcept;

    /// Return the observation dimension.
    [[nodiscard]]
    std::size_t n_features() const noexcept;

    /// Return the number of accumulated observations.
    [[nodiscard]]
    std::size_t n_observations() const noexcept;

    /// Clear all accumulated observations.
    void reset();

private:
    std::size_t n_features_;
    std::size_t n_observations_;

    std::vector<double> mean_;
    std::vector<double> M2_;
    std::vector<double> variance_;
    bool is_current = false;
};

class RunningCovariance {
public:
    explicit RunningCovariance(std::size_t n_features);

    void update(std::span<const double> x);
    void batch_update(std::span<const double> X);

    /// Return the observation dimension.
    [[nodiscard]]
    std::size_t n_features() const noexcept;

    /// Return the number of accumulated observations.
    [[nodiscard]]
    std::size_t n_observations() const noexcept;
    [[nodiscard]] std::span<const double> mean() const noexcept;

    // ddof = 1: sample covariance
    // ddof = 0: population covariance
    [[nodiscard]] std::span<const double> covariance() noexcept;

    void reset();

private:
    std::size_t n_features_;
    std::size_t n_observations_;

    std::vector<double> mean_;
    std::vector<double> M2_;
    std::vector<double> covariance_;
    bool is_current = false;
};

} 