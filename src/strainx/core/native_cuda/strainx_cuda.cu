#include "strainx_cuda.h"

#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <new>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

thread_local std::string g_last_error;

void set_error(const std::string& message) { g_last_error = message; }

bool cuda_ok(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    set_error(std::string(operation) + ": " + cudaGetErrorString(status));
    return false;
}

struct Solver {
    int radius;
    int spacing;
    int search_radius;
    int rescue_radius;
    int max_iter;
    double conv_tol;
    double corr_cutoff;
    bool mask_subsets_to_roi;

    int height = 0;
    int width = 0;
    int grid_h = 0;
    int grid_w = 0;
    int total = 0;
    int n_pixels = 0;
    bool initialized = false;
    bool has_current = false;
    bool has_state = false;

    std::vector<int> gx;
    std::vector<int> gy;
    std::vector<uint8_t> valid;
    std::vector<int8_t> state;
    std::vector<uint8_t> retry;
    std::vector<double> parameters;
    std::vector<double> correlation;
    std::vector<int2> offsets;
    std::vector<double> host_reference;
    std::vector<double> host_current;
    std::vector<double> host_reference_coeff;
    std::vector<double> host_current_coeff;

    double* d_reference = nullptr;
    double* d_current = nullptr;
    double* d_reference_coeff = nullptr;
    double* d_current_coeff = nullptr;
    double* d_grad_x = nullptr;
    double* d_grad_y = nullptr;
    uint8_t* d_roi_mask = nullptr;
    int* d_grid_x = nullptr;
    int* d_grid_y = nullptr;
    int2* d_offsets = nullptr;
    int* d_active_indices = nullptr;
    double* d_active_parameters = nullptr;
    double* d_result_parameters = nullptr;
    double* d_result_correlation = nullptr;
    uint8_t* d_result_accepted = nullptr;
    double* d_ncc_scores = nullptr;
    size_t ncc_capacity = 0;
    double reference_peak = 0.0;
    double current_peak = 0.0;
    double intensity_scale = 1.0;

    Solver(int r, int s, int search, int rescue, int iterations,
           double tolerance, double cutoff, bool mask_roi)
        : radius(r), spacing(s), search_radius(search), rescue_radius(rescue),
          max_iter(iterations), conv_tol(tolerance), corr_cutoff(cutoff),
          mask_subsets_to_roi(mask_roi) {}

    ~Solver() { release_device(); }

    void release_device() {
        cudaFree(d_reference); d_reference = nullptr;
        cudaFree(d_current); d_current = nullptr;
        cudaFree(d_reference_coeff); d_reference_coeff = nullptr;
        cudaFree(d_current_coeff); d_current_coeff = nullptr;
        cudaFree(d_grad_x); d_grad_x = nullptr;
        cudaFree(d_grad_y); d_grad_y = nullptr;
        cudaFree(d_roi_mask); d_roi_mask = nullptr;
        cudaFree(d_grid_x); d_grid_x = nullptr;
        cudaFree(d_grid_y); d_grid_y = nullptr;
        cudaFree(d_offsets); d_offsets = nullptr;
        cudaFree(d_active_indices); d_active_indices = nullptr;
        cudaFree(d_active_parameters); d_active_parameters = nullptr;
        cudaFree(d_result_parameters); d_result_parameters = nullptr;
        cudaFree(d_result_correlation); d_result_correlation = nullptr;
        cudaFree(d_result_accepted); d_result_accepted = nullptr;
        cudaFree(d_ncc_scores); d_ncc_scores = nullptr;
        ncc_capacity = 0;
        initialized = has_current = has_state = false;
    }
};

double image_peak(const double* image, size_t count) {
    double peak = 0.0;
    for (size_t i = 0; i < count; ++i)
        if (std::isfinite(image[i])) peak = std::max(peak, image[i]);
    return peak;
}

double infer_intensity_scale(double peak) {
    if (peak <= 1.5) return 1.0;
    if (peak <= 255.0) return 255.0;
    if (peak <= 65535.0) return 65535.0;
    return peak;
}

// scipy/cupyx ndimage's cubic spline filter with mode="mirror" is the
// separable solution of
//
//     c[i-1] + 4*c[i] + c[i+1] = 6*s[i]
//
// with reflected end points (the first upper and last lower coefficient are
// therefore 2).  Keeping this prefilter in the native host runtime reproduces
// the coefficient images used by the former CuPy implementation without
// putting any numerical work back into Python.
void cubic_spline_filter_line(double* data, int length, int stride,
                              std::vector<double>& upper,
                              std::vector<double>& rhs) {
    if (length <= 1) return;
    upper.resize(length);
    rhs.resize(length);
    upper[0] = 0.5;
    rhs[0] = 1.5 * data[0];
    for (int i = 1; i < length; ++i) {
        const double lower = i == length - 1 ? 2.0 : 1.0;
        const double diagonal = 4.0 - lower * upper[i - 1];
        const double next_upper = i + 1 < length ? 1.0 : 0.0;
        upper[i] = next_upper / diagonal;
        rhs[i] = (6.0 * data[i * stride] - lower * rhs[i - 1]) / diagonal;
    }
    data[(length - 1) * stride] = rhs[length - 1];
    for (int i = length - 2; i >= 0; --i)
        data[i * stride] = rhs[i] - upper[i] * data[(i + 1) * stride];
}

std::vector<double> cubic_spline_coefficients(const double* image,
                                               int height, int width) {
    std::vector<double> coefficients(
        image, image + static_cast<size_t>(height) * width);
    std::vector<double> upper, rhs;
    for (int y = 0; y < height; ++y)
        cubic_spline_filter_line(coefficients.data() + y * width, width, 1,
                                 upper, rhs);
    for (int x = 0; x < width; ++x)
        cubic_spline_filter_line(coefficients.data() + x, height, width,
                                 upper, rhs);
    return coefficients;
}

__device__ __forceinline__ int mirror_index(int value, int size) {
    if (size <= 1) return 0;
    while (value < 0 || value >= size) {
        value = value < 0 ? -value : 2 * size - 2 - value;
    }
    return value;
}

__device__ __forceinline__ double cubic_weight(double x) {
    x = fabs(x);
    if (x < 1.0) return 2.0 / 3.0 - x * x + 0.5 * x * x * x;
    if (x < 2.0) {
        const double tail = 2.0 - x;
        return tail * tail * tail / 6.0;
    }
    return 0.0;
}

__device__ __forceinline__ double sample_cubic(
    const double* image, int height, int width, double x, double y) {
    const int ix = static_cast<int>(floor(x));
    const int iy = static_cast<int>(floor(y));
    double value = 0.0;
    for (int j = -1; j <= 2; ++j) {
        const int yy = mirror_index(iy + j, height);
        const double wy = cubic_weight(y - static_cast<double>(iy + j));
        for (int i = -1; i <= 2; ++i) {
            const int xx = mirror_index(ix + i, width);
            value += image[yy * width + xx] *
                     cubic_weight(x - static_cast<double>(ix + i)) * wy;
        }
    }
    return value;
}

__device__ __forceinline__ double sample_bilinear(
    const double* image, int height, int width, double x, double y) {
    const int x0 = static_cast<int>(floor(x));
    const int y0 = static_cast<int>(floor(y));
    const double tx = x - x0;
    const double ty = y - y0;
    const int xa = mirror_index(x0, width);
    const int xb = mirror_index(x0 + 1, width);
    const int ya = mirror_index(y0, height);
    const int yb = mirror_index(y0 + 1, height);
    const double a = image[ya * width + xa] * (1.0 - tx) +
                     image[ya * width + xb] * tx;
    const double b = image[yb * width + xa] * (1.0 - tx) +
                     image[yb * width + xb] * tx;
    return a * (1.0 - ty) + b * ty;
}

__global__ void gradient_kernel(const double* coefficients, double* grad_x,
                                double* grad_y, int height, int width) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = height * width;
    if (idx >= count) return;
    const int y = idx / width;
    const int x = idx - y * width;
    constexpr double h = 1e-3;
    grad_x[idx] = (sample_cubic(coefficients, height, width, x + h, y) -
                   sample_cubic(coefficients, height, width, x - h, y)) /
                  (2.0 * h);
    grad_y[idx] = (sample_cubic(coefficients, height, width, x, y + h) -
                   sample_cubic(coefficients, height, width, x, y - h)) /
                  (2.0 * h);
}

__device__ bool invert_6x6(const double* matrix, double* inverse) {
    double a[72];
    for (int row = 0; row < 6; ++row) {
        for (int col = 0; col < 6; ++col) {
            a[row * 12 + col] = matrix[row * 6 + col];
            a[row * 12 + 6 + col] = row == col ? 1.0 : 0.0;
        }
    }
    for (int col = 0; col < 6; ++col) {
        int pivot = col;
        double largest = fabs(a[col * 12 + col]);
        for (int row = col + 1; row < 6; ++row) {
            const double candidate = fabs(a[row * 12 + col]);
            if (candidate > largest) { largest = candidate; pivot = row; }
        }
        if (!(largest > 1e-15) || !isfinite(largest)) return false;
        if (pivot != col) {
            for (int k = 0; k < 12; ++k) {
                const double tmp = a[col * 12 + k];
                a[col * 12 + k] = a[pivot * 12 + k];
                a[pivot * 12 + k] = tmp;
            }
        }
        const double scale = 1.0 / a[col * 12 + col];
        for (int k = 0; k < 12; ++k) a[col * 12 + k] *= scale;
        for (int row = 0; row < 6; ++row) {
            if (row == col) continue;
            const double factor = a[row * 12 + col];
            for (int k = 0; k < 12; ++k)
                a[row * 12 + k] -= factor * a[col * 12 + k];
        }
    }
    for (int row = 0; row < 6; ++row)
        for (int col = 0; col < 6; ++col)
            inverse[row * 6 + col] = a[row * 12 + 6 + col];
    return true;
}

__device__ __forceinline__ void raw_jacobian(
    double gx, double gy, double dx, double dy, double* out) {
    out[0] = gx; out[1] = gy;
    out[2] = gx * dx; out[3] = gx * dy;
    out[4] = gy * dx; out[5] = gy * dy;
}

__device__ bool compose_inverse_affine(const double* p, const double* dp,
                                       double* output) {
    const double a2 = 1.0 + dp[2], b2 = dp[3], c2 = dp[0];
    const double d2 = dp[4], e2 = 1.0 + dp[5], f2 = dp[1];
    const double determinant = a2 * e2 - b2 * d2;
    if (fabs(determinant) < 1e-12 || !isfinite(determinant)) return false;
    const double inv_det = 1.0 / determinant;
    const double ia = e2 * inv_det, ib = -b2 * inv_det;
    const double ic = (b2 * f2 - c2 * e2) * inv_det;
    const double id = -d2 * inv_det, ie = a2 * inv_det;
    const double iff = (c2 * d2 - a2 * f2) * inv_det;
    const double a1 = 1.0 + p[2], b1 = p[3], c1 = p[0];
    const double d1 = p[4], e1 = 1.0 + p[5], f1 = p[1];
    output[0] = a1 * ic + b1 * iff + c1;
    output[1] = d1 * ic + e1 * iff + f1;
    output[2] = a1 * ia + b1 * id - 1.0;
    output[3] = a1 * ib + b1 * ie;
    output[4] = d1 * ia + e1 * id;
    output[5] = d1 * ib + e1 * ie - 1.0;
    return true;
}

// One CUDA block cooperates on each IC-GN subset; pixels are reduced in
// parallel and only the 6x6 solve/composition remains on lane zero.
struct BlockWorkspace {
    double sums[40];
    double p[6];
    double start[6];
    double best[6];
    double correction[6];
    double hessian[36];
    double inverse[36];
    double b[6];
    double ref_mean;
    double ref_sigma;
    double r_eff;
    double score;
    double best_score;
    double rescue_scores[128];
    int rescue_candidates[128];
    int support;
    int invalid;
    int stop;
};

__device__ __forceinline__ double warp_sum(double value) {
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(0xffffffffu, value, offset);
    return value;
}

__device__ void block_accumulate(double* local, int count,
                                 BlockWorkspace* workspace) {
    for (int k = threadIdx.x; k < count; k += blockDim.x)
        workspace->sums[k] = 0.0;
    __syncthreads();
    const int lane = threadIdx.x & 31;
    for (int k = 0; k < count; ++k) {
        const double value = warp_sum(local[k]);
        if (lane == 0) atomicAdd(&workspace->sums[k], value);
    }
    __syncthreads();
}

__device__ void evaluate_parameters_block(
    const double* reference, const double* current,
    const double* grad_x, const double* grad_y,
    const uint8_t* roi_mask, double intensity_scale,
    int height, int width, int center_x, int center_y,
    const int2* offsets, int n_pixels,
    BlockWorkspace* workspace, bool need_b) {
    double first[6] = {0, 0, 0, 0, 0, 0};
    for (int i = threadIdx.x; i < n_pixels; i += blockDim.x) {
        const double dx = offsets[i].x, dy = offsets[i].y;
        const int ref_idx = (center_y + offsets[i].y) * width +
                            center_x + offsets[i].x;
        if (roi_mask && !roi_mask[ref_idx]) continue;
        first[5] += 1.0;
        const double x = center_x + dx + workspace->p[0] +
                         workspace->p[2] * dx + workspace->p[3] * dy;
        const double y = center_y + dy + workspace->p[1] +
                         workspace->p[4] * dx + workspace->p[5] * dy;
        if (x < -0.5 || x > width - 0.5 || y < -0.5 || y > height - 0.5) {
            first[4] += 1.0;
            continue;
        }
        const double value = sample_cubic(current, height, width, x, y);
        first[0] += value;
        if (value <= (12.0 / 255.0) * intensity_scale ||
            value >= (250.0 / 255.0) * intensity_scale) {
            first[1] += 1.0; first[2] += dx; first[3] += dy;
        }
    }
    block_accumulate(first, 6, workspace);
    if (threadIdx.x == 0) {
        workspace->invalid = workspace->sums[4] > 0.0 ||
                             static_cast<int>(workspace->sums[5]) !=
                                 workspace->support;
        const double occluded = workspace->sums[1];
        if (!workspace->invalid && occluded > 0.0) {
            const double fraction = occluded / workspace->support;
            const double asymmetry = hypot(workspace->sums[2] / occluded,
                                            workspace->sums[3] / occluded) /
                                     workspace->r_eff *
                                     fraction * 4.0;
            workspace->invalid = fraction > 0.12 || asymmetry > 0.35;
        }
        workspace->sums[0] /= workspace->support;
    }
    __syncthreads();
    if (workspace->invalid) {
        if (threadIdx.x == 0) workspace->score = CUDART_INF;
        __syncthreads();
        return;
    }
    const double mean = workspace->sums[0];
    double sigma_local[1] = {0.0};
    for (int i = threadIdx.x; i < n_pixels; i += blockDim.x) {
        const double dx = offsets[i].x, dy = offsets[i].y;
        const int ref_idx = (center_y + offsets[i].y) * width +
                            center_x + offsets[i].x;
        if (roi_mask && !roi_mask[ref_idx]) continue;
        const double x = center_x + dx + workspace->p[0] +
                         workspace->p[2] * dx + workspace->p[3] * dy;
        const double y = center_y + dy + workspace->p[1] +
                         workspace->p[4] * dx + workspace->p[5] * dy;
        const double centered = sample_cubic(current, height, width, x, y) - mean;
        sigma_local[0] += centered * centered;
    }
    block_accumulate(sigma_local, 1, workspace);
    if (threadIdx.x == 0) {
        workspace->sums[0] = sqrt(fmax(workspace->sums[0], 1e-24));
        workspace->invalid = !(workspace->sums[0] > 1e-12);
    }
    __syncthreads();
    if (workspace->invalid) {
        if (threadIdx.x == 0) workspace->score = CUDART_INF;
        __syncthreads();
        return;
    }
    const double sigma = workspace->sums[0];
    double totals[7] = {0, 0, 0, 0, 0, 0, 0};
    for (int i = threadIdx.x; i < n_pixels; i += blockDim.x) {
        const int dx_i = offsets[i].x, dy_i = offsets[i].y;
        const double dx = dx_i, dy = dy_i;
        const int ref_idx = (center_y + dy_i) * width + center_x + dx_i;
        if (roi_mask && !roi_mask[ref_idx]) continue;
        const double f_norm = (reference[ref_idx] - workspace->ref_mean) /
                              workspace->ref_sigma;
        const double x = center_x + dx + workspace->p[0] +
                         workspace->p[2] * dx + workspace->p[3] * dy;
        const double y = center_y + dy + workspace->p[1] +
                         workspace->p[4] * dx + workspace->p[5] * dy;
        const double residual =
            (sample_cubic(current, height, width, x, y) - mean) / sigma - f_norm;
        totals[0] += residual * residual;
        if (need_b) {
            double raw[6];
            raw_jacobian(grad_x[ref_idx], grad_y[ref_idx], dx, dy, raw);
            for (int k = 0; k < 6; ++k) {
                const double sd = (raw[k] - f_norm * workspace->correction[k]) /
                                  workspace->ref_sigma;
                totals[k + 1] += sd * residual;
            }
        }
    }
    block_accumulate(totals, need_b ? 7 : 1, workspace);
    if (threadIdx.x == 0) {
        workspace->score = workspace->sums[0];
        if (need_b)
            for (int k = 0; k < 6; ++k) workspace->b[k] = workspace->sums[k + 1];
    }
    __syncthreads();
}

__device__ double evaluate_translation_bilinear(
    const double* reference, const double* current,
    const uint8_t* roi_mask, int support,
    int height, int width, int center_x, int center_y,
    const int2* offsets, int n_pixels, double translation_u,
    double translation_v, double ref_mean, double ref_sigma,
    bool reject_out_of_bounds) {
    double mean = 0.0;
    for (int i = 0; i < n_pixels; ++i) {
        const int ref_idx = (center_y + offsets[i].y) * width +
                            center_x + offsets[i].x;
        if (roi_mask && !roi_mask[ref_idx]) continue;
        const double x = center_x + offsets[i].x + translation_u;
        const double y = center_y + offsets[i].y + translation_v;
        if (reject_out_of_bounds &&
            (x < -0.5 || x > width - 0.5 ||
             y < -0.5 || y > height - 0.5)) return CUDART_INF;
        mean += sample_bilinear(current, height, width, x, y);
    }
    mean /= support;
    double sigma2 = 0.0;
    for (int i = 0; i < n_pixels; ++i) {
        const int ref_idx = (center_y + offsets[i].y) * width +
                            center_x + offsets[i].x;
        if (roi_mask && !roi_mask[ref_idx]) continue;
        const double x = center_x + offsets[i].x + translation_u;
        const double y = center_y + offsets[i].y + translation_v;
        const double centered =
            sample_bilinear(current, height, width, x, y) - mean;
        sigma2 += centered * centered;
    }
    const double sigma = fmax(sqrt(sigma2), 1e-12);
    double score = 0.0;
    for (int i = 0; i < n_pixels; ++i) {
        const int ref_idx = (center_y + offsets[i].y) * width +
                            center_x + offsets[i].x;
        if (roi_mask && !roi_mask[ref_idx]) continue;
        const double f_norm = (reference[ref_idx] - ref_mean) / ref_sigma;
        const double x = center_x + offsets[i].x + translation_u;
        const double y = center_y + offsets[i].y + translation_v;
        const double g_norm =
            (sample_bilinear(current, height, width, x, y) - mean) / sigma;
        const double residual = g_norm - f_norm;
        score += residual * residual;
    }
    return score;
}

__global__ void solve_subsets_block_kernel(
    const double* reference, const double* current_raw,
    const double* current_coeff,
    const double* grad_x, const double* grad_y,
    const uint8_t* roi_mask, double intensity_scale,
    int height, int width, const int* grid_x, const int* grid_y,
    const int2* offsets, int n_pixels,
    const int* active_indices, const double* initial_parameters, int n_active,
    int max_iter, double conv_tol, double corr_cutoff, int rescue_radius,
    int spacing, int subset_radius, int warm_start,
    double* result_parameters, double* result_correlation,
    uint8_t* result_accepted) {
    const int row = blockIdx.x;
    if (row >= n_active) return;
    __shared__ BlockWorkspace workspace;
    const int grid_idx = active_indices[row];
    const int cx = grid_x[grid_idx], cy = grid_y[grid_idx];
    if (threadIdx.x < 6) {
        workspace.p[threadIdx.x] = initial_parameters[row * 6 + threadIdx.x];
    }
    if (threadIdx.x == 0) {
        workspace.invalid = 0; workspace.stop = 0;
        workspace.best_score = CUDART_INF;
    }
    __syncthreads();

    double ref_stats[5] = {0, 0, 0, 0, 0};
    double local_r_eff = 0.0;
    for (int i = threadIdx.x; i < n_pixels; i += blockDim.x) {
        const int idx = (cy + offsets[i].y) * width + cx + offsets[i].x;
        if (roi_mask && !roi_mask[idx]) continue;
        const double value = reference[idx];
        ref_stats[0] += value;
        ref_stats[4] += 1.0;
        local_r_eff = fmax(
            local_r_eff,
            hypot(static_cast<double>(offsets[i].x),
                  static_cast<double>(offsets[i].y)));
        if (value <= (12.0 / 255.0) * intensity_scale ||
            value >= (250.0 / 255.0) * intensity_scale) {
            ref_stats[1] += 1.0;
            ref_stats[2] += offsets[i].x;
            ref_stats[3] += offsets[i].y;
        }
    }
    block_accumulate(ref_stats, 5, &workspace);
    workspace.rescue_scores[threadIdx.x] = local_r_eff;
    __syncthreads();
    if (threadIdx.x == 0) {
        workspace.support = static_cast<int>(workspace.sums[4]);
        workspace.r_eff = 1.0;
        for (int lane = 0; lane < blockDim.x; ++lane)
            workspace.r_eff = fmax(workspace.r_eff,
                                   workspace.rescue_scores[lane]);
        workspace.invalid = workspace.support < 6 ||
                            workspace.support < 0.30 * n_pixels;
        workspace.ref_mean = workspace.support > 0
            ? workspace.sums[0] / workspace.support : 0.0;
        const double count = workspace.sums[1];
        if (!workspace.invalid && count > 0.0) {
            const double fraction = count / workspace.support;
            const double asymmetry = hypot(workspace.sums[2] / count,
                                            workspace.sums[3] / count) /
                                     workspace.r_eff *
                                     fraction * 4.0;
            workspace.invalid |= fraction > 0.12 || asymmetry > 0.35;
        }
    }
    __syncthreads();
    double sigma_local[1] = {0.0};
    for (int i = threadIdx.x; i < n_pixels; i += blockDim.x) {
        const int idx = (cy + offsets[i].y) * width + cx + offsets[i].x;
        if (roi_mask && !roi_mask[idx]) continue;
        const double centered = reference[idx] - workspace.ref_mean;
        sigma_local[0] += centered * centered;
    }
    block_accumulate(sigma_local, 1, &workspace);
    if (threadIdx.x == 0) {
        workspace.ref_sigma = sqrt(fmax(workspace.sums[0], 1e-24));
        workspace.invalid |= !(workspace.ref_sigma > 1e-12);
    }
    __syncthreads();
    if (workspace.invalid) {
        if (threadIdx.x == 0) {
            result_accepted[row] = 0; result_correlation[row] = CUDART_INF;
        }
        return;
    }

    double correction_local[6] = {0, 0, 0, 0, 0, 0};
    for (int i = threadIdx.x; i < n_pixels; i += blockDim.x) {
        const int dx = offsets[i].x, dy = offsets[i].y;
        const int idx = (cy + dy) * width + cx + dx;
        if (roi_mask && !roi_mask[idx]) continue;
        const double fn = (reference[idx] - workspace.ref_mean) / workspace.ref_sigma;
        double raw[6]; raw_jacobian(grad_x[idx], grad_y[idx], dx, dy, raw);
        for (int k = 0; k < 6; ++k) correction_local[k] += fn * raw[k];
    }
    block_accumulate(correction_local, 6, &workspace);
    if (threadIdx.x < 6) workspace.correction[threadIdx.x] = workspace.sums[threadIdx.x];
    __syncthreads();

    double hessian_local[36];
    for (int k = 0; k < 36; ++k) hessian_local[k] = 0.0;
    for (int i = threadIdx.x; i < n_pixels; i += blockDim.x) {
        const int dx = offsets[i].x, dy = offsets[i].y;
        const int idx = (cy + dy) * width + cx + dx;
        if (roi_mask && !roi_mask[idx]) continue;
        const double fn = (reference[idx] - workspace.ref_mean) / workspace.ref_sigma;
        double raw[6], sd[6]; raw_jacobian(grad_x[idx], grad_y[idx], dx, dy, raw);
        for (int k = 0; k < 6; ++k)
            sd[k] = (raw[k] - fn * workspace.correction[k]) / workspace.ref_sigma;
        for (int a = 0; a < 6; ++a)
            for (int b = 0; b < 6; ++b)
                hessian_local[a * 6 + b] += sd[a] * sd[b];
    }
    block_accumulate(hessian_local, 36, &workspace);
    if (threadIdx.x == 0) {
        for (int k = 0; k < 36; ++k) workspace.hessian[k] = workspace.sums[k];
        for (int k = 0; k < 6; ++k) workspace.hessian[k * 6 + k] += 1e-6;
        workspace.invalid = !invert_6x6(workspace.hessian, workspace.inverse);
    }
    __syncthreads();
    if (workspace.invalid) {
        if (threadIdx.x == 0) {
            result_accepted[row] = 0; result_correlation[row] = CUDART_INF;
        }
        return;
    }

    // Keep the propagated/warm parameters as the safety anchor. The former
    // CuPy path overwrote this after integer rescue, allowing an R-pixel rescue
    // plus another full IC-GN jump to pass as a small local correction. The
    // cumulative gate below must measure motion from the estimate supplied by
    // the wavefront, not from a rescued alias.
    if (threadIdx.x < 6)
        workspace.start[threadIdx.x] = workspace.p[threadIdx.x];
    __syncthreads();

    // Keep the former CuPy rescue scoring semantics: score only the
    // translational part of the propagated guess with bilinear interpolation
    // on the raw current image. Search only poor guesses, and reject
    // off-frame search candidates (the trigger score itself remains mirrored).
    if (threadIdx.x == 0) {
        workspace.score = evaluate_translation_bilinear(
            reference, current_raw, roi_mask, workspace.support,
            height, width, cx, cy, offsets, n_pixels,
            workspace.p[0], workspace.p[1], workspace.ref_mean,
            workspace.ref_sigma, false);
    }
    __syncthreads();
    if (workspace.score > 0.25 && rescue_radius > 0) {
        const int side = 2 * rescue_radius + 1;
        const int candidates = side * side;
        double local_best = CUDART_INF;
        int local_candidate = candidates;
        for (int candidate = threadIdx.x; candidate < candidates;
             candidate += blockDim.x) {
            const int du = candidate % side - rescue_radius;
            const int dv = candidate / side - rescue_radius;
            const double score = evaluate_translation_bilinear(
                reference, current_raw, roi_mask, workspace.support,
                height, width, cx, cy, offsets, n_pixels,
                workspace.p[0] + du, workspace.p[1] + dv,
                workspace.ref_mean, workspace.ref_sigma, true);
            if (score < local_best ||
                (score == local_best && candidate < local_candidate)) {
                local_best = score;
                local_candidate = candidate;
            }
        }
        workspace.rescue_scores[threadIdx.x] = local_best;
        workspace.rescue_candidates[threadIdx.x] = local_candidate;
        __syncthreads();
        if (threadIdx.x == 0) {
            double best_score = workspace.score;
            int best_candidate = candidates;
            for (int lane = 0; lane < blockDim.x; ++lane) {
                const double score = workspace.rescue_scores[lane];
                const int candidate = workspace.rescue_candidates[lane];
                if (score < best_score ||
                    (score == best_score && best_candidate < candidates &&
                     candidate < best_candidate)) {
                    best_score = score;
                    best_candidate = candidate;
                }
            }
            if (best_candidate < candidates) {
                const int du = best_candidate % side - rescue_radius;
                const int dv = best_candidate / side - rescue_radius;
                workspace.p[0] += du;
                workspace.p[1] += dv;
                if (!warm_start && hypot(static_cast<double>(du),
                                         static_cast<double>(dv)) > 4.0)
                    for (int k = 2; k < 6; ++k) workspace.p[k] = 0.0;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x < 6)
        workspace.best[threadIdx.x] = workspace.p[threadIdx.x];
    if (threadIdx.x == 0) workspace.best_score = CUDART_INF;
    __syncthreads();

    for (int iteration = 0; iteration < max_iter; ++iteration) {
        evaluate_parameters_block(reference, current_coeff, grad_x, grad_y,
                                  roi_mask, intensity_scale,
                                  height, width, cx, cy, offsets, n_pixels,
                                  &workspace, true);
        if (threadIdx.x == 0) {
            if (workspace.score < workspace.best_score) {
                workspace.best_score = workspace.score;
                for (int k = 0; k < 6; ++k) workspace.best[k] = workspace.p[k];
            }
            if (!isfinite(workspace.score)) {
                workspace.stop = 1;
            } else {
                double dp[6] = {0, 0, 0, 0, 0, 0};
                for (int a = 0; a < 6; ++a)
                    for (int k = 0; k < 6; ++k)
                        dp[a] += workspace.inverse[a * 6 + k] * workspace.b[k];
                double norm2 = 0.0;
                for (int k = 0; k < 6; ++k) norm2 += dp[k] * dp[k];
                const double disp = hypot(dp[0], dp[1]);
                const double grad = sqrt(dp[2] * dp[2] + dp[3] * dp[3] +
                                         dp[4] * dp[4] + dp[5] * dp[5]);
                double next[6];
                if (!isfinite(norm2) || disp > 5.0 || grad > 8.0 ||
                    !compose_inverse_affine(workspace.p, dp, next)) {
                    workspace.stop = 1;
                } else {
                    for (int k = 0; k < 6; ++k) workspace.p[k] = next[k];
                    workspace.stop = sqrt(norm2) < conv_tol;
                }
            }
        }
        __syncthreads();
        if (workspace.stop) break;
    }
    evaluate_parameters_block(reference, current_coeff, grad_x, grad_y,
                              roi_mask, intensity_scale,
                              height, width, cx, cy, offsets, n_pixels,
                              &workspace, false);
    if (threadIdx.x == 0) {
        if (workspace.score < workspace.best_score) {
            workspace.best_score = workspace.score;
            for (int k = 0; k < 6; ++k) workspace.best[k] = workspace.p[k];
        }
        const double displacement_limit = spacing + 1.0;
        const bool accepted = isfinite(workspace.best_score) &&
            workspace.best_score < corr_cutoff &&
            fabs(workspace.best[0] - workspace.start[0]) < displacement_limit &&
            fabs(workspace.best[1] - workspace.start[1]) < displacement_limit;
        for (int k = 0; k < 6; ++k)
            result_parameters[row * 6 + k] = workspace.best[k];
        result_correlation[row] = workspace.best_score;
        result_accepted[row] = accepted ? 1 : 0;
    }
}

__global__ void ncc_kernel(
    const double* reference, const double* current, int height, int width,
    int center_x, int center_y, int subset_radius,
    int base_u, int base_v, int search_radius, double* scores) {
    const int side = 2 * search_radius + 1;
    const int candidate = blockIdx.x * blockDim.x + threadIdx.x;
    if (candidate >= side * side) return;
    const int du = candidate % side - search_radius;
    const int dv = candidate / side - search_radius;
    const int u = base_u + du, v = base_v + dv;
    double mean_f = 0.0, mean_g = 0.0;
    const int template_side = 2 * subset_radius + 1;
    const int n_pixels = template_side * template_side;
    for (int dy_ref = -subset_radius; dy_ref <= subset_radius; ++dy_ref) {
        const int yr = center_y + dy_ref;
        for (int dx_ref = -subset_radius; dx_ref <= subset_radius; ++dx_ref) {
            const int xr = center_x + dx_ref;
            const int xc = xr + u, yc = yr + v;
            if (xc < 0 || xc >= width || yc < 0 || yc >= height) {
                scores[candidate] = CUDART_INF; return;
            }
            mean_f += reference[yr * width + xr];
            mean_g += current[yc * width + xc];
        }
    }
    mean_f /= n_pixels; mean_g /= n_pixels;
    double ff = 0.0, gg = 0.0, fg = 0.0;
    for (int dy_ref = -subset_radius; dy_ref <= subset_radius; ++dy_ref) {
        const int yr = center_y + dy_ref;
        for (int dx_ref = -subset_radius; dx_ref <= subset_radius; ++dx_ref) {
            const int xr = center_x + dx_ref;
            const double f = reference[yr * width + xr] - mean_f;
            const double g = current[(yr + v) * width + xr + u] - mean_g;
            ff += f * f; gg += g * g; fg += f * g;
        }
    }
    const double denom = sqrt(ff * gg);
    scores[candidate] = denom > 1e-20 ? 2.0 * (1.0 - fg / denom) : CUDART_INF;
}

bool ensure_ncc_capacity(Solver* solver, size_t count) {
    if (solver->ncc_capacity >= count) return true;
    cudaFree(solver->d_ncc_scores);
    solver->d_ncc_scores = nullptr;
    if (!cuda_ok(cudaMalloc(&solver->d_ncc_scores, count * sizeof(double)),
                 "allocating NCC scores")) return false;
    solver->ncc_capacity = count;
    return true;
}

bool ncc_guess(Solver* solver, int grid_index, double guess_u, double guess_v,
               double& output_u, double& output_v,
               double* output_zncc = nullptr) {
    const int radius = std::max(0, solver->search_radius);
    const int side = 2 * radius + 1;
    const size_t count = static_cast<size_t>(side) * side;
    if (!ensure_ncc_capacity(solver, count)) return false;
    // Match Python round(center + guess) exactly, including ties-to-even.
    // Rounding the guess alone is not equivalent at half-pixel ties because
    // the integer subset centre changes which even integer wins.
    const int center_x = solver->gx[grid_index];
    const int center_y = solver->gy[grid_index];
    const int base_u = static_cast<int>(std::nearbyint(center_x + guess_u)) - center_x;
    const int base_v = static_cast<int>(std::nearbyint(center_y + guess_v)) - center_y;
    const int threads = 256;
    ncc_kernel<<<static_cast<int>((count + threads - 1) / threads), threads>>>(
        solver->d_reference, solver->d_current, solver->height, solver->width,
        solver->gx[grid_index], solver->gy[grid_index], solver->radius,
        base_u, base_v, radius, solver->d_ncc_scores);
    if (!cuda_ok(cudaGetLastError(), "launching native NCC")) return false;
    std::vector<double> scores(count);
    if (!cuda_ok(cudaMemcpy(scores.data(), solver->d_ncc_scores,
                            count * sizeof(double), cudaMemcpyDeviceToHost),
                 "reading native NCC scores")) return false;
    const auto best = std::min_element(scores.begin(), scores.end());
    if (best == scores.end() || !std::isfinite(*best)) {
        set_error("Native NCC found no in-frame candidate."); return false;
    }
    const int index = static_cast<int>(best - scores.begin());
    output_u = base_u + index % side - radius;
    output_v = base_v + index / side - radius;
    if (output_zncc) *output_zncc = 1.0 - 0.5 * (*best);
    return true;
}

bool launch_gradient(Solver* solver) {
    const int count = solver->height * solver->width;
    const int threads = 256;
    gradient_kernel<<<(count + threads - 1) / threads, threads>>>(
        solver->d_reference_coeff, solver->d_grad_x, solver->d_grad_y,
        solver->height, solver->width);
    return cuda_ok(cudaGetLastError(), "launching reference gradient") &&
           cuda_ok(cudaDeviceSynchronize(), "computing reference gradient");
}

bool upload_current_with_coefficients(Solver* solver,
                                      const double* current_image,
                                      const char* operation) {
    const size_t count = static_cast<size_t>(solver->height) * solver->width;
    const size_t bytes = count * sizeof(double);
    solver->host_current.assign(current_image, current_image + count);
    solver->current_peak = image_peak(current_image, count);
    solver->intensity_scale = infer_intensity_scale(
        std::max(solver->reference_peak, solver->current_peak));
    solver->host_current_coeff = cubic_spline_coefficients(
        current_image, solver->height, solver->width);
    return cuda_ok(cudaMemcpy(solver->d_current, current_image, bytes,
                              cudaMemcpyHostToDevice), operation) &&
           cuda_ok(cudaMemcpy(solver->d_current_coeff,
                              solver->host_current_coeff.data(), bytes,
                              cudaMemcpyHostToDevice),
                   "uploading current spline coefficients");
}

void initialize_fresh(Solver* solver, int seed_index,
                      double guess_u, double guess_v) {
    std::fill(solver->state.begin(), solver->state.end(), int8_t{0});
    std::fill(solver->retry.begin(), solver->retry.end(), uint8_t{0});
    std::fill(solver->correlation.begin(), solver->correlation.end(),
              std::numeric_limits<double>::quiet_NaN());
    std::fill(solver->parameters.begin(), solver->parameters.end(), 0.0);
    for (int i = 0; i < solver->total; ++i)
        if (!solver->valid[i]) solver->state[i] = -1;

    double u = guess_u, v = guess_v;
    if (!ncc_guess(solver, seed_index, guess_u, guess_v, u, v))
        throw std::runtime_error("Native CUDA could not initialize the NCC seed.");
    solver->parameters[seed_index * 6] = u;
    solver->parameters[seed_index * 6 + 1] = v;
    solver->state[seed_index] = 1;
    solver->has_state = true;
}

void initialize_warm_start(Solver* solver) {
    if (!solver->has_state)
        throw std::runtime_error("Warm start requested before a fresh solve.");
    for (int i = 0; i < solver->total; ++i) {
        solver->retry[i] = 0;
        if (!solver->valid[i]) {
            solver->state[i] = -1;
        } else if (solver->state[i] == 2 &&
                   std::isfinite(solver->parameters[i * 6]) &&
                   std::isfinite(solver->parameters[i * 6 + 1])) {
            // Preserve the complete previous-pair affine solution. This is the
            // defining warm-start operation used by the former CuPy pipeline.
            solver->state[i] = 1;
        } else {
            // Failed points are reopened and can be reached from a reliable
            // neighbour during this pair, but are never independently NCC-seeded.
            solver->state[i] = 0;
            solver->correlation[i] = std::numeric_limits<double>::quiet_NaN();
        }
    }
}

void squared_distance_transform_1d(const std::vector<double>& input,
                                   std::vector<double>& output) {
    const int length = static_cast<int>(input.size());
    std::vector<int> sites(length);
    std::vector<double> boundaries(length + 1);
    int envelope = 0;
    sites[0] = 0;
    boundaries[0] = -std::numeric_limits<double>::infinity();
    boundaries[1] = std::numeric_limits<double>::infinity();
    for (int q = 1; q < length; ++q) {
        double crossing = 0.0;
        while (true) {
            const int site = sites[envelope];
            crossing = ((input[q] + static_cast<double>(q) * q) -
                        (input[site] + static_cast<double>(site) * site)) /
                       (2.0 * (q - site));
            if (crossing > boundaries[envelope] || envelope == 0) break;
            --envelope;
        }
        ++envelope;
        sites[envelope] = q;
        boundaries[envelope] = crossing;
        boundaries[envelope + 1] = std::numeric_limits<double>::infinity();
    }
    output.resize(length);
    envelope = 0;
    for (int q = 0; q < length; ++q) {
        while (boundaries[envelope + 1] < q) ++envelope;
        const double delta = q - sites[envelope];
        output[q] = delta * delta + input[sites[envelope]];
    }
}

int deepest_component_seed(const std::vector<int>& component,
                           int grid_h, int grid_w) {
    // Pad with a zero-valued border so components touching the grid edge still
    // have a well-defined interior distance. For ordinary failed components
    // this is the same Euclidean EDT argmax used by the former Python path.
    const int padded_h = grid_h + 2;
    const int padded_w = grid_w + 2;
    std::vector<uint8_t> inside(
        static_cast<size_t>(padded_h) * padded_w, 0);
    for (int index : component) {
        const int y = index / grid_w, x = index % grid_w;
        inside[(y + 1) * padded_w + x + 1] = 1;
    }
    constexpr double far = 1e20;
    std::vector<double> column_pass(
        static_cast<size_t>(padded_h) * padded_w, 0.0);
    std::vector<double> input, output;
    input.resize(padded_h);
    for (int x = 0; x < padded_w; ++x) {
        for (int y = 0; y < padded_h; ++y)
            input[y] = inside[y * padded_w + x] ? far : 0.0;
        squared_distance_transform_1d(input, output);
        for (int y = 0; y < padded_h; ++y)
            column_pass[y * padded_w + x] = output[y];
    }
    std::vector<double> distance2(
        static_cast<size_t>(padded_h) * padded_w, 0.0);
    input.resize(padded_w);
    for (int y = 0; y < padded_h; ++y) {
        for (int x = 0; x < padded_w; ++x)
            input[x] = column_pass[y * padded_w + x];
        squared_distance_transform_1d(input, output);
        for (int x = 0; x < padded_w; ++x)
            distance2[y * padded_w + x] = output[x];
    }
    int seed = component.front();
    double best = -1.0;
    // Row-major scan reproduces np.argmax's tie break.
    for (int y = 0; y < grid_h; ++y)
        for (int x = 0; x < grid_w; ++x) {
            const int index = y * grid_w + x;
            if (!inside[(y + 1) * padded_w + x + 1]) continue;
            const double candidate = distance2[(y + 1) * padded_w + x + 1];
            if (candidate > best) { best = candidate; seed = index; }
        }
    return seed;
}

void initialize_recovery(Solver* solver, double guess_u, double guess_v) {
    if (!solver->has_state) throw std::runtime_error("Recovery requested before a solve.");
    const int n = solver->total;
    std::vector<uint8_t> failed(n, 0), seen(n, 0);
    for (int i = 0; i < n; ++i) {
        failed[i] = solver->valid[i] && solver->state[i] != 2;
        if (failed[i]) { solver->state[i] = 0; solver->retry[i] = 0; }
    }
    std::vector<std::vector<int>> components;
    const int di[4] = {-1, 1, 0, 0};
    const int dj[4] = {0, 0, -1, 1};
    for (int start = 0; start < n; ++start) {
        if (!failed[start] || seen[start]) continue;
        components.emplace_back();
        std::queue<int> queue;
        queue.push(start); seen[start] = 1;
        while (!queue.empty()) {
            const int item = queue.front(); queue.pop();
            components.back().push_back(item);
            const int y = item / solver->grid_w, x = item % solver->grid_w;
            for (int k = 0; k < 4; ++k) {
                const int yy = y + dj[k], xx = x + di[k];
                if (yy < 0 || yy >= solver->grid_h || xx < 0 || xx >= solver->grid_w) continue;
                const int next = yy * solver->grid_w + xx;
                if (failed[next] && !seen[next]) { seen[next] = 1; queue.push(next); }
            }
        }
    }
    struct RankedComponent {
        std::vector<int> indices;
        int seed;
    };
    std::vector<RankedComponent> ranked;
    ranked.reserve(components.size());
    for (auto& component : components) {
        const int seed = deepest_component_seed(
            component, solver->grid_h, solver->grid_w);
        ranked.push_back({std::move(component), seed});
    }
    std::sort(ranked.begin(), ranked.end(), [solver](const auto& a, const auto& b) {
        if (a.indices.size() != b.indices.size())
            return a.indices.size() > b.indices.size();
        const int ay = a.seed / solver->grid_w, ax = a.seed % solver->grid_w;
        const int by = b.seed / solver->grid_w, bx = b.seed % solver->grid_w;
        return ay != by ? ay > by : ax > bx;
    });
    if (ranked.size() > 32) ranked.resize(32);
    for (const auto& component : ranked) {
        const int seed = component.seed;
        double u = guess_u, v = guess_v;
        if (!ncc_guess(solver, seed, guess_u, guess_v, u, v)) continue;
        solver->state[seed] = 1;
        for (int k = 0; k < 6; ++k) solver->parameters[seed * 6 + k] = 0.0;
        solver->parameters[seed * 6] = u;
        solver->parameters[seed * 6 + 1] = v;
    }
}

bool run_wavefront(Solver* solver, bool warm_start) {
    constexpr int max_batch = 40000;
    constexpr uint8_t max_retry = 3;
    std::vector<int> active;
    std::vector<double> initial;
    std::vector<double> output_parameters;
    std::vector<double> output_correlation;
    std::vector<uint8_t> accepted;
    std::vector<int> best_parent(solver->total, -1);
    while (true) {
        active.clear();
        for (int i = 0; i < solver->total && static_cast<int>(active.size()) < max_batch; ++i)
            if (solver->state[i] == 1) active.push_back(i);
        if (active.empty()) break;
        const int count = static_cast<int>(active.size());
        initial.resize(static_cast<size_t>(count) * 6);
        output_parameters.resize(static_cast<size_t>(count) * 6);
        output_correlation.resize(count);
        accepted.resize(count);
        for (int row = 0; row < count; ++row)
            std::copy_n(&solver->parameters[active[row] * 6], 6, &initial[row * 6]);
        if (!cuda_ok(cudaMemcpy(solver->d_active_indices, active.data(), count * sizeof(int),
                                cudaMemcpyHostToDevice), "uploading active indices") ||
            !cuda_ok(cudaMemcpy(solver->d_active_parameters, initial.data(),
                                static_cast<size_t>(count) * 6 * sizeof(double),
                                cudaMemcpyHostToDevice), "uploading active parameters")) return false;
        const int threads = 128;
        solve_subsets_block_kernel<<<count, threads>>>(
            solver->d_reference, solver->d_current,
            solver->d_current_coeff,
            solver->d_grad_x, solver->d_grad_y,
            solver->mask_subsets_to_roi ? solver->d_roi_mask : nullptr,
            solver->intensity_scale,
            solver->height, solver->width, solver->d_grid_x,
            solver->d_grid_y,
            solver->d_offsets, solver->n_pixels, solver->d_active_indices,
            solver->d_active_parameters, count, solver->max_iter, solver->conv_tol,
            solver->corr_cutoff, solver->rescue_radius, solver->spacing,
            solver->radius, warm_start ? 1 : 0,
            solver->d_result_parameters,
            solver->d_result_correlation, solver->d_result_accepted);
        if (!cuda_ok(cudaGetLastError(), "launching native IC-GN")) return false;
        if (!cuda_ok(cudaMemcpy(output_parameters.data(), solver->d_result_parameters,
                                static_cast<size_t>(count) * 6 * sizeof(double),
                                cudaMemcpyDeviceToHost), "reading IC-GN parameters") ||
            !cuda_ok(cudaMemcpy(output_correlation.data(), solver->d_result_correlation,
                                count * sizeof(double), cudaMemcpyDeviceToHost),
                     "reading IC-GN correlation") ||
            !cuda_ok(cudaMemcpy(accepted.data(), solver->d_result_accepted,
                                count * sizeof(uint8_t), cudaMemcpyDeviceToHost),
                     "reading IC-GN status")) return false;

        std::fill(best_parent.begin(), best_parent.end(), -1);
        for (int row = 0; row < count; ++row) {
            const int index = active[row];
            if (accepted[row]) {
                std::copy_n(&output_parameters[row * 6], 6,
                            &solver->parameters[index * 6]);
                solver->correlation[index] = output_correlation[row];
                solver->state[index] = 2;
            } else {
                solver->correlation[index] = std::numeric_limits<double>::quiet_NaN();
                solver->retry[index]++;
                solver->state[index] = solver->retry[index] < max_retry ? 0 : -1;
            }
        }
        for (int row = 0; row < count; ++row) {
            if (!accepted[row]) continue;
            const int parent = active[row];
            const int y = parent / solver->grid_w, x = parent % solver->grid_w;
            const int neighbours[4] = {parent - solver->grid_w, parent + solver->grid_w,
                                       parent - 1, parent + 1};
            const bool allowed[4] = {y > 0, y + 1 < solver->grid_h,
                                     x > 0, x + 1 < solver->grid_w};
            for (int k = 0; k < 4; ++k) {
                if (!allowed[k]) continue;
                const int child = neighbours[k];
                if (solver->state[child] != 0) continue;
                const int previous = best_parent[child];
                if (previous < 0 || solver->correlation[parent] < solver->correlation[previous])
                    best_parent[child] = parent;
            }
        }
        for (int child = 0; child < solver->total; ++child) {
            const int parent = best_parent[child];
            if (parent < 0) continue;
            solver->state[child] = 1;
            std::copy_n(&solver->parameters[parent * 6], 6,
                        &solver->parameters[child * 6]);
            const double dx = solver->gx[child] - solver->gx[parent];
            const double dy = solver->gy[child] - solver->gy[parent];
            const double* pp = &solver->parameters[parent * 6];
            solver->parameters[child * 6] += pp[2] * dx + pp[3] * dy;
            solver->parameters[child * 6 + 1] += pp[4] * dx + pp[5] * dy;
        }
    }
    return true;
}

__global__ void plane_fit_kernel(
    const double* vx, const double* vy, const uint8_t* mask,
    int height, int width, int radius,
    double* ux, double* uy, double* vx_out, double* vy_out) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = height * width;
    if (index >= size) return;
    const double nan = CUDART_NAN;
    ux[index] = uy[index] = vx_out[index] = vy_out[index] = nan;
    if (!mask[index]) return;
    const int cy = index / width, cx = index - cy * width;
    double n = 0.0, sx = 0.0, sy = 0.0, sx2 = 0.0, sy2 = 0.0, sxy = 0.0;
    double su = 0.0, sv = 0.0, sux = 0.0, suy = 0.0, svx = 0.0, svy = 0.0;
    for (int dy = -radius; dy <= radius; ++dy) {
        const int y = cy + dy;
        if (y < 0 || y >= height) continue;
        for (int dx = -radius; dx <= radius; ++dx) {
            const int x = cx + dx;
            if (x < 0 || x >= width) continue;
            const int item = y * width + x;
            if (!mask[item]) continue;
            const double u = vx[item], v = vy[item];
            if (!isfinite(u) || !isfinite(v)) continue;
            n += 1.0; sx += dx; sy += dy; sx2 += dx * dx; sy2 += dy * dy; sxy += dx * dy;
            su += u; sv += v; sux += u * dx; suy += u * dy;
            svx += v * dx; svy += v * dy;
        }
    }
    if (n < 6.0) return;
    const double sxx = sx2 - sx * sx / n;
    const double syy = sy2 - sy * sy / n;
    const double sxyc = sxy - sx * sy / n;
    const double suxc = sux - su * sx / n;
    const double suyc = suy - su * sy / n;
    const double svxc = svx - sv * sx / n;
    const double svyc = svy - sv * sy / n;
    const double determinant = sxx * syy - sxyc * sxyc;
    if (!(determinant > 1e-12) || !isfinite(determinant)) return;
    ux[index] = (suxc * syy - suyc * sxyc) / determinant;
    uy[index] = (suyc * sxx - suxc * sxyc) / determinant;
    vx_out[index] = (svxc * syy - svyc * sxyc) / determinant;
    vy_out[index] = (svyc * sxx - svxc * sxyc) / determinant;
}

} // namespace

extern "C" {

const char* strainx_cuda_version(void) { return "2.0.0-native"; }
uint32_t strainx_cuda_abi_version(void) { return 2; }
const char* strainx_cuda_last_error(void) { return g_last_error.c_str(); }

int strainx_cuda_device_count(void) {
    g_last_error.clear();
    int count = 0;
    const cudaError_t status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess) {
        set_error(cudaGetErrorString(status));
        cudaGetLastError();
        return 0;
    }
    return count;
}

int strainx_cuda_synchronize(void) {
    g_last_error.clear();
    return cuda_ok(cudaDeviceSynchronize(), "synchronizing CUDA") ? 0 : -1;
}

int strainx_cuda_memory_info(uint64_t* free_bytes, uint64_t* total_bytes) {
    g_last_error.clear();
    if (!free_bytes || !total_bytes) {
        set_error("CUDA memory-info outputs may not be null."); return -1;
    }
    size_t free_value = 0, total_value = 0;
    if (!cuda_ok(cudaMemGetInfo(&free_value, &total_value),
                 "reading CUDA memory information")) return -1;
    *free_bytes = static_cast<uint64_t>(free_value);
    *total_bytes = static_cast<uint64_t>(total_value);
    return 0;
}

strainx_cuda_solver_t strainx_cuda_solver_create(
    int subset_radius, int subset_spacing, int search_radius, int rescue_radius,
    int max_iterations, double convergence_tolerance, double correlation_cutoff,
    int mask_subsets_to_roi) {
    g_last_error.clear();
    if (subset_radius < 1 || subset_spacing < 1 || max_iterations < 1) {
        set_error("Invalid native CUDA solver parameters."); return nullptr;
    }
    try {
        return new Solver(subset_radius, subset_spacing, search_radius,
                          rescue_radius, max_iterations,
                          convergence_tolerance, correlation_cutoff,
                          mask_subsets_to_roi != 0);
    } catch (const std::exception& error) {
        set_error(error.what()); return nullptr;
    }
}

void strainx_cuda_solver_destroy(strainx_cuda_solver_t handle) {
    delete static_cast<Solver*>(handle);
}

int strainx_cuda_solver_precompute(strainx_cuda_solver_t handle,
                                 const double* reference_image,
                                 const uint8_t* roi_mask,
                                 int height, int width) {
    g_last_error.clear();
    auto* solver = static_cast<Solver*>(handle);
    if (!solver || !reference_image || !roi_mask || height <= 0 || width <= 0) {
        set_error("Invalid native CUDA precompute arguments."); return -1;
    }
    solver->release_device();
    solver->gx.clear(); solver->gy.clear(); solver->valid.clear();
    solver->state.clear(); solver->retry.clear(); solver->parameters.clear();
    solver->correlation.clear(); solver->offsets.clear();
    solver->host_reference.clear(); solver->host_current.clear();
    solver->host_reference_coeff.clear(); solver->host_current_coeff.clear();
    solver->height = height; solver->width = width;
    for (int y = solver->radius; y < height - solver->radius; y += solver->spacing)
        for (int x = solver->radius; x < width - solver->radius; x += solver->spacing) {
            solver->gx.push_back(x); solver->gy.push_back(y);
            solver->valid.push_back(roi_mask[y * width + x] ? 1 : 0);
        }
    solver->grid_h = (height - 2 * solver->radius + solver->spacing - 1) / solver->spacing;
    solver->grid_w = (width - 2 * solver->radius + solver->spacing - 1) / solver->spacing;
    solver->total = static_cast<int>(solver->gx.size());
    if (solver->total <= 0) { set_error("Image is too small for the selected subset radius."); return -1; }
    for (int dy = -solver->radius; dy <= solver->radius; ++dy)
        for (int dx = -solver->radius; dx <= solver->radius; ++dx)
            if (dx * dx + dy * dy <= solver->radius * solver->radius)
                solver->offsets.push_back(make_int2(dx, dy));
    solver->n_pixels = static_cast<int>(solver->offsets.size());
    solver->state.assign(solver->total, 0);
    solver->retry.assign(solver->total, 0);
    solver->parameters.assign(static_cast<size_t>(solver->total) * 6, 0.0);
    solver->correlation.assign(solver->total, std::numeric_limits<double>::quiet_NaN());
    solver->host_reference.assign(reference_image, reference_image + static_cast<size_t>(height) * width);
    solver->reference_peak = image_peak(
        reference_image, static_cast<size_t>(height) * width);
    solver->current_peak = 0.0;
    solver->intensity_scale = infer_intensity_scale(solver->reference_peak);
    solver->host_reference_coeff = cubic_spline_coefficients(
        reference_image, height, width);
    solver->host_current.resize(static_cast<size_t>(height) * width);
    solver->host_current_coeff.resize(static_cast<size_t>(height) * width);
    const size_t image_bytes = static_cast<size_t>(height) * width * sizeof(double);
    const size_t active_p_bytes = static_cast<size_t>(solver->total) * 6 * sizeof(double);
    bool ok =
        cuda_ok(cudaMalloc(&solver->d_reference, image_bytes), "allocating reference image") &&
        cuda_ok(cudaMalloc(&solver->d_current, image_bytes), "allocating current image") &&
        cuda_ok(cudaMalloc(&solver->d_reference_coeff, image_bytes), "allocating reference spline coefficients") &&
        cuda_ok(cudaMalloc(&solver->d_current_coeff, image_bytes), "allocating current spline coefficients") &&
        cuda_ok(cudaMalloc(&solver->d_grad_x, image_bytes), "allocating x gradient") &&
        cuda_ok(cudaMalloc(&solver->d_grad_y, image_bytes), "allocating y gradient") &&
        cuda_ok(cudaMalloc(&solver->d_roi_mask,
                           static_cast<size_t>(height) * width),
                "allocating ROI mask") &&
        cuda_ok(cudaMalloc(&solver->d_grid_x, solver->total * sizeof(int)), "allocating grid x") &&
        cuda_ok(cudaMalloc(&solver->d_grid_y, solver->total * sizeof(int)), "allocating grid y") &&
        cuda_ok(cudaMalloc(&solver->d_offsets, solver->offsets.size() * sizeof(int2)), "allocating subset offsets") &&
        cuda_ok(cudaMalloc(&solver->d_active_indices, solver->total * sizeof(int)), "allocating active indices") &&
        cuda_ok(cudaMalloc(&solver->d_active_parameters, active_p_bytes), "allocating active parameters") &&
        cuda_ok(cudaMalloc(&solver->d_result_parameters, active_p_bytes), "allocating result parameters") &&
        cuda_ok(cudaMalloc(&solver->d_result_correlation, solver->total * sizeof(double)), "allocating correlations") &&
        cuda_ok(cudaMalloc(&solver->d_result_accepted, solver->total * sizeof(uint8_t)), "allocating statuses") &&
        cuda_ok(cudaMemcpy(solver->d_reference, reference_image, image_bytes, cudaMemcpyHostToDevice), "uploading reference image") &&
        cuda_ok(cudaMemcpy(solver->d_reference_coeff,
                           solver->host_reference_coeff.data(), image_bytes,
                           cudaMemcpyHostToDevice),
                "uploading reference spline coefficients") &&
        cuda_ok(cudaMemcpy(solver->d_roi_mask, roi_mask,
                           static_cast<size_t>(height) * width,
                           cudaMemcpyHostToDevice), "uploading ROI mask") &&
        cuda_ok(cudaMemcpy(solver->d_grid_x, solver->gx.data(), solver->total * sizeof(int), cudaMemcpyHostToDevice), "uploading grid x") &&
        cuda_ok(cudaMemcpy(solver->d_grid_y, solver->gy.data(), solver->total * sizeof(int), cudaMemcpyHostToDevice), "uploading grid y") &&
        cuda_ok(cudaMemcpy(solver->d_offsets, solver->offsets.data(), solver->offsets.size() * sizeof(int2), cudaMemcpyHostToDevice), "uploading subset offsets");
    if (!ok || !launch_gradient(solver)) { solver->release_device(); return -1; }
    solver->initialized = true;
    return 0;
}

int strainx_cuda_solver_ncc(
    strainx_cuda_solver_t handle, const double* current_image, int grid_index,
    double guess_u, double guess_v, double* out_u, double* out_v,
    double* out_zncc) {
    g_last_error.clear();
    auto* solver = static_cast<Solver*>(handle);
    if (!solver || !solver->initialized || !current_image || !out_u ||
        !out_v || !out_zncc || grid_index < 0 ||
        grid_index >= solver->total || !solver->valid[grid_index]) {
        set_error("Invalid native CUDA NCC arguments."); return -1;
    }
    const size_t count = static_cast<size_t>(solver->height) * solver->width;
    if (!upload_current_with_coefficients(
            solver, current_image, "uploading NCC current image")) return -1;
    solver->has_current = true;
    return ncc_guess(solver, grid_index, guess_u, guess_v,
                     *out_u, *out_v, out_zncc) ? 0 : -1;
}

int strainx_cuda_solver_icgn(
    strainx_cuda_solver_t handle, const double* current_image, int grid_index,
    const double* initial_parameters, double* out_parameters,
    double* out_znssd, uint8_t* out_accepted) {
    g_last_error.clear();
    auto* solver = static_cast<Solver*>(handle);
    if (!solver || !solver->initialized || !current_image ||
        !initial_parameters || !out_parameters || !out_znssd ||
        !out_accepted || grid_index < 0 || grid_index >= solver->total ||
        !solver->valid[grid_index]) {
        set_error("Invalid native CUDA IC-GN arguments."); return -1;
    }
    if (!upload_current_with_coefficients(
            solver, current_image, "uploading isolated IC-GN current image"))
        return -1;
    if (!cuda_ok(cudaMemcpy(solver->d_active_indices, &grid_index, sizeof(int),
                            cudaMemcpyHostToDevice),
                 "uploading isolated IC-GN index") ||
        !cuda_ok(cudaMemcpy(solver->d_active_parameters, initial_parameters,
                            6 * sizeof(double), cudaMemcpyHostToDevice),
                 "uploading isolated IC-GN parameters")) return -1;
    solve_subsets_block_kernel<<<1, 128>>>(
        solver->d_reference, solver->d_current,
        solver->d_current_coeff,
        solver->d_grad_x, solver->d_grad_y,
        solver->mask_subsets_to_roi ? solver->d_roi_mask : nullptr,
        solver->intensity_scale,
        solver->height, solver->width, solver->d_grid_x, solver->d_grid_y,
        solver->d_offsets, solver->n_pixels, solver->d_active_indices,
        solver->d_active_parameters, 1, solver->max_iter, solver->conv_tol,
        solver->corr_cutoff, 0, solver->spacing, solver->radius, 0,
        solver->d_result_parameters, solver->d_result_correlation,
        solver->d_result_accepted);
    if (!cuda_ok(cudaGetLastError(), "launching isolated native IC-GN") ||
        !cuda_ok(cudaMemcpy(out_parameters, solver->d_result_parameters,
                            6 * sizeof(double), cudaMemcpyDeviceToHost),
                 "reading isolated IC-GN parameters") ||
        !cuda_ok(cudaMemcpy(out_znssd, solver->d_result_correlation,
                            sizeof(double), cudaMemcpyDeviceToHost),
                 "reading isolated IC-GN correlation") ||
        !cuda_ok(cudaMemcpy(out_accepted, solver->d_result_accepted,
                            sizeof(uint8_t), cudaMemcpyDeviceToHost),
                 "reading isolated IC-GN status")) return -1;
    return 0;
}

int strainx_cuda_solver_solve(
    strainx_cuda_solver_t handle, const double* current_image, int mode,
    int seed_index, double guess_u, double guess_v,
    double* out_u, double* out_v, double* out_du_dx, double* out_du_dy,
    double* out_dv_dx, double* out_dv_dy, double* out_correlation) {
    g_last_error.clear();
    auto* solver = static_cast<Solver*>(handle);
    if (!solver || !solver->initialized || !out_u || !out_v || !out_du_dx ||
        !out_du_dy || !out_dv_dx || !out_dv_dy || !out_correlation) {
        set_error("Invalid native CUDA solve arguments."); return -1;
    }
    try {
        const size_t count = static_cast<size_t>(solver->height) * solver->width;
        if (current_image) {
            if (!upload_current_with_coefficients(
                    solver, current_image, "uploading current image")) return -1;
            solver->has_current = true;
        } else if (!solver->has_current) {
            throw std::runtime_error("No resident current image for recovery.");
        }
        if (mode == STRAINX_CUDA_FRESH) {
            if (seed_index < 0 || seed_index >= solver->total || !solver->valid[seed_index])
                throw std::runtime_error("The native CUDA seed is outside the valid grid.");
            initialize_fresh(solver, seed_index, guess_u, guess_v);
        } else if (mode == STRAINX_CUDA_WARM_START) {
            initialize_warm_start(solver);
        } else if (mode == STRAINX_CUDA_RECOVER_FAILED) {
            initialize_recovery(solver, guess_u, guess_v);
        } else {
            throw std::runtime_error("Unknown native CUDA solve mode.");
        }
        if (!run_wavefront(solver, mode == STRAINX_CUDA_WARM_START)) return -1;
        const double nan = std::numeric_limits<double>::quiet_NaN();
        std::fill_n(out_u, count, nan); std::fill_n(out_v, count, nan);
        std::fill_n(out_du_dx, count, nan); std::fill_n(out_du_dy, count, nan);
        std::fill_n(out_dv_dx, count, nan); std::fill_n(out_dv_dy, count, nan);
        std::fill_n(out_correlation, count, nan);
        for (int i = 0; i < solver->total; ++i) {
            if (solver->state[i] != 2) continue;
            const int pixel = solver->gy[i] * solver->width + solver->gx[i];
            const double* p = &solver->parameters[i * 6];
            out_u[pixel] = p[0]; out_v[pixel] = p[1];
            out_du_dx[pixel] = p[2]; out_du_dy[pixel] = p[3];
            out_dv_dx[pixel] = p[4]; out_dv_dy[pixel] = p[5];
            out_correlation[pixel] = solver->correlation[i];
        }
        return 0;
    } catch (const std::exception& error) {
        set_error(error.what()); return -1;
    }
}

int strainx_cuda_solver_update_reference(strainx_cuda_solver_t handle,
                                       const double* new_reference_image) {
    g_last_error.clear();
    auto* solver = static_cast<Solver*>(handle);
    if (!solver || !solver->initialized) { set_error("Native CUDA solver is not initialized."); return -1; }
    const size_t count = static_cast<size_t>(solver->height) * solver->width;
    if (new_reference_image) {
        solver->host_reference.assign(new_reference_image, new_reference_image + count);
        solver->reference_peak = image_peak(new_reference_image, count);
        solver->host_reference_coeff = cubic_spline_coefficients(
            new_reference_image, solver->height, solver->width);
        if (!cuda_ok(cudaMemcpy(solver->d_reference, new_reference_image,
                                count * sizeof(double), cudaMemcpyHostToDevice),
                     "uploading updated reference") ||
            !cuda_ok(cudaMemcpy(solver->d_reference_coeff,
                                solver->host_reference_coeff.data(),
                                count * sizeof(double), cudaMemcpyHostToDevice),
                     "uploading updated reference spline coefficients")) return -1;
    } else {
        if (!solver->has_current) { set_error("No resident current image to promote."); return -1; }
        solver->host_reference.swap(solver->host_current);
        solver->host_reference_coeff.swap(solver->host_current_coeff);
        std::swap(solver->d_reference, solver->d_current);
        std::swap(solver->d_reference_coeff, solver->d_current_coeff);
        solver->reference_peak = solver->current_peak;
    }
    solver->current_peak = 0.0;
    solver->intensity_scale = infer_intensity_scale(solver->reference_peak);
    solver->has_current = false;
    return launch_gradient(solver) ? 0 : -1;
}

int strainx_cuda_plane_fit(
    const double* vx, const double* vy, const uint8_t* mask,
    int height, int width, int radius,
    double* out_dvx_dx, double* out_dvx_dy,
    double* out_dvy_dx, double* out_dvy_dy) {
    g_last_error.clear();
    if (!vx || !vy || !mask || height <= 0 || width <= 0 || radius < 1 ||
        !out_dvx_dx || !out_dvx_dy || !out_dvy_dx || !out_dvy_dy) {
        set_error("Invalid native CUDA plane-fit arguments."); return -1;
    }
    const size_t count = static_cast<size_t>(height) * width;
    const size_t bytes = count * sizeof(double);
    double *d_vx = nullptr, *d_vy = nullptr, *d_ux = nullptr, *d_uy = nullptr,
           *d_vx_out = nullptr, *d_vy_out = nullptr;
    uint8_t* d_mask = nullptr;
    auto cleanup = [&]() {
        cudaFree(d_vx); cudaFree(d_vy); cudaFree(d_mask);
        cudaFree(d_ux); cudaFree(d_uy); cudaFree(d_vx_out); cudaFree(d_vy_out);
    };
    bool ok = cuda_ok(cudaMalloc(&d_vx, bytes), "allocating plane-fit Vx") &&
              cuda_ok(cudaMalloc(&d_vy, bytes), "allocating plane-fit Vy") &&
              cuda_ok(cudaMalloc(&d_mask, count), "allocating plane-fit mask") &&
              cuda_ok(cudaMalloc(&d_ux, bytes), "allocating plane-fit dVx/dx") &&
              cuda_ok(cudaMalloc(&d_uy, bytes), "allocating plane-fit dVx/dy") &&
              cuda_ok(cudaMalloc(&d_vx_out, bytes), "allocating plane-fit dVy/dx") &&
              cuda_ok(cudaMalloc(&d_vy_out, bytes), "allocating plane-fit dVy/dy") &&
              cuda_ok(cudaMemcpy(d_vx, vx, bytes, cudaMemcpyHostToDevice), "uploading plane-fit Vx") &&
              cuda_ok(cudaMemcpy(d_vy, vy, bytes, cudaMemcpyHostToDevice), "uploading plane-fit Vy") &&
              cuda_ok(cudaMemcpy(d_mask, mask, count, cudaMemcpyHostToDevice), "uploading plane-fit mask");
    if (ok) {
        const int threads = 256;
        plane_fit_kernel<<<static_cast<int>((count + threads - 1) / threads), threads>>>(
            d_vx, d_vy, d_mask, height, width, radius,
            d_ux, d_uy, d_vx_out, d_vy_out);
        ok = cuda_ok(cudaGetLastError(), "launching native plane fit") &&
             cuda_ok(cudaMemcpy(out_dvx_dx, d_ux, bytes, cudaMemcpyDeviceToHost), "reading dVx/dx") &&
             cuda_ok(cudaMemcpy(out_dvx_dy, d_uy, bytes, cudaMemcpyDeviceToHost), "reading dVx/dy") &&
             cuda_ok(cudaMemcpy(out_dvy_dx, d_vx_out, bytes, cudaMemcpyDeviceToHost), "reading dVy/dx") &&
             cuda_ok(cudaMemcpy(out_dvy_dy, d_vy_out, bytes, cudaMemcpyDeviceToHost), "reading dVy/dy");
    }
    cleanup();
    return ok ? 0 : -1;
}

} // extern "C"
