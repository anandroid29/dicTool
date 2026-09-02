#pragma once

#include <stdint.h>

#if defined(_WIN32)
#  if defined(STRAINX_CUDA_EXPORTS)
#    define STRAINX_CUDA_API __declspec(dllexport)
#  else
#    define STRAINX_CUDA_API __declspec(dllimport)
#  endif
#else
#  define STRAINX_CUDA_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef void* strainx_cuda_solver_t;

enum strainx_cuda_solve_mode {
    STRAINX_CUDA_FRESH = 0,
    STRAINX_CUDA_WARM_START = 1,
    STRAINX_CUDA_RECOVER_FAILED = 2,
};

STRAINX_CUDA_API const char* strainx_cuda_version(void);
STRAINX_CUDA_API uint32_t strainx_cuda_abi_version(void);
STRAINX_CUDA_API const char* strainx_cuda_last_error(void);
STRAINX_CUDA_API int strainx_cuda_device_count(void);
STRAINX_CUDA_API int strainx_cuda_synchronize(void);
STRAINX_CUDA_API int strainx_cuda_memory_info(uint64_t* free_bytes,
                                          uint64_t* total_bytes);

STRAINX_CUDA_API strainx_cuda_solver_t strainx_cuda_solver_create(
    int subset_radius,
    int subset_spacing,
    int search_radius,
    int rescue_radius,
    int max_iterations,
    double convergence_tolerance,
    double correlation_cutoff,
    int mask_subsets_to_roi);

STRAINX_CUDA_API void strainx_cuda_solver_destroy(strainx_cuda_solver_t solver);

STRAINX_CUDA_API int strainx_cuda_solver_precompute(
    strainx_cuda_solver_t solver,
    const double* reference_image,
    const uint8_t* roi_mask,
    int height,
    int width);

/* Run only the integer-pixel square-template ZNCC seed search. */
STRAINX_CUDA_API int strainx_cuda_solver_ncc(
    strainx_cuda_solver_t solver,
    const double* current_image,
    int grid_index,
    double guess_u,
    double guess_v,
    double* out_u,
    double* out_v,
    double* out_zncc);

/* Run one IC-GN subset from caller-supplied parameters, with no NCC,
 * propagation, rescue search, or warm-start state transition. */
STRAINX_CUDA_API int strainx_cuda_solver_icgn(
    strainx_cuda_solver_t solver,
    const double* current_image,
    int grid_index,
    const double* initial_parameters,
    double* out_parameters,
    double* out_znssd,
    uint8_t* out_accepted);

/*
 * current_image may be NULL only for STRAINX_CUDA_RECOVER_FAILED, which reuses
 * the current frame already resident in the solver. Fresh mode performs its
 * integer NCC seed search inside the native runtime around (guess_u, guess_v).
 */
STRAINX_CUDA_API int strainx_cuda_solver_solve(
    strainx_cuda_solver_t solver,
    const double* current_image,
    int mode,
    int seed_index,
    double guess_u,
    double guess_v,
    double* out_u,
    double* out_v,
    double* out_du_dx,
    double* out_du_dy,
    double* out_dv_dx,
    double* out_dv_dy,
    double* out_correlation);

/* Passing NULL reuses the current device/host image without another upload. */
STRAINX_CUDA_API int strainx_cuda_solver_update_reference(
    strainx_cuda_solver_t solver,
    const double* new_reference_image);

STRAINX_CUDA_API int strainx_cuda_plane_fit(
    const double* vx,
    const double* vy,
    const uint8_t* component_mask,
    int height,
    int width,
    int radius,
    double* out_dvx_dx,
    double* out_dvx_dy,
    double* out_dvy_dx,
    double* out_dvy_dy);

#ifdef __cplusplus
}
#endif
