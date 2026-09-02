#pragma once

#include <stdint.h>

#if defined(_WIN32)
#  if defined(PYDIC_CUDA_EXPORTS)
#    define PYDIC_CUDA_API __declspec(dllexport)
#  else
#    define PYDIC_CUDA_API __declspec(dllimport)
#  endif
#else
#  define PYDIC_CUDA_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef void* pydic_cuda_solver_t;

enum pydic_cuda_solve_mode {
    PYDIC_CUDA_FRESH = 0,
    PYDIC_CUDA_WARM_START = 1,
    PYDIC_CUDA_RECOVER_FAILED = 2,
};

PYDIC_CUDA_API const char* pydic_cuda_version(void);
PYDIC_CUDA_API uint32_t pydic_cuda_abi_version(void);
PYDIC_CUDA_API const char* pydic_cuda_last_error(void);
PYDIC_CUDA_API int pydic_cuda_device_count(void);
PYDIC_CUDA_API int pydic_cuda_synchronize(void);
PYDIC_CUDA_API int pydic_cuda_memory_info(uint64_t* free_bytes,
                                          uint64_t* total_bytes);

PYDIC_CUDA_API pydic_cuda_solver_t pydic_cuda_solver_create(
    int subset_radius,
    int subset_spacing,
    int search_radius,
    int rescue_radius,
    int max_iterations,
    double convergence_tolerance,
    double correlation_cutoff,
    int mask_subsets_to_roi);

PYDIC_CUDA_API void pydic_cuda_solver_destroy(pydic_cuda_solver_t solver);

PYDIC_CUDA_API int pydic_cuda_solver_precompute(
    pydic_cuda_solver_t solver,
    const double* reference_image,
    const uint8_t* roi_mask,
    int height,
    int width);

/* Run only the integer-pixel square-template ZNCC seed search. */
PYDIC_CUDA_API int pydic_cuda_solver_ncc(
    pydic_cuda_solver_t solver,
    const double* current_image,
    int grid_index,
    double guess_u,
    double guess_v,
    double* out_u,
    double* out_v,
    double* out_zncc);

/* Run one IC-GN subset from caller-supplied parameters, with no NCC,
 * propagation, rescue search, or warm-start state transition. */
PYDIC_CUDA_API int pydic_cuda_solver_icgn(
    pydic_cuda_solver_t solver,
    const double* current_image,
    int grid_index,
    const double* initial_parameters,
    double* out_parameters,
    double* out_znssd,
    uint8_t* out_accepted);

/*
 * current_image may be NULL only for PYDIC_CUDA_RECOVER_FAILED, which reuses
 * the current frame already resident in the solver. Fresh mode performs its
 * integer NCC seed search inside the native runtime around (guess_u, guess_v).
 */
PYDIC_CUDA_API int pydic_cuda_solver_solve(
    pydic_cuda_solver_t solver,
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
PYDIC_CUDA_API int pydic_cuda_solver_update_reference(
    pydic_cuda_solver_t solver,
    const double* new_reference_image);

PYDIC_CUDA_API int pydic_cuda_plane_fit(
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
