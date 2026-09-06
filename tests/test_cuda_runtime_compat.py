from strainx.core.cuda_native import NativeCudaError, NativeCudaSolver


def test_neighbour_recovery_falls_back_for_an_older_native_runtime():
    solver = object.__new__(NativeCudaSolver)
    solver._neighbour_recovery_supported = True
    calls = []
    expected = ("recovered",)

    def solve(_current, mode, _seed, _u, _v):
        calls.append(mode)
        if mode == NativeCudaSolver.RECOVER_NEIGHBOURS:
            raise NativeCudaError("Unknown native CUDA solve mode.")
        return expected

    solver._solve = solve

    assert solver.recover_failed(strategy="neighbour") is expected
    assert calls == [NativeCudaSolver.RECOVER_NEIGHBOURS,
                     NativeCudaSolver.RECOVER_FAILED]
    assert not solver._neighbour_recovery_supported

    calls.clear()
    assert solver.recover_failed(strategy="neighbour") is expected
    assert calls == [NativeCudaSolver.RECOVER_FAILED]


def test_neighbour_recovery_does_not_hide_unrelated_native_errors():
    solver = object.__new__(NativeCudaSolver)
    solver._neighbour_recovery_supported = True

    def fail(*_args):
        raise NativeCudaError("CUDA launch failed")

    solver._solve = fail

    try:
        solver.recover_failed(strategy="neighbour")
    except NativeCudaError as exc:
        assert str(exc) == "CUDA launch failed"
    else:
        raise AssertionError("unrelated CUDA errors must propagate")
