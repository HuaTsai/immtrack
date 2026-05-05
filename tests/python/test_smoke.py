import numpy as np
import pytest

from immtrack import (
    CovarianceNotPsd,
    InvalidArgument,
    NumericalError,
    UkfPosVxyzYawCV,
)


def test_default_construction() -> None:
    f = UkfPosVxyzYawCV()
    assert f.state.shape == (8,)
    assert f.covariance.shape == (8, 8)
    np.testing.assert_array_equal(f.state, np.zeros(8))
    np.testing.assert_array_equal(f.covariance, np.eye(8))


def test_class_level_dimensions() -> None:
    assert UkfPosVxyzYawCV.N == 8
    assert UkfPosVxyzYawCV.M == 4


def test_init_round_trip() -> None:
    f = UkfPosVxyzYawCV()
    x0 = np.arange(8, dtype=np.float64)
    p0 = np.eye(8) * 2.0
    f.init(state=x0, cov=p0)
    np.testing.assert_array_equal(f.state, x0)
    np.testing.assert_array_equal(f.covariance, p0)


def test_predict_advances_position_under_cv() -> None:
    f = UkfPosVxyzYawCV()
    x0 = np.zeros(8)
    x0[3] = 1.0
    f.init(state=x0, cov=np.eye(8))
    f.predict(dt=0.5)
    assert f.state[0] == pytest.approx(0.5)
    assert f.state[1] == pytest.approx(0.0)
    assert f.state[2] == pytest.approx(0.0)


def test_predict_rejects_negative_dt() -> None:
    f = UkfPosVxyzYawCV()
    f.init(state=np.zeros(8), cov=np.eye(8))
    with pytest.raises(ValueError):
        f.predict(dt=-0.1)


def test_update_returns_nis_and_shrinks_cov() -> None:
    f = UkfPosVxyzYawCV()
    f.init(state=np.zeros(8), cov=np.eye(8))

    trace_before = np.trace(f.covariance)
    nis = f.update(measurement=np.zeros(4))
    trace_after = np.trace(f.covariance)

    assert nis >= 0.0
    assert trace_after < trace_before


def test_exception_classes_importable() -> None:
    assert issubclass(InvalidArgument, ValueError)
    assert issubclass(CovarianceNotPsd, RuntimeError)
    assert issubclass(NumericalError, RuntimeError)


def test_predict_with_nonpsd_cov_raises_covariance_not_psd() -> None:
    f = UkfPosVxyzYawCV()
    f.init(state=np.zeros(8), cov=-np.eye(8))
    with pytest.raises(CovarianceNotPsd):
        f.predict(dt=0.1)


def test_invalid_argument_is_value_error() -> None:
    f = UkfPosVxyzYawCV()
    f.init(state=np.zeros(8), cov=np.eye(8))
    with pytest.raises(ValueError):
        f.predict(dt=-1.0)
