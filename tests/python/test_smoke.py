import numpy as np
import pytest

from immtrack import CtrvBBox3DUKF, CvBBox3DUKF

FILTERS = [CvBBox3DUKF, CtrvBBox3DUKF]


@pytest.mark.parametrize("filter_cls", FILTERS)
def test_default_construction(filter_cls: type) -> None:
    f = filter_cls()
    assert f.state.shape == (10,)
    assert f.covariance.shape == (10, 10)
    np.testing.assert_array_equal(f.state, np.zeros(10))
    np.testing.assert_array_equal(f.covariance, np.eye(10))


@pytest.mark.parametrize("filter_cls", FILTERS)
def test_init(filter_cls: type) -> None:
    f = filter_cls()
    x0 = np.arange(10, dtype=np.float64)
    p0 = np.eye(10) * 2.0
    f.init(state=x0, cov=p0)
    np.testing.assert_array_equal(f.state, x0)
    np.testing.assert_array_equal(f.covariance, p0)


def test_cv_predict_advances_position() -> None:
    f = CvBBox3DUKF()
    x0 = np.zeros(10)
    x0[7] = 1.0  # vx = 1 m/s
    f.init(state=x0, cov=np.eye(10))
    f.predict(dt=0.5)
    assert f.state[0] == pytest.approx(0.5)
    assert f.state[1] == 0.0
    assert f.state[2] == 0.0


def test_ctrv_predict_straight_line() -> None:
    f = CtrvBBox3DUKF()
    x0 = np.zeros(10)
    x0[7] = 2.0  # v = 2 m/s
    # yaw = 0, yaw_rate = 0 → moves along +x
    f.init(state=x0, cov=np.eye(10))
    f.predict(dt=1.0)
    assert f.state[0] == pytest.approx(2.0)
    assert f.state[1] == pytest.approx(0.0)


def test_update_not_implemented() -> None:
    f = CtrvBBox3DUKF()
    f.init(state=np.zeros(10), cov=np.eye(10))
    with pytest.raises(RuntimeError):
        f.update(measurement=np.zeros(7))
