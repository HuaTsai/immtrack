import numpy as np

from immtrack import CtrvBBox3DUKF


def main() -> None:
    ukf = CtrvBBox3DUKF()
    x0 = np.zeros(10)
    x0[7] = 1.0  # v = 1 m/s
    ukf.init(state=x0, cov=np.eye(10))
    ukf.predict(dt=0.1)
    print(f"state after one predict: {ukf.state}")


if __name__ == "__main__":
    main()
