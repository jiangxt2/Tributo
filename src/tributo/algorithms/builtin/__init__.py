"""First-party implementations of the formal distributed algorithm SPI."""

from tributo.algorithms.builtin.multinomial_nb import (
    MULTINOMIAL_NB_DESCRIPTOR as MULTINOMIAL_NB_DESCRIPTOR,
)
from tributo.algorithms.builtin.multinomial_nb import (
    MULTINOMIAL_NB_REGISTRATION as MULTINOMIAL_NB_REGISTRATION,
)
from tributo.algorithms.builtin.multinomial_nb import (
    DistributedMultinomialNB,
)
from tributo.algorithms.builtin.torch_collective import (
    DNN_DESCRIPTOR as DNN_DESCRIPTOR,
)
from tributo.algorithms.builtin.torch_collective import (
    DNN_REGISTRATION as DNN_REGISTRATION,
)
from tributo.algorithms.builtin.torch_collective import (
    PU_DESCRIPTOR as PU_DESCRIPTOR,
)
from tributo.algorithms.builtin.torch_collective import (
    PU_REGISTRATION as PU_REGISTRATION,
)
from tributo.algorithms.builtin.torch_collective import (
    DistributedDNN,
    DistributedPU,
)
from tributo.algorithms.builtin.xgboost_native import (
    XGBOOST_DESCRIPTOR as XGBOOST_DESCRIPTOR,
)
from tributo.algorithms.builtin.xgboost_native import (
    XGBOOST_REGISTRATION as XGBOOST_REGISTRATION,
)
from tributo.algorithms.builtin.xgboost_native import (
    DistributedXGBoost,
)

__all__ = [
    "DistributedMultinomialNB",
    "DistributedDNN",
    "DistributedPU",
    "DistributedXGBoost",
]
