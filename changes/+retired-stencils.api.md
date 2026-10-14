The ``"biased"`` and ``"centered"`` choices of ``config.loss_stencil_method`` are
removed; the default ``"expfit_low_upwind2"`` and the ``"expfit_low_upwind"``,
``"expfit"``, ``"upwind"`` and ``"upwind2"`` stencils remain. The unused Legendre
helpers ``moments``, ``forward_weights`` and ``hemisphere_projection`` are removed
from ``MCEq.operators.legendre``.
