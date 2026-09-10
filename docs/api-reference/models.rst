.. _models:

************************************************
models (:mod:`MCEq.models`)
************************************************
.. currentmodule:: MCEq.models


Physics plug-ins that modify yields, imported by the driver only
(contract C4). The data-driven muon model lives under ``models/ddm``;
import it from ``MCEq.models.ddm.ddm`` / ``MCEq.models.ddm.ddm_utils``
(the flat ``MCEq.ddm`` / ``MCEq.ddm_utils`` shims were removed at 2.0;
see :doc:`/migration_v2_layout`).

Reference/API
=============
.. automodule:: MCEq.models.ddm.ddm
   :members:
.. automodule:: MCEq.models.ddm.ddm_utils
   :members:
