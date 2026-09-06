"""The data-driven muon production model (moved from the flat ``MCEq.ddm``).

``ddm.py`` holds the spline database and the model class, ``ddm_utils.py`` the
matrix-generation helpers; both moved here verbatim by §13.2 item 7, with the
flat ``MCEq.ddm`` / ``MCEq.ddm_utils`` left as re-export shims (D14, deleted by
Phase 7). Names are reached through the submodules, not re-exported here, so
``automodapi`` and the surface test keep seeing exactly the old flat surface.
"""
