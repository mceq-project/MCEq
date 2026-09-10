The ``expfit_low_upwind``/``expfit_low_upwind2`` composites now widen their
monotone low-energy layer per species. With ``generic_losses_all_charged`` on
(ships default-True), the shared 8-row layer left the freshly-equipped hadron
blocks in the unstable regime of the explicit loss stencil under the
diagonal-only ETD split, and a coupled wide-grid run blew up from round-off
(p+ first, around X=440 at ``dX_max=1``; muon block later with the flag off).
``cont_loss_operator`` now builds the differential operator per particle with
the smallest upwind layer covering every row whose
``dX_max * |dEdX| / (E * dlnE)`` reaches 0.5 (floor 8, cap dim-2), one
operator per distinct row count, cached. The shared ``op_matrix``, every
other stencil family, and every grid where the criterion stays under the
floor (all golden fixtures, the shipped e_min=0.1 default) are bitwise
unchanged.
