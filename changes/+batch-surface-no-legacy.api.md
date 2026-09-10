``MCEqBatchResult`` **no longer unpacks as a tuple** (the legacy
``__iter__``/``__len__``/``__getitem__`` over the producing method's old
return shape are gone). Read the named attributes instead:
``res.sol``, ``res.grid_sol``, ``res.nsteps_per_col``. A new
``res.shared_path`` boolean names what the tuple layout used to encode
implicitly (``grid_sol`` is only ever populated when it is True).

``solve_multirhs`` now emits the ``DeprecationWarning`` its docstring has
always claimed (``-W error::DeprecationWarning`` catches it).

``MCEqRun.solve_batch`` / ``solve_multirhs`` / ``solve_fullsky`` /
``_build_condition_paths`` are now direct bindings of the
:mod:`MCEq.driver.batch` / :mod:`MCEq.driver.paths` functions instead of
per-method delegators. The public surface is identical; the fullsky 2-D
phi0 cutoff warning lost one stack frame (``stacklevel=2`` attributes it
to the caller line, verified).
