"""Structural types shared by the channel-table classes.

A *channel table* is any object the species layer can wire itself from:
it exposes the parent list, the parent→children relations, and a matrix
lookup, and it supports membership tests. ``MCEqParticle`` consumes
channel tables through this interface only
(:meth:`MCEq.species.particle.MCEqParticle._set_channels`), which lets the
hadronic and decay wiring share one routine.

Two shipped classes satisfy the protocol structurally — no inheritance
is introduced anywhere:

* ``MCEq.data.interaction_tables.Interactions`` — 1D matrices and, on
  2D runs, ``(n_k, dim, dim)`` blocks. Keys are ``(pdg, helicity)``
  tuples; ``get_matrix`` applies the absolute channel overrides before the
  ``mod_pprod`` production modifications on the way out.
* ``MCEq.data.decay_tables.Decays`` — plain decay matrices, same key
  shape.

The cross-section and continuous-loss tables are single-parent lookup
tables (``get_cs`` / ``__getitem__``, no child axis), so they live
outside this protocol deliberately: "channel" here means a two-index
yield matrix, which is hadronic/decay domain.

The protocol is ``runtime_checkable`` so tests can assert membership of
a table object structurally (``isinstance`` checks member presence, not
signatures).
"""

from typing import Any, Dict, List, Protocol, runtime_checkable

__all__ = ["ChannelTable"]


@runtime_checkable
class ChannelTable(Protocol):
    """Anything ``MCEqParticle`` can wire hadronic/decay channels from."""

    #: Parent keys the table knows, in load order. Tuple ``(pdg,
    #: helicity)`` on the shipped tables; ``key in table`` iff ``key in
    #: parents``.
    parents: List[Any]

    #: Parent key → list of child keys, i.e. the channels to wire.
    relations: Dict[Any, List[Any]]

    def get_matrix(self, parent: Any, child: Any) -> Any:
        """Yield/decay matrix for one channel (1D ``(dim, dim)``, or
        ``(n_k, dim, dim)`` blocks on 2D runs)."""
        ...

    def __contains__(self, key: Any) -> bool:
        """``True`` if *key* is a parent of this table."""
        ...
