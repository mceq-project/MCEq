"""Bug-ledger pins for the config package (ledger B26).

B26 was found by the 2026-09-02 audit; Phase 6 pins it because the Phase 7
decision (make ``config['x'] = v`` forward to the module attribute, or
document the no-op and deprecate the shim) needs a recorded baseline.
"""

import pytest

import MCEq.config as cfg


def test_b26_dict_write_updates_the_dict_but_never_the_attribute():
    """B26: ``config['x'] = v`` writes only into the dict superclass.

    ``MCEqConfigCompatibility.__init__`` copies the module namespace into
    ``self.__dict__`` — the dict itself starts empty, and
    ``__setitem__`` delegates to ``dict.__setitem__`` — while every flat
    ``config.x`` read goes through the module attribute the dict write never
    touches. The deprecated dictionary assignment has therefore been a no-op
    for the access it claims to serve since it was introduced (the audit's
    wording "writes only into the dict superclass" is exact). Pinned as-is;
    the forward/no-op ruling is Phase 7 (D14).
    """
    assert dict(cfg.config) == {}  # __init__ filled __dict__, not the dict
    original = cfg.print_module
    flipped = not original
    try:
        cfg.config["print_module"] = flipped
        assert cfg.config["print_module"] is flipped  # dict sees it
        assert cfg.print_module is original  # the attribute does not
    finally:
        del cfg.config["print_module"]  # leave the dict as found


def test_b26_unknown_key_raises_and_keys_are_lowered():
    """B26 legs (ii)/(iii): unknown key raises ``Exception``, keys lower-cased.

    ``__setitem__`` raises ``Exception('Unknown config key', key)`` (not
    ``KeyError``) for a name absent from the ``__dict__`` snapshot, and
    silently lower-cases before the membership test.
    """
    with pytest.raises(Exception, match="Unknown config key"):
        cfg.config["definitely_not_a_config_key"] = 1
    cfg.config["PRINT_MODULE"] = cfg.print_module  # accepted via lower()
    try:
        assert "print_module" in cfg.config
    finally:
        del cfg.config["print_module"]
