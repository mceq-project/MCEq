`misc.caller_name` fetches its frame with `sys._getframe` instead of
`inspect.stack()`, which materialises `FrameInfo` records with source context
for the whole stack. `info()` reaches it on every call while
`override_debug_fcn` is set, so the saving is large where it matters: per call,
1.60 us instead of 157.69 us suppressed with an override list active, 1.95 vs
183.95 emitted, and 3.33 vs 368.78 when the override promotes the level. The
suppressed path with no override list is untouched. Console output is
byte-identical.
