import warnings

from MCEq.config import *  # noqa: F403

warnings.warn(
    "The module 'mceq_config' has been moved to 'MCEq.config'. "
    "Please update your `import mceq_config` to `import MCEq.config` accordingly.",
    DeprecationWarning,
    stacklevel=2,
)
# The above import is to maintain backward compatibility for the old module name.
