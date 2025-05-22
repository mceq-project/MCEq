import sys
import warnings

# Import the actual configuration module
from MCEq import config as _config

# Warn users about the deprecated module path
warnings.warn(
    "The module 'mceq_config' has been moved to 'MCEq.config'. "
    "Please update your `import mceq_config` to `import MCEq.config` accordingly.",
    DeprecationWarning,
    stacklevel=2,
)

# Replace this module in sys.modules so that any attribute access
# or subsequent imports refer to the real config module
sys.modules[__name__] = _config
