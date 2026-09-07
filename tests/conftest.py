"""Applied before any test imports MuJoCo.

``isaacgrasp.bootstrap`` picks a render backend and, on the software-rendering
fallback, loads torch and Triton before a GL context exists. Without this the
suite dies with a bare segfault as soon as one test renders a frame and a later
one imports torchvision, and every file passes when run on its own.
"""

from __future__ import annotations

import isaacgrasp.bootstrap  # noqa: F401
