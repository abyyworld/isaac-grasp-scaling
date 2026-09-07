"""A stand-in Isaac Lab API, so the backend's own logic can be executed off a GPU.

This does **not** verify that the real Isaac Lab behaves the way these stubs do.
It cannot: that is what ``scripts/check_setup.py --isaac`` on a real machine is
for. What it does verify is everything on this side of the boundary, which was
otherwise entirely unexecuted code:

* every code path in ``IsaacBackend`` runs start to finish;
* the TCP-to-hand-body offset and the world-to-base-frame conversion produce the
  poses they are supposed to, checked against the arithmetic rather than against
  a screenshot;
* objects are placed, scaled and given the right mass;
* the batch is assembled and unpacked in the right order, so environment *i*'s
  result belongs to environment *i*'s grasp;
* outcomes are classified by the shared rule.

The stubs record what the backend asked them to do, which is what the assertions
inspect. Where a stub returns a value the backend depends on, it returns
something dimensionally correct and otherwise inert.
"""

from .isaac import install_fake_isaac, uninstall_fake_isaac

__all__ = ["install_fake_isaac", "uninstall_fake_isaac"]
