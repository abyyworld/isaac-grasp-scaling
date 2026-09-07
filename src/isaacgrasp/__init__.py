"""Does the orientation failure survive more data?

The predecessor project (github.com/abyyworld/Simulated-grasping) trained a
fully-convolutional grasp-quality network on self-supervised grasp attempts in
MuJoCo and lost to a hand-written depth heuristic on unseen objects, 58.4%
against 75.3%. An ablation traced the cause to the network learning grasp
position but not orientation.

This project asks whether that failure is a data problem. Isaac Lab runs
thousands of environments in parallel on one GPU, so one to two orders of
magnitude more grasp attempts is practical. If more data closes the gap, the
original negative result is explained. If it does not, the failure is
architectural, which is a sharper claim than the original made.

The comparison is only meaningful if everything except the training set size is
held fixed, so the model, the training loop, the dataset format, the heuristic
baseline and the evaluation harness are imported from ``simgrasp`` at a pinned
commit rather than reimplemented here. See :mod:`isaacgrasp.parity`.
"""

__version__ = "0.1.0"
