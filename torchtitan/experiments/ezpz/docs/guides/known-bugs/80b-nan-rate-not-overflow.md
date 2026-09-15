# MOVED -- and the title was wrong

This page was called "the NaN is an accelerating RATE of transient non-finite
gradients". **That conclusion is withdrawn.** It came from experiments that
ran at ~1000x the documented learning-rate ceiling, and its replicate showed
the event counts were noise (2 vs 8 in the identical configuration).

The current write-up, with conclusions first and the invalidated work kept
only as a log, is:

**[80b-nan-what-we-know.md](80b-nan-what-we-know.md)**

Short version: the 80B NaN is NOT overflow (measured: activations 36 orders
below the bf16 ceiling), NOT the residual stream (fp32-ing it fails its own
test), and NOT root-caused. It is real, reproducible, optimizer-independent,
and prevented by fp32 activations at 3.4x cost. The site is unmeasured.
