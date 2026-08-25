"""Run the optcmp report generator with h5py stubbed out.

wandb's lazy-loader walks sys.modules during report.save() and touches h5py,
which is installed in the conda env but whose libhdf5.so.310 is absent on
Sunspot. Nothing in this report uses h5py.

The stub must be INERT, not raising: pydantic's docstring extraction calls
inspect.getmodule(), which probes hasattr(module, "__file__"). A stub that
raises on attribute access turns that probe into a hard error.
"""
import sys, types

for _m in ("h5py", "h5py._errors", "h5py._hl"):
    _mod = types.ModuleType(_m)
    _mod.__file__ = f"<stubbed {_m}>"
    _mod.__spec__ = None
    sys.modules[_m] = _mod

exec(open("torchtitan/experiments/ezpz/scripts/make_optcmp_report.py").read(),
     {"__name__": "__main__", "__file__": "torchtitan/experiments/ezpz/scripts/make_optcmp_report.py"})
