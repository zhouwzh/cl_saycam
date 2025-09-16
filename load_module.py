from contextlib import contextmanager
import os, sys
@contextmanager
def temp_sys_path(path):
    sys_path_backup = list(sys.path)
    sys.path.insert(0, path)
    try:
        yield
    finally:
        sys.path[:] = sys_path_backup

SC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..","slotcontrast"))
STEVE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..","steve"))

with temp_sys_path(SC_ROOT):
    from load_slotcontrast_model import load_model as load_slotcontrast_model
# with temp_sys_path(STEVE_ROOT):
#     from load_steve import GetSlot