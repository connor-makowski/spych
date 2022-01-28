import sys
if sys.version_info[0] == 3:
    from spych.core import *
elif sys.version_info[0] < 3:
    from core import *
