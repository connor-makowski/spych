import sys
if sys.version_info[0] == 3:
    from spych.core import spych
    from spych.wake import spych_wake
elif sys.version_info[0] < 3:
    from core import spych
    from wake import spych_wake
