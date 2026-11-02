import os
if os.environ.get("SHIFT_BABY_COMPAT") == "1":
    from shift_full.compat import install_baby_event_fix, install_managed_callback_fix
    install_baby_event_fix()
    install_managed_callback_fix()
