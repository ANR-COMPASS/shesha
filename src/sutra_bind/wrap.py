import importlib


def smart_import(mod, cls, verbose=False, silent=False):
    try:
        if verbose:
            print("trying from " + mod + " import " + cls)
        my_module = __import__(mod)
        return getattr(my_module, cls)

    except ImportError as err:
        if not silent:
            import warnings
            warnings.warn("Error importing %s, it will be simulated due to: %s" %
                        (cls, err.msg), Warning)

        class tmp_cls:

            def __init__(self, *args, **kwargs):
                self.is_fake = True

        return tmp_cls


Dms = smart_import("Dms", "Dms")
Rtc = smart_import("Rtc", "Rtc")
Rtc_brama = smart_import("Rtc_brama", "Rtc_brama", silent=True)
Sensors = smart_import("Sensors", "Sensors")
Atmos = smart_import("Atmos", "Atmos")
Telescope = smart_import("Telescope", "Telescope")
Target = smart_import("Target", "Target")
Target_brama = smart_import("Target_brama", "Target_brama", silent=True)

naga_context = smart_import("naga", "naga_context")
