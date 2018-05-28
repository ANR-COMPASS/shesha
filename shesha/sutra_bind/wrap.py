import importlib


def smart_import(mod, cls, verbose=False, silent=False):
    try:
        if verbose:
            print("trying from " + mod + " import " + cls)
        my_module = __import__(mod, globals(), locals(), [cls], 0)
        return getattr(my_module, cls)

    except (ImportError, AttributeError) as err:
        if not silent:
            import warnings
            warnings.warn("Error importing %s, it will be simulated due to: %s" %
                          (cls, err.msg), Warning)

        class tmp_cls:

            def __init__(self, *args, **kwargs):
                raise RuntimeError("Can not initilize the simulation with fake objects")

        return tmp_cls


Dms = smart_import("shesha.sutra_bind.Dms", "Dms")
Rtc = smart_import("shesha.sutra_bind.Rtc", "Rtc")
Rtc_brahma = smart_import("shesha.sutra_bind.Rtc", "Rtc_brahma", silent=True)
Sensors = smart_import("shesha.sutra_bind.Sensors", "Sensors")
Atmos = smart_import("shesha.sutra_bind.Atmos", "Atmos")
Telescope = smart_import("shesha.sutra_bind.Telescope", "Telescope")
Target = smart_import("shesha.sutra_bind.Target", "Target")
Target_brahma = smart_import("shesha.sutra_bind.Target", "Target_brahma", silent=True)

naga_context = smart_import("naga.context", "context")
