import importlib


def smart_import(mod, cls, verbose=False, silent=False):
    try:
        if verbose:
            print("trying from " + mod + " import " + cls)
        # my_module = __import__(mod, globals(), locals(), [cls], 0)
        my_module = importlib.import_module(mod)
        return getattr(my_module, cls)

    except ImportError as err:
        if not silent:
            import warnings
            warnings.warn(
                    "Error importing %s, it will be simulated due to: %s" %
                    (cls, err.msg), Warning)

        class tmp_cls:

            def __init__(self, *args, **kwargs):
                raise RuntimeError("Can not initilize the simulation with fake objects")

        return tmp_cls

    except AttributeError as err:
        if not silent:
            import warnings
            warnings.warn(
                    "Error importing %s, it will be simulated due to: %s" %
                    (cls, err.args), Warning)

        class tmp_cls:

            def __init__(self, *args, **kwargs):
                raise RuntimeError("Can not initilize the simulation with fake objects")

        return tmp_cls


# The import of carmaWrap MUST be done first
# since MAGMA >= 2.5.0
# Otherwise, it causes huge memory leak
# plus not working code
# Why ? We don't know... TB check with further version of MAGMA
carmaWrap_context = smart_import("carmaWrap", "context")

Dms = smart_import("sutraWrap", "Dms")
Rtc_FFF = smart_import("sutraWrap", "Rtc_FFF")
Rtc_FHF = smart_import("sutraWrap", "Rtc_FHF", silent=True)
Rtc_UFF = smart_import("sutraWrap", "Rtc_UFF", silent=True)
Rtc_UHF = smart_import("sutraWrap", "Rtc_UHF", silent=True)
Rtc_FFU = smart_import("sutraWrap", "Rtc_FFU", silent=True)
Rtc_FHU = smart_import("sutraWrap", "Rtc_FHU", silent=True)
Rtc_UFU = smart_import("sutraWrap", "Rtc_UFU", silent=True)
Rtc_UHU = smart_import("sutraWrap", "Rtc_UHU", silent=True)
Rtc_brahma = smart_import("sutraWrap", "Rtc_brahma", silent=True)
Rtc_cacao_FFF = smart_import("sutraWrap", "Rtc_cacao_FFF", silent=True)
Rtc_cacao_UFF = smart_import("sutraWrap", "Rtc_cacao_UFF", silent=True)
Rtc_cacao_FFU = smart_import("sutraWrap", "Rtc_cacao_FFU", silent=True)
Rtc_cacao_UFU = smart_import("sutraWrap", "Rtc_cacao_UFU", silent=True)
Rtc_cacao_FHF = smart_import("sutraWrap", "Rtc_cacao_FHF", silent=True)
Rtc_cacao_UHF = smart_import("sutraWrap", "Rtc_cacao_UHF", silent=True)
Rtc_cacao_FHU = smart_import("sutraWrap", "Rtc_cacao_FHU", silent=True)
Rtc_cacao_UHU = smart_import("sutraWrap", "Rtc_cacao_UHU", silent=True)
Sensors = smart_import("sutraWrap", "Sensors")
Atmos = smart_import("sutraWrap", "Atmos")
Telescope = smart_import("sutraWrap", "Telescope")
Target = smart_import("sutraWrap", "Target")
Target_brahma = smart_import("sutraWrap", "Target_brahma", silent=True)
Gamora = smart_import("sutraWrap", "Gamora")
Groot = smart_import("sutraWrap", "Groot")
