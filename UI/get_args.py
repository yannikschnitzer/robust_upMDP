import sys

def run():
    from opts import opt_settings
    opts = {opt: setting["default"] for opt, setting in opt_settings.items()}
    for opt in opts:
        settings = opt_settings[opt]
        for flag in settings["flags"]:
            if flag in sys.argv:
                opts[opt] = settings["parser"](flag, *settings["args"])
                break
    return opts
