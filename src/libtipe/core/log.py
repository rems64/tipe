import colorit

class logSeverity:
    """The color the log is printed in (matches colorit colors)
    """
    info = (92, 252, 71)        # Green
    timer = (71, 177, 252)      # Blue
    warning = (245, 252, 71)    # Yellow
    error = (245, 90, 66)       # Red
    trace = (255, 255, 255)     # White

class logColor:
    blue = (71, 177, 251)

def log(msg="", type:logSeverity=logSeverity.trace, no_prefix=False) -> None:
    prefix = ""
    if type==logSeverity.info:      prefix = "[INFO ] "
    elif type==logSeverity.timer:   prefix = "[TIME ] "
    elif type==logSeverity.warning: prefix = "[WARN ] "
    elif type==logSeverity.error:   prefix = "[ERROR] "
    elif type==logSeverity.trace:   prefix = "[TRACE] "
    if no_prefix: prefix=""
    print(colorit.color("{}{}".format(prefix, msg), type))

def trace(msg=""):
    log(msg, logSeverity.trace)

def info(msg=""):
    log(msg, logSeverity.info)

def section(msg=""):
    count = 30
    pref = "-"*(count-len(msg)//2)
    suf = "-"*(count + len(msg)//2-len(msg))
    log(f"\n{pref} {msg} {suf}\n", logSeverity.info, True)

def warn(msg=""):
    log(msg, logSeverity.warning)

def error(msg=""):
    log(msg, logSeverity.error)

def log_list(sect, list, decorator="n°"):
    section(sect)
    for i, v in enumerate(list):
        info(f"{decorator}{i}")
        print(v)