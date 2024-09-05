# round datetime object down to nearest 15 minutes using datetime
def round_down_15(dt):
    minutes = (dt.minute // 15) * 15
    return dt.replace(minute=minutes, second=0, microsecond=0)

# custom function for printing depending on verbose setting
def v_print(msg, verbose=1, level=1):
    if verbose >= level:
        print(msg)