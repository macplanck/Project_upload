def LOD(value):

    logv = 1
    log2 = 0

    while 2 * logv < value:
        logv *= 2
        log2 += 1

    return log2
