import datetime
def removeSysFiles(lsFiles0):
    lsFiles = lsFiles0
    lsSysFiles = ["desktop.ini", ".DS_Store", ".ipynb_checkpoints"]
    for sysFn in lsSysFiles:
        if sysFn in lsFiles:
            lsFiles.remove(sysFn)
    return lsFiles

DATEFORMAT = ["%Y-%m-%d %H:%M", "%d/%m/%Y %H:%M", "%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M:%S", "%m/%d/%Y %H:%M", "%m/%d/%Y %H:%M:%S", "%Y/%m/%d %H:%S"]