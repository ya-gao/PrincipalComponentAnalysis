# A series of useful functions to be added to $PYTHONPATH for use in various
# scripts and programming tools

from __future__ import print_function, division


# ============================================================================
# MAIN DIRECTORY FUNCTION FOR WHATEVER PROJECT
# ============================================================================


def get_maindir():
    import os.path
    d = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    return d + "/"


# ============================================================================
# OUTPUT FUNCTIONS
# ============================================================================


# -----------------------------------------------------------------------
# Progress / loading bar
def progress(count, total, status=''):
    import sys
    barlen = 59
    filledlen = int(round(barlen * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = "=" * filledlen + ' ' * (barlen - filledlen)

    sys.stdout.write(' |%s| %s%s\t%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()
    if count == total:
        sys.stdout.write("\n\n")


# -----------------------------------------------------------------------
# Sets stdout to /dev/null
def hide_output():
    import sys
    sys.stdout = open("/dev/null", 'w')


# -----------------------------------------------------------------------
# sets stdout back to normal
def show_output():
    import sys
    sys.stdout = sys.__stdout__


# -----------------------------------------------------------------------
# Raises whichever Error message it's inputted with exclamation points
# then exits if set to.
def printh(mssg, h=1):
    if h == 1 or h == 2:
        topchar = "="
        bottomchar = "=" if h == 1 else "-"
    elif h == 3:
        topchar = "-"
        bottomchar = "-"
    elif h > 3:
        topchar = " "
        bottomchar = "-"

    print("{}\n{}\n{}\n".format(topchar * 79, mssg, bottomchar * 79))


# ============================================================================
# GREPPING FUNCTIONS
# ============================================================================


# -----------------------------------------------------------------------
# Homemade grep function: returns true/false
def grep_q(mask, filemask):
    import subprocess
    command = "grep --quiet '{}' {}".format(mask, filemask)
    ps = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    try:
        output = int(ps.communicate()[0])
    except ValueError:
        output = 0
    return output


# -----------------------------------------------------------------------
# Homemade grep function: returns number of files
def grep_fc(mask, filemask):
    import subprocess
    command = "grep -l '{}' {} | wc -l".format(mask, filemask)
    ps = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    try:
        output = int(ps.communicate()[0])
    except ValueError:
        output = 0
    return output


# -----------------------------------------------------------------------
# Homemade grep function: returns total found number of lines
def grep_lc(mask, filemask):
    import subprocess
    command = "grep '{}' {} | wc -l".format(mask, filemask)
    ps = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        output = int(ps.communicate()[0])
    except ValueError:
        output = 0
    return output


# ============================================================================
# DIRECTORY + SEARCH FUNCTIONS
# ============================================================================


# ----------------------------------------------------------------------------
# Finds files in subdirectories which mask any number of given masks
def ensureDir(directory):
    import os
    directory = directory if directory[-1] != "/" else directory[:-1]
    if os.path.exists(directory) and os.path.isdir(directory):
        return
    elif not os.path.exists(directory):
        os.makedirs(directory)
    elif os.path.exists(directory) and not os.path.isdir(directory):
        raise RuntimeError("Inputted directory path exists but is not directory")
    else:
        raise RuntimeError("Unknown case for directory check")


# ----------------------------------------------------------------------------
# Finds files in subdirectories which mask any number of given masks
def find_files(directory, masks):
    import os
    import fnmatch

    directory = directory if directory[-1] != "/" else directory[:-1]
    items = []
    iappend = items.append

    if "list" not in str(type(masks)):
        masks = [masks]

    for tmpitem in os.listdir(directory):
        if os.path.isdir("{}/{}".format(directory, tmpitem)):
            items.extend(find_files("{}/{}".format(directory, tmpitem), masks))
        else:
            if any([fnmatch.fnmatch(tmpitem, mask) for mask in masks]):
                iappend("{}/{}".format(directory, tmpitem))

    return items


# -----------------------------------------------------------------------
# Counts number of lines given a file + path
def line_count(filename):
    with open(filename) as f:
        linecount = len(f.readlines())
    return linecount
