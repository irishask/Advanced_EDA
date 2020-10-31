# https://pypi.org/project/colored/
# http://ozzmaker.com/add-colour-to-text-in-python/
# https://plumbum.readthedocs.io/en/latest/colors.html

"""
The above ANSI escape code will set the text colour to bright green. The format is;
\033[  Escape code, this is always the same
1 = Style, 1 for normal.
32 = Text colour, 32 for bright green.
40m = Background colour, 40 is for black.
"""

class ColorsStyles:
    reset='\033[0m'
    bold='\033[01m'
    disable='\033[02m'
    underline='\033[04m'
    reverse='\033[07m'
    strikethrough='\033[09m'
    invisible='\033[08m'

    class fg:   #foreground
        black='\033[30m'
        red='\033[31m'
        green='\033[32m'
        orange = '\033[43m'
        blue='\033[34m'
        purple='\033[35m'
        cyan='\033[36m'
        lightgrey='\033[37m'
        darkgrey='\033[90m'
        lightred='\033[91m'
        lightgreen='\033[92m'
        magenta = '\033[35m'
        yellow='\033[93m'
        lightblue='\033[94m'
        pink='\033[95m'
        lightcyan='\033[96m'
        white = '\033[37m'

    class bg:    #Background
        black='\033[40m'
        red='\033[41m'
        green='\033[42m'
        orange='\033[43m'
        blue='\033[44m'
        purple='\033[45m'
        cyan='\033[46m'
        #lightGrey='\033[47m'
        lightPurple='\033[94m'
        white ='\033[37m'

# # Examples:-------------------
# print(colors.colors.fg.blue+ f"\tHello")
# print (colors.colors.bold + f"Linear Regression for X2:" + colors.colors.reset)
# print(colors.colors.fg.magenta + f"\tHello")
# print(colors.colors.reset)

#-PANDAS DISPLEY OPTIONS -----------------------------------------------------------------------

import pandas as pd

# Change display_options_for_pandas
# For all display options=>  https://pandas.pydata.org/pandas-docs/version/0.22/options.html
# Change display_options_for_pandas
# For all display options=>  https://pandas.pydata.org/pandas-docs/version/0.22/options.html
def display_options_for_pandas():
    pd.set_option('display.max_rows', 5000)   #Decrease it to
    pd.set_option('display.max_columns', 500)
    pd.set_option('max_colwidth', 300)     # Sets the maximum width of columns. Cells of this length or longer will be truncated with an ellipsis.
    pd.set_option('display.width', 1000000)
    pd.set_option('column_space', 50)
    pd.set_option('precision', 4)   #sets the output display precision in terms of decimal places
    pd.get_option("display.max_row", 1000)
    # pd.set_option("display.max_rows", 101)  #default = 500
    pd.set_option('max_info_columns', 100)

    # Reset ALL display options:
    #pd.reset_option("^display")
    # Reset wanted display option:
    #pd.reset_option('display.width')

#-------------------------------------------------------------


