class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
    # Additional colors
    LIGHTGRAY = '\033[37m'
    DARKGRAY = '\033[90m'
    WHITE = '\033[97m'
    BLACK = '\033[30m'
    
    # New colors
    ORANGE = '\033[38;5;208m'  # Orange
    PURPLE = '\033[38;5;128m'  # Purple
    YELLOW = '\033[38;5;226m'  # Yellow
    PINK = '\033[38;5;213m'    # Pink
    TEAL = '\033[38;5;36m'     # Teal
    BROWN = '\033[38;5;94m'     # Brown
    LIGHTBLUE = '\033[38;5;12m' # Light Blue
    LIGHTGREEN = '\033[38;5;82m' # Light Green


import random

class TerminalColor:

    chosen_color = {}
    unchosen_color = [
            bcolors.OKBLUE, bcolors.OKGREEN, bcolors.HEADER, bcolors.WARNING, 
            bcolors.FAIL, bcolors.DARKGRAY,  bcolors.OKCYAN,
            bcolors.ORANGE, bcolors.PURPLE, bcolors.YELLOW, bcolors.PINK, bcolors.TEAL, bcolors.BROWN, bcolors.LIGHTBLUE, bcolors.LIGHTGREEN
        ]

    def assignColor(self, key):
        self.chosen_color[key] = random.choice(self.unchosen_color)
    
    def colorText(self, text, key):
        if not key in self.chosen_color:
            self.assignColor(key)
        return self.chosen_color[key] + text + bcolors.ENDC