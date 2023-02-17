import sys
from math import ceil

completed = -1
completedUpdated = 0
completedBars = -1
completedUpdatedBars = 0



                               # [Not-Important] This is just for interface
def showProgress(start, all):  # This is nothing but user-friendly interface
    global completed
    global completedUpdated
    if(int(completedUpdated) != 0):
        if (completed != completedUpdated):
            sys.stdout.write('\x1b[1A')  # Moves up one line on terminal
            sys.stdout.write('\r')       # Moves the cursor to the beginning of the line
            sys.stdout.write('Please wait while image is processing... {}% completed'
                             .format(completedUpdated))  # Overwrite previous line to show progress in one line

            completed = completedUpdated
    completedUpdated = int(start * 100 / all)
    if(completedUpdated > 99):
        completed = -1
        completedUpdated = 0




def showProgressForDijkstra(start, all):  # We need to show more than %
    global completed
    global completedUpdated
    start = all - start
    sys.stdout.write('\x1b[1A')
    sys.stdout.write('\r')
    sys.stdout.write('Please wait while image is processing... {} / {} completed'.format(start, all))



                                       # [Not-Important] This is just for interface
def showProgressWithBars(start, all):  # This is nothing but user-friendly interface
    global completedBars
    global completedUpdatedBars
    multiplier = 1 if all >= 50 else int(ceil(50 / all))
    if(int(completedUpdatedBars) != 0 or multiplier == 50):
        if (completedBars != completedUpdatedBars):
            for i in range(multiplier):
                print('|', end='')
            completedBars = completedUpdatedBars
    completedUpdatedBars = int(ceil(start * 50 / all))
    if(completedUpdatedBars > 49 or start == all-1):
        completedBars = -1
        completedUpdatedBars = 0




def showRemainingTime(time, format = "short", message = "default"): # This method directly get from Internet; it converts milliseconds to hours/minutes/seconds and show them properly
    if message == "default":
        message = 'Estimated Remaining Time: '

    if format == "full":
        days = int(time // (24 * 3600))
        time = time % (24 * 3600)
        hours = int(time // 3600)
        time %= 3600
        minutes = int(time // 60)
        time %= 60
        seconds = int(time)

        print('\n' + message + str(days) + "day(s) " + str(hours) + "hour(s) " + str(minutes) + "minute(s) " + str(seconds) + "second(s)")
    else:
        hours = int(time // 3600)
        time = time - 3600 * hours
        minutes = int(time // 60)
        seconds = int(time - 60 * minutes)
        print('\n' + message + str(hours) + ":" + str(minutes) + ":" + str(seconds))

    # print('\nEstimated Remaining Time: %d:%d:%d' % (hours, minutes, seconds))