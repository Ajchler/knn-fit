class SkipError(Exception):
    pass


class QuitError(Exception):
    pass


def quit_or_proceed(crs):
    key = crs.getch()

    while key not in (ord(c) for c in "cCqQ"):
        key = crs.getch()

    if key == ord("c") or key == ord("C"):
        return "proceed"

    elif key == ord("q") or key == ord("Q"):
        return "quit"


def accept_or_reject(crs, question_string):
    crs.addstr(question_string)

    done = False
    while not done:
        key = crs.getch()
        while key not in (ord(c) for c in "yYnNsS"):
            key = crs.getch()

        if key == ord("n") or key == ord("N"):
            return False
        elif key == ord("y") or key == ord("Y"):
            return True
        else:
            crs.addstr("\nAre you sure you want to skip this text? [Y/n] ")
            key = crs.getch()
            while key not in [ord("y"), ord("Y"), ord("n"), ord("N")]:
                key = crs.getch()

            crs.addstr("\n")
            if key == ord("y") or key == ord("Y"):
                raise SkipError


def redo_or_proceed(crs):
    """
    Asks the user if they want to redo the current text or proceed to the next one.
    :param crs: Curses window
    :return: String "redo" or "proceed"
    :raises SkipError: If the user wants to skip the current text
    :raises QuitError: If the user wants to quit the program
    """
    key = crs.getch()
    if key in (ord(c) for c in "qQcCrRsS"):
        crs.addstr("\n")
        if key == ord("q") or key == ord("Q"):  # Quit
            crs.addstr("Are you sure you want to quit? [Y/n] ")
            key = crs.getch()
            if key == ord("y") or key == ord("Y"):
                raise QuitError
        elif key == ord("r") or key == ord("R"):  # Redo
            return "redo"

        elif key == ord("c") or key == ord("C"):
            return "continue"

        elif key == ord("s") or key == ord("S"):
            crs.addstr("\nAre you sure you want to skip this text? [Y/n] ")
            key = crs.getch()
            while key not in [ord("y"), ord("Y"), ord("n"), ord("N")]:
                key = crs.getch()

            crs.addstr("\n")
            if key == ord("y") or key == ord("Y"):
                raise SkipError


def redo_accept(crs):
    crs.addstr("Relevant? [Y/n] ")
    choice = crs.getch()
    while choice not in (ord(c) for c in "yYnN"):
        choice = crs.getch()

    if choice == ord("n") or choice == ord("N"):
        return False
    elif choice == ord("y") or choice == ord("Y"):
        return True
