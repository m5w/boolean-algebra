# Copyright (C) 2022 Matthew Marting
# SPDX-License-Identifier: GPL-3.0-or-later

from parser import *

__all__ = ["s", "errno", "main"]

s = """\
0 (a & ~b | ~a & b) & ~(c & ~d | ~c & d) | ~(a & ~b | ~a & b) & (c & ~d | ~c & d foo)
"""


errno = None


def main():
    try:
        stack = parse(s)
        print(stack.pop())
    except ParseError as e:
        handle(e, "s")
        global errno
        errno = e


if __name__ == "__main__":
    main()
