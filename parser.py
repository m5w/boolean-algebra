# Copyright (C) 2022 Matthew Marting
# SPDX-License-Identifier: GPL-3.0-or-later

from collections import namedtuple
from enum import Enum
import regex as re

__all__ = [
    "ParseError",
    "handle",
    "CheckedParseError",
    "ParseSyntaxError",
    "ParseTrailingSyntaxError",
    "ParseRuleSyntaxError",
    "ParseRuleBranchSyntaxError",
    "parse",
]


class StopParsing(Exception):
    pass


class ParseError(RuntimeError):
    def __init__(self, *args, parser, **kwargs):
        super().__init__(*args, **kwargs)
        self.history = parser.history
        self.line_number = parser.line_number
        self.column = parser.column
        self.stack = parser.stack
        self.rule_identifier = parser.rule_identifier
        self.branch_number = parser.branch_number
        self.item_number = parser.item_number


def branch_number_str(rule_identifier, branch_number):
    if len(rules[rule_identifier].branches) == 1:
        return ""
    return f"branch {branch_number} of "


def state_str(rule_identifier, branch_number, item_number, *, stack, end):
    rule = rules[rule_identifier]
    branch_count = len(rule.branches)
    if (not stack or end) and branch_number == branch_count:
        if stack:
            start = "while trying to parse"
        else:
            start = "failed to parse"
        return f"{start} rule {rule_identifier!s}"
    if branch_number < 0 or branch_number > branch_count or stack and not end and branch_number == branch_count:
        if stack:
            start = "while trying to parse"
        else:
            start = "possibly trying to parse"
        return (
            f"""{start}"""
            f""" item {item_number} of """
            f"""invalid branch {branch_number}"""
            f""" [of {branch_count} {"branch" if branch_count == 1 else "branches"}] of """
            f"""rule {rule_identifier!s}"""
        )
    branch = rule.branches[branch_number]
    item_count = len(branch)
    if (not stack or end) and item_number == item_count:
        if stack:
            start = "after parsing"
        else:
            start = "parsed"
        return f"{start} {branch_number_str(rule_identifier, branch_number)}rule {rule_identifier!s}"
    if item_number < 0 or item_number > item_count or stack and not end and item_number == item_count:
        if stack:
            start = "while trying to parse"
        else:
            start = "possibly trying to parse"
        return (
            f"""{start}"""
            f""" invalid item {item_number}"""
            f""" [of {item_count} {"item" if item_count == 1 else "items"}] of """
            f"""{branch_number_str(rule_identifier, branch_number)}"""
            f"""rule {rule_identifier!s}"""
        )
    start = "trying to parse"
    if stack:
        start = f"while {start}"
    return (
        f"{start}"
        f" item {item_number} of "
        f"{branch_number_str(rule_identifier, branch_number)}"
        f"rule {rule_identifier!s}"
    )


def edge_str(edge):
    return f"""got {edge[0]!s}{f" `{edge[1]}'" if edge[0].append_match else ""}"""


def error_str(source_file, line_number, column, message):
    return f"{source_file}:{line_number + 1}:{column + 1}: {message}"


def handle(e, source_file):
    print(error_str(source_file, e.line_number, e.column, f"error: {e}"))
    print(
        error_str(
            source_file,
            e.line_number,
            e.column,
            state_str(e.rule_identifier, e.branch_number, e.item_number, stack=True, end=True),
        )
    )
    for line_number, column, rule_identifier, branch_number, item_number in reversed(e.stack):
        print(
            error_str(
                source_file,
                line_number,
                column,
                state_str(rule_identifier, branch_number, item_number, stack=True, end=False),
            )
        )
    print()
    print("History:")
    print()
    for line_number, column, rule_identifier, branch_number, item_number, stack, edge in e.history:
        print(
            error_str(
                source_file,
                line_number,
                column,
                state_str(rule_identifier, branch_number, item_number, stack=False, end=False),
            )
        )
        if edge is not None:
            print(error_str(source_file, line_number, column, edge_str(edge)))
    print(
        error_str(
            source_file,
            e.line_number,
            e.column,
            state_str(e.rule_identifier, e.branch_number, e.item_number, stack=False, end=True),
        )
    )


class CheckedParseError(ParseError):
    pass


class ParseSyntaxError(CheckedParseError):
    pass


class ParseTrailingSyntaxError(ParseSyntaxError):
    def __init__(self, **kwargs):
        super().__init__("trailing tokens", **kwargs)


def oxford_join(iterable, s, s2=","):
    sequence = tuple(iterable)
    if len(sequence) <= 2:
        return f" {s} ".join(sequence)
    return f"""{f"{s2} ".join(sequence[:-1])}{s2} {s} {sequence[-1]}"""


class ParseRuleSyntaxError(ParseSyntaxError):
    def __init__(self, *, expected, **kwargs):
        super().__init__(f"""expected {oxford_join(map(str, expected), "or")}""", **kwargs)
        self.expected = expected


class ParseRuleBranchSyntaxError(ParseSyntaxError):
    def __init__(self, *, expected, **kwargs):
        super().__init__(f"""expected {oxford_join(map(str, expected), "or")}""", **kwargs)
        self.expected = expected


class Item:
    def is_token(self):
        return False


class TokenIdentifier(namedtuple("_TokenIdentifier", ("description", "append_match")), Item, Enum):
    def __new__(cls, description, append_match=True):
        obj = super().__new__(cls, description, append_match)
        obj._value_ = len(cls.__members__) + 1
        return obj

    def __str__(self):
        return self.description

    def is_token(self):
        return True

    NEWLINE = "newline", False
    WHITE_SPACE = "white space"
    OR = "`|'", False
    AND = "`&'", False
    NOT = "`~'", False
    LEFT_PARENTHESIS = "`('", False
    RIGHT_PARENTHESIS = "`)'", False
    IDENTIFIER = "identifier"


tokens = {
    TokenIdentifier.NEWLINE: re.compile("\\n|\\r\\n|\\r"),
    TokenIdentifier.WHITE_SPACE: re.compile("[ \\t\\v\\f]+"),
    TokenIdentifier.OR: re.compile(re.escape("|")),
    TokenIdentifier.AND: re.compile(re.escape("&")),
    TokenIdentifier.NOT: re.compile(re.escape("~")),
    TokenIdentifier.LEFT_PARENTHESIS: re.compile(re.escape("(")),
    TokenIdentifier.RIGHT_PARENTHESIS: re.compile(re.escape(")")),
    TokenIdentifier.IDENTIFIER: re.compile("\\p{XID_Start}\\p{XID_Continue}*"),
}


class RuleIdentifier(Item, Enum):
    def __new__(cls):
        obj = object.__new__(cls)
        obj._value_ = len(cls.__members__) + 1
        return obj

    def __str__(self):
        return self.name

    START = ()
    EXPRESSION = ()
    OR_EXPRESSION = ()
    OR_ARGUMENT = ()
    OR_ARGUMENT_LIST = ()
    OR_ARGUMENT_LIST2 = ()
    AND_EXPRESSION = ()
    AND_ARGUMENT = ()
    AND_ARGUMENT_LIST = ()
    AND_ARGUMENT_LIST2 = ()
    NOT_EXPRESSION = ()
    NOT_EXPRESSION2 = ()
    NOT_ARGUMENT = ()
    ARGUMENT = ()
    IDENTIFIER = ()


def nop_on_enter(stack):
    pass


def nop_on_exit(stack, branch_number, tokens):
    pass


class IdentifierExpression(namedtuple("_IdentifierExpression", ("identifier",))):
    def __str__(self):
        return self.identifier


class NotExpression(namedtuple("_NotExpression", ("a",))):
    def __str__(self):
        return f"not({self.a})"


class AndExpression(namedtuple("_AndExpression", ("a", "b"))):
    def __str__(self):
        return f"and({self.a}, {self.b})"


class OrExpression(namedtuple("_OrExpression", ("a", "b"))):
    def __str__(self):
        return f"or({self.a}, {self.b})"


def identifier_on_exit(stack, branch_number, tokens):
    stack.append(IdentifierExpression(tokens[-1]))
    tokens.clear()


def not_expression2_on_exit(stack, branch_number, tokens):
    a = stack.pop()
    stack.append(NotExpression(a))


def and_argument_list2_on_enter(stack):
    b = stack.pop()
    a = stack.pop()
    stack.append(AndExpression(a, b))


def or_argument_list2_on_enter(stack):
    b = stack.pop()
    a = stack.pop()
    stack.append(OrExpression(a, b))


class Rule(namedtuple("_Rule", ("branches", "on_enter", "on_exit"))):
    def __new__(cls, *branches, on_enter=None, on_exit=None):
        if len(branches) < 1:
            raise ValueError("a rule must have at least 1 branch")
        if on_enter is None:
            on_enter = nop_on_enter
        if on_exit is None:
            on_exit = nop_on_exit
        return super().__new__(cls, branches, on_enter, on_exit)


rules = {
    RuleIdentifier.START: Rule(
        [RuleIdentifier.EXPRESSION],
    ),
    RuleIdentifier.EXPRESSION: Rule(
        [RuleIdentifier.OR_EXPRESSION],
    ),
    RuleIdentifier.OR_EXPRESSION: Rule(
        [RuleIdentifier.OR_ARGUMENT, RuleIdentifier.OR_ARGUMENT_LIST],
    ),
    RuleIdentifier.OR_ARGUMENT: Rule(
        [RuleIdentifier.AND_EXPRESSION],
    ),
    RuleIdentifier.OR_ARGUMENT_LIST: Rule(
        [TokenIdentifier.OR, RuleIdentifier.OR_ARGUMENT, RuleIdentifier.OR_ARGUMENT_LIST2],
        [],
    ),
    RuleIdentifier.OR_ARGUMENT_LIST2: Rule(
        [TokenIdentifier.OR, RuleIdentifier.OR_ARGUMENT, RuleIdentifier.OR_ARGUMENT_LIST2],
        [],
        on_enter=or_argument_list2_on_enter,
    ),
    RuleIdentifier.AND_EXPRESSION: Rule(
        [RuleIdentifier.AND_ARGUMENT, RuleIdentifier.AND_ARGUMENT_LIST],
    ),
    RuleIdentifier.AND_ARGUMENT: Rule(
        [RuleIdentifier.NOT_EXPRESSION],
    ),
    RuleIdentifier.AND_ARGUMENT_LIST: Rule(
        [TokenIdentifier.AND, RuleIdentifier.AND_ARGUMENT, RuleIdentifier.AND_ARGUMENT_LIST2],
        [],
    ),
    RuleIdentifier.AND_ARGUMENT_LIST2: Rule(
        [TokenIdentifier.AND, RuleIdentifier.AND_ARGUMENT, RuleIdentifier.AND_ARGUMENT_LIST2],
        [],
        on_enter=and_argument_list2_on_enter,
    ),
    RuleIdentifier.NOT_EXPRESSION: Rule(
        [RuleIdentifier.NOT_EXPRESSION2],
        [RuleIdentifier.NOT_ARGUMENT],
    ),
    RuleIdentifier.NOT_EXPRESSION2: Rule(
        [TokenIdentifier.NOT, RuleIdentifier.NOT_EXPRESSION],
        on_exit=not_expression2_on_exit,
    ),
    RuleIdentifier.NOT_ARGUMENT: Rule(
        [RuleIdentifier.ARGUMENT],
    ),
    RuleIdentifier.ARGUMENT: Rule(
        [TokenIdentifier.LEFT_PARENTHESIS, RuleIdentifier.EXPRESSION, TokenIdentifier.RIGHT_PARENTHESIS],
        [RuleIdentifier.IDENTIFIER],
    ),
    RuleIdentifier.IDENTIFIER: Rule(
        [TokenIdentifier.IDENTIFIER],
        on_exit=identifier_on_exit,
    ),
}


class Parser:
    def __init__(self, s):
        self.lines = tokens[TokenIdentifier.NEWLINE].split(s)
        self.line_number = 0
        self.column = 0
        self.rule_identifier = RuleIdentifier.START
        self.branch_number = 0
        self.item_number = 0
        self.stack = []
        self.history = []

    def append(self, edge):
        self.history.append(
            (
                self.line_number,
                self.column,
                self.rule_identifier,
                self.branch_number,
                self.item_number,
                self.stack,
                edge,
            )
        )

    def eof(self):
        return self.line_number >= len(self.lines)

    def match(self, token_identifier):
        pattern = tokens[token_identifier]
        while True:
            if self.eof():
                return None
            line = self.lines[self.line_number]
            if self.column < len(line):
                break
            self.append(None)
            self.line_number += 1
            self.column = 0
        match = pattern.match(line, self.column)
        if match:
            if token_identifier.append_match:
                edge = (token_identifier, match.group())
            else:
                edge = (token_identifier,)
            self.append(edge)
            self.column = match.end()
        return match

    def push(self):
        self.append(None)
        self.stack.append(
            (self.line_number, self.column, self.rule_identifier, self.branch_number, self.item_number)
        )

    def pop(self):
        self.append(None)
        _, _, self.rule_identifier, self.branch_number, self.item_number = self.stack.pop()

    def next_branch(self, expected):
        unwound_stack = []
        while True:
            if self.item_number > 0:
                raise RuntimeError("grammar must be LL(1)")
            self.append(None)
            self.branch_number += 1
            self.item_number = 0
            if self.branch_number < len(rules[self.rule_identifier].branches):
                return
            unwound_stack.append(
                (self.line_number, self.column, self.rule_identifier, self.branch_number, self.item_number)
            )
            try:
                assert self.line_number == self.stack[-1][0] and self.column == self.stack[-1][1]
                self.pop()
            except IndexError:
                unwound_stack.pop()
                self.append(None)
                while unwound_stack:
                    self.branch_number -= 1
                    self.stack.append(
                        (self.line_number, self.column, self.rule_identifier, self.branch_number, self.item_number)
                    )
                    _, _, self.rule_identifier, self.branch_number, self.item_number = unwound_stack.pop()
                raise ParseRuleSyntaxError(parser=self, expected=expected)
            if len(rules[self.rule_identifier].branches) > 1:
                unwound_stack.clear()


def parse(s):
    stack = []
    parser = Parser(s)

    def parse_error(type_, *args, **kwargs):
        return type_(*args, parser=parser, **kwargs)

    try:
        expected = []
        tokens = []
        while True:
            while parser.match(TokenIdentifier.WHITE_SPACE):
                pass
            while True:
                while True:
                    branch = rules[parser.rule_identifier].branches[parser.branch_number]
                    if parser.item_number < len(branch):
                        break
                    rules[parser.rule_identifier].on_exit(stack, parser.branch_number, tokens)
                    try:
                        parser.pop()
                    except IndexError:
                        raise StopParsing
                    parser.item_number += 1
                item = branch[parser.item_number]
                if item.is_token():
                    break
                parser.push()
                parser.rule_identifier = item
                parser.branch_number = 0
                parser.item_number = 0
                rules[parser.rule_identifier].on_enter(stack)
            match = parser.match(item)
            if not match:
                expected.append(item)
                if parser.item_number > 0:
                    raise parse_error(ParseRuleBranchSyntaxError, expected=expected)
                parser.next_branch(expected)
            else:
                expected.clear()
                parser.item_number += 1
                tokens.append(match.group())
    except StopParsing:
        if not parser.eof():
            raise parse_error(ParseTrailingSyntaxError)
        return stack
    except CheckedParseError:
        raise
    except Exception as e:
        raise ParseError(e, parser=parser) from e
