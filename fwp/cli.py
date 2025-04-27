import sys
from optparse import OptionParser
from pathlib import Path
from typing import Generator, TextIO

from lark.exceptions import UnexpectedInput

from .ir import Block
from .parser import Parser


prog = Path(sys.argv[0])

optparser = OptionParser(
    usage="Usage: %prog [-P|-S] [OPTION]... [-o FILE] [FILE]...",
    prog=(
        f"{Path(sys.executable).name} -m {__package__}"
        if prog.stem == "__main__"
        else prog.name
    ),
)

optparser.add_option(
    "-P",
    dest="stage",
    action="store_const",
    const=0,
    help="stop at the parse stage; dump the parse tree",
)
optparser.add_option(
    "-r",
    "--raw-tree",
    action="store_true",
    help="use the raw format when dumping parse tree",
)

optparser.add_option(
    "-S",
    dest="stage",
    action="store_const",
    const=1,
    default=2,
    help="stop at the compile stage; dump the IR tree",
)

optparser.add_option(
    "-o", "--output", metavar="FILE", help="output to FILE instead of standard output"
)


def main() -> int:
    opts, args = optparser.parse_args()

    def getfiles() -> Generator[TextIO]:
        nonlocal args

        if not args:
            args = ["-"]
            print("No arguments supplied; reading from standard input", file=sys.stderr)

        for filename in args:
            if filename == "-":
                yield sys.stdin
            else:
                with open(filename, "r") as io:
                    yield io

    failed = False

    try:
        parser = Parser()

        for io in getfiles():
            try:
                parse_tree = parser.parse(io.read())
            except (OSError, UnexpectedInput) as e:
                failed = True
                print(e, file=sys.stderr)
                continue

            if opts.stage < 1:
                print(parse_tree if opts.raw_tree else parse_tree.pretty())
                return 0

            ir_tree = Block(parse_tree)

            if opts.stage < 2:
                print(ir_tree)
                return 0

            try:
                output = open(opts.output, "w") if opts.output else sys.stdout
            except OSError as e:
                print(e, file=sys.stderr)
                return 1

            print("Nothing to do (yet)! Please specify -P or -S and try again.")

            if output != sys.stdout:
                output.close()
    except KeyboardInterrupt:
        return 130

    return int(failed)
