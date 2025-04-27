import importlib.resources

from lark import Lark, ParseTree


class Parser:
    def __init__(self, /) -> None:
        assert __package__

        self.lark = Lark(
            importlib.resources.read_text(__package__, "grammar.lark"),
            parser="earley",
            strict=False,
            start="block",
            propagate_positions=True,
        )

    def parse(self, /, code: str) -> ParseTree:
        return self.lark.parse(code)
