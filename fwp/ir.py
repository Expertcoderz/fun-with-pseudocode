# pylint: disable=missing-class-docstring,super-init-not-called,too-few-public-methods

from __future__ import annotations

import re
from collections.abc import Callable
from enum import Enum, auto
from typing import Self, cast

from lark import Tree, Token


class EntityKind(Enum):
    NORMAL = auto()
    SINGLETON = auto()
    GENERIC = auto()


class Entity:
    __slots__ = ()

    # This regex is used for converting entity (sub)class names (PascalCase)
    # into parse tree node names (snake_case), for registration purposes.
    NAME_CONVERSION_PATTERN = re.compile("(?<!^)(?=[A-Z])")

    _kind: EntityKind = EntityKind.NORMAL
    _operand_class: type[Entity] | None = None
    _node_name: Callable[[type[Self]], str] | None = None

    _global_class_registry: set[str] = set()
    _class_registry: dict[str, type[Self]] = {}
    _instance_registry: dict[str, Self] = {}

    class UnhandledNode(Exception): ...

    def __new__(cls, parser_node: Tree | Token, /) -> Self:
        if cls._kind != EntityKind.GENERIC:
            if cls._kind != EntityKind.SINGLETON:
                return super().__new__(cls)

            assert isinstance(
                parser_node, Token
            ), f"attempt to construct singleton Entity {cls!r} from non-Token node: {parser_node}"

            obj = cls._instance_registry.get(parser_node.value)
            if not obj:
                obj = cls._instance_registry[parser_node.value] = super().__new__(cls)

            return obj

        node_name = (
            parser_node.data if isinstance(parser_node, Tree) else parser_node.type
        )

        if specific_class := cls._class_registry.get(node_name):
            return specific_class(parser_node)

        raise cls.UnhandledNode(
            f"Could not find a class for node {parser_node!r} under {cls!r}"
        )

    def __init_subclass__(
        cls,
        /,
        *,
        kind: EntityKind = EntityKind.NORMAL,
        node_name: str | Callable[[type[Self]], str] | None = None,
        terminal: bool = False,
    ) -> None:
        assert (
            cls.__name__.isalnum()
        ), f"entity class {cls!r} has invalid name: {cls.__name__}"

        # Non-generic entity classes should define __slots__ to specify
        # instance attributes. For generic classes, __slots__ is irrelevant
        # as they are never instantianted; see the implementation of
        # Entity.__new__() for details. Furthermore, Entity.__str__() relies
        # on __slots__ to enumerate instance attributes.
        assert (
            kind == EntityKind.GENERIC or cls.__slots__ != Entity.__slots__
        ), f"entity class {cls!r} is missing a dedicated __slots__"

        # Note: this also (intentionally) resets _kind to EntityKind.GENERIC for all subclasses.
        cls._kind = kind

        cls._class_registry = {}
        cls._instance_registry = {}

        if kind == EntityKind.GENERIC:
            assert node_name is None or isinstance(
                node_name, Callable
            ), f"node_name for generic Entity class {cls!r} is set to a non-callable"
            cls._node_name = node_name
            return

        node_name = cls._node_name

        if not node_name:
            node_name = cls.NAME_CONVERSION_PATTERN.sub("_", cls.__name__)
            node_name = (node_name.upper if terminal else node_name.lower)()
        elif isinstance(node_name, Callable):
            # pylint: disable=not-callable
            node_name = node_name(cls)

        assert issubclass(
            cls.__bases__[0], Entity
        ), f"parent of Entity class {cls!r} is not Entity"

        cls._register_class(node_name)

    @classmethod
    def _register_class(cls, node_name: str, /):
        assert (
            node_name not in cls._global_class_registry
        ), f"duplicate registration of entity class for {node_name}"

        cls._global_class_registry.add(node_name)

        def recurse_base(base: type[Entity], /) -> None:
            if base == Entity:
                return

            base._class_registry[node_name] = cls

            for parent in base.__bases__:
                recurse_base(parent)

        recurse_base(cls)

    def __str__(self, /, *, depth: int = 0) -> str:
        result = ["(" + self.__class__.__name__]
        depth += 1

        # __str__() output format inspired by S-expressions

        for value in (getattr(self, attr) for attr in self.__slots__):
            is_entity = isinstance(value, Entity)

            if is_entity or isinstance(value, list):
                for child in (value,) if is_entity else value:
                    result.append(
                        child.__str__(depth=depth)
                        if type(child).__str__ == Entity.__str__
                        else str(child)
                    )
            else:
                result.append(repr(value))

        result[-1] += ")"

        if len(result) == 2 and "\n" not in result[1]:
            result[0] += " " + result.pop()

        return f"\n{" " * depth * 2}".join(result)


class Expression(Entity, kind=EntityKind.GENERIC): ...


class DataType(Entity, kind=EntityKind.SINGLETON):
    __slots__ = ("type_name",)

    def __init__(self, token: Token, /) -> None:
        self.type_name = cast(str, token.value)

    def __str__(self, /, *, depth: int = 0) -> str:
        del depth
        return self.type_name

    def __repr__(self, /) -> str:
        return self.type_name


class DataObject(
    Expression,
    kind=EntityKind.GENERIC,
    node_name=lambda cls: cls.TYPE_NAME + "_LITERAL",
):
    TYPE_NAME: str = "OBJECT"

    __slots__ = ("representation",)

    def __init__(self, literal: Token, /) -> None:
        self.representation: str = literal.value

    def __str__(self, /, *, depth: int = 0) -> str:
        del depth
        return f"{self.TYPE_NAME}:{self.representation}"


class IntegerDataObject(DataObject):
    TYPE_NAME = "INTEGER"


class RealDataObject(DataObject):
    TYPE_NAME = "REAL"


class CharDataObject(DataObject):
    TYPE_NAME = "CHAR"

    def __init__(self, literal: Token, /) -> None:
        self.representation = literal[1:-1]

    # Override __str__() to add surrounding quotes and handle escapes.
    def __str__(self, *, depth: int = 0) -> str:
        del depth
        return f"{self.TYPE_NAME}:{self.representation!r}"


class StringDataObject(CharDataObject):
    TYPE_NAME = "STRING"


class BooleanDataObject(DataObject):
    TYPE_NAME = "BOOLEAN"


class BinaryOperation(Expression):
    __slots__ = ("operands",)

    def __init__(self, expr: Tree, /) -> None:
        self.operands = list(map(Expression, expr.children))


class Add(BinaryOperation): ...


class Sub(BinaryOperation): ...


class Mul(BinaryOperation): ...


class Div(BinaryOperation): ...


class Gt(BinaryOperation): ...


class Lt(BinaryOperation): ...


class Ge(BinaryOperation): ...


class Le(BinaryOperation): ...


class Eq(BinaryOperation): ...


class Ne(BinaryOperation): ...


class Concat(BinaryOperation): ...


class Intdiv(BinaryOperation): ...


class Mod(BinaryOperation): ...


class And(BinaryOperation): ...


class Or(BinaryOperation): ...


class UnaryOperation(Expression):
    __slots__ = ("operand",)

    def __init__(self, expr: Tree, /) -> None:
        self.operand = Expression(expr.children[0])


class Neg(UnaryOperation): ...


class Not(UnaryOperation): ...


class Statement(Entity, kind=EntityKind.GENERIC): ...


class Comment(Statement):
    __slots__ = ("text",)

    def __init__(self, stmt: Tree, /) -> None:
        self.text = cast(Token, stmt.children[0]).value[2:]


class Primary(Expression, kind=EntityKind.GENERIC): ...


class Identifier(Primary, kind=EntityKind.SINGLETON, terminal=True):
    __slots__ = ("name",)

    def __init__(self, expr: Token, /) -> None:
        self.name = expr.value


class ConstantDeclaration(Statement):
    __slots__ = ("identifier", "value")

    def __init__(self, stmt: Tree, /) -> None:
        self.identifier = Identifier(cast(Token, stmt.children[0]))
        self.value = Expression(stmt.children[1])


class VariableDeclaration(Statement):
    __slots__ = ("identifier", "dimensions", "datatype")

    def __init__(self, stmt: Tree, /) -> None:
        self.identifier = Identifier(cast(Token, stmt.children[0]))

        self.dimensions = (
            list(map(ArrayRange, stmt.children[2:-1]))
            if stmt.children[1] == "ARRAY"
            else None
        )

        self.datatype = DataType(cast(Token, stmt.children[-1]))


class ArrayRange(Entity):
    __slots__ = ("start", "end")

    def __init__(self, array_range: Tree, /) -> None:
        self.start = Expression(array_range.children[0])
        self.end = Expression(array_range.children[1])


class VariableAssignment(Statement):
    __slots__ = ("target", "value")

    def __init__(self, tree: Tree, /) -> None:
        self.target = Primary(tree.children[0])
        self.value = Expression(tree.children[1])


class ArraySubscription(Primary):
    __slots__ = ("target", "subscript")

    def __init__(self, pri: Tree, /) -> None:
        self.target = Primary(pri.children[0])
        self.subscript = Expression(pri.children[1])


class Block(Entity):
    __slots__ = ("statements",)

    def __init__(self, block: Tree, /) -> None:
        self.statements = list(map(Statement, block.children))


class If(Statement):
    __slots__ = ("predicate", "main_block", "else_block")

    def __init__(self, stmt: Tree, /) -> None:
        self.predicate = Expression(stmt.children[0])

        self.main_block = Block(stmt.children[1])
        self.else_block = stmt.children[2] and Block(stmt.children[2])


class For(Statement):
    __slots__ = ("identifier", "start", "stop", "step", "body")

    def __init__(self, stmt: Tree, /) -> None:
        self.identifier = Identifier(cast(Token, stmt.children[0]))

        self.start = Expression(stmt.children[1])
        self.stop = Expression(stmt.children[2])
        self.step = stmt.children[3] and Expression(stmt.children[3])

        self.body = Block(stmt.children[4])


class While(Statement):
    __slots__ = ("predicate", "block")

    def __init__(self, stmt: Tree, /) -> None:
        self.predicate = Expression(stmt.children[0])
        self.block = Block(stmt.children[1])


class Repeat(Statement):
    __slots__ = ("block", "predicate")

    def __init__(self, stmt: Tree, /) -> None:
        self.block = Block(stmt.children[0])
        self.predicate = Expression(stmt.children[1])


class Case(Statement):
    __slots__ = ("predicate", "blocks")

    def __init__(self, stmt: Tree, /) -> None:
        self.predicate = Expression(stmt.children[0])
        self.blocks = list(map(CaseBlock, stmt.children[1:-1]))


class CaseBlock(Entity):
    __slots__ = ("predicate_start", "predicate_end", "block")

    def __init__(self, block: Tree, /) -> None:
        self.predicate_start = Expression(block.children[0])
        self.predicate_end = block.children[1] and Expression(block.children[1])
        self.block = Block(block.children[2])


class Procedure(Statement):
    __slots__ = ("identifier", "parameters", "body")

    def __init__(self, tree: Tree, /) -> None:
        self.identifier = Identifier(cast(Token, tree.children[0]))

        self.parameters = []
        for param in tree.children[1:]:
            if not isinstance(param, Tree) or param.data != "parameter":
                break
            self.parameters.append(Parameter(param))

        self.body = Block(tree.children[-1])


class Function(Procedure):
    __slots__ = ("return_datatype",)

    def __init__(self, tree: Tree, /) -> None:
        super().__init__(tree)
        self.return_datatype = DataType(cast(Token, tree.children[-2]))


class Parameter(Entity):
    __slots__ = ("identifier", "datatype", "is_byref")

    def __init__(self, param: Tree, /) -> None:
        self.identifier = Identifier(cast(Token, param.children[1]))
        self.datatype = DataType(cast(Token, param.children[2]))
        self.is_byref = param.children[0] == "BYREF"


class Return(Statement):
    __slots__ = ("value",)

    def __init__(self, return_stmt: Tree, /) -> None:
        self.value = Expression(return_stmt.children[0]) if return_stmt else None


class GenericCall(Statement):
    __slots__ = ("target", "arguments")

    def __init__(self, tree: Tree, /) -> None:
        self.target = Primary(cast(Token, tree.children[0]))
        self.arguments = list(map(Expression, tree.children[1:]))


class Call(GenericCall): ...


class CallExpression(Expression, GenericCall): ...


class Class(Statement):
    __slots__ = ("identifier", "parent", "members")

    def __init__(self, class_decl: Tree, /) -> None:
        self.identifier = Identifier(cast(Token, class_decl.children[0]))
        self.parent = class_decl.children[1] and Identifier(
            cast(Token, class_decl.children[1])
        )
        self.members = list(map(MemberDeclaration, class_decl.children[2:]))


class MemberDeclaration(Statement):
    __slots__ = ("is_private", "declaration")

    def __init__(self, memb_decl: Tree, /) -> None:
        self.is_private = memb_decl.children[0] == "PRIVATE"
        self.declaration = Statement(memb_decl.children[1])


class RecordAccess(Primary):
    __slots__ = ("container", "item")

    def __init__(self, record_access: Tree, /) -> None:
        self.container = Primary(record_access.children[0])
        self.item = Identifier(cast(Token, record_access.children[1]))


class Input(Statement):
    __slots__ = ("target",)

    def __init__(self, input_stmt: Tree, /) -> None:
        self.target = Primary(cast(Token, input_stmt.children[0]))


class GenericCommand(Statement):
    __slots__ = ("arguments",)

    def __init__(self, cmd: Tree, /) -> None:
        self.arguments = list(map(Expression, cmd.children))


class Output(GenericCommand): ...


class New(Expression):
    __slots__ = ("identifier", "arguments")

    def __init__(self, stmt: Tree, /) -> None:
        self.identifier = Identifier(cast(Token, stmt.children[0]))
        self.arguments = list(map(Expression, stmt.children[1:]))
