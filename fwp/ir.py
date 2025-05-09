# pylint: disable=missing-class-docstring,super-init-not-called,too-few-public-methods

from __future__ import annotations

import re
from collections.abc import Callable
from enum import Enum, auto
from typing import Iterable, Self, cast

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
        def recurse_base(base: type[Entity], /) -> None:
            if base == Entity:
                return

            assert (
                node_name not in base._class_registry
            ), f"duplicate registration of entity class for {node_name} under {cls!r}"

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

    def __init__(self, tree: Tree, /) -> None:
        self.operands = list(map(Expression, tree.children))


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

    def __init__(self, tree: Tree, /) -> None:
        self.operand = Expression(tree.children[0])


class Neg(UnaryOperation): ...


class Not(UnaryOperation): ...


class Statement(Entity, kind=EntityKind.GENERIC): ...


class Comment(Statement):
    __slots__ = ("text",)

    def __init__(self, tree: Tree, /) -> None:
        self.text = cast(Token, tree.children[0]).value[2:]


class Primary(Expression, kind=EntityKind.GENERIC): ...


class Identifier(Primary, kind=EntityKind.SINGLETON, terminal=True):
    __slots__ = ("name",)

    def __init__(self, tree: Token, /) -> None:
        self.name = tree.value


class ConstantDeclaration(Statement):
    __slots__ = ("identifier", "value")

    def __init__(self, tree: Tree, /) -> None:
        self.identifier = Identifier(cast(Token, tree.children[0]))
        self.value = Expression(tree.children[1])


class VariableDeclaration(Statement):
    __slots__ = ("identifier", "dimensions", "datatype")

    def __init__(self, tree: Tree, /) -> None:
        self.identifier = Identifier(cast(Token, tree.children[0]))

        self.dimensions = (
            list(map(ArrayRange, tree.children[2:-1]))
            if tree.children[1] == "ARRAY"
            else None
        )

        self.datatype = DataType(cast(Token, tree.children[-1]))


class ArrayRange(Entity):
    __slots__ = ("start", "end")

    def __init__(self, tree: Tree, /) -> None:
        self.start = Expression(tree.children[0])
        self.end = Expression(tree.children[1])


class VariableAssignment(Statement):
    __slots__ = ("target", "value")

    def __init__(self, tree: Tree, /) -> None:
        self.target = Primary(tree.children[0])
        self.value = Expression(tree.children[1])


class ArraySubscription(Primary):
    __slots__ = ("target", "subscript")

    def __init__(self, tree: Tree, /) -> None:
        self.target = Primary(tree.children[0])
        self.subscript = Expression(tree.children[1])


class Block(Entity):
    __slots__ = ("statements",)

    def __init__(self, tree: Tree, /) -> None:
        self.statements = list(map(Statement, tree.children))


class If(Statement):
    __slots__ = ("predicate", "main_block", "else_block")

    def __init__(self, tree: Tree, /) -> None:
        self.predicate = Expression(tree.children[0])

        self.main_block = Block(tree.children[1])
        self.else_block = tree.children[2] and Block(tree.children[2])


class For(Statement):
    __slots__ = ("identifier", "start", "stop", "step", "body")

    def __init__(self, tree: Tree, /) -> None:
        self.identifier = Identifier(cast(Token, tree.children[0]))

        self.start = Expression(tree.children[1])
        self.stop = Expression(tree.children[2])
        self.step = tree.children[3] and Expression(tree.children[3])

        self.body = Block(tree.children[4])


class While(Statement):
    __slots__ = ("predicate", "block")

    def __init__(self, tree: Tree, /) -> None:
        self.predicate = Expression(tree.children[0])
        self.block = Block(tree.children[1])


class Repeat(Statement):
    __slots__ = ("block", "predicate")

    def __init__(self, tree: Tree, /) -> None:
        self.block = Block(tree.children[0])
        self.predicate = Expression(tree.children[1])


class Case(Statement):
    __slots__ = ("predicate", "blocks")

    def __init__(self, tree: Tree, /) -> None:
        self.predicate = Expression(tree.children[0])
        self.blocks = list(map(CaseBlock, tree.children[1:-1]))


class CaseBlock(Entity):
    __slots__ = ("predicate_start", "predicate_end", "block")

    def __init__(self, tree: Tree, /) -> None:
        self.predicate_start = Expression(tree.children[0])
        self.predicate_end = tree.children[1] and Expression(tree.children[1])
        self.block = Block(tree.children[2])


class Procedure(Statement):
    __slots__ = ("identifier", "parameters", "body")

    def __init__(self, tree: Tree, /) -> None:
        self.identifier = Identifier(cast(Token, tree.children[0]))

        self.parameters = []

        byref = False
        for param in tree.children[1:]:
            if not isinstance(param, Tree) or param.data != "parameter":
                break

            byref = param.children[0] == "BYREF" if param.children[0] else byref

            param_obj = Parameter(param)
            param_obj.is_byref = byref
            self.parameters.append(param_obj)

        self.body = Block(tree.children[-1])


class Function(Procedure):
    __slots__ = ("return_datatype",)

    def __init__(self, tree: Tree, /) -> None:
        super().__init__(tree)
        self.return_datatype = DataType(cast(Token, tree.children[-2]))


class Parameter(Entity):
    __slots__ = ("identifier", "datatype", "is_byref")

    def __init__(self, tree: Tree, /) -> None:
        self.identifier = Identifier(cast(Token, tree.children[1]))
        self.datatype = DataType(cast(Token, tree.children[2]))
        self.is_byref: bool


class Return(Statement):
    __slots__ = ("value",)

    def __init__(self, tree: Tree, /) -> None:
        self.value = Expression(tree.children[0]) if tree else None


class GenericCall(Statement):
    __slots__ = ("target", "arguments")

    def __init__(self, tree: Tree, /) -> None:
        self.target = Primary(cast(Token, tree.children[0]))
        self.arguments = list(map(Expression, tree.children[1:]))


class Call(GenericCall): ...


class CallExpression(Expression, GenericCall): ...


class Typedef(Statement):
    __slots__ = ("identifier", "definition")

    def __init__(self, tree: Tree, /) -> None:
        self.identifier = Identifier(cast(Token, tree.children[0]))
        self.definition = TypeDefinition(tree.children[1])


class TypeDefinition(Entity, kind=EntityKind.GENERIC): ...


class EnumValues(TypeDefinition):
    __slots__ = ("values",)

    def __init__(self, tree: Tree, /) -> None:
        self.values = list(map(Identifier, cast(Iterable[Token], tree.children)))


class SetDefinition(Statement):
    __slots__ = ("identifier", "elements")

    def __init__(self, tree: Tree, /) -> None:
        self.identifier = Identifier(cast(Token, tree.children[0]))
        self.elements = list(map(Expression, tree.children[1:]))


class PointerType(TypeDefinition):
    __slots__ = ("type",)

    def __init__(self, tree: Tree, /) -> None:
        self.type = DataType(cast(Token, tree.children[0]))


class GenericPointer(Entity):
    __slots__ = ("target",)

    def __init__(self, tree: Tree, /) -> None:
        self.target = Expression(tree.children[0])


class Pointer(Primary, GenericPointer): ...


class PointerDereference(Expression, GenericPointer): ...


class Record(TypeDefinition):
    __slots__ = ("members",)

    def __init__(self, tree: Tree, /) -> None:
        self.members = list(map(Statement, tree.children))


class SetType(TypeDefinition):
    __slots__ = ("type",)

    def __init__(self, tree: Tree, /) -> None:
        self.type = DataType(cast(Token, tree.children[0]))


class Class(Statement):
    __slots__ = ("identifier", "parent", "members")

    def __init__(self, tree: Tree, /) -> None:
        self.identifier = Identifier(cast(Token, tree.children[0]))
        self.parent = tree.children[1] and Identifier(cast(Token, tree.children[1]))
        self.members = list(map(MemberDeclaration, tree.children[2:]))


class MemberDeclaration(Statement):
    __slots__ = ("is_private", "declaration")

    def __init__(self, tree: Tree, /) -> None:
        self.is_private = tree.children[0] == "PRIVATE"
        self.declaration = Statement(tree.children[1])


class RecordAccess(Primary):
    __slots__ = ("container", "item")

    def __init__(self, tree: Tree, /) -> None:
        self.container = Primary(tree.children[0])
        self.item = Identifier(cast(Token, tree.children[1]))


class New(Expression):
    __slots__ = ("identifier", "arguments")

    def __init__(self, tree: Tree, /) -> None:
        self.identifier = Identifier(cast(Token, tree.children[0]))
        self.arguments = list(map(Expression, tree.children[1:]))


class Input(Statement):
    __slots__ = ("target",)

    def __init__(self, tree: Tree, /) -> None:
        self.target = Primary(cast(Token, tree.children[0]))


class GenericCommand(Statement):
    __slots__ = ("arguments",)

    def __init__(self, tree: Tree, /) -> None:
        self.arguments = list(map(Expression, tree.children))


class Output(GenericCommand): ...


class OpenFile(Statement):
    __slots__ = ("name", "mode")

    def __init__(self, tree: Tree, /) -> None:
        self.name = StringDataObject(cast(Token, tree.children[0]))
        self.mode = str(tree.children[1])


class ReadFile(Statement):
    __slots__ = ("name", "destination")

    def __init__(self, tree: Tree, /) -> None:
        self.name = StringDataObject(cast(Token, tree.children[0]))
        self.destination = Primary(tree.children[1])


class WriteFile(Statement):
    __slots__ = ("name", "data")

    def __init__(self, tree: Tree, /) -> None:
        self.name = StringDataObject(cast(Token, tree.children[0]))
        self.data = Expression(tree.children[1])


class CloseFile(Statement):
    __slots__ = ("name",)

    def __init__(self, tree: Tree, /) -> None:
        self.name = StringDataObject(cast(Token, tree.children[0]))


class Seek(Statement):
    __slots__ = ("name", "target")

    def __init__(self, tree: Tree, /) -> None:
        self.name = StringDataObject(cast(Token, tree.children[0]))
        self.target = Expression(tree.children[1])


class FileRecordOperation(Statement, kind=EntityKind.GENERIC):
    __slots__ = ("name", "target")

    def __init__(self, tree: Tree, /) -> None:
        self.name = StringDataObject(cast(Token, tree.children[0]))
        self.target = Primary(tree.children[1])


class GetRecord(FileRecordOperation): ...


class PutRecord(FileRecordOperation): ...
