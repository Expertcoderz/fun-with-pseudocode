# Fun with Pseudocode

Fun with Pseudocode (or `fwp` for short) is a Python utility to parse
"pseudocode" with syntax based on the Cambridge International A & AS Level
Computer Science [Pseudocode Guide for Teachers](https://www.cambridgeinternational.org/Images/697401-2026-pseudocode-guide-for-teachers.pdf).

This project is a fun exercise. It has no intended practical application. It is
also largely unfinished as of now; not all language features have been
implemented or rigorously tested.

## Usage

```sh
$ python -m fwp --help
```

Requires [Lark](https://github.com/lark-parser/lark) to be installed.

Sample pseudocode programs are provided under the `examples` directory.

## The Process

```txt
                   +--------------+
                   | EBNF grammar |
                   +---|----------+
                       |
+-------------------+  V   +------------+     +----+
| Pseudocode source | ---> | Parse tree | --> | IR | ---> ???
+-------------------+      +------------+     +----+
```

1. **Parsing:** the pseudocode source is first run through the parser to
   generate a parse tree. [Lark](https://github.com/lark-parser/lark) is used as
   the parser generator, and supplied with the EBNF definition of the pseudocode
   language (see `fwp/grammar.lark`). The parser performs lexical analysis
   (tokenization) and parsing of the pseudocode source, resulting in a parse
   tree. Pass the `-P` option to stop at the parsing stage and view the parse
   tree.

2. **Intermediate representation (IR) generation:** the IR is implemented as a
   hierarchy of Python objects that comprise a tree structure. It is essentially
   a parse tree with some semantics added. However, unlike most other IRs there
   is no actual text or binary format for which to store such generated IR,
   which exists purely as a Python object. To view the IR as a nicely formatted
   S-expression, pass the `-S` option.

3. Now all that's left is to do something useful with the IR. This has yet to be
   decided upon.

## License

Fun with Pseudocode is licensed under the AGPLv3. See the LICENSE file for more
information.
