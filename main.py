import os
import re
import sys
from typing import List, Tuple, Optional
from PySide6.QtGui import QFont, QAction
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPlainTextEdit, QTabWidget,
    QTableWidget, QVBoxLayout, QFileDialog, QMessageBox, QTableWidgetItem
)
from PySide6.QtCore import Signal
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass(frozen=True)
class Error:
    line: int
    column: int
    message: str


@dataclass(frozen=True)
class Token:
    type: str
    value: str
    line: int
    column: int


class ILexer(ABC):
    @abstractmethod
    def tokenize(self, text: str) -> Tuple[List[Token], List[Error]]:
        ...


class IParser(ABC):
    @abstractmethod
    def parse(self, tokens: List[Token]) -> Tuple[List[str], List[Error]]:
        ...


class ICompilerModel(ABC):
    @abstractmethod
    def lex(self, text: str) -> Tuple[List[Token], List[Error]]:
        ...

    @abstractmethod
    def parse(self, tokens: List[Token]) -> Tuple[List[str], List[Error]]:
        ...


class ICompilerController(ABC):
    @abstractmethod
    def open_file(self, path: str) -> None:
        ...

    @abstractmethod
    def save_file(self, path: str) -> None:
        ...

    @abstractmethod
    def parse_expression(self) -> None:
        ...


class PolizGenerator:
    def __init__(self):
        self.output = []
        self.stack = []
        self.errors = []

    def generate(self, tokens: List[Token]) -> Tuple[List[str], List[Error]]:
        self.output = []
        self.stack = []
        self.errors = []

        if not tokens:
            self.errors.append(Error(0, 0, "Пустое выражение"))
            return [], self.errors

        try:
            self._parse_e(tokens)
            while self.stack:
                op = self.stack.pop()
                if op == '(':
                    self.errors.append(Error(0, 0, "Несбалансированные скобки"))
                self.output.append(op)
        except Exception as e:
            self.errors.append(Error(0, 0, f"Ошибка генерации ПОЛИЗ: {str(e)}"))

        return self.output, self.errors

    def _parse_e(self, tokens: List[Token]) -> int:
        pos = self._parse_t(tokens, 0)
        pos = self._parse_a(tokens, pos)
        return pos

    def _parse_a(self, tokens: List[Token], pos: int) -> int:
        while pos < len(tokens) and tokens[pos].type in ("PLUS", "MINUS"):
            op = tokens[pos]
            while (self.stack and self.stack[-1] != '(' and
                   self._get_precedence(self.stack[-1]) >= self._get_precedence(op.value)):
                self.output.append(self.stack.pop())
            self.stack.append(op.value)
            pos += 1
            pos = self._parse_t(tokens, pos)
        return pos

    def _parse_t(self, tokens: List[Token], pos: int) -> int:
        pos = self._parse_o(tokens, pos)
        while pos < len(tokens) and tokens[pos].type in ("MULTIPLY", "DIVIDE"):
            op = tokens[pos]
            while (self.stack and self.stack[-1] != '(' and
                   self._get_precedence(self.stack[-1]) >= self._get_precedence(op.value)):
                self.output.append(self.stack.pop())
            self.stack.append(op.value)
            pos += 1
            pos = self._parse_o(tokens, pos)
        return pos

    def _parse_o(self, tokens: List[Token], pos: int) -> int:
        if pos >= len(tokens):
            return pos

        token = tokens[pos]
        if token.type == "LPAREN":
            self.stack.append('(')
            pos += 1
            pos = self._parse_e(tokens[pos:])
            if pos < len(tokens) and tokens[pos].type == "RPAREN":
                while self.stack and self.stack[-1] != '(':
                    self.output.append(self.stack.pop())
                if self.stack and self.stack[-1] == '(':
                    self.stack.pop()
                else:
                    self.errors.append(Error(token.line, token.column, "Несбалансированные скобки"))
                pos += 1
            else:
                self.errors.append(Error(token.line, token.column, "Ожидается закрывающая скобка"))
        elif token.type == "NUM":
            self.output.append(token.value)
            pos += 1
        else:
            self.errors.append(Error(token.line, token.column,
                                  f"Неожиданный токен: {token.value}"))
        return pos

    def _get_precedence(self, op: str) -> int:
        if op in ("*", "/"):
            return 2
        if op in ("+", "-"):
            return 1
        return 0

    def evaluate_poliz(self, poliz: List[str]) -> Tuple[Optional[float], List[Error]]:
        stack = []
        errors = []

        for item in poliz:
            if item in "+-*/":
                if len(stack) < 2:
                    errors.append(Error(0, 0, f"Недостаточно операндов для оператора {item}"))
                    return None, errors
                b = stack.pop()
                a = stack.pop()
                try:
                    if item == "+":
                        stack.append(a + b)
                    elif item == "-":
                        stack.append(a - b)
                    elif item == "*":
                        stack.append(a * b)
                    elif item == "/":
                        if b == 0:
                            errors.append(Error(0, 0, "Деление на ноль"))
                            return None, errors
                        stack.append(a / b)
                except Exception as e:
                    errors.append(Error(0, 0, f"Ошибка вычисления: {str(e)}"))
                    return None, errors
            else:
                try:
                    stack.append(float(item))
                except ValueError:
                    errors.append(Error(0, 0, f"Неверное число: {item}"))
                    return None, errors

        if len(stack) != 1:
            errors.append(Error(0, 0, "Некорректное выражение"))
            return None, errors

        return stack[0], errors


class RegexLexer:
    _TOKEN_REGEX: List[Tuple[str, str]] = [
        (r"\+", "PLUS"),
        (r"-", "MINUS"),
        (r"\*", "MULTIPLY"),
        (r"/", "DIVIDE"),
        (r"\(", "LPAREN"),
        (r"\)", "RPAREN"),
        (r"\s+", "WHITESPACE"),
    ]

    def tokenize(self, text: str) -> Tuple[List['Token'], List['Error']]:
        if not text:
            return [], [Error(0, 0, "Входная строка пуста")]
        tokens = []
        errors = []
        pos = 0
        line, column = 1, 1

        while pos < len(text):
            if text[pos].isdigit():
                start_line = line
                start_column = column
                num_value = []
                current_pos = pos
                while current_pos < len(text):
                    c = text[current_pos]
                    if c.isdigit():
                        num_value.append(c)
                        current_pos += 1
                    else:
                        other_matched = False
                        for pattern, _ in self._TOKEN_REGEX:
                            if re.match(pattern, text[current_pos:]):
                                other_matched = True
                                break
                        if other_matched:
                            break
                        errors.append(Error(line, column, f"Неверный символ в числе: '{c}'"))
                        current_pos += 1
                    if c == '\n':
                        line += 1
                        column = 1
                    else:
                        column += 1 if c != '\t' else 4
                if num_value:
                    tokens.append(Token("NUM", ''.join(num_value), start_line, start_column))
                pos = current_pos
                continue

            matched = False
            for pattern, ttype in self._TOKEN_REGEX:
                match = re.match(pattern, text[pos:])
                if match:
                    value = match.group(0)
                    if ttype != "WHITESPACE":
                        tokens.append(Token(ttype, value, line, column))
                    lines = value.split('\n')
                    line += len(lines) - 1
                    column = len(lines[-1]) + 1 if len(lines) > 1 else column + len(value)
                    pos += len(value)
                    matched = True
                    break
            if matched:
                continue

            errors.append(Error(line, column, f"Неверный символ: '{text[pos]}'"))
            if text[pos] == '\n':
                line += 1
                column = 1
            else:
                column += 1
            pos += 1

        return tokens, errors


class RecursiveParser(IParser):
    def __init__(self) -> None:
        self.tokens: List[Token] = []
        self.pos: int = 0
        self.current: Optional[Token] = None
        self.errors: List[Error] = []
        self.poliz: List[str] = []

    def parse(self, tokens: List[Token]) -> Tuple[List[str], List[Error]]:
        self.tokens = tokens
        self.pos = 0
        self.current = tokens[0] if tokens else None
        self.errors.clear()
        self.poliz.clear()

        if not tokens:
            self.errors.append(Error(0, 0, "Пустое выражение"))
            return [], self.errors

        self._parse_e()
        if self.current is not None:
            self.errors.append(Error(self.current.line, self.current.column,
                                     f"Лишний токен '{self.current.value}'"))
        return self.poliz, self.errors

    def _advance(self) -> None:
        self.pos += 1
        self.current = self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _parse_e(self) -> None:
        self._parse_t()
        self._parse_a()

    def _parse_a(self) -> None:
        while self.current and self.current.type in ("PLUS", "MINUS"):
            op = self.current.value
            self._advance()
            self._parse_t()
            self.poliz.append(op)

    def _parse_t(self) -> None:
        self._parse_o()
        self._parse_b()

    def _parse_b(self) -> None:
        while self.current and self.current.type in ("MULTIPLY", "DIVIDE"):
            op = self.current.value
            self._advance()
            self._parse_o()
            self.poliz.append(op)

    def _parse_o(self) -> None:
        if not self.current:
            last = self.tokens[-1] if self.tokens else Token("", "", 0, 0)
            self.errors.append(Error(last.line, last.column,
                               "Ожидается число или скобка"))
            return
        tok = self.current
        if tok.type == "LPAREN":
            self._advance()
            self._parse_e()
            if self.current and self.current.type == "RPAREN":
                self._advance()
            else:
                self.errors.append(Error(tok.line, tok.column, "Незакрытая скобка"))
        elif tok.type == "NUM":
            self.poliz.append(tok.value)
            self._advance()
        else:
            self.errors.append(Error(tok.line, tok.column,
                                  f"Неожиданный токен '{tok.value}'"))
            self._advance()


class CompilerModel(ICompilerModel):
    def __init__(self, lexer: ILexer = None, parser: IParser = None) -> None:
        self.lexer: ILexer = lexer or RegexLexer()
        self.parser: IParser = parser or RecursiveParser()
        self.poliz_gen = PolizGenerator()

    def lex(self, text: str) -> Tuple[List[Token], List[Error]]:
        return self.lexer.tokenize(text)

    def parse(self, tokens: List[Token]) -> Tuple[List[str], List[Error]]:
        return self.parser.parse(tokens)

    def generate_poliz(self, tokens: List[Token]) -> Tuple[List[str], List[Error]]:
        return self.poliz_gen.generate(tokens)

    def evaluate_poliz(self, poliz: List[str]) -> Tuple[Optional[float], List[Error]]:
        return self.poliz_gen.evaluate_poliz(poliz)


class CompilerController(ICompilerController):
    def __init__(self, view: 'MainWindow', model: Optional[CompilerModel] = None) -> None:
        self.view = view
        self.model = model or CompilerModel()
        self._connect_signals()

    def _connect_signals(self) -> None:
        self.view.open_file_signal.connect(self.open_file)
        self.view.save_file_signal.connect(self.save_file)
        self.view.parse_signal.connect(self.parse_expression)

    def parse_expression(self) -> None:
        text = self.view.get_text().strip()
        if not text:
            self.view.show_error("Нечего анализировать: текст пуст")
            return

        tokens, lex_errors = self.model.lex(text)
        poliz, parse_errors = self.model.parse(tokens)
        result, eval_errors = self.model.evaluate_poliz(poliz)
        all_errors = lex_errors + parse_errors + eval_errors
        self.view.display_results(poliz, result, all_errors)

    def open_file(self, path: str) -> None:
        if not os.path.exists(path):
            self.view.show_error(f"Файл не найден: {path}")
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                self.view.set_text(f.read())
        except Exception as e:
            self.view.show_error(f"Ошибка при открытии: {e}")

    def save_file(self, path: str) -> None:
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.view.get_text())
        except Exception as e:
            self.view.show_error(f"Ошибка при сохранении: {e}")


class TextEditor(QPlainTextEdit):
    def __init__(self) -> None:
        super().__init__()
        self.setFont(QFont("Fira Code", 12))


class DocumentWidget(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.editor = TextEditor()
        self.tabs = QTabWidget()
        self._init_ui()

    def _init_ui(self) -> None:
        self._init_tables()
        self.layout.addWidget(self.editor)
        self.layout.addWidget(self.tabs)

    def _init_tables(self) -> None:
        self.error_table = self._make_table(["Строка", "Колонка", "Сообщение"], 3)
        self.poliz_table = self._make_table(["ПОЛИЗ"], 1)
        self.result_table = self._make_table(["Результат"], 1)
        self.tabs.addTab(self.poliz_table, "ПОЛИЗ")
        self.tabs.addTab(self.error_table, "Ошибки")
        self.tabs.addTab(self.result_table, "Результат")

    def _make_table(self, headers: List[str], cols: int) -> QTableWidget:
        tbl = QTableWidget()
        tbl.setColumnCount(cols)
        tbl.setHorizontalHeaderLabels(headers)
        tbl.horizontalHeader().setStretchLastSection(True)
        return tbl

    def display_results(self, poliz: List[str], result: Optional[float], errors: List[Error]) -> None:
        self._fill_table(self.error_table, errors, lambda e: [e.line, e.column, e.message])

        if errors:
            self._fill_table(self.poliz_table, ["Невозможно выполнить"], lambda p: [p])
            self.result_table.setRowCount(1)
            self.result_table.setItem(0, 0, QTableWidgetItem("Невозможно выполнить"))
        else:
            self._fill_table(self.poliz_table, poliz, lambda p: [p])
            self.result_table.setRowCount(1 if result is not None else 0)
            if result is not None:
                self.result_table.setItem(0, 0, QTableWidgetItem(f"{result}"))
            else:
                self.result_table.setRowCount(0)

    def _fill_table(self, table: QTableWidget, data: list, row_func) -> None:
        table.setRowCount(len(data))
        for i, item in enumerate(data):
            cells = row_func(item)
            for j, cell in enumerate(cells):
                table.setItem(i, j, QTableWidgetItem(str(cell)))

    def set_text(self, text: str) -> None:
        self.editor.setPlainText(text)

    def get_text(self) -> str:
        return self.editor.toPlainText()


class MainWindow(QMainWindow):
    open_file_signal = Signal(str)
    save_file_signal = Signal(str)
    parse_signal = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ПОЛИЗ")
        self.resize(800, 600)
        self.doc = DocumentWidget()
        self.setCentralWidget(self.doc)
        self._make_menu()

    def _make_menu(self) -> None:
        menu = self.menuBar()
        file_menu = menu.addMenu("Файл")
        open_act = QAction("Открыть", self)
        open_act.triggered.connect(self._on_open)
        save_act = QAction("Сохранить", self)
        save_act.triggered.connect(self._on_save)
        file_menu.addAction(open_act)
        file_menu.addAction(save_act)
        file_menu.addSeparator()
        exit_act = QAction("Выход", self)
        exit_act.triggered.connect(self.close)
        file_menu.addAction(exit_act)

        run_menu = menu.addMenu("Выполнить")
        run_act = QAction("Анализировать", self)
        run_act.triggered.connect(self.parse_signal.emit)
        run_menu.addAction(run_act)

    def _on_open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Открыть", "", "Текст (*.txt)")
        if path:
            self.open_file_signal.emit(path)

    def _on_save(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить", "", "Текст (*.txt)")
        if path:
            self.save_file_signal.emit(path)

    def set_text(self, text: str) -> None:
        self.doc.set_text(text)

    def get_text(self) -> str:
        return self.doc.get_text()

    def display_results(self, poliz: List[str], result: Optional[float], errors: List[Error]) -> None:
        self.doc.display_results(poliz, result, errors)

    def show_error(self, msg: str) -> None:
        QMessageBox.critical(self, "Ошибка", msg)


def main() -> None:
    app = QApplication(sys.argv)
    view = MainWindow()
    controller = CompilerController(view)
    view.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
