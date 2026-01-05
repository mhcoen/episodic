import pytest

import episodic.unified_streaming_format as usf


class OutputCapture:
    def __init__(self):
        self.parts = []

    def secho(self, text=None, fg=None, nl=True, **kwargs):
        if text is None:
            text = ""
        self.parts.append(text)
        if nl:
            self.parts.append("\n")

    def echo(self, text=None):
        if text:
            self.parts.append(text)
        self.parts.append("\n")

    def write(self, text):
        self.parts.append(text)

    def flush(self):
        pass

    def output(self):
        return "".join(self.parts)


@pytest.fixture
def output_capture(monkeypatch):
    cap = OutputCapture()
    monkeypatch.setattr(usf.typer, "secho", cap.secho)
    monkeypatch.setattr(usf.typer, "echo", cap.echo)
    monkeypatch.setattr(usf.sys.stdout, "write", cap.write)
    monkeypatch.setattr(usf.sys.stdout, "flush", cap.flush)
    monkeypatch.setattr(usf, "process_stream_response", lambda gen, model: gen)
    return cap


def _bold(text, color="cyan"):
    color_codes = {
        'cyan': '\033[36m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'red': '\033[31m',
        'white': '\033[37m',
        'bright_cyan': '\033[96m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_red': '\033[91m',
        'bright_white': '\033[97m',
    }
    color_code = color_codes.get(color, '\033[96m')
    return f"{color_code}\033[1m{text}\033[0m"


def test_wrap_preserving_indent_no_wrap():
    assert usf._wrap_preserving_indent("short line", 20) == ["short line"]


def test_wrap_preserving_indent_exact_width():
    line = "abcdefghij"
    assert usf._wrap_preserving_indent(line, 10) == [line]


def test_wrap_preserving_indent_one_over():
    assert usf._wrap_preserving_indent("abcdefghijk", 10) == ["abcdefghij", "k"]


def test_wrap_preserving_indent_wrap_on_space():
    line = "word1 word2 word3"
    assert usf._wrap_preserving_indent(line, 10) == ["word1", "word2", "word3"]


def test_wrap_preserving_indent_no_spaces():
    line = "averylongword"
    assert usf._wrap_preserving_indent(line, 5) == ["avery", "longw", "ord"]


def test_wrap_preserving_indent_with_indent():
    line = "    word1 word2 word3"
    assert usf._wrap_preserving_indent(line, 10) == [
        "    word1",
        "    word2",
        "    word3",
    ]


def test_wrap_preserving_indent_multiple_wraps():
    line = "alpha beta gamma delta epsilon"
    assert usf._wrap_preserving_indent(line, 12) == [
        "alpha beta",
        "gamma delta",
        "epsilon",
    ]


def test_print_formatted_line_no_bold(output_capture):
    usf._print_formatted_line("plain text", "cyan")
    assert output_capture.output() == "plain text\n"


def test_print_formatted_line_single_bold(output_capture):
    usf._print_formatted_line("hello **bold** world", "cyan")
    assert output_capture.output() == f"hello {_bold('bold', 'cyan')} world\n"


def test_print_formatted_line_multiple_bold_regions(output_capture):
    usf._print_formatted_line("**one** and **two**", "cyan")
    assert output_capture.output() == f"{_bold('one', 'cyan')} and {_bold('two', 'cyan')}\n"


def test_print_formatted_line_unclosed_bold(output_capture):
    usf._print_formatted_line("**text without closing", "cyan")
    assert output_capture.output() == "**text without closing\n"


def test_print_formatted_line_empty_bold(output_capture):
    usf._print_formatted_line("****", "cyan")
    assert output_capture.output() == "\n"


def test_print_formatted_line_bold_at_start(output_capture):
    usf._print_formatted_line("**bold** start", "cyan")
    assert output_capture.output() == f"{_bold('bold', 'cyan')} start\n"


def test_print_formatted_line_bold_at_end(output_capture):
    usf._print_formatted_line("end **bold**", "cyan")
    assert output_capture.output() == f"end {_bold('bold', 'cyan')}\n"


def test_print_formatted_line_entire_line_bold(output_capture):
    usf._print_formatted_line("**bold**", "cyan")
    assert output_capture.output() == f"{_bold('bold', 'cyan')}\n"


def test_print_formatted_line_bullet_with_bold(output_capture):
    usf._print_formatted_line("- **important** item", "cyan")
    assert output_capture.output() == f"- {_bold('important', 'cyan')} item\n"


def test_stream_bold_single_chunk(output_capture):
    chunks = ["Hello **bold** world\n"]
    response = usf.stream_with_format_preservation(
        (c for c in chunks), model="test", prefix=None, color="cyan", wrap_width=None
    )
    assert response == "".join(chunks)
    assert output_capture.output() == f"Hello {_bold('bold', 'cyan')} world\n\n"


def test_stream_bold_split_across_chunks(output_capture):
    chunks = ["**bol", "d**\n"]
    response = usf.stream_with_format_preservation(
        (c for c in chunks), model="test", prefix=None, color="cyan", wrap_width=None
    )
    assert response == "".join(chunks)
    assert output_capture.output() == f"{_bold('bold', 'cyan')}\n\n"


def test_stream_bold_marker_split_across_chunks(output_capture):
    chunks = ["*", "*text**\n"]
    response = usf.stream_with_format_preservation(
        (c for c in chunks), model="test", prefix=None, color="cyan", wrap_width=None
    )
    assert response == "".join(chunks)
    assert output_capture.output() == f"{_bold('text', 'cyan')}\n\n"


def test_stream_small_chunks(output_capture):
    chunks = ["H", "i", "\n"]
    response = usf.stream_with_format_preservation(
        (c for c in chunks), model="test", prefix=None, color="cyan", wrap_width=None
    )
    assert response == "".join(chunks)
    assert output_capture.output() == "Hi\n\n"


def test_stream_chunks_ending_midword(output_capture):
    chunks = ["Hello ", "wor", "ld\n"]
    response = usf.stream_with_format_preservation(
        (c for c in chunks), model="test", prefix=None, color="cyan", wrap_width=None
    )
    assert response == "".join(chunks)
    assert output_capture.output() == "Hello world\n\n"


def test_stream_bullet_and_numbered_lines(output_capture):
    chunks = ["- item\n* item\n1. item\n"]
    response = usf.stream_with_format_preservation(
        (c for c in chunks), model="test", prefix=None, color="cyan", wrap_width=None
    )
    assert response == "".join(chunks)
    assert output_capture.output() == "- item\n* item\n1. item\n\n"


def test_stream_wrap_on_spaces(output_capture):
    chunks = ["word1 word2 word3\n"]
    response = usf.stream_with_format_preservation(
        (c for c in chunks), model="test", prefix=None, color="cyan", wrap_width=10
    )
    assert response == "".join(chunks)
    assert output_capture.output() == "word1\nword2\nword3\n\n"


def test_stream_wrap_no_spaces(output_capture):
    chunks = ["abcdefghijk\n"]
    response = usf.stream_with_format_preservation(
        (c for c in chunks), model="test", prefix=None, color="cyan", wrap_width=5
    )
    assert response == "".join(chunks)
    assert output_capture.output() == "abcde\nfghij\nk\n\n"


def test_stream_wrap_preserves_indent(output_capture):
    chunks = ["    word1 word2 word3\n"]
    response = usf.stream_with_format_preservation(
        (c for c in chunks), model="test", prefix=None, color="cyan", wrap_width=10
    )
    assert response == "".join(chunks)
    assert output_capture.output() == "    word1\n    word2\n    word3\n\n"


def test_stream_wrap_width_uses_wrap_width_not_80(output_capture):
    def gen():
        yield "A" * 90
        assert output_capture.output() == ""
        yield "\n"

    response = usf.stream_with_format_preservation(
        gen(), model="test", prefix=None, color="cyan", wrap_width=120
    )
    assert response == "A" * 90 + "\n"


def test_stream_wrap_boundary_at_bold_marker(output_capture):
    line = "aaaaaa**bold**bbbbbb\n"
    response = usf.stream_with_format_preservation(
        (c for c in [line]), model="test", prefix=None, color="cyan", wrap_width=10
    )
    assert response == line
    expected = f"aaaaaa{_bold('bold', 'cyan')}\nbbbbbb\n\n"
    assert output_capture.output() == expected


def test_stream_split_bold_marker_across_chunks_with_early_flush(output_capture):
    def gen():
        yield "A" * 81 + "*"
        yield "*bold**\n"

    response = usf.stream_with_format_preservation(
        gen(), model="test", prefix=None, color="cyan", wrap_width=120
    )
    assert response == "A" * 81 + "**bold**\n"
    assert _bold("bold", "cyan") in output_capture.output()
