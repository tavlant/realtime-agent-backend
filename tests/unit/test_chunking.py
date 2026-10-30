import itertools

import pytest

from agent_backend.rag.chunking import chunk_text, split_sections

LOREM = (
    "Redis keeps the hot conversation state close to the API. Postgres stores the full "
    "history and is the source of truth. A per-conversation lock prevents interleaving. "
)


def test_empty_and_whitespace_input() -> None:
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []


def test_short_text_is_one_chunk() -> None:
    chunks = chunk_text("Just a short note.")
    assert [c.text for c in chunks] == ["Just a short note."]
    assert chunks[0].index == 0
    assert chunks[0].heading is None


@pytest.mark.parametrize(("max_chars", "overlap"), [(200, 0), (200, 40), (500, 100), (80, 20)])
def test_chunks_respect_max_size(max_chars: int, overlap: int) -> None:
    text = "# Guide\n\n" + "\n\n".join(LOREM * 3 for _ in range(10))
    chunks = chunk_text(text, max_chars=max_chars, overlap=overlap)
    assert len(chunks) > 1
    assert all(len(c.text) <= max_chars for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_neighbouring_chunks_overlap() -> None:
    words = " ".join(f"word{i}" for i in range(400))
    chunks = chunk_text(words, max_chars=200, overlap=50)
    for previous, current in itertools.pairwise(chunks):
        tail_words = previous.text.split()[-3:]
        assert " ".join(tail_words) in current.text


def test_no_content_is_lost_without_overlap() -> None:
    words = [f"w{i}" for i in range(300)]
    chunks = chunk_text(" ".join(words), max_chars=120, overlap=0)
    rebuilt = " ".join(c.text for c in chunks).split()
    assert rebuilt == words


def test_markdown_sections_carry_heading_path() -> None:
    text = (
        "# Handbook\nIntro paragraph.\n\n"
        "## Deployment\nUse docker compose.\n\n"
        "### Secrets\nNever commit keys.\n\n"
        "## Testing\nRun pytest.\n"
    )
    sections = split_sections(text)
    assert [h for h, _ in sections] == [
        "Handbook",
        "Handbook > Deployment",
        "Handbook > Deployment > Secrets",
        "Handbook > Testing",
    ]
    chunks = chunk_text(text)
    assert chunks[2].text.startswith("Handbook > Deployment > Secrets\n")
    assert "Never commit keys." in chunks[2].text


def test_headings_inside_code_fences_are_ignored() -> None:
    text = "# Real\n```bash\n# not a heading\necho hi\n```\n"
    sections = split_sections(text)
    assert len(sections) == 1
    assert sections[0][0] == "Real"
    assert "# not a heading" in sections[0][1]


def test_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="max_chars"):
        chunk_text("x", max_chars=0)
    with pytest.raises(ValueError, match="overlap"):
        chunk_text("x", max_chars=100, overlap=100)
