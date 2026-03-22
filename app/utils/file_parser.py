from pathlib import Path


def parse_file(path: str | Path, file_type: str) -> str:
    # 路由层只关心“能不能解析成功”，具体解析细节统一收口到这里。
    normalized_file_type = file_type.lower()
    file_path = Path(path)

    if normalized_file_type == "txt":
        return _parse_text_file(file_path)
    if normalized_file_type == "md":
        return _parse_markdown_file(file_path)
    if normalized_file_type == "pdf":
        return _parse_pdf_file(file_path)

    raise ValueError(f"Unsupported file type: {file_type}")


def _parse_text_file(path: Path) -> str:
    # 文本文件统一按 UTF-8 读取，不合法时明确抛错，避免后续出现脏数据。
    return path.read_text(encoding="utf-8")


def _parse_markdown_file(path: Path) -> str:
    # Day 2 只需要拿到原始文本，Markdown 暂时不做额外结构化处理。
    return path.read_text(encoding="utf-8")


def _parse_pdf_file(path: Path) -> str:
    # 这里只支持文本型 PDF，所以直接提取文本；库缺失或提取失败都明确报错。
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "PDF parsing dependency is missing, please install pypdf first"
        ) from exc

    reader = PdfReader(str(path))
    pages_text: list[str] = []

    for page in reader.pages:
        page_text = page.extract_text() or ""
        pages_text.append(page_text)

    return "\n".join(pages_text)
