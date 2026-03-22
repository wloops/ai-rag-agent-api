import re


ZERO_WIDTH_CHARS = {
    "\ufeff",
    "\u200b",
    "\u200c",
    "\u200d",
    "\u2060",
}


def clean_text(text: str | None) -> str:
    if text is None:
        return ""

    # PDF 解析后的文本经常会混入控制字符、零宽字符和异常空白。
    # 如果不先清洗，这些脏数据会直接进入 chunk 和 embedding，影响检索质量。
    normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized_text = normalized_text.replace("\t", " ")
    normalized_text = normalized_text.replace("\u00a0", " ")
    normalized_text = normalized_text.replace("\u3000", " ")

    cleaned_chars: list[str] = []
    for char in normalized_text:
        if char == "\n":
            cleaned_chars.append(char)
            continue

        if char in ZERO_WIDTH_CHARS:
            continue

        if ord(char) < 32 or ord(char) == 127:
            continue

        cleaned_chars.append(char)

    cleaned_text = "".join(cleaned_chars)

    # 逐行压缩多余空白，但保留必要换行，这样既能保住段落边界，也不会把文本打散得过碎。
    cleaned_lines = []
    for line in cleaned_text.split("\n"):
        collapsed_line = re.sub(r" +", " ", line).strip()
        cleaned_lines.append(collapsed_line)

    cleaned_text = "\n".join(cleaned_lines)

    # 连续 3 个及以上换行压成 2 个换行，避免 PDF 抽取造成过多空段落。
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)

    return cleaned_text.strip()
