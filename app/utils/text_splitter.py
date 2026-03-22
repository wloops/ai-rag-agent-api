def _validate_split_args(chunk_size: int, overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0:
        raise ValueError("overlap must be greater than or equal to 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size")


def iter_chunk_ranges(
    text: str | None, chunk_size: int = 700, overlap: int = 100
) -> list[tuple[int, int]]:
    _validate_split_args(chunk_size, overlap)

    if text is None:
        return []

    if not text.strip():
        return []

    ranges: list[tuple[int, int]] = []
    text_length = len(text)
    start = 0
    step = chunk_size - overlap

    while start < text_length:
        end = min(start + chunk_size, text_length)
        ranges.append((start, end))

        if end >= text_length:
            break

        next_start = start + step
        # chunk_size 表示单个 chunk 的最大字符窗口。
        # overlap 表示相邻 chunk 之间保留的重叠字符数，用来减少语义断裂。
        # 这里强制保证下一轮起点持续前进，避免边界条件导致死循环。
        if next_start <= start:
            next_start = end

        start = next_start

    return ranges


def split_text(text: str | None, chunk_size: int = 700, overlap: int = 100) -> list[str]:
    if text is None:
        return []

    if not text.strip():
        return []

    return [text[start:end] for start, end in iter_chunk_ranges(text, chunk_size, overlap)]
