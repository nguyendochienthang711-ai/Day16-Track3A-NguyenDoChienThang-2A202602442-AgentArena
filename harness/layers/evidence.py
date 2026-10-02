"""BẰNG CHỨNG CHUNG cho `critic` và `citation_checker`.

Hai lớp trước đây mỗi lớp tự cài một bản `_is_doc_seen` giống hệt nhau, và
cả hai đều lấy `seen_ids` bằng cách quét regex `doc-\\d{4}` trên TOÀN BỘ
`ctx.messages`. Như vậy một tài liệu được coi là "đã đọc" nếu mã của nó chỉ
xuất hiện trong system prompt (phụ lục có ví dụ `doc-0004`), trong ACTION
của chính mô hình (kể cả khi lượt fetch đó hỏng), hay trong một câu chữ bất
kỳ. Scorer thì chấm `UNRETRIEVED` cho đúng những trường hợp ấy.

Module này gom logic về một chỗ và chỉ tin vào thứ agent THỰC SỰ đã thấy:

  * `ctx.fetched`          — fetch_doc thành công, sạch (agent ghi lại)
  * `ctx.fetched_partial`  — fetch_doc ok=True nhưng bị cắt/nhiễu
  * `ctx.observed_text`    — mọi quan sát (kể cả snippet của search)

Nếu `ctx` không có sổ `fetched` (ví dụ test dùng ctx giả) thì quay về luật
cũ: toàn văn tài liệu nằm trong quan sát. Không đụng vào `claim["text"]`.
"""

from __future__ import annotations

import re

_DOC_ID_RE = re.compile(r"doc-\d{4}")

try:  # cùng một hàm làm sạch với injection_guard
    from harness.injection_guard import sanitize_untrusted as _sanitize
except Exception:  # pragma: no cover - chỉ để module độc lập khi test
    def _sanitize(text: str) -> str:
        return text


class Evidence:
    """Ảnh chụp bằng chứng của MỘT lượt chạy, dựng một lần ở `after_agent`."""

    def __init__(self, ctx) -> None:
        self.observed: str = getattr(ctx, "observed_text", "") or ""
        corpus = getattr(ctx, "corpus", None)
        self.docs: list = list(getattr(corpus, "docs", None) or []) if corpus is not None else []
        self.by_id: dict = {d.doc_id: d for d in self.docs}
        self.order: dict = {d.doc_id: i for i, d in enumerate(self.docs)}
        self._lines: dict = {}

        clean = getattr(ctx, "fetched", None)
        partial = getattr(ctx, "fetched_partial", None)
        self.has_ledger = isinstance(clean, dict)
        self.clean_ids: set = set(clean) if isinstance(clean, dict) else set()
        self.partial_ids: set = set(partial) if isinstance(partial, dict) else set()
        # Mã tài liệu xuất hiện trong QUAN SÁT (snippet của search) — nhưng
        # không phải trong system prompt hay lời của chính mô hình.
        self.listed_ids: set = set(_DOC_ID_RE.findall(self.observed))

    # -- tài liệu -------------------------------------------------------

    def get(self, doc_id: str):
        return self.by_id.get(doc_id)

    def lines(self, doc) -> list:
        cached = self._lines.get(doc.doc_id)
        if cached is None:
            cached = doc.body.splitlines() if doc and doc.body else []
            self._lines[doc.doc_id] = cached
        return cached

    def in_line(self, text: str, doc) -> bool:
        """`text` là một đoạn nguyên văn nằm gọn trong MỘT DÒNG của `doc`?"""
        if not doc or not text:
            return False
        return any(text in line for line in self.lines(doc))

    def is_clean(self, doc) -> bool:
        """Toàn văn tài liệu đã về sạch (fetch sạch, hoặc nằm nguyên trong quan sát)."""
        if not doc:
            return False
        if doc.doc_id in self.clean_ids:
            return True
        if self.has_ledger:
            # Có sổ đọc thì CHỈ tin sổ. Luật "toàn văn nằm trong quan sát" sai
            # với bản nhái: một tài liệu lookalike chép lại câu của bản gốc có
            # body là tập con của quan sát, nên bị coi là đã fetch sạch.
            return False
        body = doc.body or ""
        return bool(body) and (body in self.observed or _sanitize(body) in self.observed)

    def is_seen(self, doc) -> bool:
        """Tài liệu đã được agent thực sự đọc (hoặc ít nhất liệt kê trong quan sát)."""
        if not doc:
            return False
        if self.is_clean(doc):
            return True
        return doc.doc_id in self.partial_ids or doc.doc_id in self.listed_ids

    # -- câu trích ------------------------------------------------------

    def candidates(self, text: str) -> list:
        """Các tài liệu ĐÃ THẤY có một dòng chứa nguyên văn `text`.

        Sắp xếp: tài liệu về sạch trước, rồi theo thứ tự trong corpus (ổn định)."""
        if not text:
            return []
        found = [d for d in self.docs if self.is_seen(d) and self.in_line(text, d)]
        found.sort(key=lambda d: (not self.is_clean(d), self.order.get(d.doc_id, 0)))
        return found

    def grounded(self, text: str) -> bool:
        """Câu có trong quan sát VÀ nằm gọn trong một dòng của tài liệu đã thấy.

        Không có corpus (test) thì rơi về luật cũ: `text in observed`."""
        if not isinstance(text, str) or not text.strip() or text not in self.observed:
            return False
        if not self.docs:
            return True
        return bool(self.candidates(text))