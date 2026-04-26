from backend.document_processor import (
    _collect_repeated_margin_signatures,
    _remove_repeated_margin_lines,
    _split_lines,
    chunk_text,
)


def test_pdf_margin_cleanup_conservative() -> None:
    pages = [
        """ACME CORP
Bao cao quy trinh

Noi dung trang 1
Thong tin chi tiet ve quy trinh thuc hien.
Trang 1""",
        """ACME CORP
Bao cao quy trinh

Noi dung trang 2
Thong tin chi tiet ve quy trinh thuc hien.
Trang 2""",
        """ACME CORP
Bao cao quy trinh

Noi dung trang 3
Thong tin chi tiet ve quy trinh thuc hien.
Trang 3""",
    ]

    page_lines = [_split_lines(page) for page in pages]
    repeated = _collect_repeated_margin_signatures(page_lines)

    assert "acme corp" in repeated
    assert "bao cao quy trinh" in repeated

    cleaned = [_remove_repeated_margin_lines(lines, repeated) for lines in page_lines]
    assert all("ACME CORP" not in line for page in cleaned for line in page)
    assert all("Bao cao quy trinh" not in line for page in cleaned for line in page)
    # Real content should remain.
    assert any("Thong tin chi tiet ve quy trinh thuc hien." in line for page in cleaned for line in page)


def test_chunk_text_quality() -> None:
    sample = """MUC LUC

1. Quy dinh chung
2. Quy trinh thuc hien

Quy trinh thuc hien: Nhan vien nop de xuat. Truong bo phan xem xet. Ban giam doc phe duyet.

- Buoc 1: Nop form
- Buoc 2: Kiem tra thong tin
- Buoc 3: Phe duyet

Bang tong hop | Gia tri | Ghi chu
A | 10 | Hop le
B | 20 | Can bo sung
"""

    chunks = chunk_text(sample, chunk_size=140, overlap=40)

    assert len(chunks) >= 2
    assert all(chunk.strip() for chunk in chunks)
    assert all(len(chunk) <= 175 for chunk in chunks)
    assert any("Bang tong hop | Gia tri | Ghi chu" in chunk for chunk in chunks)


def main() -> None:
    test_pdf_margin_cleanup_conservative()
    test_chunk_text_quality()
    print("Chunking tests passed.")


if __name__ == "__main__":
    main()
