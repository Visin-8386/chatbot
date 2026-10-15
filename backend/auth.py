"""
Auth - RBAC-lite bằng API key phân quyền (không DB, không dependency mới).

Ngữ nghĩa:
- Không đặt key nào -> mở hoàn toàn (giữ hành vi dev cũ).
- Chỉ đặt API_KEY -> yêu cầu key này cho mọi endpoint được bảo vệ (hành vi cũ).
- Đặt API_READ_KEYS / API_WRITE_KEYS (CSV):
  - endpoint đọc (GET) cần key thuộc read HOẶC write;
  - endpoint ghi (POST/DELETE) cần key thuộc write.
  - API_KEY (nếu có) được tính như write-level để tương thích ngược.

Chuẩn hóa: strip khoảng trắng, bỏ rỗng, so sánh hằng số thời gian (hmac).
"""
import hmac
from typing import Optional, Set


def parse_keys(raw: Optional[str]) -> Set[str]:
    if not raw:
        return set()
    return {part.strip() for part in raw.split(",") if part.strip()}


def _matches(provided: Optional[str], allowed: Set[str]) -> bool:
    if not provided or not allowed:
        return False
    return any(hmac.compare_digest(provided, candidate) for candidate in allowed)


def is_read_allowed(
    provided: Optional[str],
    api_key: str,
    read_keys: Set[str],
    write_keys: Set[str],
) -> bool:
    """True khi: hệ mở (không key nào) / khớp API_KEY / khớp read / khớp write."""
    if not api_key and not read_keys and not write_keys:
        return True
    if api_key and provided and hmac.compare_digest(provided, api_key):
        return True
    return _matches(provided, read_keys) or _matches(provided, write_keys)


def is_write_allowed(
    provided: Optional[str],
    api_key: str,
    read_keys: Set[str],
    write_keys: Set[str],
) -> bool:
    """True khi: hệ mở / khớp API_KEY (full) / khớp write. Read-only KHÔNG đủ."""
    if not api_key and not read_keys and not write_keys:
        return True
    if api_key and provided and hmac.compare_digest(provided, api_key):
        return True
    return _matches(provided, write_keys)


def describe_auth(api_key: str, read_keys: Set[str], write_keys: Set[str]) -> str:
    if not api_key and not read_keys and not write_keys:
        return "open"
    if read_keys or write_keys:
        return "rbac"
    return "single-key"
