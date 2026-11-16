"""Human-readable metadata for each PDF in `documents/`.

File names such as `000039238.pdf` mean nothing to a user, so every citation is
shown with the official title and the publishing ministry instead.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DocMeta:
    doc_id: str  # file stem
    title: str
    ministry: str
    ministry_en: str


CATALOG: dict[str, DocMeta] = {
    m.doc_id: m
    for m in [
        DocMeta(
            "000039238",
            "令和5年版 環境白書・循環型社会白書・生物多様性白書（要約）",
            "環境省",
            "Ministry of the Environment",
        ),
        DocMeta("001124666", "令和5年版 厚生労働白書（概要）", "厚生労働省", "Ministry of Health, Labour and Welfare"),
        DocMeta(
            "01point",
            "令和5年版 情報通信白書（情報通信に関する現状報告の概要）",
            "総務省",
            "Ministry of Internal Affairs and Communications",
        ),
        DocMeta("2023_gaiyo", "通商白書2023（概要）", "経済産業省", "Ministry of Economy, Trade and Industry"),
        DocMeta(
            "20240719-mxt_soseisk01-000037016_07",
            "文部科学白書2023（令和5年度）第2部第1章 抜粋",
            "文部科学省",
            "Ministry of Education, Culture, Sports, Science and Technology",
        ),
    ]
}


def get_meta(doc_id: str) -> DocMeta:
    """Return catalog metadata, falling back gracefully for PDFs not yet catalogued."""
    return CATALOG.get(doc_id, DocMeta(doc_id, doc_id, "不明", "Unknown"))
