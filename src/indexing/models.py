from dataclasses import dataclass

@dataclass
class IndexingReport:
    collection_name: str
    chunks_inserted: int
    dimension: int
