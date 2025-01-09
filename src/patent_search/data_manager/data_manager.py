from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import pickle
from typing import List, Optional
from tqdm import tqdm
from loguru import logger
import threading
from src.patent_search.core.processor import ProcessedText
from src.patent_search.config import BASE_FOLDER


@dataclass
class ProcessingConfig:
    use_cache: bool = True
    cache_path: Path = BASE_FOLDER / "data" / "processed" / "processed_patents.pkl"
    force_reprocess: bool = False

    @property
    def should_use_cache(self) -> bool:
        return self.use_cache and not self.force_reprocess and self.cache_path.exists()


class DataManager:
    def __init__(self, service, config: Optional[ProcessingConfig] = None):
        self.service = service
        self.config = config or ProcessingConfig()
        self._cache_lock = threading.Lock()  # Add thread safety

    def load_or_process_data(self, data_path: Path) -> List[ProcessedText]:
        """Load data from cache if available, otherwise process and cache it."""
        if self.config.should_use_cache:
            try:
                processed_texts = self._load_from_cache()
                if processed_texts:
                    logger.info(f"Loaded {len(processed_texts)} processed patents from cache")
                    return processed_texts
            except Exception as e:
                logger.error(f"Error loading cache: {e}. Will reprocess data.")

        return self._process_and_cache_data(data_path)

    def _load_from_cache(self) -> Optional[List[ProcessedText]]:
        """Load processed data from cache."""
        if self.config.cache_path.exists():
            with open(self.config.cache_path, "rb") as f:
                return pickle.load(f)
        return None

    def _process_and_cache_data(self, data_path: Path) -> List[ProcessedText]:
        """Process raw data and cache the results."""
        logger.info("Processing new data...")

        # Read raw data
        try:
            with open(data_path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(data_path, "r", encoding="latin-1") as f:
                content = f.read()

        patent_texts = [abstract for abstract in content.split("\n")]

        # Process texts
        processed_texts = []
        for text in tqdm(patent_texts, desc="Processing patents", unit="patent"):
            try:
                language = self.service.detect_language(text)
                embedding = self.service.model.encode(text)
                processed_text = ProcessedText(
                    text=text,
                    embedding=embedding,
                    language=language,
                    metadata={
                        "processed_date": datetime.now().isoformat(),
                        "char_length": len(text),
                        "word_count": len(text.split()),
                    },
                )
                processed_texts.append(processed_text)
            except Exception as e:
                logger.error(f"Error processing patent: {e}")
                continue

        # Cache the results
        if self.config.use_cache:
            self._save_to_cache(processed_texts)

        return processed_texts

    def _save_to_cache(self, processed_texts: List[ProcessedText]) -> None:
        """Save processed data to cache."""
        logger.info(f"Saving {len(processed_texts)} processed patents to cache...")
        self.config.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config.cache_path, "wb") as f:
            pickle.dump(processed_texts, f)
        logger.info("Cache saved successfully")

    def append_to_cache(self, new_processed_texts: List[ProcessedText]) -> None:
        """Append new processed texts to the existing cache."""
        if not self.config.use_cache:
            return

        with self._cache_lock:
            try:
                # Load existing cache
                existing_texts = self._load_from_cache() or []

                # Append new texts
                updated_texts = existing_texts + new_processed_texts

                # Save updated cache
                self._save_to_cache(updated_texts)
                logger.info(
                    f"Successfully appended {len(new_processed_texts)} new patents to cache"
                )
            except Exception as e:
                logger.error(f"Error updating cache: {e}")
                raise
