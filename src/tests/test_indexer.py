import pytest
import tempfile
import os
import uuid
from src.patent_search.core.indexer import TextIndexer, ProcessedText
from src.patent_search.config import EMBEDDING_SIZE

@pytest.fixture
def sample_texts():
    return [
        ProcessedText(
            text="This is the first test document.",
            embedding=[0.1] * EMBEDDING_SIZE,
            language="en",
            metadata={"id": 3}
        )
    ]

@pytest.fixture
def indexer():
    return TextIndexer(collection_name="test_collection")

def test_add_texts(indexer, sample_texts):
    """Test adding texts to the index."""
    # Add texts
    indexer.add_texts(sample_texts)

    # Check if texts were added
    assert len(indexer.metadata) == 1
    assert indexer.ntotal == 1

def test_save_load_index(indexer, sample_texts):
    """Test saving and loading the index."""
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = os.path.join(tmpdir, "test_index")

        # Add texts and save index
        indexer.add_texts(sample_texts)

        # Verify data was added
        assert indexer.ntotal > 0

        # Save index
        indexer.save_index(index_path)

        # Verify files were created
        assert os.path.exists(os.path.join(index_path, "state.json"))
        assert os.path.exists(os.path.join(index_path, "vectors.json"))

        # Create new indexer and load saved index
        new_indexer = TextIndexer(
            collection_name=f"test_collection_new_{uuid.uuid4()}",
            dimension=EMBEDDING_SIZE,
            load_from=index_path
        )

        # Verify metadata was loaded
        assert len(new_indexer.metadata) == len(indexer.metadata)
        assert new_indexer.ntotal == indexer.ntotal

        # Verify search still works
        query_embedding = sample_texts[0].embedding
        results = new_indexer.search(query_embedding, top_k=1)
        assert len(results) == 1
        assert results[0].text == sample_texts[0].text

def test_search(indexer, sample_texts):
    """Test searching in the index."""
    # Add texts
    indexer.add_texts(sample_texts)

    # Search
    results = indexer.search(
        query_embedding=[0.1] * EMBEDDING_SIZE,
        top_k=1
    )

    assert len(results) == 1
    assert results[0].text == sample_texts[0].text
