from rag_core.utils.config import load_config


def test_load_config():
    config = load_config()
    assert config.project.name == "reliable-rag"
    assert len(config.data.urls) > 0
    assert config.llms.generator.model

