from utils import config

def test_config_values():
    assert hasattr(config, 'EDSDK_PATH')
    assert config.GALLERY_FILE.endswith('.json')
    assert config.YOLO_MODEL.endswith('.pt')
