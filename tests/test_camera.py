from camera.canon_sdk import CanonCamera

def test_camera_init_no_dll():
    cam = CanonCamera()
    assert hasattr(cam, 'available')
    assert cam.get_frame() is None
