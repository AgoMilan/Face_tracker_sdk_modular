# camera/canon_sdk.py
import os, time, ctypes, numpy as np, cv2
from utils.config import EDSDK_PATH
edsdk = None
try:
    edsdk = ctypes.WinDLL(EDSDK_PATH)
except Exception:
    edsdk = None
EDS_OK = 0x00000000
kEdsPropID_Evf_OutputDevice = 0x00000500
kEdsEvfOutputDevice_PC = 0x00000002
kEdsPropID_Record = 0x00000501
kEdsRecord_Stop = 0
kEdsRecord_Start = 4
def check_error(err, action=""):
    if err != EDS_OK:
        raise RuntimeError(f"Canon SDK Error {hex(err)} při {action}")
class CanonCamera:
    def __init__(self):
        if edsdk is None:
            self.available = False
            self.edsdk = None
        else:
            self.available = True
            self.edsdk = edsdk
        self.cam_ref = None
    def init(self):
        if not self.available:
            raise RuntimeError("EDSDK DLL not available")
        print("📸 Inicializuji Canon EDSDK...")
        check_error(self.edsdk.EdsInitializeSDK(), "EdsInitializeSDK")
        cam_list = ctypes.c_void_p()
        check_error(self.edsdk.EdsGetCameraList(ctypes.byref(cam_list)), "EdsGetCameraList")
        cam_ref = ctypes.c_void_p()
        check_error(self.edsdk.EdsGetChildAtIndex(cam_list, 0, ctypes.byref(cam_ref)), "EdsGetChildAtIndex")
        try:
            self.edsdk.EdsRelease(cam_list)
        except Exception:
            pass
        check_error(self.edsdk.EdsOpenSession(cam_ref), "EdsOpenSession")
        self.cam_ref = cam_ref
        print("✅ Session otevřena.")
        return cam_ref
    def start_liveview(self):
        if not self.available:
            print("EDSDK not available: start_liveview skipped")
            return
        output_device = ctypes.c_int(kEdsEvfOutputDevice_PC)
        try:
            check_error(self.edsdk.EdsSetPropertyData(self.cam_ref, kEdsPropID_Evf_OutputDevice, 0,
                                                     ctypes.sizeof(output_device), ctypes.byref(output_device)),
                        "EdsSetPropertyData(Evf_OutputDevice)")
        except Exception as e:
            print(f"[CANON] ⚠ Nešlo nastavit Evf_OutputDevice: {e}")
        record_state = ctypes.c_int(kEdsRecord_Start)
        try:
            err = self.edsdk.EdsSetPropertyData(self.cam_ref, kEdsPropID_Record, 0,
                                               ctypes.sizeof(record_state), ctypes.byref(record_state))
            if err != EDS_OK:
                print(f"⚠️ Nelze spustit Movie Mode (err={hex(err)}) – pokračuji jen s LiveView.")
        except Exception:
            pass
        time.sleep(1.0)
        print("✅ Movie Mode aktivní, LiveView běží.")
    def get_frame(self):
        if not self.available:
            return None
        try:
            stream_ref = ctypes.c_void_p()
            self.edsdk.EdsCreateMemoryStream(0, ctypes.byref(stream_ref))
            evf_image = ctypes.c_void_p()
            self.edsdk.EdsCreateEvfImageRef(stream_ref, ctypes.byref(evf_image))
            err = self.edsdk.EdsDownloadEvfImage(self.cam_ref, evf_image)
            if err != EDS_OK:
                try:
                    self.edsdk.EdsRelease(evf_image)
                    self.edsdk.EdsRelease(stream_ref)
                except Exception:
                    pass
                return None
            pointer = ctypes.c_void_p()
            size = ctypes.c_uint64()
            self.edsdk.EdsGetPointer(stream_ref, ctypes.byref(pointer))
            self.edsdk.EdsGetLength(stream_ref, ctypes.byref(size))
            if size.value == 0:
                try:
                    self.edsdk.EdsRelease(evf_image)
                    self.edsdk.EdsRelease(stream_ref)
                except Exception:
                    pass
                return None
            data = (ctypes.c_ubyte * size.value).from_address(pointer.value)
            img_array = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            try:
                self.edsdk.EdsRelease(evf_image)
                self.edsdk.EdsRelease(stream_ref)
            except Exception:
                pass
            return frame
        except Exception:
            return None
    def stop_liveview(self):
        if not self.available:
            return
        try:
            record_state = ctypes.c_int(kEdsRecord_Stop)
            self.edsdk.EdsSetPropertyData(self.cam_ref, kEdsPropID_Record, 0, ctypes.sizeof(record_state),
                                         ctypes.byref(record_state))
        except Exception:
            pass
        time.sleep(0.3)
    def close(self):
        if not self.available:
            return
        try:
            self.edsdk.EdsCloseSession(self.cam_ref)
        except Exception:
            pass
        try:
            self.edsdk.EdsTerminateSDK()
        except Exception:
            pass
