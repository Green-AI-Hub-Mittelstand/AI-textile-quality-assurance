import sys

from PIL import Image
import sane

class Scanner:
    def __init__(self):
        ver = sane.init()
        print('SANE version:', ver)
        while True:
            try:
                devices = sane.get_devices()  # this is slow, so only do it once
                print('Available scanners:', devices)
                self.device = devices[0][0]  # Use first device
                break
            except KeyError:
                print("No scanner found, trying again!", file=sys.stderr)

    def scan_document(self, dpi: int = 600) -> Image:
        dev = sane.open(self.device)
        try:
            dev.resolution = dpi
            image = dev.scan()
        finally:
            dev.close()
        return image
