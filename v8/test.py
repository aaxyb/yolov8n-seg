from tests import CFG,  SOURCE
from ultralytics import RTDETR, YOLO
from ultralytics.utils import ROOT

CFG = 'ultralytics/cfg/models/v8/MobileNetV4.yaml'
SOURCE = ROOT / 'assets/bus.jpg'


def test_model_forward():
    """Test the forward pass of the YOLO model."""
    model = YOLO(CFG)
    model(SOURCE)  # also test no source and augment
