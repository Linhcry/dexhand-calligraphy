"""Run with .venv (the OCR environment), independent of task control."""
import cv2
import numpy as np
from experiments.ocr_camera_worker import rectify_displays


def test_display_rectification_uses_observed_corners():
    image=np.zeros((600,900,3),np.uint8)+60
    corners=np.array([[390,80],[820,110],[740,300],[330,260]],np.int32)
    cv2.fillConvexPoly(image,corners,(245,245,245))
    cv2.line(image,(420,130),(730,150),(0,0,0),5)
    # A portrait sheet is deliberately present; it should not replace the display.
    cv2.rectangle(image,(30,80),(200,500),(255,255,255),-1)
    panels=rectify_displays(image)
    assert len(panels)==1
    panel,observed,matrix=panels[0]
    assert panel.shape==(420,1000,3)
    assert (panel.mean(axis=2)<100).sum()>500
    assert np.linalg.norm(observed-corners,axis=1).max()<4


def test_dark_or_absent_display_has_no_invented_region():
    assert rectify_displays(np.zeros((600,900,3),np.uint8))==[]
