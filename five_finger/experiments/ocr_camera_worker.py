"""Image-only OCR worker. No examiner, scenario metadata, expected answer or bank input."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import sys
import cv2
import numpy as np
from fivefinger.vision import OfflineOCR


def rectify_displays(image):
    """Find bright quadrilateral display panels in pixels, without model poses or text hints."""
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    mask=cv2.morphologyEx((gray>215).astype(np.uint8)*255,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    result=[]
    for contour in contours:
        if cv2.contourArea(contour)<5000:continue
        poly=cv2.approxPolyDP(contour,.025*cv2.arcLength(contour,True),True)
        if len(poly)!=4 or not cv2.isContourConvex(poly):continue
        p=poly[:,0,:].astype(np.float32);total=p.sum(axis=1);diff=np.diff(p,axis=1).ravel()
        corners=np.array([p[np.argmin(total)],p[np.argmin(diff)],p[np.argmax(total)],p[np.argmax(diff)]],np.float32)
        width=max(np.linalg.norm(corners[1]-corners[0]),np.linalg.norm(corners[2]-corners[3]))
        height=max(np.linalg.norm(corners[3]-corners[0]),np.linalg.norm(corners[2]-corners[1]))
        if not 1.5<width/max(height,1)<5:continue
        # Aspect is the calibrated display format; detected corners all come from RGB.
        matrix=cv2.getPerspectiveTransform(corners,np.array([[0,0],[999,0],[999,419],[0,419]],np.float32))
        result.append((cv2.warpPerspective(image,matrix,(1000,420)),corners,matrix))
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('image');parser.add_argument('output');args=parser.parse_args()
    def image_only_boundary(event,arguments):
        if event=='open' and isinstance(arguments[0],(str,bytes)):
            name=str(arguments[0])
            if any(v in name for v in ('examiner_private','example_poems.json','planned_text.txt','scene.json')):
                raise PermissionError('OCR worker may read camera pixels and model weights, not task answers')
    sys.addaudithook(image_only_boundary)
    raw=Path(args.image).read_bytes();image=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
    if image is None:raise ValueError('Invalid camera image')
    start=time.perf_counter();ocr=OfflineOCR();panels=rectify_displays(image);detections=[];regions=[]
    for number,(panel,corners,matrix) in enumerate(panels):
        region_file=Path(args.output).with_name(f'ocr_rectified_{number}.png')
        cv2.imencode('.png',panel)[1].tofile(str(region_file))
        regions.append({'camera_corners_px':corners.tolist(),'rectified_image':region_file.name})
        for detection in ocr.read(panel):
            detection['box']=cv2.perspectiveTransform(np.asarray(detection['box'],np.float32)[None],np.linalg.inv(matrix))[0].tolist()
            detection['source']='RGB detected quadrilateral, perspective rectified';detections.append(detection)
    if not detections:detections=ocr.read(image)
    # Spatial order only, with no vocabulary-based correction.
    detections.sort(key=lambda x:(np.mean(np.asarray(x['box'])[:,1]),np.mean(np.asarray(x['box'])[:,0])))
    result={'input_image_sha256':hashlib.sha256(raw).hexdigest(),'image_shape':list(image.shape),
            'detections':detections,'display_regions':regions,'elapsed_wall_s':time.perf_counter()-start}
    Path(args.output).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
