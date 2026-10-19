"""Local feature registration to an annotated template; no arbitrary-circle selection."""
import cv2
import numpy as np


def locate_eye(template_rgb, camera_rgb, template_eye_px, template_radius_px):
    detector=cv2.ORB_create(nfeatures=2500,fastThreshold=12)
    ka,da=detector.detectAndCompute(cv2.cvtColor(template_rgb,cv2.COLOR_RGB2GRAY),None)
    kb,db=detector.detectAndCompute(cv2.cvtColor(camera_rgb,cv2.COLOR_RGB2GRAY),None)
    if da is None or db is None:raise ValueError('Template or image has insufficient texture')
    matches=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da,db,k=2)
    good=[pair[0] for pair in matches if len(pair)==2 and pair[0].distance<.7*pair[1].distance]
    if len(good)<16:raise ValueError('Template not recognized')
    a=np.float32([ka[g.queryIdx].pt for g in good]);b=np.float32([kb[g.trainIdx].pt for g in good])
    H,mask=cv2.findHomography(a,b,cv2.RANSAC,2.)
    if H is None or mask.mean()<.6:raise ValueError('Ambiguous template registration')
    inliers=mask.ravel().astype(bool)
    extent=np.ptp(a[inliers],axis=0)/np.array(template_rgb.shape[:2][::-1])
    if min(extent)<.35:raise ValueError('Matches clustered; eye extrapolation unsafe')
    projected=cv2.perspectiveTransform(a.reshape(-1,1,2),H).reshape(-1,2)
    error=float(np.quantile(np.linalg.norm(projected[inliers]-b[inliers],axis=1),.95))
    x,y=template_eye_px;r=template_radius_px
    points=np.float32([[x,y],[x+r,y],[x,y+r]])
    q=cv2.perspectiveTransform(points.reshape(-1,1,2),H).reshape(-1,2)
    radius=float(min(np.linalg.norm(q[1]-q[0]),np.linalg.norm(q[2]-q[0])))
    h,w=camera_rgb.shape[:2]
    if not 0<=q[0,0]<w or not 0<=q[0,1]<h or error>2 or radius<=0:raise ValueError('Invalid eye projection')
    return dict(center_px=q[0].tolist(),minimum_projected_radius_px=radius,
                registration_p95_error_px=error,inliers=int(inliers.sum()),
                template_verified=True,official_template=False)
