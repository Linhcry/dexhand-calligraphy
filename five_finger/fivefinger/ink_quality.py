"""Detect ink outside planned strokes, including unintended connecting marks."""
import numpy as np
from scipy.spatial import cKDTree


def measure_ink_precision(ink, planned_xy, tolerance_m=.0025):
    ink=np.asarray(ink,float);planned_xy=np.asarray(planned_xy,float)
    if ink.ndim!=2 or ink.shape[1]<3 or planned_xy.ndim!=2 or planned_xy.shape[1]!=2 or not len(planned_xy):
        raise ValueError('Invalid ink or planned stroke data')
    if not np.isfinite(ink[:,:3]).all() or not np.isfinite(planned_xy).all() or not np.isfinite(tolerance_m) or tolerance_m<=0:
        raise ValueError('Invalid ink geometry')
    tree=cKDTree(planned_xy);active=ink[:,0]>0;points=ink[:,1:3]
    distances=tree.query(points[active])[0] if active.any() else np.array([])
    total=0.;outside=0.
    for i in np.flatnonzero(active[1:]&active[:-1])+1:
        a,b=points[i-1],points[i];length=float(np.linalg.norm(b-a))
        if length<1e-10:continue
        # Check interpolated ink, so two valid endpoints cannot hide a bridge.
        count=max(1,int(np.ceil(length/.0005)))
        centers=a+(np.arange(count)+.5)[:,None]/count*(b-a)
        total+=length;outside+=length*float(np.mean(tree.query(centers)[0]>tolerance_m))
    return {'ink_sample_precision_within_2_5mm':float(np.mean(distances<=tolerance_m)) if len(distances) else 0.,
            'ink_length_precision_within_2_5mm':1.-outside/total if total>0 else 0.,
            'recorded_connected_ink_length_m':total,'outside_stroke_ink_length_m':outside,
            'ink_precision_note':'geometric centerline diagnostic, 0.5 mm segment sampling; not official calligraphy score'}
