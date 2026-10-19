"""Metric pen-tip paths from ordered stroke medians, with pen-up transitions."""
import json
from dataclasses import dataclass
import numpy as np
from .paths import ASSETS


def smoothstep5(u):
    u = np.clip(np.asarray(u), 0, 1)
    return u**3 * (10 - 15*u + 6*u*u)


@dataclass
class PenPath:
    xyz: np.ndarray
    pen_down: np.ndarray
    time_s: np.ndarray


def make_path(text, origin_xy=(0., 0.), character_size=0.05, gap=0.01,
              paper_z=0., clearance=0.015, speed=0.05, travel_speed=0.15, dt=0.01, columns=5,continuous_strokes=False,stroke_timing='quintic',landing_speed=None,landing_band=.002):
    if not text or character_size <= 0 or speed <= 0 or travel_speed <= 0 or dt <= 0 or clearance <= 0 or columns < 1:
        raise ValueError("Invalid path settings")
    points, flags = [], []
    if landing_speed is not None and (not np.isfinite([landing_speed,landing_band]).all()
            or not 0<landing_speed<=travel_speed or not 0<landing_band<clearance):
        raise ValueError('Landing speed must be positive and at most travel speed, with band inside clearance')

    def segment(a, b, down, velocity_override=None):
        # Each piece stops at its ends, prioritizing an inspectable baseline over speed.
        # Max derivative of smoothstep5 is 1.875, so duration enforces the speed bound.
        velocity_limit = speed if down else travel_speed
        if velocity_override is not None:velocity_limit=velocity_override
        duration = max(dt, np.linalg.norm(b-a) * 1.875 / velocity_limit)
        n = max(2, int(np.ceil(duration/dt)) + 1)
        p = a + smoothstep5(np.linspace(0, 1, n))[:, None] * (b-a)
        chunk = p[1:] if points else p
        points.extend(chunk)
        flags.extend([down] * len(chunk))

    for index, char in enumerate(text):
        src = ASSETS / "hanzi" / f"{char}.json"
        if not src.exists():
            raise ValueError(f"No offline stroke data for {char}; run scripts/fetch_hanzi.py --text TEXT")
        data = json.loads(src.read_text(encoding="utf-8"))
        offset = np.array(origin_xy) + np.array([index % columns, -(index // columns)]) * (character_size + gap)
        for median in data["medians"]:
            xy = np.asarray(median, float) / 1024 * character_size + offset
            # Font coordinates have Y up; world plane here also has Y up.
            stroke = np.c_[xy, np.full(len(xy), paper_z)]
            above = stroke[0] + [0, 0, clearance]
            if points:
                segment(np.array(points[-1]), above, False)
            else:
                points.append(above); flags.append(False)
            if landing_speed is None:
                segment(above, stroke[0], False)
            else:
                landing=stroke[0]+[0,0,landing_band]
                segment(above,landing,False)
                segment(landing,stroke[0],False,velocity_override=landing_speed)
            if continuous_strokes and len(stroke)>1:
                from scipy.interpolate import PchipInterpolator
                keep=np.r_[True,np.linalg.norm(np.diff(stroke,axis=0),axis=1)>1e-9]
                distinct=stroke[keep]
                if len(distinct)>1:
                    arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(distinct,axis=0),axis=1))]
                    dense=PchipInterpolator(arc,distinct,axis=0)(np.linspace(0,arc[-1],max(20,int(np.ceil(arc[-1]/.0001)))))
                    distances=np.r_[0,np.cumsum(np.linalg.norm(np.diff(dense,axis=0),axis=1))]
                    if stroke_timing=='cosine_cruise':
                        ramp=.08;duration=max(2*ramp,distances[-1]/speed+ramp)
                        n=max(2,int(np.ceil(duration/dt))+1);duration=(n-1)*dt
                        peak=distances[-1]/(duration-ramp);t=np.linspace(0,duration,n)
                        query=peak*(t-ramp/2)
                        start=t<ramp;finish=t>duration-ramp;left=duration-t[finish]
                        query[start]=peak*.5*(t[start]-ramp/np.pi*np.sin(np.pi*t[start]/ramp))
                        query[finish]=distances[-1]-peak*.5*(left-ramp/np.pi*np.sin(np.pi*left/ramp))
                    elif stroke_timing=='quintic':
                        n=max(2,int(np.ceil(distances[-1]*1.875/speed/dt))+1)
                        query=smoothstep5(np.linspace(0,1,n))*distances[-1]
                    else:raise ValueError('Unknown stroke timing')
                    chunk=np.column_stack([np.interp(query,distances,dense[:,j]) for j in range(3)])[1:]
                    points.extend(chunk);flags.extend([True]*len(chunk))
            else:
                for a, b in zip(stroke[:-1], stroke[1:]):
                    segment(a, b, True)
            segment(stroke[-1], stroke[-1] + [0, 0, clearance], False)
    xyz = np.asarray(points)
    if len(flags) != len(xyz):
        raise RuntimeError("Path flag alignment error")
    return PenPath(xyz, np.asarray(flags, bool), np.arange(len(xyz))*dt)
