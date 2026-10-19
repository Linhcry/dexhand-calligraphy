"""Build an offline, browsable evidence gallery and paper-relative plan artifacts."""
import html
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from experiments.scene_matrix import scenarios
from fivefinger.autonomy.geometry import Calibration
from fivefinger.autonomy.planning import PaperFrame,poem_path
from fivefinger.paths import ROOT


def project(points,cal):
    p=(np.asarray(points)-cal.T_base_camera[:3,3])@cal.T_base_camera[:3,:3]
    q,_=cv2.projectPoints(p,np.zeros(3),np.zeros(3),cal.K,np.asarray(cal.distortion,float))
    return q.reshape(-1,2)


def main():
    out=ROOT/'outputs/adaptive_scenes';rows=[];tiles=[]
    bank=json.loads((ROOT/'config/adaptive_poems.json').read_text(encoding='utf-8'))
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
    for spec in scenarios():
        folder=out/spec['name'];e=json.loads((folder/'evaluation.json').read_text(encoding='utf-8'))
        result=json.loads((folder/'perception.json').read_text(encoding='utf-8')) if (folder/'perception.json').exists() else None
        cal=Calibration.from_dict(json.loads((folder/'calibration.json').read_text(encoding='utf-8')))
        picture=Image.open(folder/'rgb.png').convert('RGB');draw=ImageDraw.Draw(picture)
        if result:
            for det in result['paper']:
                coords=project(det['corners_m'],cal);draw.line([tuple(p) for p in np.vstack([coords,coords[0]])],fill='#00e8a1',width=4)
            for key,color in [('brush','#ffcf54'),('inkstone','#50caff')]:
                for det in result[key]:
                    pos=det.get('center_m',det.get('visible_center_m'));x,y=project([pos],cal)[0]
                    draw.ellipse((x-10,y-10,x+10,y+10),outline=color,width=3)
        plans=[]
        if result and result['scene_geometry_ready'] and result['paper'][0]['size']=='A3' and not spec.get('xy_drift'):
            paper=PaperFrame.from_detection(result['paper'][0])
            for title,text in bank.items():
                try:
                    path,meta=poem_path(text,paper,uncertainty_m=result['paper'][0]['uncertainty_m'])
                    np.savez_compressed(folder/f'plan_{len(text)}.npz',xyz=path.xyz,pen_down=path.pen_down,time_s=path.time_s)
                    meta.update(title=title,characters=len(text),coordinate_frame=cal.base_frame,planning_only=True,physical_execution=False,
                                nominal_round_with_60s_overhead=bool(meta['path_seconds']+60<=300))
                    plans.append(meta)
                    if len(text)==20:
                        uv=project(path.xyz,cal)
                        # Blue is a PLAN overlay, never represented as ink or physical execution.
                        for i in range(2,len(uv),4):
                            if path.pen_down[i] and path.pen_down[i-1]:draw.line([tuple(uv[i-1]),tuple(uv[i])],fill='#1988ff',width=2)
                except ValueError as exc:plans.append(dict(title=title,error=str(exc)))
        e['plans']=plans;e['full_task_verified']=False;e['real_hardware_ready']=False
        if spec.get('xy_drift'):
            e['status']='UNDETECTED_BY_TABLE_PLANE';e['explanation']='桌面平面检查无法发现相机水平漂移；必须通过固定特征重投影检查才允许运动。'
        elif e.get('scene_geometry_ready'):e['status']='GEOMETRY_ONLY'
        else:e['status']='REJECTED'
        picture.save(folder/'annotated.png')
        thumb=picture.resize((426,320));tile=Image.new('RGB',(426,385),'#18222f');tile.paste(thumb,(0,0))
        td=ImageDraw.Draw(tile);td.text((10,325),spec['name'],font=font,fill='white');td.text((10,353),e['status'],font=font,fill='#7fdbad' if e['status']=='GEOMETRY_ONLY' else '#ffbc76');tiles.append(tile)
        rows.append(e)
    mosaic=Image.new('RGB',(426*4,385*int(np.ceil(len(tiles)/4))),'#18222f')
    for i,tile in enumerate(tiles):mosaic.paste(tile,((i%4)*426,(i//4)*385))
    mosaic.save(out/'场景总览.jpg',quality=90)
    summary=dict(scene_count=len(rows),geometry_candidates=sum(r['status']=='GEOMETRY_ONLY' for r in rows),
        rejected=sum(r['status']=='REJECTED' for r in rows),known_missed_faults=sum(r['status']=='UNDETECTED_BY_TABLE_PLANE' for r in rows),
        completed_physical_tasks_in_this_suite=0,hardware_ready=False,cases=rows)
    (out/'coverage.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    cards=[]
    for row in rows:
        name=html.escape(row['name']);details=html.escape(json.dumps(row,ensure_ascii=False,indent=2))
        cards.append(f'<article data-status="{row["status"]}"><h2>{name} · {row["status"]}</h2><img loading="lazy" src="{name}/annotated.png"><details><summary>参数、定位误差与轨迹预算</summary><pre>{details}</pre></details></article>')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>五指比赛场景验证</title><style>
    body{background:#111a25;color:#e9f0fa;font:16px system-ui;margin:28px;line-height:1.7}h1{font-size:28px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:20px}article{background:#1c2939;padding:18px;border-radius:12px}img{width:100%}h2{font-size:17px}pre{white-space:pre-wrap;font-size:13px}select{padding:8px;font-size:16px;margin-bottom:20px}p{max-width:1000px;color:#c2d0df}
    </style><h1>五指比赛 · 场景变化与拒绝执行测试</h1><p>这些图像由 MuJoCo 渲染。绿色框为视觉估计的纸张，黄色为笔杆可见部分，浅蓝色为砚台；深蓝线是规划预览，不是实际写出的墨迹。本组验证感知与规划，未执行完整抓笔书写。水平标定漂移案例专门保留为失效证据。</p><select onchange="document.querySelectorAll('article').forEach(e=>e.hidden=this.value!=='all'&&e.dataset.status!==this.value)"><option value="all">全部场景</option><option>GEOMETRY_ONLY</option><option>REJECTED</option><option>UNDTECTED_BY_TABLE_PLANE</option></select><main>'''+''.join(cards)+'</main></html>'
    page=page.replace('UNDTECTED_BY_TABLE_PLANE','UNDETECTED_BY_TABLE_PLANE')
    (out/'index.html').write_text(page,encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='cases'},ensure_ascii=False))

if __name__=='__main__':main()
