"""Download public official documents; never logs into registration or submits anything."""
import hashlib,json,re,urllib.request,urllib.parse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'deliverables/报名准备_20260910/官方原件'
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    base='https://rcccaa.drct-caa.org.cn/'
    page=urllib.request.urlopen(base+'article_info.php?id=139',timeout=20).read().decode('utf-8')
    links=re.findall('href=[\"\']([^\"\']+)',page)
    doc=next(x for x in links if '.docx' in x.lower())
    files=[('第一轮比赛通知.pdf',base+'image/file/20260803/1785748120164370.pdf'),
           ('五指赛项规则.pdf',base+'image/file/20260827/1787795937567742.pdf'),
           ('官方推荐晋级汇总表.docx',urllib.parse.urljoin(base,doc))]
    results=[]
    for name,url in files:
        content=urllib.request.urlopen(url,timeout=25).read();(OUT/name).write_bytes(content)
        results.append(dict(file=name,url=url,bytes=len(content),sha256=hashlib.sha256(content).hexdigest()))
    (OUT/'来源.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(results,ensure_ascii=False))
if __name__=='__main__':main()
