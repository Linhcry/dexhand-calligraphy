"""Assemble source/resources and channel-specific candidate submission attachments."""
import hashlib,json,shutil,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'deliverables/报名准备_20260910'
OUT=ROOT/'deliverables/五指参赛提交材料_20260910'
STAGE=ROOT/'outputs/submission_audit_20260910/build/five_finger'
def copy(source,dest):
    dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
def manifest(folder,name='PACKAGE_MANIFEST.json'):
    rows=[dict(file=p.relative_to(folder).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(folder.rglob('*')) if p.is_file() and p.name!=name and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts and 'outputs' not in p.relative_to(folder).parts]
    (folder/name).write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8');return rows
def archive(folder,dest):
    dest.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(folder.rglob('*')):
            rel=p.relative_to(folder)
            if p.is_file() and not any(x in rel.parts for x in ('__pycache__','.pytest_cache','outputs')):z.write(p,folder.name+'/'+rel.as_posix())
    with zipfile.ZipFile(dest) as z:assert z.testzip() is None
def main():
    OUT.mkdir(parents=True,exist_ok=True);STAGE.mkdir(parents=True,exist_ok=True)
    for folder in ('fivefinger','experiments','config','tests','lessons','scripts','assets'):
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.mp4'):copy(p,STAGE/p.relative_to(ROOT))
    for p in ROOT.iterdir():
        if p.is_file() and p.suffix in ('.py','.ps1','.cmd','.toml','.txt','.md'):copy(p,STAGE/p.name)
    # Preserve all vendor Python/config source and robot models/meshes used by this project.
    # Unrelated furniture demo meshes, demo films and SolidWorks CAD are not runtime inputs.
    for folder in ('vendor/dexrobot_mujoco','vendor/dex-retargeting'):
        for p in (ROOT/folder).rglob('*'):
            if not p.is_file() or '.git' in p.parts or '__pycache__' in p.parts:continue
            rel=p.relative_to(ROOT/folder)
            if p.suffix in ('.py','.yaml','.yml','.json','.toml','.xml','.md','.pkl') or p.name.startswith(('LICENSE','NOTICE')) or ('meshes' in rel.parts and 'scenes' not in rel.parts):copy(p,STAGE/p.relative_to(ROOT))
    for p in (ROOT.parent/'.agents/skills').rglob('*'):
        if p.is_file() and p.suffix=='.md':copy(p,STAGE/'.agents/skills'/p.relative_to(ROOT.parent/'.agents/skills'))
    (STAGE/'README_FIRST.txt').write_text('先读 参赛代码运行说明.md。\n本包为当前参考仿真代码与资源；没有官方真机SDK，不是接机即用程序。\n保留了第三方许可与来源；不含Python虚拟环境、FFmpeg、无关演示视频或历史抽题答案。\nsetup_submission.ps1在赛前联网安装依赖，main.py为本项目复现入口。\n',encoding='utf-8')
    manifest(STAGE)
    archive(STAGE,OUT/'03_赛前代码审核/五指参赛代码与资源.zip')
    for name in ['队伍信息采集表.docx','校赛组织初步计划.txt']:copy(OLD/name,OUT/'01_报名系统材料'/name)
    (OUT/'01_报名系统材料/正式表格尚缺.txt').write_text('以下两份必须由报名系统队伍管理页面生成，不可用自制文档代替：\n1. 正式报名表\n2. 免责声明\n全体成员含指导老师签字，加盖二级学院公章，扫描为PDF后分别上传。现未获得原表及个人信息，未伪造签名或印章。\n取得晋级资格后办理；第一轮通知截止为9月15日24时。\n',encoding='utf-8')
    for name in ['资格演示_参考仿真.mp4','技术资格仿真说明.docx','技术资格邮件草稿.txt']:
        copy(OLD/name,OUT/'02_技术资格邮箱材料'/name)
    (OUT/'02_技术资格邮箱材料/发送前须知.txt').write_text('收件人：robot_2026@126.com\n主题：五指灵巧手非遗艺术技能赛项_队伍名_学校名\n规则第20页：报名时同时提交；不得晚于正赛前三周。\n视频105秒、2437531字节，满足文件时长和大小上限，但这是参考仿真片段剪辑；规则未明确仿真或剪辑可否接受，应向技术组确认。\n技术说明为辅助材料，规则没有明确要求提交本DOCX或PPT。发送前填写真实队伍信息。未自动发邮件。\n',encoding='utf-8')
    for name in ['完整复核视频_非资格短片.mp4','视频与实验验证.json','完整视频验证.json','优化结果与复现.md']:
        copy(OLD/name,OUT/'04_复核参考不要整包当视频附件'/name)
    for folder in ('实验依据','优化试验记录'):
        for p in (OLD/folder).rglob('*'):
            if p.is_file():copy(p,OUT/'04_复核参考不要整包当视频附件'/folder/p.relative_to(OLD/folder))
    copy(ROOT/'模型核对_20260910.md',OUT/'04_复核参考不要整包当视频附件/模型核对.md')
    copy(ROOT/'outputs/model_check_20260910/verification.json',OUT/'04_复核参考不要整包当视频附件/模型来源核验.json')
    for p in (OLD/'官方原件').iterdir():
        if p.is_file():copy(p,OUT/'05_官方依据'/p.name)
    for source,name in [(ROOT.parent/'tmp/pdfs/new-five-08.png','五指规则第8页.png'),(ROOT.parent/'tmp/pdfs/new-five-20.png','五指规则第20页.png'),(ROOT/'outputs/submission_audit_20260910/notice-7.png','第一轮通知第7页.png'),(ROOT/'outputs/submission_audit_20260910/notice-8.png','第一轮通知第8页.png')]:copy(source,OUT/'05_官方依据/条款页'/name)
    guide='''# 五指参赛材料交付清单

本包整理截至2026年9月10日在官网和PDF中核查到的要求。材料尚未发送或上传；视频、代码为当前参考仿真成果，尚无官方技术资格通过证明。正式表格、真实队伍信息、签字盖章仍需补齐。

## 按提交渠道办理

| 提交事项 | 官方要求和依据 | 去哪里交 | 已备文件与缺项 |
|---|---|---|---|
| 校赛报名信息 | 第一轮通知p7：8月20日至9月10日填报 | 官方报名系统 | 信息采集表与校赛计划仅供整理；实际人数和时间必须核实 |
| 晋级名单 | 第一轮通知p7：校赛负责人统一提交，9月10日截止 | 负责人按官方流程办理 | 官方推荐晋级汇总表在05目录；不是队伍自行填一张就算晋级 |
| 报名表、免责声明 | 通知p7–8：取得资格后，从系统生成；全体成员含老师签字、二级学院盖章、扫描PDF；至9月15日24时 | 队伍管理→当前赛事总决赛→对应表格生成/下载/上传 | 原表、真实信息及签章尚缺；自制信息表不可替代 |
| 技术资格视频 | 五指p20：报名时同时提交，且不得晚于正赛前三周；展示抓笔和写字，≤2分钟、≤30M | robot_2026@126.com | 02目录105秒MP4与邮件草稿；学校队伍信息待填，仿真/剪辑可接受性待技术组确认 |
| 全部代码 | 五指p8：所有代码赛前提交，由组委会资格审查；p19要求备份及自主开发算法 | 规则未写明代码渠道、确切截止时间或压缩格式 | 03目录代码和资源ZIP已备；确认官方接入与提交方式后再交，不能声称ZIP为官方标准 |
| 注册费及凭证 | 通知p8：审核通过后每队1200元，9月20日24时前付款；对公转账须上传有效回单/截图 | 系统费用缴纳；对公支付选择其他支付上传凭证 | 未付款、无凭证；按系统要求办理，不生成虚假回单 |

技术资格邮件主题：`五指灵巧手非遗艺术技能赛项_队伍名_学校名`。此主题属于视频资格材料要求，不能据此假定代码也提交同一邮箱。

**时间更正：**原先按10月16日倒推三周得到9月25日，只是最晚界限的推算，不能覆盖“报名时同时提交”的要求。应在报名时同步处理资格材料，若已漏交须立即向技术组确认补交办法。官网最新公开通知列表仍为8月3日第一轮通知，未在已核查页面看到本赛项专门的延长期或代码接口公告；群内通知可能另有补充。

## 本次给你的文件

01：信息采集表、校赛计划、缺少正式表格的说明。02：105秒资格候选视频、两页技术说明和邮件草稿。03：当前自有代码、模型及网格、OCR权重、笔顺素材、来源和许可证、锁定依赖及运行说明。04：约232秒全程视频及成功/失败实验记录，仅供复核。05：官网原件和关键条款页。

规则第20页的表头提到demo、影音、ppt模板等，但具体条文只明确以展示操笔与写字的视频为主要资格材料；没有明确另交PPT、论文或某种演示文稿。本包不把自制DOCX当作强制材料。

## 代码与模型的实际范围

代码ZIP包含当前项目源代码与运行资源，Python环境和FFmpeg需赛前安装。运行入口及安装命令见包内`参赛代码运行说明.md`。解压复核使用本机已安装的两套Python环境，不能替代空白机器安装验证。官方设备SDK、机械臂型号及控制映射未确认，因此当前程序只能用于参考仿真。

已验证的物理任务是固定参考工位的20字示例古诗；存在换题握笔失败与断墨。初赛点睛、28/40字全文、变化纸笔位置的完整闭环未完成验收。不能把过程检查通过、文件格式满足限制或提供代码包称作官方资格合格。

## 依据版本

官网五指文件现为22.2，2026年8月27日发布链接；用户提供的旧文件名为23.2。两份文件哈希不同，但已对照第20页，资格提交条款一致。本包以官网当前链接为主并保留页码证据。

- 通知：https://rcccaa.drct-caa.org.cn/article_info.php?id=139
- 通知PDF：https://rcccaa.drct-caa.org.cn/image/file/20260803/1785748120164370.pdf
- 规则发布页：https://rcccaa.drct-caa.org.cn/article_info.php?id=138
- 五指PDF：https://rcccaa.drct-caa.org.cn/image/file/20260827/1787795937567742.pdf
- 报名系统：https://robotreg.drct-caa.org.cn/

最终交付ZIP只是本机整理文件的容器，不要整包发送到视频资格邮箱。请按上表选择对应附件。
'''
    (OUT/'00_先看提交清单.md').write_text(guide,encoding='utf-8')
    manifest(OUT,'文件清单.json')
    archive(OUT,ROOT/'deliverables/五指参赛提交材料_20260910.zip')
    print(json.dumps(dict(code_stage=str(STAGE),delivery=str(OUT),code_bytes=(OUT/'03_赛前代码审核/五指参赛代码与资源.zip').stat().st_size,total_bytes=(ROOT/'deliverables/五指参赛提交材料_20260910.zip').stat().st_size),ensure_ascii=False))
if __name__=='__main__':main()
