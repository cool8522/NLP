"""Render the project explanation and reproducible training figure.

Install requirements-dev.txt. Run from the extracted project root.
"""
import html
import json
from pathlib import Path
import re
import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, Preformatted, Image, PageBreak, KeepTogether, CondPageBreak)

ROOT=Path(__file__).resolve().parents[1]
DOCS=ROOT/"docs"
FIGS=DOCS/"figures";FIGS.mkdir(exist_ok=True)
report=json.loads((ROOT/"runs/training.json").read_text())
history=report["history"]
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10})
fig,ax=plt.subplots(figsize=(8,3.5),layout="constrained")
ax.plot([x['step'] for x in history],[x['train_loss'] for x in history],"o-",label="Training",color="#225e47",lw=2,ms=4)
ax.plot([x['step'] for x in history],[x['validation_loss'] for x in history],"o-",label="Validation",color="#b36c37",lw=2,ms=4)
ax.axvline(report['best_step'],color="#65786e",ls=":",label="Saved checkpoint")
ax.set(xlabel="Optimizer steps",ylabel="Cross-entropy (nats per BPE token)",title="Mini-GPT learns, then overfits the small training corpus")
ax.spines[['top','right']].set_visible(False);ax.grid(axis="y",alpha=.18);ax.legend(frameon=False)
fig.savefig(FIGS/"training_curve.png",dpi=180);plt.close(fig)

font_dir=Path('/usr/share/fonts/truetype/dejavu')
if (font_dir/'DejaVuSans.ttf').exists():
    for name,file in [('Body','DejaVuSans.ttf'),('BodyBold','DejaVuSans-Bold.ttf'),('Mono','DejaVuSansMono.ttf')]:
        pdfmetrics.registerFont(TTFont(name,str(font_dir/file)))
    pdfmetrics.registerFontFamily('Body',normal='Body',bold='BodyBold',italic='Body',boldItalic='BodyBold')
else:
    # Portable fallback for readers rebuilding the PDF outside Linux.
    for name,target in [('Body','Helvetica'),('BodyBold','Helvetica-Bold'),('Mono','Courier')]:
        pdfmetrics.registerFont(pdfmetrics.Font(name,target,'WinAnsiEncoding'))
    pdfmetrics.registerFontFamily('Body',normal='Body',bold='BodyBold',italic='Body',boldItalic='BodyBold')

styles={
 'title':ParagraphStyle('Title',fontName='BodyBold',fontSize=23,leading=29,spaceAfter=18,textColor=colors.black),
 'h2':ParagraphStyle('H2',fontName='BodyBold',fontSize=13,leading=18,spaceBefore=19,spaceAfter=9,keepWithNext=True,textColor=colors.black),
 'body':ParagraphStyle('Body',fontName='Body',fontSize=9.6,leading=14.4,spaceAfter=8,alignment=TA_LEFT,textColor=colors.HexColor('#202b28')),
 'small':ParagraphStyle('Small',fontName='Body',fontSize=8,leading=11,spaceAfter=7,textColor=colors.HexColor('#53625a')),
 'cell':ParagraphStyle('Cell',fontName='Body',fontSize=8,leading=11,wordWrap='LTR'),
 'thead':ParagraphStyle('THead',fontName='BodyBold',fontSize=8,leading=11),
 'code':ParagraphStyle('Code',fontName='Mono',fontSize=7.8,leading=11,spaceAfter=12,leftIndent=9,rightIndent=9,backColor=colors.HexColor('#f0f3f1'))
}

def rich(text):
    value=html.escape(text)
    value=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',value)
    value=re.sub(r'`([^`]+)`',r'<font name="Mono">\1</font>',value)
    return value

def para(text,style='body'):
    return Paragraph(rich(text),styles[style])

WIDTH=A4[0]-96
story=[]
lines=(DOCS/'PROJECT_GUIDE.md').read_text().splitlines()
i=0
while i<len(lines):
    line=lines[i].strip()
    if not line:i+=1;continue
    if line.startswith('# '):
        story.extend([Spacer(1,12),para(line[2:],'title'),para('Complete source code, measured experiments and a component-by-component explanation','small'),para('Prepared for Pratham Swarnkar | 24 November 2026','small')]);i+=1
    elif line.startswith('## '):
        title=line[3:]
        if title.startswith('1 What'):
            story.append(para('Reading map','h2'))
            story.append(para('Sections 1-5: purpose, architecture, technologies and data. Sections 6-10: BPE, Mini-GPT and training. Sections 11-16: PDF retrieval, answer verification and application flow. Sections 17-21: complexity, measured performance, failures and testing. Sections 22-27: code map, value, demonstration, troubleshooting, viva and references.'))
            story.append(para('Run the project using README.md. The ZIP includes the trained educational model and starter data. FLAN model files are downloaded separately.','small'))
            story.append(PageBreak())
        story.append(CondPageBreak(115))
        story.append(para(title,'h2'));i+=1
        if title.startswith('19 Measured'):
            story.append(Image(str(FIGS/'training_curve.png'),width=WIDTH,height=WIDTH*3.5/8))
            story.append(para('Figure 1. The saved checkpoint uses the minimum measured validation loss at step 250. The later decrease in training loss does not correspond to better validation performance.','small'))
    elif line.startswith('```'):
        i+=1;code=[]
        while i<len(lines) and not lines[i].strip().startswith('```'):
            for part in textwrap.wrap(lines[i],width=88,replace_whitespace=False,drop_whitespace=False) or ['']:
                code.append(part)
            i+=1
        story.append(Preformatted('\n'.join(code),styles['code']));i+=1
    elif line.startswith('|'):
        rows=[]
        while i<len(lines) and lines[i].strip().startswith('|'):
            cols=[c.strip() for c in lines[i].strip().strip('|').split('|')]
            if not all(re.fullmatch(r':?-+:?',c.replace(' ','')) for c in cols):rows.append(cols)
            i+=1
        n=len(rows[0]);first=rows[0][0]
        if n==2:widths=[WIDTH*.30,WIDTH*.70]
        elif n==3:widths=[WIDTH*.30,WIDTH*.23,WIDTH*.47]
        else:widths=[WIDTH/n]*n
        # Numeric result tables need a wider metric-name column.
        if rows[0][0]=='Answer measure':widths=[WIDTH*.48,WIDTH*.26,WIDTH*.26]
        content=[[para(c,'thead' if r==0 else 'cell') for c in row] for r,row in enumerate(rows)]
        t=Table(content,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e5ece7')),
                              ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f7f9f7')]),
                              ('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#b5c4b8')),
                              ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
                              ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
        if len(rows)<=4:
            story.extend([KeepTogether([t]),Spacer(1,10)])
        else:
            if story and isinstance(story[-1],Paragraph) and story[-1].style.name=='H2':
                story[-1].keepWithNext=False
            story.extend([t,Spacer(1,10)])
    elif re.match(r'^\d+\.\s',line):
        story.append(para(line,'small'));i+=1
    else:
        group=[line];i+=1
        while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','```')) and not re.match(r'^\d+\.\s',lines[i]):
            group.append(lines[i].strip());i+=1
        story.append(para(' '.join(group)))

def footer(canvas,doc):
    canvas.saveState();canvas.setFont('Body',7)
    canvas.setFillColor(colors.HexColor('#637068'))
    canvas.drawString(48,28,'PaperLab | Complete project explanation')
    canvas.drawRightString(A4[0]-48,28,str(doc.page));canvas.restoreState()

out=DOCS/'PaperLab_Project_Guide.pdf'
doc=SimpleDocTemplate(str(out),pagesize=A4,rightMargin=48,leftMargin=48,topMargin=43,bottomMargin=48,
                     title='PaperLab Research Paper Assistant',author='PaperLab project documentation')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print('Wrote',out)
