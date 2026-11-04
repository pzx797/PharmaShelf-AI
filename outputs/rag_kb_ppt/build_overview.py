from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)
W, H = 2400, 1350
im = Image.new("RGB", (W, H), "#F5F7FB")
d = ImageDraw.Draw(im)
NAVY, INK, MUTED, BLUE, TEAL = "#10233F", "#1D304C", "#62718A", "#2563EB", "#087F8C"


def font(size, bold=False):
    return ImageFont.truetype("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc", size)


def text(x, y, s, size=30, color=INK, bold=False):
    d.text((x, y), s, font=font(size, bold), fill=color)


def box(x, y, w, h, fill="#FFFFFF", outline=None, r=24):
    d.rounded_rectangle((x, y, x + w, y + h), radius=r, fill=fill, outline=outline, width=2)


def arrow(x1, y, x2):
    d.line((x1, y, x2 - 13, y), fill="#8FA2BD", width=5)
    d.polygon([(x2, y), (x2 - 16, y - 10), (x2 - 16, y + 10)], fill="#8FA2BD")


d.rectangle((0, 0, W, 12), fill=BLUE)
text(100, 62, "RAG 实验 · 药品 SKU 知识库", 64, NAVY, True)
text(104, 158, "为药店货架商品识别提供可检索的标准商品信息", 32, MUTED)

for x, num, label, detail in [
    (100, "2,019", "有效 SKU", "每条记录对应一个商品编码"),
    (850, "17", "商品类别", "涵盖药品、器械及健康相关商品"),
    (1600, "14", "结构化字段", "连接视觉线索与标准商品信息"),
]:
    box(x, 237, 700, 170)
    text(x + 34, 253, num, 62, BLUE, True)
    text(x + 270, 279, label, 34, NAVY, True)
    text(x + 35, 347, detail, 27, MUTED)

box(100, 443, 1080, 440)
text(138, 475, "知识库里有什么", 37, NAVY, True)
groups = [
    ("身份信息", "SKU 编码、条形码"),
    ("商品属性", "药品名、通用名、规格、厂家、类别"),
    ("检索线索", "别名、视觉关键词、检索文本"),
    ("溯源信息", "原始商品名 / 规格 / 类别、货架位置"),
]
for i, (a, b) in enumerate(groups):
    y = 555 + i * 77
    box(138, y, 157, 49, "#EAF0FD", r=10)
    text(155, y + 4, a, 27, BLUE, True)
    text(320, y + 5, b, 28, INK)

box(1220, 443, 1080, 440, NAVY)
text(1263, 475, "一条真实商品记录", 35, "#CCD9ED")
text(1263, 538, "氯雷他定片", 48, "#FFFFFF", True)
text(1265, 610, "规格   10毫克*6片", 32, "#FFFFFF")
text(1265, 667, "厂家   江苏亚邦", 32, "#FFFFFF")
d.line((1265, 735, 2255, 735), fill="#354863", width=2)
text(1265, 766, "SKU  2112594", 29, "#CEE0F9")
text(1700, 766, "条码  6934666301079", 29, "#CEE0F9")
text(1265, 821, "记录来源：本地知识库", 24, "#A8BBD6")

text(103, 918, "如何辅助 RAG 识别", 36, NAVY, True)
steps = [
    (100, "01", "图像识别线索", "药名、规格、厂家等"),
    (668, "02", "检索 Top-5 候选", "字符级 TF-IDF + 字段匹配"),
    (1236, "03", "结合图片复核", "Qwen 从候选中选择 SKU"),
    (1804, "04", "标准信息回填", "名称、规格、厂家、条码"),
]
for i, (x, n, title, sub) in enumerate(steps):
    box(x, 987, 496, 162, "#E8F3F4" if i == 3 else "#FFFFFF")
    text(x + 25, 1005, n, 25, TEAL if i == 3 else BLUE, True)
    text(x + 80, 1001, title, 33, NAVY, True)
    text(x + 25, 1078, sub, 25, MUTED)
    if i < 3:
        arrow(x + 511, 1067, x + 551)

text(104, 1182, "作用：用知识库候选辅助确定 SKU，并补全标准商品属性。", 32, TEAL, True)
text(104, 1258, "数据口径：原表 2,020 条；1 条缺少 SKU 未入索引。检索与推理不使用人工真值。", 24, MUTED)
im.save(OUT / "rag_knowledge_base_overview.png", dpi=(200, 200))

intro = """PPT 标题：RAG 实验中的药品 SKU 知识库

介绍文字：
本实验使用本地药品 SKU 知识库，包含 2,019 个有效商品 SKU、17 类商品和 14 个结构化字段，涵盖名称、规格、厂家、条码及检索线索。系统根据图像识别结果检索 Top-5 候选，再由 Qwen 结合图片复核，确定 SKU 并回填标准商品信息。

讲解一句话：
知识库相当于一份标准商品档案，帮助模型将图像中的商品线索匹配到具体 SKU。

核对依据：
知识库：rag_knowledge_base/sku_kb_cleaned.xlsx（工作表 sku_kb_cleaned）
实际实验配置：qwen37plus_rag_experiment/runs/qwen37plus_rag_v1/config.yaml
实际索引元数据：qwen37plus_rag_experiment/runs/qwen37plus_rag_v1/index/index_metadata.json
索引构建逻辑：qwen37plus_rag_experiment/02_build_rag_index.py
推理逻辑：qwen37plus_rag_experiment/03_run_qwen37plus_rag.py

数据口径：原表 2,020 条记录，1 条缺少 SKU 被跳过；2,019 个有效 SKU 均唯一。17 类商品按有效 SKU 统计。图中示例逐字段取自知识库原记录。检索和推理不使用人工真值。这里描述知识库作用与实验方法，不代表已验证的准确率提升。
"""
(OUT / "PPT介绍文字.txt").write_text(intro, encoding="utf-8-sig")
print(OUT / "rag_knowledge_base_overview.png")
