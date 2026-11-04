# Qwen3.7-Plus药品裁剪图RAG实验

## 1.实验定位

该实验与`qwen37plus_direct_experiment`构成严格对照：

- Direct基线：裁剪图→Qwen3.7-Plus→直接结构化识别结果。
- RAG实验：冻结Direct第一阶段视觉结果→检索药品知识库Top-5→裁剪图+候选记录→Qwen3.7-Plus复核并选择SKU。

RAG推理阶段不读取人工真值表。人工真值仅在`04`和`05`评价脚本中使用。

## 2.默认路径

```text
项目根目录：E:\EdgeDownload\ultralytics-main
直接基线：E:\EdgeDownload\ultralytics-main\qwen37plus_direct_experiment
RAG实验：E:\EdgeDownload\ultralytics-main\qwen37plus_rag_experiment
知识库：E:\EdgeDownload\ultralytics-main\rag_knowledge_base\sku_kb_cleaned.xlsx
真值表：E:\EdgeDownload\ultralytics-main\runs\MSA_SKU_to_MRS_stage2_seed3_best_crops\instances_gt_v1_enriched.xlsx
裁剪图：E:\EdgeDownload\ultralytics-main\runs\MSA_SKU_to_MRS_stage2_seed3_best_crops\crops
```

## 3.安装依赖

进入现有Python环境后执行：

```powershell
pip install -r requirements.txt
```

## 4.设置API密钥

PowerShell当前窗口临时设置：

```powershell
$env:DASHSCOPE_API_KEY="你的API-Key"
```

## 5.运行顺序

### 第一步：准备RAG实验

```powershell
python 01_prepare_rag_experiment.py
```

该步骤校验直接基线结果、知识库、真值表和裁剪图路径；复制基线输入清单；冻结API参数和提示词。

### 第二步：构建知识库索引

```powershell
python 02_build_rag_index.py
```

默认使用字符级TF-IDF加字段精确匹配加权。索引仅基于知识库，不读取真值表。

### 第三步：先试跑3个样本

```powershell
python 03_run_qwen37plus_rag.py --limit 3
```

检查以下文件：

```text
runs\qwen37plus_rag_v1\predictions\rag_predictions.csv
runs\qwen37plus_rag_v1\retrieval\retrieval_results.jsonl
runs\qwen37plus_rag_v1\responses\raw_responses.jsonl
runs\qwen37plus_rag_v1\logs\api_errors.log
```

确认3个样本没有`api_error`或大面积`format_error`后继续全量运行。

### 第四步：全量运行

```powershell
python 03_run_qwen37plus_rag.py
```

脚本支持断点续跑。已标记为`ok`、`format_error`和`input_error`的样本不会重复调用API；`api_error`样本会在下次运行时重试。

### 第五步：计算与Direct一致的主识别指标

```powershell
python 04_evaluate_rag_results.py
```

输出药品名称、规格、厂家、严格SKU、拒识、错误确定识别率、延迟和Token等指标。评价规则直接沿用基线评价脚本。

### 第六步：计算RAG专属检索指标

```powershell
python 05_evaluate_retrieval_results.py
```

输出：

- Top-1、Top-3、Top-5召回率；
- MRR；
- SKU选择覆盖率；
- 最终SKU选择准确率；
- 真值已进入Top-K条件下的候选重排准确率；
- 检索失败和候选选择失败样本。

## 6.主要输出目录

```text
qwen37plus_rag_experiment
└─ runs
   └─ qwen37plus_rag_v1
      ├─ config.yaml
      ├─ input_manifest.csv
      ├─ prompts
      ├─ index
      ├─ retrieval
      ├─ responses
      ├─ predictions
      ├─ evaluation
      └─ logs
```

## 7.实验控制原则

1.两组实验使用相同裁剪图片、sample_id、Qwen模型、温度、思考模式和高分辨率参数。
2.RAG检索查询来自冻结的Direct预测，不重新调用第一阶段模型，降低随机性和API成本。
3.检索和推理不读取`gt_sku_id`、`gt_drug_name`等真值字段。
4.候选记录只是召回结果，提示词明确禁止模型仅按Top-1排名猜测。
5.模型只能选择Top-K中的SKU；选中后由程序从知识库确定性回填规范名称、规格、厂家和条码。
6.主识别指标使用`04`脚本，确保与Direct基线同口径；检索质量使用`05`脚本单独报告。

## 8.首次运行建议

正式全量运行前，先检查3个试跑样本中的：

- `retrieval_query`是否包含可见药名、品牌、规格等有效线索；
- `retrieved_candidates_json`中的候选是否合理；
- `pred_sku_id`是否确实属于Top-5；
- `visible_text_json`是否只记录图片文字，没有抄写知识库内容；
- `candidate_consistency`和`reason`是否符合图像证据。

若Top-5候选明显不合理，应先调整检索字段权重，再重新构建索引和清空RAG预测结果，不能在部分样本上临时修改规则。
