# Receipt Recognition Tool（票据识别工具） 
## 一、工具简介 
本工具是一个 **本地运行的票据 / 单据识别程序**，用于对图片或 PDF 文件进行自动化处理，包括： 
- OCR 文字识别
- 单据类型自动分类（行程单 / 酒店发票 / 支付凭证等）
- 按单据类型调用对应解析策略进行字段抽取
- 输出结构化 JSON 结果 
- 通过 .env 文件进行配置 
## 二、运行环境要求 
- 操作系统：**Linux（x86_64）**
- 无需安装 Python
- 需要网络访问权限（用于字段抽取 API）
## 三、目录结构（推荐） 将可执行文件放在一个独立目录中，例如：
```text
receipt_recognizer/
├── receipt_recognizer        # 可执行文件
├── .env                      # 配置文件（必须）
├── input/                    # 输入目录（用户放图片 / PDF）
├── processed/                # 中间处理目录（自动生成）
└── output/                   # 输出目录（结果 JSON）
```
如果未在 .env 中显式指定路径，程序将默认使用以上目录结构。

## 四、配置说明（.env）
在可执行文件同目录 下创建 .env 文件。

示例 .env
```env
API_KEY=your_api_key_here
INPUT_DIR=./input
PROCESSED_DIR=./processed
OUTPUT_DIR=./output
```
配置项说明
|配置项	|说明|
|---|---|
|API_KEY|	用于调用字段抽取服务|
|INPUT_DIR	|	输入文件目录|
|PROCESSED_DIR	|	中间处理目录|
|OUTPUT_DIR	|输出结果目录|

📌 所有路径均支持 相对路径或绝对路径。

## 五、使用步骤
1️⃣ 准备输入文件
将需要识别的图片或 PDF 文件放入输入目录，例如：

```text
input/
├── receipt_1.jpg
├── receipt_2.png
└── invoice.pdf
```
2️⃣ 运行程序
在当前目录执行：

```bash
./receipt_recognizer
```

3️⃣ 查看输出结果
程序运行完成后，会在输出目录生成一个 JSON 文件，例如：

```text
output/
└── output_20251218_203556.json
```
## 六、输出结果说明
输出文件为 JSON 格式，示例如下：

```json
{
  "meta": {
    "input_dir": "input",
    "file_count": 3,
    "total_time_sec": 28.4,
    "ocr_time_sec": 10.1,
    "classification_time_sec": 1.9
  },
  "results": [
    {
      "file": "receipt_1.jpg",
      "type": "hotel_invoice",
      "api_time_seconds": 3.2,
      "result": {
        "output": {
          "...": "..."
        }
      }
    }
  ]
}
```
字段说明
- meta：本次运行的整体统计信息
- results：每个文件对应的识别与解析结果
- type：识别出的单据类型
- result.output：字段抽取后的结构化结果

## 七、常见问题（FAQ）
**Q1：启动时报错 “API_KEY not found”**
**原因：**
.env 文件不存在，或未配置 API_KEY。

**解决方式：**
- 确保 .env 与可执行文件在同一目录

- 在 .env 中配置 API_KEY=...

**Q2：提示 “No input files found”**
**原因：**
输入目录为空。

**解决方式：**

- 检查 INPUT_DIR 路径是否正确

- 确保目录中存在图片或 PDF 文件

**Q3：出现 onnxruntime 的 GPU 警告**
```text

GPU device discovery failed
```
说明：

- 当前机器无可用 GPU

- 程序会自动使用 CPU

- 不影响使用，可直接忽略

## 八、注意事项
本工具为本地批处理工具，不提供 HTTP API 服务

每次运行会生成一个新的输出 JSON 文件

输入文件数量较多时，处理时间会相应增加
