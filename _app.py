import gradio as gr
from pathlib import Path
import asyncio
import json

from src.pre_processor import preprocess_file
from src.rulebased_classifier import rule_classify, run_ocr_async
from src.run_model import run_one_file


PROMPT_MAP = {
    "itinerary": "prompts/itinerary_prompt.txt",
    "hotel_invoice": "prompts/hotel_prompt.txt",
    "payment": "prompts/payment_prompt.txt",
}


async def process_one(upload_file, progress=gr.Progress(track_tqdm=True)):
    progress(0.1, "预处理文件...")
    file_path = Path(upload_file.name)

    processed = preprocess_file(file_path, Path("data/processed"))

    progress(0.3, "OCR 识别中...")
    text = await run_ocr_async(processed)

    progress(0.5, "类型识别中...")
    doc_type = await rule_classify(text)

    progress(0.7, "字段抽取中...")
    prompt_path = PROMPT_MAP.get(doc_type)
    result = await run_one_file(processed, prompt_path)

    try:
        fields = json.loads(result["output"])
    except:
        fields = {}

    progress(1.0, "完成")

    return {
        "file_name": file_path.name,
        "file_type": doc_type,
        "fields": fields
    }


async def process_files_async(files):
    tasks = [process_one(f) for f in files]
    return await asyncio.gather(*tasks)


def process_all(files):
    if not files:
        raise gr.Error("⚠️ 请先上传文件！")

    data = asyncio.run(process_files_async(files))

    # Popup content
    preview = [
        {"文件名": item["file_name"], "字段": item["fields"]}
        for item in data
    ]
    preview_json = json.dumps(preview, ensure_ascii=False, indent=2)

    return (
        data,                     # files_state
        preview_json,             # popup_json
        gr.update(visible=True),  # show popup
    )


with gr.Blocks() as demo:
    gr.Markdown("# 📄 票据识别系统")

    file_input = gr.File(label="上传文件（可多选）", file_count="multiple")
    process_btn = gr.Button("开始处理", variant="primary", interactive=False)

    files_state = gr.State([])

    file_input.change(
        lambda f: gr.update(interactive=True) if f else gr.update(interactive=False),
        inputs=file_input,
        outputs=process_btn
    )

    popup_group = gr.Group(visible=False)
    with popup_group:
        gr.Markdown("### 🔍 识别预览")
        popup_json = gr.JSON()

    process_btn.click(
        fn=process_all,
        inputs=file_input,
        outputs=[files_state, popup_json, popup_group]
    )

demo.launch(share=True, server_name="0.0.0.0")