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
    "other": "prompts/other_prompt.txt"
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
        "fields": fields     # 直接返回模型字段
    }



async def process_files_async(files):
    tasks = [process_one(f) for f in files]
    return await asyncio.gather(*tasks)


def process_files(files):
    if not files:
        raise gr.Error("⚠️ 请先上传文件！")
    return asyncio.run(process_files_async(files))


def build_popup(files_data):
    preview = []
    for item in files_data:
        preview.append({
            "文件名": item["file_name"],
            "字段": item["fields"],
        })
    return json.dumps(preview, ensure_ascii=False, indent=2)


def build_final_output(files_data):
    return files_data



with gr.Blocks() as demo:
    gr.Markdown("# 📄 票据识别系统")

    file_input = gr.File(label="上传文件（可多选）", file_count="multiple")
    process_btn = gr.Button("开始处理", variant="primary", interactive=False)

    files_state = gr.State([])

    # 上传文件 → 激活按钮
    file_input.change(
        lambda files: gr.update(interactive=True, variant="primary") if files else gr.update(interactive=False, variant="secondary"),
        inputs=file_input,
        outputs=process_btn
    )

    # Popup (modal)
    popup_group = gr.Group(visible=False)
    with popup_group:
        gr.Markdown("### 🔍 识别预览")
        popup_json = gr.JSON()
        # confirm_btn = gr.Button("确认", variant="primary")

    # Final output
    final_json = gr.JSON(label="最终结果", visible=False)

    # Step 1: Run processing
    process_btn.click(
        fn=process_files,
        inputs=file_input,
        outputs=files_state
    )

    # Step 2: Fill popup
    process_btn.click(
        fn=build_popup,
        inputs=files_state,
        outputs=popup_json
    )

    # Step 3: Show popup
    process_btn.click(
        lambda: gr.update(visible=True),
        None,
        popup_group
    )

    # # Confirm → produce final output
    # confirm_btn.click(
    #     fn=build_final_output,
    #     inputs=files_state,
    #     outputs=final_json
    # )

    # # Close popup + show final table
    # confirm_btn.click(
    #     lambda: (gr.update(visible=False), gr.update(visible=True)),
    #     None,
    #     [popup_group, final_json]
    # )

demo.launch(share=True, server_name="0.0.0.0")

