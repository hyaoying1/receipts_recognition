import os
from glob import glob

from docx2pdf import convert as docx2pdf_convert
from pdf2image import convert_from_path


# ====== 根据你自己的路径改这三个 ======
INPUT_DIR = "/Users/liuran/Projects/OCR_test_2/artifacts/docx"   # 存 docx 的目录
OUTPUT_DIR = "/Users/liuran/Projects/OCR_test_2/artifacts/images"   # 输出 JPG 的目录
DPI = 200                                        # 图片清晰度
# 每个平台本次最多处理多少个 DOCX
N_PER_PLATFORM = 40  # 你按需要改，比如 20 / 100 都行


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def docx_to_pdf(docx_path: str, pdf_path: str):
    """
    用 Word (通过 docx2pdf) 把 docx 转成 pdf
    docx2pdf 在 Mac 上会调用 Word.app，
    所以前提是你本地安装了 Microsoft Word。
    """
    ensure_dir(os.path.dirname(pdf_path))
    print(f"[DOCX→PDF] {docx_path} -> {pdf_path}")
    docx2pdf_convert(docx_path, pdf_path)


def pdf_to_jpg(pdf_path: str, out_dir: str, dpi: int = 200):
    """
    用 pdf2image 把 pdf 转成 jpg
    - 单页：<base>.jpg
    - 多页：<base>_page_1.jpg, <base>_page_2.jpg, ...
    """
    ensure_dir(out_dir)
    base = os.path.splitext(os.path.basename(pdf_path))[0]

    print(f"[PDF→JPG] {pdf_path} -> {out_dir}")
    pages = convert_from_path(pdf_path, dpi=dpi, fmt="jpeg")

    jpg_paths = []
    if len(pages) == 1:
        out_jpg = os.path.join(out_dir, f"{base}.jpg")
        pages[0].save(out_jpg, "JPEG")
        jpg_paths.append(out_jpg)
        print("  生成：", out_jpg)
    else:
        for i, img in enumerate(pages, start=1):
            out_jpg = os.path.join(out_dir, f"{base}_page_{i}.jpg")
            img.save(out_jpg, "JPEG")
            jpg_paths.append(out_jpg)
            print("  生成：", out_jpg)
    return jpg_paths



def get_platform_from_filename(path: str) -> str:
    """
    根据 docx 文件名推断平台名：
    默认假设文件名形如: didi_xxx.docx / caocao_xxx.docx，
    即“下划线前面的部分”是平台名。
    """
    name = os.path.basename(path)
    prefix = name.split("_", 1)[0].lower()
    return prefix


# def main():
#     input_dir = os.path.abspath(INPUT_DIR)
#     output_dir = os.path.abspath(OUTPUT_DIR)
#     print(f"[INFO] 输入目录: {input_dir}")
#     print(f"[INFO] 输出目录: {output_dir}")

#     docx_files = sorted(glob(os.path.join(input_dir, "*.docx")))
#     print(f"[INFO] 共发现 {len(docx_files)} 个 DOCX")

#     if not docx_files:
#         return

#     for idx, docx_path in enumerate(docx_files, start=1):
#         print(f"\n=== ({idx}/{len(docx_files)}) 处理: {docx_path} ===")
#         base = os.path.splitext(os.path.basename(docx_path))[0]
#         pdf_path = os.path.join(output_dir, base + ".pdf")

#         # 1) 先 docx -> pdf
#         docx_to_pdf(docx_path, pdf_path)

#         # 2) 再 pdf -> jpg
#         pdf_to_jpg(pdf_path, output_dir, dpi=DPI)

#         # 如果不想保留 pdf，可以删掉
#         os.remove(pdf_path)



def main(force_regen: bool = False):
    input_dir = os.path.abspath(INPUT_DIR)
    output_dir = os.path.abspath(OUTPUT_DIR)
    print(f"[INFO] 输入目录: {input_dir}")
    print(f"[INFO] 输出目录: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    # 1) 收集所有 docx
    docx_files = sorted(glob(os.path.join(input_dir, "*.docx")))
    print(f"[INFO] 共发现 {len(docx_files)} 个 DOCX")

    if not docx_files:
        print("[WARN] 没有发现 docx，直接结束。")
        return

    # 2) 收集已有 jpg（按 base 名来）
    existing_jpg_bases = set(
        os.path.splitext(name)[0]
        for name in os.listdir(output_dir)
        if name.lower().endswith(".jpg")
    )
    print(f"[INFO] 当前已有 JPG 数量: {len(existing_jpg_bases)}")

    # 3) 先按平台分组 docx
    platform_buckets = {}
    for docx_path in docx_files:
        platform = get_platform_from_filename(docx_path)
        platform_buckets.setdefault(platform, []).append(docx_path)

    # 每个平台内部按文件名排序，方便增量
    for platform, files in platform_buckets.items():
        platform_buckets[platform] = sorted(files)

    # 4) 为每个平台选出「本次要处理」的 docx 列表
    to_convert = []
    for platform, files in platform_buckets.items():
        if force_regen:
            # 强制重建：直接取前 N_PER_PLATFORM 个
            selected = files[:N_PER_PLATFORM]
            need_cnt = len(files)
        else:
    # 增量模式：先“纳入”前 N_PER_PLATFORM 个文件，
    # 具体有没有生成 JPG，在后面统一用 existing_jpg_bases 跳过
            candidate_files = files[:N_PER_PLATFORM]

            # 统计这 N 个候选里，有多少个目前还没有生成 JPG（只是用于打印信息）
            need_cnt = sum(
                1
                for docx_path in candidate_files
                if os.path.splitext(os.path.basename(docx_path))[0] not in existing_jpg_bases
            )

            # 本次要处理/检查的 docx 就是这 N 个候选
            selected = candidate_files

        print(
            f"[INFO] 平台 {platform}: 共 {len(files)} 个 DOCX，"
            f"其中 {need_cnt} 个尚未生成 JPG，本次准备转换 {len(selected)} 个。"
        )

        to_convert.extend(selected)

    if not to_convert:
        print("[INFO] 所有平台都已无待转换的 DOCX，结束。")
        return

    # 5) 统一遍历“本次要转换”的 docx（跨平台合并后）
    total = len(to_convert)
    for idx, docx_path in enumerate(to_convert, start=1):
        base = os.path.splitext(os.path.basename(docx_path))[0]
        pdf_path = os.path.join(output_dir, base + ".pdf")
        jpg_path = os.path.join(output_dir, base + ".jpg")  # 单页输出的情况

        print(f"\n=== ({idx}/{total}) 处理: {docx_path} ===")

        # ---- 增量模式：如果已经有 jpg 且不强制重建，则跳过 ----
        if (not force_regen) and (base in existing_jpg_bases):
            print(f"[SKIP] 已存在同名 JPG，跳过: {jpg_path}")
            continue

        # 1) docx -> pdf
        print(f"[DOCX→PDF] {docx_path} -> {pdf_path}")
        docx_to_pdf(docx_path, pdf_path)

        # 2) pdf -> jpg
        print(f"[PDF→JPG] {pdf_path} -> {output_dir}")
        pdf_to_jpg(pdf_path, output_dir, dpi=DPI)

        # 3) 如不想保留 pdf，则删除
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
            print(f"[CLEAN] 删除中间 PDF: {pdf_path}")

    print("\n[DONE] DOCX → JPG 转换流程结束。")


if __name__ == "__main__":
    main()