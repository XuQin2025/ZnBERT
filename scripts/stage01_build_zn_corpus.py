import os
import glob
import re

# ============ 可配置部分 ============
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BUNDLE_ROOT = os.path.dirname(SCRIPT_DIR)
INPUT_PATTERN = os.path.join(BUNDLE_ROOT, "corpus", "raw_wos", "*.txt")
OUTPUT_FILE = os.path.join(BUNDLE_ROOT, "corpus", "zn_corpus.txt")

LOWERCASE     = False
JOIN_TOKEN    = " "

# ============ Parse a single WOS file ============

def parse_wos_file(path):
    """
    Based on the structure exported from Web of Science, extract the TI (Title) and AB (Abstract) for each record.
    back list：[{ "TI": "...", "AB": "..." }, ...]
    """
    records = []

    current_ti = ""
    current_ab = ""
    in_record = False        # 是否已经进入一条记录
    current_field = None     # 正在追加的字段: "TI" / "AB" / None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            # 完全空行直接跳过
            if not line.strip():
                continue

            # 记录结束：行是单独的 "ER"（可能后面有空格）
            if line.strip() == "ER":
                if in_record:
                    # 保存一条记录
                    records.append({
                        "TI": current_ti.strip(),
                        "AB": current_ab.strip()
                    })
                # 重置状态
                current_ti = ""
                current_ab = ""
                in_record = False
                current_field = None
                continue

            # 记录开始：PT 开头
            if line.startswith("PT "):
                # 如果上一条没遇到 ER 就又遇到 PT，先保存上一条
                if in_record and (current_ti or current_ab):
                    records.append({
                        "TI": current_ti.strip(),
                        "AB": current_ab.strip()
                    })
                in_record = True
                current_ti = ""
                current_ab = ""
                current_field = None
                continue

            # 标题行
            if line.startswith("TI "):
                in_record = True
                current_field = "TI"
                current_ti = line[3:].strip()
                continue

            # 摘要行
            if line.startswith("AB "):
                in_record = True
                current_field = "AB"
                current_ab = line[3:].strip()
                continue

            # 其他以两位大写字母开头的字段都忽略
            # （FN、VR、AU、SO 等）
            if re.match(r"^[A-Z]{2}\s", line):
                current_field = None
                continue

            # 如果前面是 TI/AB，这里就是续行（有缩进）
            if current_field == "TI":
                current_ti = (current_ti + " " + line.strip()).strip()
            elif current_field == "AB":
                current_ab = (current_ab + " " + line.strip()).strip()
            else:
                # 其他字段的续行，直接忽略
                pass

    # 文件结尾如果最后一条没遇到 ER，也保存一下
    if in_record and (current_ti or current_ab):
        records.append({
            "TI": current_ti.strip(),
            "AB": current_ab.strip()
        })

    return records


# ============ 文本清洗 ============

def clean_text(text: str) -> str:
    """
    简单预处理：压缩空白，去掉前后空格，必要时转小写。
    """
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    if LOWERCASE:
        text = text.lower()
    return text


# ============ 主函数：合并所有 txt 生语料 ============

def build_corpus(input_pattern=INPUT_PATTERN, output_file=OUTPUT_FILE):
    files = glob.glob(input_pattern)
    if not files:
        print("未在当前文件夹找到匹配的 txt 文件。")
        return

    # 按文件名中的起始数字排序
    def sort_key(fname):
        base = os.path.basename(fname)
        m = re.match(r"(\d+)", base)
        return int(m.group(1)) if m else 0

    files = sorted(files, key=sort_key)

    print(f"找到 {len(files)} 个 txt 文件：")
    for f in files:
        print("  -", f)

    total_records = 0
    total_written = 0

    with open(output_file, "w", encoding="utf-8") as out:
        for path in files:
            recs = parse_wos_file(path)
            total_records += len(recs)

            for rec in recs:
                ti = clean_text(rec.get("TI", ""))
                ab = clean_text(rec.get("AB", ""))

                if not ti and not ab:
                    continue

                if ti and ab:
                    text = ti + JOIN_TOKEN + ab
                else:
                    text = ti or ab

                text = clean_text(text)
                if text:
                    out.write(text + "\n")
                    total_written += 1

    print(f"\n解析记录总数（含无 TI/AB 的）：{total_records}")
    print(f"写入语料条数（有标题或摘要的）：{total_written}")
    print(f"已生成：{output_file}")


if __name__ == "__main__":
    build_corpus()
