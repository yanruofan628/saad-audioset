import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors


def parse_stereo_filename(filename):
    """
    解析合成立体声原始文件名，提取左右小类名。
    期望格式: nn_左类_左ID+右类_右ID.wav 或 rn_...
    返回: (left_category, right_category) 或 (None, None)
    """
    try:
        name_without_ext = os.path.splitext(os.path.basename(str(filename)))[0]
        if not (name_without_ext.startswith('rn_') or name_without_ext.startswith('nn_')):
            return None, None
        name_without_prefix = name_without_ext[3:]
        if '+' not in name_without_prefix:
            return None, None
        left_part, right_part = name_without_prefix.split('+', 1)
        left_underscore_pos = left_part.find('_')
        right_underscore_pos = right_part.find('_')
        if left_underscore_pos == -1 or right_underscore_pos == -1:
            return None, None
        left_category = left_part[:left_underscore_pos]
        right_category = right_part[:right_underscore_pos]
        return left_category, right_category
    except Exception:
        return None, None


def collect_selection(excel_path, log_path, output_dir):
    """
    从 Excel 与日志收集 trial，按原始文件名前缀拆分 nn_/rn_ 两组，并各自绘制一张左选比例热力图。
    仅输出热力图，不导出其他文件。
    """
    os.makedirs(output_dir, exist_ok=True)

    target_categories = [
        'Telephone bell ringing', 'Ambulance (siren)', 'Baby cry, infant cry',
        'Computer keyboard', 'Helicopter', 'Chicken, rooster',
        'Male speech, man speaking', 'Female speech, woman speaking', 'Child speech, kid speaking',
        'Bass drum', 'Funny music', 'Sad music',
        'Pulse', 'Whack, thwack', 'Crumpling, crinkling'
    ]
    idx_map = {cat: i for i, cat in enumerate(target_categories)}
    n = len(target_categories)

    # 读取 Excel 映射: 标准名 aXXX -> (left_cat, right_cat, original_filename)
    try:
        # 尝试使用 xlrd 引擎读取 Excel
        df_map = pd.read_excel(excel_path, sheet_name='filename_list', header=None, engine='xlrd')
    except Exception as e1:
        try:
            # 如果 xlrd 失败，尝试 openpyxl
            df_map = pd.read_excel(excel_path, sheet_name='filename_list', header=None, engine='openpyxl')
        except Exception as e2:
            print(f"读取 Excel 失败: xlrd错误={e1}, openpyxl错误={e2}")
            print("请安装依赖: pip install xlrd 或 pip install openpyxl")
            return
    if df_map.shape[1] < 2:
        print("Excel 'filename_list' 至少需要两列：原始名 与 标准名")
        return

    std_to_pair = {}
    for _, row in df_map.iterrows():
        orig_name = str(row.iloc[0]).strip()
        std_name = str(row.iloc[1]).strip()
        left_cat, right_cat = parse_stereo_filename(orig_name)
        if left_cat is None or right_cat is None:
            continue
        std_base = os.path.splitext(std_name)[0].lower()
        # 同时登记无扩展名与带 .wav 的键，适配两种情况
        std_to_pair[std_base] = (left_cat, right_cat, orig_name)
        std_to_pair[std_base + '.wav'] = (left_cat, right_cat, orig_name)

    if not std_to_pair:
        print("未从 Excel 解析到任何映射")
        return
    else:
        sample_keys = list(std_to_pair.keys())[:10]
        print(f"从 Excel 读取到映射数量: {len(std_to_pair)}，样例键: {sample_keys}")

    # 解析日志: 提取 aXXX(.wav) 与选择 1/2
    trials = []
    current_std = None
    current_list = None
    std_pattern = re.compile(r"\b(a\d{3}(?:\.wav)?)\b", re.IGNORECASE)
    resp_pattern = re.compile(r"ImageDisplay\d*\.RESP\s*[:=]\s*(\d)", re.IGNORECASE)
    list_pattern = re.compile(r"List1\s*[:=]\s*(\d+)", re.IGNORECASE)
    matched_wav_lines = 0
    matched_resp_lines = 0
    try:
        # 尝试多种编码读取日志
        encodings = ['utf-8', 'utf-16', 'utf-16le', 'utf-16be', 'gbk', 'ansi']
        file_text = None
        used_encoding = None
        with open(log_path, 'rb') as fb:
            raw = fb.read()
        for enc in encodings:
            try:
                file_text = raw.decode(enc)
                used_encoding = enc
                break
            except Exception:
                continue
        if file_text is None:
            # 最后退回忽略错误的 utf-8 解码
            file_text = raw.decode('utf-8', errors='ignore')
            used_encoding = 'utf-8(ignore)'

        print(f"日志采用编码读取: {used_encoding}")

        for line in file_text.splitlines():
            s = line.strip()
            if not s:
                continue
            # List 索引
            m_list = list_pattern.search(s)
            if m_list:
                try:
                    current_list = int(m_list.group(1))
                except Exception:
                    current_list = None
            # 提取标准名（允许不在 wavfile 行上，也能识别 a### 或 a###.wav）
            m_std = std_pattern.search(s)
            if m_std:
                current_std = m_std.group(1).lower()
                matched_wav_lines += 1
            # 提取 RESP
            m_resp = resp_pattern.search(s)
            if m_resp:
                try:
                    resp = int(m_resp.group(1))
                except Exception:
                    resp = None
                matched_resp_lines += 1
                if current_std and resp in (1, 2):
                    trials.append({'std_name': current_std, 'resp': resp, 'list_index': current_list})
                    current_std = None
    except Exception as e:
        print(f"读取日志失败: {e}")
        return

    if not trials:
        print("日志中未解析到任何选择记录")
        print(f"调试: 匹配到标准名行数={matched_wav_lines}, 匹配到RESP行数={matched_resp_lines}")
        return
    else:
        print(f"解析到试次数: {len(trials)}")
        print(f"调试: 匹配到标准名行数={matched_wav_lines}, 匹配到RESP行数={matched_resp_lines}")
        print(f"日志中解析到的标准名样例: {[t['std_name'] for t in trials[:10]]}")

    # 组装 trial 明细（仅内存使用，不导出）
    detailed = []
    unmatched = []
    for t in trials:
        std = t['std_name']
        std_key = os.path.splitext(std)[0].lower()
        # 先用无扩展名匹配，不行再尝试带 .wav
        key_try = std_key if std_key in std_to_pair else (std_key + '.wav')
        if key_try not in std_to_pair:
            unmatched.append(std)
            continue
        left_cat, right_cat, orig_name = std_to_pair[key_try]
        detailed.append({
            'original_filename': orig_name,
            'left_category': left_cat,
            'right_category': right_cat,
            'selected_side': 'Left' if t['resp'] == 1 else 'Right'
        })

    if not detailed:
        print("无有效的 trial 可用于统计（Excel 映射与日志不匹配）")
        if unmatched:
            print(f"未匹配标准名样本数: {len(unmatched)}，示例: {unmatched[:10]}")
        return
    if unmatched:
        print(f"警告：有 {len(unmatched)} 个标准名未在 Excel 中找到映射，示例: {unmatched[:10]}")

    # 映射主/子类别与色系
    main_category_groups = {
        'High Ecology': ['Telephone bell ringing', 'Ambulance (siren)', 'Baby cry, infant cry'],
        'Low Ecology': ['Computer keyboard', 'Helicopter', 'Chicken, rooster'],
        'speech': ['Male speech, man speaking', 'Female speech, woman speaking', 'Child speech, kid speaking'],
        'music': ['Bass drum', 'Funny music', 'Sad music'],
        'Unknown Source': ['Pulse', 'Whack, thwack', 'Crumpling, crinkling']
    }

    main_category_colors = {
        'High Ecology': '#FF6B6B',
        'Low Ecology': '#DDA0DD',
        'speech': '#45B7D1',
        'music': '#96CEB4',
        'Unknown Source': '#FFEAA7',
    }

    def get_main_category(sub_category):
        for main_cat, subs in main_category_groups.items():
            if sub_category in subs:
                return main_cat
        return 'Unknown Source'

    # 为每个子类别设置同一色系不同明度
    category_colors = {}
    for main_cat, subs in main_category_groups.items():
        base_color = main_category_colors.get(main_cat, '#DDA0DD')
        base_rgb = mcolors.to_rgb(base_color)
        for idx, sub in enumerate(subs):
            brightness = 0.4 + 0.6 * (idx / (len(subs) - 1)) if len(subs) > 1 else 0.7
            category_colors[sub] = tuple(c * brightness for c in base_rgb)

    # 构建颜色热力图：每个配对单元用被选中子类别对应颜色填充
    def plot_color_heatmap(prefix, title, filename):
        # 颜色矩阵 (n, n, 3) 初始化为 NaN
        color_matrix = np.full((n, n, 3), np.nan, dtype=float)

        for r in detailed:
            name_wo_ext = os.path.splitext(os.path.basename(r['original_filename']))[0]
            if not name_wo_ext.startswith(prefix):
                continue
            left_cat = r['left_category']
            right_cat = r['right_category']
            if left_cat not in idx_map or right_cat not in idx_map:
                continue
            selected_cat = left_cat if r['selected_side'] == 'Left' else right_cat
            color = category_colors.get(selected_cat, (0.8, 0.8, 0.8))
            i = idx_map[left_cat]
            j = idx_map[right_cat]
            color_matrix[i, j] = color

        # 绘制：按单元格填充对应颜色
        try:
            plt.style.use('seaborn-v0_8')
        except OSError:
            try:
                plt.style.use('seaborn')
            except OSError:
                plt.style.use('default')

        fig, ax = plt.subplots(1, 1, figsize=(18, 16))
        # 先画白色背景
        ax.imshow(np.ones((n, n, 3)))
        for i in range(n):
            for j in range(n):
                if not np.isnan(color_matrix[i, j, 0]):
                    rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                         facecolor=tuple(color_matrix[i, j]),
                                         edgecolor='black', linewidth=0.5)
                    ax.add_patch(rect)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        x_labels = ax.set_xticklabels(target_categories, rotation=45, ha='right')
        y_labels = ax.set_yticklabels(target_categories)
        # 轴标签颜色与小类颜色一致（同色系不同明度）
        for i, lab in enumerate(x_labels):
            cat = target_categories[i]
            col = category_colors.get(cat, (0.2, 0.2, 0.2))
            lab.set_color(col)
            lab.set_fontweight('bold')
        for i, lab in enumerate(y_labels):
            cat = target_categories[i]
            col = category_colors.get(cat, (0.2, 0.2, 0.2))
            lab.set_color(col)
            lab.set_fontweight('bold')
        ax.set_xlabel('Right Channel Category')
        ax.set_ylabel('Left Channel Category')
        ax.set_title(title, fontsize=16, fontweight='bold')
        # 网格
        ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.3, alpha=0.3)
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(-0.5, n - 0.5)
        plt.tight_layout()
        out_path = os.path.join(output_dir, filename)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"热力图已保存: {out_path}")

    # 绘图样式
    try:
        plt.style.use('seaborn-v0_8')
    except OSError:
        try:
            plt.style.use('seaborn')
        except OSError:
            plt.style.use('default')

    # 生成两张颜色热力图（nn_/rn_）
    plot_color_heatmap('nn_', 'Selection Color Heatmap - nn_', 'collect_color_heatmap_nn.png')
    plot_color_heatmap('rn_', 'Selection Color Heatmap - rn_', 'collect_color_heatmap_rn.png')


def main():
    # 默认使用你提供的路径；可按需修改
    excel_path = r"D:\D\research\数据采集\video_timestamps2.xlsx"
    log_path = r"D:\D\research\数据采集\Jiachen_session2_20250926_052722.mff\benchmark_1_10-1-1.txt"
    output_dir = r"D:\D\research\数据采集\collect_out2"
    collect_selection(excel_path, log_path, output_dir)


if __name__ == "__main__":
    main()


