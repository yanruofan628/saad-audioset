import os
import numpy as np
import re
import mne
from datetime import datetime, timedelta
from extract_channel_info import make_montage
from collections import defaultdict
import mffpy
import logging
from mne.preprocessing import ICA
import json
import pandas as pd

# 导入 PyPREP 用于自动检测坏通道
try:
    from pyprep import PrepPipeline
    PYPREP_AVAILABLE = True
except ImportError:
    print("警告: 未安装 PyPREP，将使用手动指定的坏电极列表")
    PYPREP_AVAILABLE = False

# 配置参数
BASE_DATA_DIRS = [r"A:\\", r"A:/standard_data_noica"]
SAMPLING_RATE = 250

# 定义受试者的多个MFF文件配置
SUBJECT_CONFIG = {
    "subject_name": "zhangyufei",
    "mff_files": [
        {
            "mff": "zhangyufei0106_1_20260106_054423.mff",
            "benchmark": "benchmark_1_10-1-1.txt",
            "max_trials": 160,  # 第一个文件取前159个trials
            "reverse_order": False  # 不倒序
        },
        {
            "mff": "zhangyufei0106_2_20260106_060959.mff",
            "benchmark": "benchmark_1_10-1-2.txt",
            "max_trials": 160,  # 第二个文件取前159个trials
            "reverse_order": False  # 不倒序
        },
        {
            "mff": "zhangyufei0106_3_20260106_063511.mff",
            "benchmark": "benchmark_1_10-1-3.txt",
            "max_trials": 160,  # 第三个文件取最后160个trials
            "reverse_order": False   # 倒序
        }
    ],
    "trial_duration": 6  # 每个trial提取6秒数据
}

# 如果 PyPREP 不可用，使用手动指定的坏电极列表作为备用
MANUAL_BAD_ELECTRODES = ['E48', 'E119', 'E49', 'E113', 'E44', 'E114',
                         'E131', 'E132', 'E56', 'E107', 'E126', 'E127',
                         'E43', 'E120', 'E45', 'E108', 'E57', 'E100', 'E99', 'E63']


def extract_labels(benchmark_file_path):
    """从benchmark文件提取trial的标签"""
    if not os.path.exists(benchmark_file_path):
        print(f"错误: benchmark文件不存在: {benchmark_file_path}")
        return {}

    try:
        # 尝试不同的编码方式读取文件
        encodings = ['utf-16-le', 'utf-16', 'utf-16-be', 'utf-8-sig', 'utf-8', 'gbk', 'gb2312', 'latin-1', 'cp1252']
        file_content = None

        for encoding in encodings:
            try:
                with open(benchmark_file_path, 'r', encoding=encoding, errors='strict') as f:
                    candidate = f.read()
                # 校验是否为合理内容（包含关键字段）
                if re.search(r"List1:|RESP|ImageDisplay", candidate):
                    file_content = candidate
                    print(f"成功使用 {encoding} 编码读取benchmark文件")
                    break
            except UnicodeDecodeError:
                continue

        if file_content is None:
            # 退化方案：忽略错误读取
            with open(benchmark_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                file_content = f.read()
            print("以 utf-8(ignore) 方式读取benchmark文件（可能存在部分字符丢失）")

        # 归一化特殊换行符并按行分割
        file_content = file_content.replace('\u2028', '\n').replace('\u2029', '\n')
        lines = re.split(r"\r\n|\r|\n", file_content)
        print(f"benchmark文件总共有 {len(lines)} 行")

        # 查找trial和标签
        labels = {}
        current_trial = None

        # 匹配格式: List1: 1 和 ImageDisplay1.RESP: 1/2
        list_pattern = r'List1:\s*(\d+)'
        resp_pattern = r'ImageDisplay1\.RESP:\s*([12])'

        for i, line in enumerate(lines):
            line = line.strip()

            # 查找List1: trial编号
            list_match = re.match(list_pattern, line)
            if list_match:
                current_trial = int(list_match.group(1))
                continue

            # 查找ImageDisplay1.RESP: 标签
            resp_match = re.match(resp_pattern, line)
            if resp_match and current_trial is not None:
                resp_value = int(resp_match.group(1))
                # 映射: 1->0, 2->1
                label = 0 if resp_value == 1 else 1
                labels[current_trial] = label
                current_trial = None  # 重置，避免重复匹配

        print(f"总共提取到 {len(labels)} 个trial的标签")
        return labels

    except Exception as e:
        print(f"读取benchmark文件错误: {e}")
        return {}




def find_trial_times(log_file_path):
    """从log文件找到trial的开始时间，匹配PLTECIEvents格式"""
    if not os.path.exists(log_file_path):
        print(f"错误: log文件不存在: {log_file_path}")
        return []

    try:
        # 优先采用二进制级别匹配
        try:
            with open(log_file_path, 'rb') as fb:
                raw_bytes = fb.read()
            # 仅匹配包含 PLTECIEvents 和 TBEG 的事件行
            bexpr = re.compile(rb"(\d{2}:\d{2}:\d{2}\.\d{3})\.\d{3}:[^\r\n]*?PLTECIEvents[^\r\n]*?TBEG", re.IGNORECASE)
            trial_times = []
            for m in bexpr.finditer(raw_bytes):
                time_str = m.group(1).decode('ascii', errors='ignore')
                time_parts = time_str.split(':')
                hours = int(time_parts[0])
                minutes = int(time_parts[1])
                seconds_parts = time_parts[2].split('.')
                seconds = int(seconds_parts[0])
                milliseconds = int(seconds_parts[1])
                trial_start_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0
                snippet = m.group(0)[:200]
                try:
                    snippet_text = snippet.decode('utf-8', errors='ignore')
                except Exception:
                    snippet_text = repr(snippet)
                trial_times.append({
                    'trial_num': len(trial_times) + 1,
                    'time': time_str,
                    'start_seconds': trial_start_seconds,
                    'line': snippet_text,
                    'line_number': None,
                })
            if len(trial_times) > 0:
                print(f"二进制扫描找到 {len(trial_times)} 个TBEG 事件")
                # 排序并编号
                trial_times.sort(key=lambda x: x['start_seconds'])
                for i, t in enumerate(trial_times):
                    t['trial_num'] = i + 1
                return trial_times
        except Exception as e:
            print(f"二进制扫描失败，改用文本解析: {e}")

        # 尝试不同的编码方式读取文件
        encodings = ['utf-16-le', 'utf-16', 'utf-16-be', 'utf-8-sig', 'utf-8', 'gbk', 'gb2312', 'latin-1', 'cp1252']
        file_content = None

        for encoding in encodings:
            try:
                with open(log_file_path, 'r', encoding=encoding, errors='strict') as f:
                    candidate = f.read()
                if re.search(r"PLTECI|TBEG|Event:", candidate, re.IGNORECASE):
                    file_content = candidate
                    print(f"成功使用 {encoding} 编码读取log文件")
                    break
            except UnicodeDecodeError:
                continue

        if file_content is None:
            with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                file_content = f.read()
            print("以 utf-8(ignore) 方式读取log文件（可能存在部分字符丢失）")

        # 归一化特殊换行符并按行分割
        file_content = file_content.replace('\u2028', '\n').replace('\u2029', '\n')
        lines = re.split(r"\r\n|\r|\n", file_content)
        print(f"log文件总共有 {len(lines)} 行")

        # 查找trial事件
        trial_times = []
        pattern = r'(\d{2}:\d{2}:\d{2}\.\d{3})\.\d{3}:\s*[^\r\n]*PLTECIEvents[^\r\n]*TBEG'

        for i, line in enumerate(lines):
            raw_line = line
            line = line.strip()
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                time_str = match.group(1)
                time_parts = time_str.split(':')
                hours = int(time_parts[0])
                minutes = int(time_parts[1])
                seconds_parts = time_parts[2].split('.')
                seconds = int(seconds_parts[0])
                milliseconds = int(seconds_parts[1])
                trial_start_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0

                trial_times.append({
                    'trial_num': len(trial_times) + 1,
                    'time': time_str,
                    'start_seconds': trial_start_seconds,
                    'line': raw_line,
                    'line_number': i + 1
                })

        # 如果按行解析未找到，退化为全文正则搜索
        if len(trial_times) == 0:
            print("按行解析未找到TBEG，尝试全文扫描...")
            trial_times = []
            iter_pattern = re.compile(r"(\d{2}:\d{2}:\d{2}\.\d{3})\.\d{3}:\s*[^\n\r]*PLTECIEvents[^\n\r]*TBEG", re.IGNORECASE)
            for m in iter_pattern.finditer(file_content):
                time_str = m.group(1)
                time_parts = time_str.split(':')
                hours = int(time_parts[0])
                minutes = int(time_parts[1])
                seconds_parts = time_parts[2].split('.')
                seconds = int(seconds_parts[0])
                milliseconds = int(seconds_parts[1])
                trial_start_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0
                trial_times.append({
                    'trial_num': len(trial_times) + 1,
                    'time': time_str,
                    'start_seconds': trial_start_seconds,
                    'line': m.group(0)[:200],
                    'line_number': None
                })
            print(f"全文扫描找到 {len(trial_times)} 个TBEG 事件")

        print(f"总共找到 {len(trial_times)} 个trial")

        # 按时间顺序排序
        trial_times.sort(key=lambda x: x['start_seconds'])

        # 重新编号
        for i, trial in enumerate(trial_times):
            trial['trial_num'] = i + 1

        return trial_times

    except Exception as e:
        print(f"读取log文件错误: {e}")
        return []




def process_subject(subject_config, output_root="A:/standard_data_noica", apply_ica=True):
    """处理单个受试者的多个MFF文件数据"""
    subject_name = subject_config["subject_name"]
    mff_files = subject_config["mff_files"]
    trial_duration = subject_config["trial_duration"]

    print(f"\n{'='*60}")
    print(f"开始处理受试者: {subject_name}")
    print(f"{'='*60}\n")
    print(f"ICA 去除 {'启用' if apply_ica else '未启用'}")
    print(f"输出目录: {output_root}")
    print(f"MFF文件配置: {len(mff_files)} 个文件")

    # 创建输出目录
    processed_data_path = os.path.join(output_root, subject_name)
    if not os.path.exists(processed_data_path):
        os.makedirs(processed_data_path, exist_ok=True)
    print(f"输出目录: {processed_data_path}")

    # 收集所有文件的trial时间和标签
    all_trial_times = []
    all_labels = []
    start_datetimes = []
    raw_objects = []
    log_files = []
    selected_trials_list = []
    file_info = []

    # 逐个处理每个MFF文件
    for i, mff_config in enumerate(mff_files):
        print(f"\n{'-'*40}")
        print(f"处理文件 {i+1}/{len(mff_files)}: {mff_config['mff']}")
        print(f"{'-'*40}")
        print(f"目标: 提取 {mff_config['max_trials']} 个trials, {'倒序' if mff_config['reverse_order'] else '正序'}")

        # 在多个目录中查找MFF文件
        mff_dir = None
        for base_dir in BASE_DATA_DIRS:
            candidate_path = os.path.join(base_dir, mff_config["mff"])
            if os.path.exists(candidate_path):
                mff_dir = candidate_path
                print(f"在 {base_dir} 中找到MFF文件")
                break

        if mff_dir is None:
            print(f"错误: 在任何配置的目录中都找不到MFF文件 {mff_config['mff']}")
            print(f"搜索的目录: {BASE_DATA_DIRS}")
            continue

        benchmark_file = os.path.join(mff_dir, mff_config["benchmark"])
        print(f"MFF目录: {mff_dir}")
        print(f"Benchmark文件: {benchmark_file}")
        print(f"Benchmark文件存在: {os.path.exists(benchmark_file)}")

        # 查找log文件
        log_file = None
        mff_name = os.path.basename(mff_dir).replace('.mff', '')

        possible_log_names = [
            f"log_{mff_name}.txt",
            f"{mff_name}.txt",
            "log.txt"
        ]

        for log_name in possible_log_names:
            candidate = os.path.join(mff_dir, log_name)
            if os.path.exists(candidate):
                log_file = candidate
                break

        if log_file is None:
            print(f"警告: 未找到log文件，尝试在目录中查找...")
            if os.path.exists(mff_dir):
                txt_files = [f for f in os.listdir(mff_dir) if f.lower().endswith('.txt')]
                print(f"目录中的TXT文件: {txt_files}")
                for f in os.listdir(mff_dir):
                    if f.lower().endswith('.txt') and 'log' in f.lower():
                        log_file = os.path.join(mff_dir, f)
                        print(f"找到log文件: {log_file}")
                        break

        # 检查文件是否存在
        if not os.path.exists(mff_dir):
            print(f"错误: MFF目录不存在，跳过文件 {mff_config['mff']}")
            continue

        # 读取MFF文件
        print(f"正在读取MFF文件...")
        raw = mne.io.read_raw_egi(mff_dir, preload=True)
        fo = mffpy.Reader(mff_dir)
        start_datetime = fo.startdatetime
        start_datetimes.append(start_datetime)

        print(f"文件数据形状: {raw.get_data().shape}")
        print(f"通道数量: {len(raw.ch_names)}")
        print(f"采样率: {raw.info['sfreq']} Hz")
        print(f"开始时间: {start_datetime}")

        raw_objects.append(raw)
        log_files.append(log_file)

        # 查找trial时间
        print(f"正在查找trial时间...")
        trial_times = find_trial_times(log_file) if log_file else []
        print(f"找到 {len(trial_times)} 个trials")

        # 根据配置选择trials
        max_trials = mff_config['max_trials']
        reverse_order = mff_config['reverse_order']

        if len(trial_times) >= max_trials:
            if reverse_order:
                # 倒序选择最后max_trials个
                selected_trials = trial_times[-max_trials:]
                print(f"选择最后 {max_trials} 个trials")
            else:
                # 正序选择前max_trials个
                selected_trials = trial_times[:max_trials]
                print(f"选择前 {max_trials} 个trials")
        else:
            selected_trials = trial_times
            print(f"警告: trials数量不足({len(trial_times)} < {max_trials})，使用全部trials")

        # 标记来源文件
        for trial in selected_trials:
            trial['source_file'] = i + 1

        selected_trials_list.append(selected_trials)
        all_trial_times.append(trial_times)  # 保存所有trials，用于标签对齐

        # 提取标签
        print(f"正在提取标签...")
        labels = extract_labels(benchmark_file)
        print(f"提取到 {len(labels)} 个标签")
        all_labels.append(labels)

        # 保存文件信息
        file_info.append({
            'file_index': i + 1,
            'mff_file': mff_config["mff"],
            'benchmark_file': mff_config["benchmark"],
            'log_file': log_file,
            'start_datetime': start_datetime,
            'total_trials': len(trial_times),
            'selected_trials': len(selected_trials),
            'max_trials': max_trials,
            'reverse_order': reverse_order,
            'num_labels': len(labels),
            'raw_shape': raw.get_data().shape
        })

        print(f"文件 {i+1} 处理完成！")

    if not raw_objects:
        print(f"错误: 没有成功读取任何MFF文件")
        return

    print(f"\n{'='*40}")
    print("所有文件处理完成，开始合并数据")
    print(f"{'='*40}")

    # 打印汇总信息
    print("\n文件处理汇总:")
    total_selected = 0
    for info in file_info:
        print(f"  文件{info['file_index']}: {info['selected_trials']}/{info['total_trials']} trials 选择")
        total_selected += info['selected_trials']
    print(f"总共选择: {total_selected} 个trials")

    # 使用第一个成功读取的文件的montage
    coordinates_path = None
    for base_dir in BASE_DATA_DIRS:
        candidate_path = os.path.join(base_dir, mff_files[0]["mff"], "coordinates.xml")
        if os.path.exists(candidate_path):
            coordinates_path = candidate_path
            print(f"在 {base_dir} 中找到坐标文件")
            break
    montage_path = os.path.join(processed_data_path, "electrode_montage.fif")

    if os.path.exists(coordinates_path):
        make_montage(coordinates_path, montage_path)
        montage = mne.channels.read_dig_fif(montage_path)
        print(f"电极数量: {len(montage.dig)}")
    else:
        print("警告: 坐标文件不存在，将使用默认montage")
        montage = None

    # 预处理函数
    def preprocess_raw(raw_data, montage=None, apply_ica=True, bad_channels_to_remove=None):
        """对单个raw对象进行预处理"""
        channels_before = len(raw_data.ch_names)

        # 仅保留前128个EEG通道
        eeg_picks = mne.pick_types(raw_data.info, eeg=True)
        if len(eeg_picks) == 0:
            print("警告: 未检测到EEG通道，保持原通道不变")
        else:
            if len(eeg_picks) > 128:
                eeg_picks = eeg_picks[:128]
            raw_data.pick(eeg_picks)
            print(f"已选择前{len(raw_data.ch_names)}个EEG通道")

        # 使用传入的统一坏通道列表
        if bad_channels_to_remove is not None:
            channels_to_remove_list = []
            for electrode in bad_channels_to_remove:
                if electrode in raw_data.ch_names:
                    channels_to_remove_list.append(electrode)

            if channels_to_remove_list:
                raw_data.drop_channels(channels_to_remove_list)
                print(f"已移除 {len(channels_to_remove_list)} 个坏电极（使用全局坏通道列表）")

        # 设置 Montage（如果有的话）
        if montage is not None:
            # 重命名通道以匹配montage格式 (E1 -> EEG001, E2 -> EEG002, etc.)
            channel_rename_dict = {}
            for ch_name in raw_data.ch_names:
                if ch_name.startswith('E') and ch_name[1:].isdigit():
                    num = int(ch_name[1:])
                    new_name = f'EEG{num:03d}'  # EEG001, EEG002, etc.
                    channel_rename_dict[ch_name] = new_name

            if channel_rename_dict:
                raw_data.rename_channels(channel_rename_dict)
                print(f"已重命名 {len(channel_rename_dict)} 个通道以匹配montage格式")

            raw_data.set_montage(montage, on_missing='warn')
            print("已设置电极 montage")
        else:
            print("警告: 未设置 montage，将使用默认布局")

        channels_after = len(raw_data.ch_names)
        print(f"处理前通道数: {channels_before}, 处理后通道数: {channels_after}")
        if channels_before != channels_after:
            print(f"移除了 {channels_before - channels_after} 个坏通道")

        raw_data.filter(l_freq=0.1, h_freq=45.0, method="fir", fir_window="hamming")
        raw_data.notch_filter(freqs=[50])
        raw_data.set_eeg_reference("average")

        if apply_ica:
            ica = ICA(
                n_components=15,
                random_state=97,
                method="infomax",
                max_iter=800,
            )

            try:
                ica.fit(raw_data)
                eog_channels = ["EEG017"]
                eog_inds = []
                for ch in eog_channels:
                    if ch in raw_data.ch_names:
                        inds, _scores = ica.find_bads_eog(
                            raw_data, ch_name=ch, threshold=2.0
                        )
                        eog_inds.extend(inds)

                eog_inds = sorted(set(eog_inds))
                if eog_inds:
                    ica.exclude = eog_inds
                    raw_data = ica.apply(raw_data.copy())
            except Exception as exc:
                print(f"ICA处理失败，跳过ICA: {exc}")

        return raw_data

    # 首先对第一个文件进行坏通道检测，确定统一的坏通道列表
    print("\n开始坏通道检测...")
    bad_channels_global = []

    if PYPREP_AVAILABLE:
        print("使用 PyPREP 对第一个文件进行坏通道检测...")
        # 复制第一个raw对象用于检测
        raw_for_detection = raw_objects[0].copy()

        # 简单的预处理用于检测
        eeg_picks = mne.pick_types(raw_for_detection.info, eeg=True)
        if len(eeg_picks) > 128:
            eeg_picks = eeg_picks[:128]
        raw_for_detection.pick(eeg_picks)

        # 设置 VREF 通道类型
        vref_candidates = ['VREF', 'EEG129', 'EEG128']
        for vref_name in vref_candidates:
            if vref_name in raw_for_detection.ch_names:
                try:
                    vref_type = raw_for_detection.get_channel_types(picks=[vref_name])[0]
                    if vref_type == 'eeg':
                        raw_for_detection.set_channel_types({vref_name: 'misc'})
                    break
                except:
                    continue

        # 设置 Montage
        if montage is not None:
            channel_rename_dict = {}
            for ch_name in raw_for_detection.ch_names:
                if ch_name.startswith('E') and ch_name[1:].isdigit():
                    num = int(ch_name[1:])
                    new_name = f'EEG{num:03d}'
                    channel_rename_dict[ch_name] = new_name

            if channel_rename_dict:
                raw_for_detection.rename_channels(channel_rename_dict)

            raw_for_detection.set_montage(montage, on_missing='warn')

        # PyPREP 参数配置
        prep_params = {
            "ref_chs": "eeg",
            "reref_chs": "eeg",
            "line_freqs": [50],
            "max_iterations": 4
        }

        try:
            # 创建并运行 PyPREP
            prep = PrepPipeline(raw_for_detection, prep_params, montage, ransac=True)
            prep.fit()

            # 获取检测到的坏道
            bad_channels = prep.noisy_channels_original['bad_all']
            bad_channels_eeg_format = [str(ch) for ch in bad_channels]

            # 将EEG格式转换为E格式，以便后续匹配
            bad_channels_global = []
            for ch in bad_channels_eeg_format:
                if ch.startswith('EEG') and ch[3:].isdigit():
                    num = int(ch[3:])
                    e_format = f'E{num}'
                    bad_channels_global.append(e_format)
                else:
                    bad_channels_global.append(ch)

            print(f"PyPREP检测到坏通道(EEG格式): {bad_channels_eeg_format}")
            print(f"转换为E格式用于移除: {bad_channels_global}")
            print(f"全局坏通道检测结果: {len(bad_channels_global)} 个坏道")

        except Exception as e:
            print(f"PyPREP 检测失败: {e}")
            print("将使用手动指定的坏电极列表作为全局坏通道...")
            bad_channels_global = MANUAL_BAD_ELECTRODES.copy()
            print(f"手动坏通道列表: {bad_channels_global}")
    else:
        print("PyPREP 不可用，使用手动指定的坏电极列表作为全局坏通道...")
        bad_channels_global = MANUAL_BAD_ELECTRODES.copy()
        print(f"手动坏通道列表: {bad_channels_global}")

    # 预处理所有raw对象，使用统一的坏通道列表
    print("\n开始预处理所有文件...")
    processed_raws = []
    for i, raw in enumerate(raw_objects):
        print(f"预处理文件 {i+1}...")
        processed_raw = preprocess_raw(raw, montage=montage, apply_ica=apply_ica,
                                     bad_channels_to_remove=bad_channels_global)
        processed_raws.append(processed_raw)
    print("预处理完成")

    # 拼接所有选择的trials（简单拼接，不考虑时间）
    print(f"\n拼接所有文件的trials...")
    merged_trials = []
    current_global_trial_num = 1

    for file_idx, selected_trials in enumerate(selected_trials_list):
        print(f"文件{file_idx+1}: 添加 {len(selected_trials)} 个trials")
        for trial in selected_trials:
            new_trial = trial.copy()
            new_trial['global_trial_num'] = current_global_trial_num
            merged_trials.append(new_trial)
            current_global_trial_num += 1

    print(f"拼接后总共有 {len(merged_trials)} 个trials")

    # 合并标签 - 按照选择的trials顺序
    print(f"合并标签...")
    merged_labels = {}
    global_trial_num = 1

    for file_idx, (selected_trials, labels) in enumerate(zip(selected_trials_list, all_labels)):
        print(f"文件{file_idx+1}: 处理 {len(selected_trials)} 个trials的标签")

        for trial in selected_trials:
            original_trial_num = trial['trial_num']
            # 从对应文件的标签中查找
            if original_trial_num in labels:
                merged_labels[global_trial_num] = labels[original_trial_num]
            else:
                merged_labels[global_trial_num] = None  # 没有找到标签

            global_trial_num += 1

    print(f"合并后总共有 {len(merged_labels)} 个标签")

    # 提取trial数据
    samples_per_trial = int(trial_duration * SAMPLING_RATE)
    epochs = []
    kept_trials_info = []

    print(f"\n开始提取trial数据，每个trial {trial_duration}秒...")

    for idx, trial in enumerate(merged_trials, start=1):
        source_file = trial['source_file']
        original_start_seconds = trial['start_seconds']

        # 选择对应的raw对象（文件索引从1开始，列表从0开始）
        raw_to_use = processed_raws[source_file - 1]

        start_sample = int(original_start_seconds * SAMPLING_RATE)
        stop_sample = start_sample + samples_per_trial

        if original_start_seconds < 0 or stop_sample / SAMPLING_RATE > raw_to_use.times[-1]:
            print(f"跳过 Trial {idx} (文件{source_file}): 超出数据范围 ({original_start_seconds:.3f}s -> {(stop_sample / SAMPLING_RATE):.3f}s)")
            continue

        data = raw_to_use.get_data(start=start_sample, stop=stop_sample)
        if data.shape[0] != len(raw_to_use.ch_names) or data.shape[1] != samples_per_trial:
            print(f"警告 Trial {idx} (文件{source_file}): 形状异常 {data.shape}，将尝试继续")

        epochs.append(data)
        kept_trials_info.append({
            'trial_num': idx,
            'global_trial_num': trial['global_trial_num'],
            'original_trial_num': trial['trial_num'],
            'source_file': source_file,
            'time': trial['time'],
            'start_seconds': original_start_seconds,
            'line': trial.get('line', ''),
            'line_number': trial.get('line_number', None)
        })

    if len(epochs) == 0:
        print("未能提取到有效trial数据，跳过数据保存。")
    else:
        # 合并所有epochs
        epochs_array = np.stack(epochs, axis=0)
        print(f"提取的epochs形状: {epochs_array.shape}  (trial, channel, time)")

        # 保存trial数据
        np.save(os.path.join(processed_data_path, f"{subject_name}_trials.npy"), epochs_array)

        # 保存trial信息
        trial_info = {
            'subject_name': subject_name,
            'data_name': f"{subject_name}",
            'sampling_rate': SAMPLING_RATE,
            'trial_duration': trial_duration,
            'num_trials': len(kept_trials_info),
            'channels': processed_raws[0].ch_names,  # 使用第一个文件的通道信息
            'trial_times': kept_trials_info,
            'mff_files': [config["mff"] for config in mff_files],
            'log_files': log_files,
        }

        # 对齐并保存标签
        if merged_labels:
            label_list = []
            for i in range(1, len(kept_trials_info) + 1):
                orig_trial_num = kept_trials_info[i-1]['original_trial_num']
                label_list.append({'Trial': i, 'OriginalTrial': orig_trial_num, 'Label': merged_labels.get(orig_trial_num, None)})

            labels_df = pd.DataFrame(label_list)
            labels_df.to_csv(os.path.join(processed_data_path, f"{subject_name}_labels.csv"), index=False)
            trial_info['labels'] = {int(row.Trial): (None if pd.isna(row.Label) else int(row.Label)) for _, row in labels_df.iterrows()}
            trial_info['num_labels'] = int(labels_df['Label'].notna().sum())

            # 打印标签分布
            if labels_df['Label'].notna().any():
                label_counts = labels_df.dropna()['Label'].value_counts().sort_index()
                print("\n标签分布:")
                for label, count in label_counts.items():
                    print(f"  标签 {int(label)}: {int(count)} 个trial")

        with open(os.path.join(processed_data_path, f"{subject_name}_trial_info.json"), 'w', encoding='utf-8') as f:
            json.dump(trial_info, f, ensure_ascii=False, indent=2)

        print(f"\n已保存trial数据到: {processed_data_path}")
        print(f"- 数据文件: {subject_name}_trials.npy (形状: {epochs_array.shape})")
        print(f"- 信息文件: {subject_name}_trial_info.json")
        if merged_labels:
            print(f"- 标签文件: {subject_name}_labels.csv")

        # 显示前几个trial的详细信息
        print(f"\n前5个trial的详细信息:")
        for i in range(min(5, len(kept_trials_info))):
            trial = kept_trials_info[i]
            orig_num = trial['original_trial_num']
            lab = merged_labels.get(orig_num, 'N/A') if merged_labels else 'N/A'
            print(f"  Trial {trial['trial_num']} (原始: {orig_num}, 文件: {trial['source_file']}): {trial['time']} - 标签: {lab}")

    print(f"\n受试者 {subject_name} 的数据处理完成！\n")


def main():
    """主函数：处理受试者的EEG数据"""
    print("="*60)
    print("开始处理受试者hanglei的EEG数据")
    print("="*60)

    for output_root, apply_ica in (
        ("A:/standard_data_noica", False),
    ):
        try:
            process_subject(SUBJECT_CONFIG, output_root=output_root, apply_ica=apply_ica)
        except Exception as e:
            mode = "ICA" if apply_ica else "无ICA"
            print(f"\n处理受试者 hanglei（{mode}）时发生错误: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("="*60)
    print("受试者hanglei的数据处理完成！")
    print("="*60)


if __name__ == "__main__":
    main()
