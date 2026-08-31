#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的EEG数据读取脚本
用于快速查看EEG数据的基本信息
"""

import os
import numpy as np
import pandas as pd
import json
from typing import Dict, List, Tuple, Optional

def read_eeg_info(data_dir: str, subject_id: str = "sub-01", 
                  session: str = "ses-MMN", task: str = "Optimum1") -> Dict:
    """
    读取EEG数据的基本信息
    
    Args:
        data_dir: EEG数据目录路径
        subject_id: 被试ID
        session: 会话名称
        task: 任务名称
        
    Returns:
        包含EEG数据信息的字典
    """
    # 构建文件路径
    base_name = f"{subject_id}_{session}_task-{task}_eeg"
    eeg_file = os.path.join(data_dir, f"{base_name}.eeg")
    vhdr_file = os.path.join(data_dir, f"{base_name}.vhdr")
    json_file = os.path.join(data_dir, f"{base_name}.json")
    channels_file = os.path.join(data_dir, f"{base_name}_channels.tsv")
    events_file = os.path.join(data_dir, f"{base_name}_events.tsv")
    
    print(f"正在读取EEG数据信息: {subject_id} - {session} - {task}")
    print(f"数据目录: {data_dir}")
    
    # 检查文件是否存在
    files_info = {}
    for file_path, file_name in [(eeg_file, "EEG数据文件"), 
                                (vhdr_file, "头文件"), 
                                (json_file, "JSON元数据文件"),
                                (channels_file, "通道信息文件"),
                                (events_file, "事件文件")]:
        exists = os.path.exists(file_path)
        files_info[file_name] = exists
        print(f"{file_name}: {'存在' if exists else '不存在'}")
    
    # 读取头文件信息
    header_info = read_vhdr_file(vhdr_file)
    
    # 读取EEG数据大小
    eeg_size = get_eeg_file_size(eeg_file, header_info)
    
    # 读取通道信息
    channel_info = None
    if os.path.exists(channels_file):
        channel_info = pd.read_csv(channels_file, sep='\t')
    
    # 读取事件信息
    events_info = None
    if os.path.exists(events_file):
        events_df = pd.read_csv(events_file, sep='\t')
        events_info = {
            'count': len(events_df),
            'columns': list(events_df.columns),
            'first_few': events_df.head(3).to_dict('records') if len(events_df) > 0 else []
        }
    
    # 读取JSON元数据
    metadata = None
    if os.path.exists(json_file):
        with open(json_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    
    # 计算数据统计信息
    sampling_rate = header_info.get('sampling_rate', 2500)
    n_channels = header_info.get('n_channels', 64)
    n_samples = eeg_size['n_samples']
    duration = n_samples / sampling_rate
    
    # 构建结果
    result = {
        '文件信息': files_info,
        '数据维度': {
            '通道数': n_channels,
            '样本数': n_samples,
            '采样率': f"{sampling_rate} Hz",
            '记录时长': f"{duration:.2f} 秒",
            '数据形状': f"{n_channels} × {n_samples}"
        },
        '通道信息': {
            '通道数': len(channel_info) if channel_info is not None else n_channels,
            '通道名称': list(channel_info['name']) if channel_info is not None else [],
            '通道类型': list(channel_info['type'].unique()) if channel_info is not None else ['EEG']
        } if channel_info is not None else None,
        '事件信息': events_info,
        '元数据': {
            '任务名称': metadata.get('TaskName', '未知'),
            '记录时长': f"{metadata.get('RecordingDuration', duration):.2f} 秒",
            '通道数': metadata.get('EEGChannelCount', n_channels),
            '采样率': f"{metadata.get('SamplingFrequency', sampling_rate)} Hz",
            '参考电极': metadata.get('EEGReference', '未知'),
            '滤波器设置': metadata.get('SoftwareFilters', {})
        } if metadata is not None else None,
        '头文件信息': header_info
    }
    
    return result

def read_vhdr_file(vhdr_file: str) -> Dict:
    """读取.vhdr头文件"""
    header_info = {}
    
    if not os.path.exists(vhdr_file):
        print(f"警告: 头文件不存在: {vhdr_file}")
        return header_info
    
    with open(vhdr_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    current_section = None
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith(';'):
            continue
            
        if line.startswith('[') and line.endswith(']'):
            current_section = line[1:-1].lower()
            continue
            
        if '=' in line:
            key, value = line.split('=', 1)
            key = key.strip().lower()
            value = value.strip()
            
            if current_section == 'common infos':
                if key == 'datafile':
                    header_info['data_file'] = value
                elif key == 'markerfile':
                    header_info['marker_file'] = value
                elif key == 'dataformat':
                    header_info['data_format'] = value
                elif key == 'dataorientation':
                    header_info['data_orientation'] = value
                elif key == 'numberofchannels':
                    header_info['n_channels'] = int(value)
                elif key == 'samplinginterval':
                    # 采样间隔（微秒）
                    sampling_interval_us = float(value)
                    header_info['sampling_rate'] = 1000000 / sampling_interval_us
                    
            elif current_section == 'binary infos':
                if key == 'binaryformat':
                    header_info['binary_format'] = value
                    
            elif current_section == 'channel infos':
                if key.startswith('ch'):
                    ch_num = int(key[2:])
                    if 'channels' not in header_info:
                        header_info['channels'] = {}
                    header_info['channels'][ch_num] = value
                    
    return header_info

def get_eeg_file_size(eeg_file: str, header_info: Dict) -> Dict:
    """获取EEG文件的大小信息"""
    if not os.path.exists(eeg_file):
        return {'n_samples': 0, 'file_size_mb': 0}
    
    # 获取文件大小
    file_size_bytes = os.path.getsize(eeg_file)
    file_size_mb = file_size_bytes / (1024 * 1024)
    
    # 计算样本数
    n_channels = header_info.get('n_channels', 64)
    binary_format = header_info.get('binary_format', 'INT_16')
    
    if binary_format == 'INT_16':
        bytes_per_sample = 2
    elif binary_format == 'FLOAT_32':
        bytes_per_sample = 4
    else:
        bytes_per_sample = 2  # 默认
    
    n_samples = file_size_bytes // (n_channels * bytes_per_sample)
    
    return {
        'n_samples': n_samples,
        'file_size_mb': file_size_mb,
        'bytes_per_sample': bytes_per_sample
    }

def print_eeg_info(info: Dict):
    """打印EEG信息"""
    print("\n" + "="*60)
    print("EEG数据信息摘要")
    print("="*60)
    
    # 文件信息
    print("\n📁 文件状态:")
    for file_name, exists in info['文件信息'].items():
        status = "✅" if exists else "❌"
        print(f"  {status} {file_name}")
    
    # 数据维度
    print(f"\n📊 数据维度:")
    for key, value in info['数据维度'].items():
        print(f"  {key}: {value}")
    
    # 通道信息
    if info['通道信息']:
        print(f"\n🔌 通道信息:")
        print(f"  通道数: {info['通道信息']['通道数']}")
        print(f"  通道类型: {', '.join(info['通道信息']['通道类型'])}")
        if len(info['通道信息']['通道名称']) <= 10:
            print(f"  通道名称: {', '.join(info['通道信息']['通道名称'])}")
        else:
            print(f"  通道名称: {', '.join(info['通道信息']['通道名称'][:10])} ... (共{len(info['通道信息']['通道名称'])}个)")
    
    # 事件信息
    if info['事件信息']:
        print(f"\n📅 事件信息:")
        print(f"  事件数量: {info['事件信息']['count']}")
        print(f"  事件列: {', '.join(info['事件信息']['columns'])}")
        if info['事件信息']['first_few']:
            print("  前几个事件:")
            for i, event in enumerate(info['事件信息']['first_few'][:3]):
                print(f"    {i+1}: {event}")
    
    # 元数据
    if info['元数据']:
        print(f"\n📋 实验元数据:")
        for key, value in info['元数据'].items():
            print(f"  {key}: {value}")
    
    print("\n" + "="*60)

def main():
    """主函数"""
    # EEG数据目录
    eeg_dir = r"D:\D\research\audioset下载\download_audioset-master\download_audioset-master\auditory oddball\sub-01\ses-MMN\eeg"
    
    try:
        # 读取EEG信息
        info = read_eeg_info(eeg_dir)
        
        # 打印信息
        print_eeg_info(info)
        
        # 额外统计信息
        print(f"\n💾 存储信息:")
        if info['数据维度']['样本数'] > 0:
            bytes_per_sample = 2  # INT_16
            total_bytes = info['数据维度']['通道数'] * info['数据维度']['样本数'] * bytes_per_sample
            print(f"  理论数据大小: {total_bytes / (1024*1024):.2f} MB")
            print(f"  每个通道样本数: {info['数据维度']['样本数']:,}")
            print(f"  每秒数据量: {info['数据维度']['通道数'] * 2500 * 2 / (1024*1024):.2f} MB/s")
        
    except Exception as e:
        print(f"❌ 读取数据时出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
