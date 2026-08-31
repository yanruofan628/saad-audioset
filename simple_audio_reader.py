#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的声音刺激读取工具
专门用于读取ds005429数据集中的声音刺激文件
"""

import numpy as np
import pandas as pd
from pathlib import Path

def read_audio_stimuli(subject="sub-01", session="ses-MMN"):
    """
    读取指定被试和会话的声音刺激数据
    
    Parameters:
    subject: 被试ID (默认: sub-01)
    session: 会话名称 (默认: ses-MMN)
    
    Returns:
    dict: 包含刺激数据和事件数据的字典
    """
    
    # 设置路径
    base_path = Path(r"D:\D\research\audioset下载\ds005429-main\ds005429-main\sourcedata")
    session_path = base_path / subject / session
    
    if not session_path.exists():
        print(f"路径不存在: {session_path}")
        return None
    
    print(f"读取 {subject}/{session} 的声音刺激数据...")
    
    result = {
        'subject': subject,
        'session': session,
        'stimuli_data': None,
        'events_data': None,
        'stimuli_file': None,
        'events_file': None
    }
    
    # 读取刺激序列文件
    sequence_files = list(session_path.glob("*stimulation_SEQUENCE.mat"))
    if sequence_files:
        sequence_file = sequence_files[0]
        print(f"读取刺激序列: {sequence_file.name}")
        
        # 尝试使用h5py读取
        try:
            import h5py
            with h5py.File(sequence_file, 'r') as f:
                stimuli_data = {}
                for key in f.keys():
                    if not key.startswith('#'):
                        stimuli_data[key] = f[key][:]
                        print(f"  - {key}: {stimuli_data[key].shape}")
                result['stimuli_data'] = stimuli_data
                result['stimuli_file'] = sequence_file.name
        except ImportError:
            print("h5py未安装，请运行: pip install h5py")
        except Exception as e:
            print(f"读取刺激序列失败: {e}")
    
    # 读取事件文件
    events_files = list(session_path.glob("*critical_events.mat"))
    if events_files:
        events_file = events_files[0]
        print(f"读取事件文件: {events_file.name}")
        
        # 尝试使用h5py读取
        try:
            import h5py
            with h5py.File(events_file, 'r') as f:
                events_data = {}
                for key in f.keys():
                    if not key.startswith('#'):
                        events_data[key] = f[key][:]
                        print(f"  - {key}: {events_data[key].shape}")
                result['events_data'] = events_data
                result['events_file'] = events_file.name
        except ImportError:
            print("h5py未安装，请运行: pip install h5py")
        except Exception as e:
            print(f"读取事件文件失败: {e}")
    
    return result

def analyze_stimuli(stimuli_data):
    """
    分析刺激数据
    
    Parameters:
    stimuli_data: 刺激数据字典
    
    Returns:
    dict: 分析结果
    """
    if not stimuli_data:
        return None
    
    print("\n=== 刺激数据分析 ===")
    
    analysis = {}
    
    for key, value in stimuli_data.items():
        print(f"\n变量: {key}")
        print(f"  形状: {value.shape}")
        print(f"  类型: {value.dtype}")
        
        # 如果是小数组，显示内容
        if value.size < 20:
            print(f"  内容: {value}")
        else:
            print(f"  前5个值: {value.flat[:5]}")
            print(f"  后5个值: {value.flat[-5:]}")
        
        # 保存基本信息
        analysis[key] = {
            'shape': value.shape,
            'dtype': str(value.dtype),
            'size': value.size,
            'min': float(np.min(value)) if value.size > 0 else None,
            'max': float(np.max(value)) if value.size > 0 else None,
            'mean': float(np.mean(value)) if value.size > 0 else None
        }
    
    return analysis

def analyze_events(events_data):
    """
    分析事件数据
    
    Parameters:
    events_data: 事件数据字典
    
    Returns:
    dict: 分析结果
    """
    if not events_data:
        return None
    
    print("\n=== 事件数据分析 ===")
    
    analysis = {}
    
    for key, value in events_data.items():
        print(f"\n变量: {key}")
        print(f"  形状: {value.shape}")
        print(f"  类型: {value.dtype}")
        
        # 如果是小数组，显示内容
        if value.size < 20:
            print(f"  内容: {value}")
        else:
            print(f"  前5个值: {value.flat[:5]}")
            print(f"  后5个值: {value.flat[-5:]}")
        
        # 保存基本信息
        analysis[key] = {
            'shape': value.shape,
            'dtype': str(value.dtype),
            'size': value.size,
            'min': float(np.min(value)) if value.size > 0 else None,
            'max': float(np.max(value)) if value.size > 0 else None,
            'mean': float(np.mean(value)) if value.size > 0 else None
        }
    
    return analysis

def save_analysis_results(result, output_dir="audio_stimuli_analysis"):
    """
    保存分析结果
    
    Parameters:
    result: 分析结果字典
    output_dir: 输出目录
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n=== 保存结果到 {output_dir} ===")
    
    # 保存刺激数据
    if result['stimuli_data']:
        stimuli_dir = output_dir / "stimuli"
        stimuli_dir.mkdir(exist_ok=True)
        
        for key, value in result['stimuli_data'].items():
            np.save(stimuli_dir / f"{key}.npy", value)
            print(f"✓ 保存刺激数据: {key}.npy")
    
    # 保存事件数据
    if result['events_data']:
        events_dir = output_dir / "events"
        events_dir.mkdir(exist_ok=True)
        
        for key, value in result['events_data'].items():
            np.save(events_dir / f"{key}.npy", value)
            print(f"✓ 保存事件数据: {key}.npy")
    
    # 保存分析摘要
    summary = {
        'subject': result['subject'],
        'session': result['session'],
        'stimuli_file': result['stimuli_file'],
        'events_file': result['events_file'],
        'stimuli_variables': list(result['stimuli_data'].keys()) if result['stimuli_data'] else [],
        'events_variables': list(result['events_data'].keys()) if result['events_data'] else []
    }
    
    import json
    with open(output_dir / "summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 保存摘要: summary.json")

def main():
    """主函数"""
    print("=== 简单声音刺激读取工具 ===")
    
    # 读取sub-01/ses-MMN的声音刺激数据
    result = read_audio_stimuli("sub-01", "ses-MMN")
    
    if result:
        # 分析刺激数据
        if result['stimuli_data']:
            stimuli_analysis = analyze_stimuli(result['stimuli_data'])
        
        # 分析事件数据
        if result['events_data']:
            events_analysis = analyze_events(result['events_data'])
        
        # 保存结果
        save_analysis_results(result)
        
        print("\n=== 读取完成 ===")
        print(f"被试: {result['subject']}")
        print(f"会话: {result['session']}")
        print(f"刺激文件: {result['stimuli_file']}")
        print(f"事件文件: {result['events_file']}")
    else:
        print("读取失败")

if __name__ == "__main__":
    main()
