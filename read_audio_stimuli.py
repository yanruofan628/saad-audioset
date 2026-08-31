#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读取ds005429数据集中的声音刺激文件
支持MATLAB v7.3+格式的HDF5文件
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def read_matlab_hdf5(file_path):
    """
    使用h5py读取MATLAB v7.3+格式的HDF5文件
    
    Parameters:
    file_path: .mat文件路径
    
    Returns:
    dict: 包含所有变量的字典
    """
    try:
        import h5py
        print(f"使用h5py读取: {file_path.name}")
        
        data = {}
        with h5py.File(file_path, 'r') as f:
            print("文件中的变量:")
            
            def read_dataset(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"  - {name}: {obj.shape} {obj.dtype}")
                    try:
                        # 读取数据
                        data[name] = obj[:]
                        print(f"    ✓ 成功读取 {name}")
                    except Exception as e:
                        print(f"    ✗ 读取 {name} 失败: {e}")
                elif isinstance(obj, h5py.Group):
                    print(f"  - {name}: Group")
            
            f.visititems(read_dataset)
        
        return data
        
    except ImportError:
        print("h5py未安装，请运行: pip install h5py")
        return None
    except Exception as e:
        print(f"h5py读取失败: {e}")
        return None

def read_matlab_mat73(file_path):
    """
    使用mat73读取MATLAB v7.3+格式文件
    
    Parameters:
    file_path: .mat文件路径
    
    Returns:
    dict: 包含所有变量的字典
    """
    try:
        import mat73
        print(f"使用mat73读取: {file_path.name}")
        
        data = mat73.loadmat(file_path)
        print("成功读取文件")
        
        print("文件中的变量:")
        for key, value in data.items():
            if not key.startswith('__'):
                print(f"  - {key}: {type(value)}")
                if hasattr(value, 'shape'):
                    print(f"    形状: {value.shape}")
        
        return data
        
    except ImportError:
        print("mat73未安装，请运行: pip install mat73")
        return None
    except Exception as e:
        print(f"mat73读取失败: {e}")
        return None

def read_matlab_scipy(file_path):
    """
    使用scipy.io.loadmat读取传统MATLAB格式文件
    
    Parameters:
    file_path: .mat文件路径
    
    Returns:
    dict: 包含所有变量的字典
    """
    try:
        from scipy.io import loadmat
        print(f"使用scipy读取: {file_path.name}")
        
        data = loadmat(file_path)
        print("成功读取文件")
        
        print("文件中的变量:")
        for key, value in data.items():
            if not key.startswith('__'):
                print(f"  - {key}: {type(value)}")
                if hasattr(value, 'shape'):
                    print(f"    形状: {value.shape}")
        
        return data
        
    except ImportError:
        print("scipy未安装，请运行: pip install scipy")
        return None
    except Exception as e:
        print(f"scipy读取失败: {e}")
        return None

def read_matlab_file(file_path):
    """
    尝试多种方法读取MATLAB文件
    
    Parameters:
    file_path: .mat文件路径
    
    Returns:
    dict: 包含所有变量的字典
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"文件不存在: {file_path}")
        return None
    
    print(f"\n=== 读取文件: {file_path.name} ===")
    print(f"文件大小: {file_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # 检查文件类型
    with open(file_path, 'rb') as f:
        header = f.read(32)
        if header.startswith(b'\x89HDF'):
            print("文件类型: HDF5格式 (MATLAB v7.3+)")
            # 尝试h5py
            data = read_matlab_hdf5(file_path)
            if data is not None:
                return data
            # 尝试mat73
            data = read_matlab_mat73(file_path)
            if data is not None:
                return data
        else:
            print("文件类型: 传统MATLAB格式")
            # 尝试scipy
            data = read_matlab_scipy(file_path)
            if data is not None:
                return data
    
    print("所有方法都失败了")
    return None

def analyze_stimuli_sequence(data):
    """
    分析刺激序列数据
    
    Parameters:
    data: 从MATLAB文件读取的数据字典
    """
    print("\n=== 分析刺激序列 ===")
    
    if not data:
        print("没有数据可分析")
        return None
    
    # 查找可能的刺激相关变量
    stimuli_vars = {}
    
    # 常见的刺激变量名
    possible_names = [
        'stimuli', 'sounds', 'tones', 'frequencies', 'freqs',
        'durations', 'durs', 'amplitudes', 'amps', 'sequence', 'seq',
        'stim', 'stimulus', 'sound_sequence', 'tone_sequence',
        'standard', 'deviant', 'deviants', 'oddball'
    ]
    
    for key, value in data.items():
        if not key.startswith('__'):
            # 检查是否是刺激相关变量
            for name in possible_names:
                if name.lower() in key.lower():
                    stimuli_vars[key] = value
                    print(f"找到刺激变量: {key}")
                    print(f"  形状: {value.shape if hasattr(value, 'shape') else 'N/A'}")
                    print(f"  类型: {type(value)}")
                    if hasattr(value, 'dtype'):
                        print(f"  数据类型: {value.dtype}")
                    break
    
    # 显示所有变量的基本信息
    print(f"\n所有变量 ({len(data)} 个):")
    for key, value in data.items():
        if not key.startswith('__'):
            print(f"  {key}: {type(value)}")
            if hasattr(value, 'shape'):
                print(f"    形状: {value.shape}")
            if hasattr(value, 'dtype'):
                print(f"    类型: {value.dtype}")
    
    return stimuli_vars

def analyze_events(data):
    """
    分析事件数据
    
    Parameters:
    data: 从MATLAB文件读取的数据字典
    """
    print("\n=== 分析事件数据 ===")
    
    if not data:
        print("没有数据可分析")
        return None
    
    # 查找可能的事件相关变量
    events_vars = {}
    
    # 常见的事件变量名
    possible_names = [
        'events', 'timestamps', 'times', 'onsets', 'durations',
        'types', 'triggers', 'markers', 'event_times', 'event_types',
        'critical_events', 'stimulus_events', 'response_events'
    ]
    
    for key, value in data.items():
        if not key.startswith('__'):
            # 检查是否是事件相关变量
            for name in possible_names:
                if name.lower() in key.lower():
                    events_vars[key] = value
                    print(f"找到事件变量: {key}")
                    print(f"  形状: {value.shape if hasattr(value, 'shape') else 'N/A'}")
                    print(f"  类型: {type(value)}")
                    if hasattr(value, 'dtype'):
                        print(f"  数据类型: {value.dtype}")
                    break
    
    return events_vars

def save_data(data, output_dir="extracted_audio_stimuli"):
    """
    保存读取的数据
    
    Parameters:
    data: 数据字典
    output_dir: 输出目录
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n=== 保存数据到 {output_dir} ===")
    
    for key, value in data.items():
        if not key.startswith('__'):
            try:
                # 保存为NumPy格式
                output_file = output_dir / f"{key}.npy"
                np.save(output_file, value)
                print(f"✓ 保存 {key} 到 {output_file}")
                
                # 如果是小数组，也保存为CSV
                if hasattr(value, 'shape') and value.size < 10000:
                    if value.ndim == 1:
                        df = pd.DataFrame({key: value})
                    elif value.ndim == 2:
                        df = pd.DataFrame(value)
                    else:
                        continue
                    
                    csv_file = output_dir / f"{key}.csv"
                    df.to_csv(csv_file, index=False)
                    print(f"✓ 保存 {key} 到 {csv_file}")
                    
            except Exception as e:
                print(f"✗ 保存 {key} 失败: {e}")

def main():
    """主函数"""
    print("=== ds005429 声音刺激读取工具 ===")
    
    # 设置文件路径
    base_path = Path(r"D:\D\research\audioset下载\ds005429-main\ds005429-main\sourcedata")
    sub01_mmn_path = base_path / "sub-01" / "ses-MMN"
    
    if not sub01_mmn_path.exists():
        print(f"路径不存在: {sub01_mmn_path}")
        return
    
    # 查找MATLAB文件
    mat_files = list(sub01_mmn_path.glob("*.mat"))
    print(f"找到 {len(mat_files)} 个MATLAB文件:")
    for f in mat_files:
        print(f"  - {f.name}")
    
    # 读取每个文件
    all_data = {}
    
    for mat_file in mat_files:
        data = read_matlab_file(mat_file)
        if data:
            all_data[mat_file.stem] = data
            
            # 分析数据
            if "stimulation_SEQUENCE" in mat_file.name:
                stimuli_vars = analyze_stimuli_sequence(data)
            elif "critical_events" in mat_file.name:
                events_vars = analyze_events(data)
    
    # 保存所有数据
    if all_data:
        save_data(all_data)
        print(f"\n✓ 所有数据已保存")
    else:
        print("\n✗ 没有成功读取任何数据")

if __name__ == "__main__":
    main()
