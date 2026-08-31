#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
获取ds005429数据集中的声音刺激数据
最简单的版本，直接读取并保存
"""

import numpy as np
import json
from pathlib import Path

def get_audio_stimuli():
    """获取声音刺激数据"""
    
    # 文件路径
    stimuli_file = Path(r"D:\D\research\audioset下载\ds005429-main\ds005429-main\sourcedata\sub-01\ses-MMN\aMMN_stimulation_SEQUENCE.mat")
    events_file = Path(r"D:\D\research\audioset下载\ds005429-main\ds005429-main\sourcedata\sub-01\ses-MMN\aMMN_critical_events.mat")
    
    print("=== 获取声音刺激数据 ===")
    print(f"刺激文件: {stimuli_file.name}")
    print(f"事件文件: {events_file.name}")
    
    result = {
        'stimuli_data': {},
        'events_data': {},
        'success': False
    }
    
    # 检查文件是否存在
    if not stimuli_file.exists():
        print(f"❌ 刺激文件不存在: {stimuli_file}")
        return result
    
    if not events_file.exists():
        print(f"❌ 事件文件不存在: {events_file}")
        return result
    
    # 检查文件类型
    print(f"\n检查文件类型...")
    with open(stimuli_file, 'rb') as f:
        header = f.read(32)
        print(f"刺激文件头: {header}")
        
        if header.startswith(b'../../.git/annex/') or header.startswith(b'../../../.git/annex/'):
            print("❌ 这是git annex管理的文件，但实际文件不在本地")
            print("数据集可能下载不完整，需要重新获取")
            print("\n解决方案:")
            print("1. 重新下载完整的数据集")
            print("2. 或者访问原始数据集页面获取实际文件")
            print("3. 数据集来源: https://openneuro.org/datasets/ds005429")
            return result
        elif header.startswith(b'\x89HDF'):
            print("✓ 检测到HDF5格式")
            file_type = 'hdf5'
        elif header.startswith(b'MATLAB'):
            print("✓ 检测到传统MATLAB格式")
            file_type = 'matlab'
        else:
            print(f"❓ 未知格式: {header}")
            file_type = 'unknown'
    
    # 尝试多种方法读取
    methods = [
        ('mat73', _read_with_mat73),
        ('h5py', _read_with_h5py),
        ('scipy', _read_with_scipy)
    ]
    
    for method_name, method_func in methods:
        print(f"\n尝试方法: {method_name}")
        try:
            # 读取刺激文件
            stimuli_data = method_func(stimuli_file)
            if stimuli_data:
                result['stimuli_data'] = stimuli_data
                print(f"✓ 刺激序列读取成功 ({method_name})")
            
            # 读取事件文件
            events_data = method_func(events_file)
            if events_data:
                result['events_data'] = events_data
                print(f"✓ 事件文件读取成功 ({method_name})")
            
            if result['stimuli_data'] or result['events_data']:
                result['success'] = True
                break
                
        except Exception as e:
            print(f"✗ {method_name} 失败: {e}")
    
    return result

def _read_with_mat73(file_path):
    """使用mat73读取"""
    try:
        import mat73
        data = mat73.loadmat(file_path)
        result = {}
        for key, value in data.items():
            if not key.startswith('__'):
                result[key] = value
                print(f"  {key}: {value.shape if hasattr(value, 'shape') else 'N/A'}")
        return result
    except ImportError:
        print("  mat73未安装")
        return None
    except Exception as e:
        print(f"  mat73错误: {e}")
        return None

def _read_with_h5py(file_path):
    """使用h5py读取"""
    try:
        import h5py
        data = {}
        with h5py.File(file_path, 'r') as f:
            for key in f.keys():
                if not key.startswith('#'):
                    value = f[key][:]
                    data[key] = value
                    print(f"  {key}: {value.shape} {value.dtype}")
        return data
    except ImportError:
        print("  h5py未安装")
        return None
    except Exception as e:
        print(f"  h5py错误: {e}")
        return None

def _read_with_scipy(file_path):
    """使用scipy读取"""
    try:
        from scipy.io import loadmat
        data = loadmat(file_path)
        result = {}
        for key, value in data.items():
            if not key.startswith('__'):
                result[key] = value
                print(f"  {key}: {value.shape} {value.dtype}")
        return result
    except ImportError:
        print("  scipy未安装")
        return None
    except Exception as e:
        print(f"  scipy错误: {e}")
        return None

def save_data(data, output_dir="audio_stimuli"):
    """保存数据"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n=== 保存数据到 {output_dir} ===")
    
    # 保存刺激数据
    if data['stimuli_data']:
        for key, value in data['stimuli_data'].items():
            output_file = output_dir / f"stimuli_{key}.npy"
            np.save(output_file, value)
            print(f"✓ 保存: {output_file}")
    
    # 保存事件数据
    if data['events_data']:
        for key, value in data['events_data'].items():
            output_file = output_dir / f"events_{key}.npy"
            np.save(output_file, value)
            print(f"✓ 保存: {output_file}")
    
    # 保存摘要
    summary = {
        'stimuli_variables': list(data['stimuli_data'].keys()),
        'events_variables': list(data['events_data'].keys()),
        'stimuli_count': len(data['stimuli_data']),
        'events_count': len(data['events_data'])
    }
    
    with open(output_dir / "summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 保存摘要: {output_dir}/summary.json")

def main():
    """主函数"""
    print("=== ds005429 声音刺激读取工具 ===")
    print("如果读取失败，请先安装依赖:")
    print("pip install h5py scipy mat73")
    print()
    
    # 获取数据
    data = get_audio_stimuli()
    
    if data['success']:
        # 保存数据
        save_data(data)
        print("\n✅ 声音刺激数据获取完成！")
        print(f"刺激变量: {list(data['stimuli_data'].keys())}")
        print(f"事件变量: {list(data['events_data'].keys())}")
    else:
        print("\n❌ 获取声音刺激数据失败")
        print("\n🔧 解决方案:")
        print("1. 安装依赖: pip install h5py scipy mat73")
        print("2. 如果是git annex文件，运行:")
        print("   cd \"D:\\D\\research\\audioset下载\\ds005429-main\\ds005429-main\"")
        print("   git annex get sourcedata/sub-01/ses-MMN/*.mat")
        print("3. 检查文件是否损坏")

if __name__ == "__main__":
    main()
