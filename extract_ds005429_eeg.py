#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
专门用于提取ds005429数据集中EEG数据的脚本
支持sub-01/ses-MMN/eeg目录中的BrainVision格式数据
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def extract_ds005429_eeg(subject_id="sub-01", session="ses-MMN", 
                        base_path=r"D:\D\research\audioset下载\ds005429-main\ds005429-main"):
    """
    提取ds005429数据集中的EEG数据
    
    Parameters:
    subject_id: 被试ID (默认: sub-01)
    session: 会话名称 (默认: ses-MMN)
    base_path: 数据集根目录路径
    """
    
    # 构建数据路径
    data_path = Path(base_path) / subject_id / session / "eeg"
    output_dir = f"extracted_{subject_id}_{session}_eeg"
    
    print(f"=== 提取 {subject_id}/{session} 的EEG数据 ===")
    print(f"数据路径: {data_path}")
    print(f"输出目录: {output_dir}")
    
    if not data_path.exists():
        print(f"错误: 数据路径不存在: {data_path}")
        return None
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 检查MNE是否可用
        try:
            import mne
            print("✓ 检测到MNE-Python")
        except ImportError:
            print("错误: 需要安装MNE-Python来读取EEG数据")
            print("请运行: pip install mne")
            return None
        
        # 查找.vhdr文件
        vhdr_files = list(data_path.glob("*.vhdr"))
        if not vhdr_files:
            print(f"错误: 在 {data_path} 中未找到.vhdr文件")
            return None
        
        vhdr_file = vhdr_files[0]
        print(f"正在加载: {vhdr_file.name}")
        
        # 使用MNE读取BrainVision数据
        raw = mne.io.read_raw_brainvision(vhdr_file, preload=True)
        
        # 提取基本信息
        eeg_data = raw.get_data()  # 形状: (n_channels, n_samples)
        sampling_rate = raw.info['sfreq']
        channel_names = raw.ch_names
        
        print(f"✓ 数据加载成功:")
        print(f"  - 通道数: {len(channel_names)}")
        print(f"  - 采样率: {sampling_rate} Hz")
        print(f"  - 数据长度: {eeg_data.shape[1] / sampling_rate:.2f} 秒")
        print(f"  - 数据形状: {eeg_data.shape}")
        
        # 读取JSON元数据
        json_files = list(data_path.glob("*_eeg.json"))
        metadata = {}
        if json_files:
            with open(json_files[0], 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            print(f"✓ 元数据加载完成")
        
        # 读取事件信息
        events_file = list(data_path.glob("*_events.tsv"))
        events_df = None
        if events_file:
            events_df = pd.read_csv(events_file[0], sep='\t')
            print(f"✓ 事件信息加载完成: {len(events_df)} 个事件")
        
        # 读取通道信息
        channels_file = list(data_path.glob("*_channels.tsv"))
        channels_df = None
        if channels_file:
            channels_df = pd.read_csv(channels_file[0], sep='\t')
            print(f"✓ 通道信息加载完成: {len(channels_df)} 个通道")
        
        # 保存EEG数据
        print("\n=== 保存数据 ===")
        
        # 保存原始EEG数据
        np.save(os.path.join(output_dir, 'eeg_data.npy'), eeg_data)
        print(f"✓ EEG数据已保存: {output_dir}/eeg_data.npy")
        
        # 保存元数据
        if metadata:
            with open(os.path.join(output_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            print(f"✓ 元数据已保存: {output_dir}/metadata.json")
        
        # 保存事件信息
        if events_df is not None:
            events_df.to_csv(os.path.join(output_dir, 'events.csv'), index=False)
            print(f"✓ 事件信息已保存: {output_dir}/events.csv")
        
        # 保存通道信息
        if channels_df is not None:
            channels_df.to_csv(os.path.join(output_dir, 'channels.csv'), index=False)
            print(f"✓ 通道信息已保存: {output_dir}/channels.csv")
        
        # 保存通道名称列表
        channel_names_df = pd.DataFrame({
            'channel_index': range(len(channel_names)),
            'channel_name': channel_names
        })
        channel_names_df.to_csv(os.path.join(output_dir, 'channel_names.csv'), index=False)
        print(f"✓ 通道名称已保存: {output_dir}/channel_names.csv")
        
        # 创建数据摘要
        summary = {
            'subject_id': subject_id,
            'session': session,
            'data_shape': eeg_data.shape,
            'sampling_rate': sampling_rate,
            'duration_seconds': eeg_data.shape[1] / sampling_rate,
            'n_channels': len(channel_names),
            'n_events': len(events_df) if events_df is not None else 0,
            'channel_names': channel_names,
            'metadata': metadata
        }
        
        with open(os.path.join(output_dir, 'data_summary.json'), 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"✓ 数据摘要已保存: {output_dir}/data_summary.json")
        
        # 如果有事件，提取一些示例片段
        if events_df is not None and len(events_df) > 0:
            print("\n=== 提取事件片段示例 ===")
            try:
                # 提取前10个事件的片段（刺激前0.1秒，刺激后0.5秒）
                pre_samples = int(0.1 * sampling_rate)
                post_samples = int(0.5 * sampling_rate)
                
                segments = []
                segment_info = []
                
                for i in range(min(10, len(events_df))):  # 只提取前10个事件作为示例
                    event = events_df.iloc[i]
                    start_sample = int(event['begsample']) - pre_samples
                    end_sample = int(event['endsample']) + post_samples
                    
                    # 检查边界
                    if start_sample >= 0 and end_sample < eeg_data.shape[1]:
                        segment = eeg_data[:, start_sample:end_sample]
                        segments.append(segment)
                        segment_info.append({
                            'event_index': i,
                            'onset_time': event['onset'],
                            'duration': event['duration'],
                            'start_sample': start_sample,
                            'end_sample': end_sample,
                            'segment_length_samples': end_sample - start_sample,
                            'segment_length_seconds': (end_sample - start_sample) / sampling_rate
                        })
                
                if segments:
                    np.save(os.path.join(output_dir, 'eeg_segments_sample.npy'), np.array(segments))
                    pd.DataFrame(segment_info).to_csv(
                        os.path.join(output_dir, 'segment_info_sample.csv'), index=False
                    )
                    print(f"✓ 提取了 {len(segments)} 个事件片段示例")
                    print(f"  - 片段长度: {segment_info[0]['segment_length_seconds']:.2f} 秒")
                    print(f"  - 片段形状: {segments[0].shape}")
            except Exception as e:
                print(f"提取事件片段时出错: {e}")
        
        print(f"\n✓ 数据提取完成！所有文件保存在: {output_dir}")
        return summary
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_extracted_data(output_dir):
    """
    加载已提取的EEG数据
    
    Parameters:
    output_dir: 输出目录路径
    
    Returns:
    dict: 包含所有数据的字典
    """
    output_dir = Path(output_dir)
    
    if not output_dir.exists():
        print(f"错误: 输出目录不存在: {output_dir}")
        return None
    
    data = {}
    
    try:
        # 加载EEG数据
        eeg_file = output_dir / 'eeg_data.npy'
        if eeg_file.exists():
            data['eeg_data'] = np.load(eeg_file)
            print(f"✓ 加载EEG数据: {data['eeg_data'].shape}")
        
        # 加载元数据
        metadata_file = output_dir / 'metadata.json'
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                data['metadata'] = json.load(f)
            print(f"✓ 加载元数据")
        
        # 加载事件信息
        events_file = output_dir / 'events.csv'
        if events_file.exists():
            data['events'] = pd.read_csv(events_file)
            print(f"✓ 加载事件信息: {len(data['events'])} 个事件")
        
        # 加载通道信息
        channels_file = output_dir / 'channels.csv'
        if channels_file.exists():
            data['channels'] = pd.read_csv(channels_file)
            print(f"✓ 加载通道信息: {len(data['channels'])} 个通道")
        
        # 加载数据摘要
        summary_file = output_dir / 'data_summary.json'
        if summary_file.exists():
            with open(summary_file, 'r', encoding='utf-8') as f:
                data['summary'] = json.load(f)
            print(f"✓ 加载数据摘要")
        
        return data
        
    except Exception as e:
        print(f"加载数据时出错: {e}")
        return None

def main():
    """主函数"""
    print("=== ds005429 EEG数据提取工具 ===")
    
    # 提取sub-01/ses-MMN的EEG数据
    summary = extract_ds005429_eeg(
        subject_id="sub-01",
        session="ses-MMN",
        base_path=r"D:\D\research\audioset下载\ds005429-main\ds005429-main"
    )
    
    if summary:
        print("\n=== 数据摘要 ===")
        print(f"被试: {summary['subject_id']}")
        print(f"会话: {summary['session']}")
        print(f"数据形状: {summary['data_shape']}")
        print(f"采样率: {summary['sampling_rate']} Hz")
        print(f"时长: {summary['duration_seconds']:.2f} 秒")
        print(f"通道数: {summary['n_channels']}")
        print(f"事件数: {summary['n_events']}")
        
        # 显示前几个通道名称
        if summary['channel_names']:
            print(f"前10个通道: {summary['channel_names'][:10]}")

if __name__ == "__main__":
    main()
