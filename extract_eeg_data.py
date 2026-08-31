#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EEG数据提取工具
支持BrainVision格式(.eeg, .vhdr, .vmrk)和BIDS格式的EEG数据提取
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False
    print("警告: MNE-Python未安装，将使用基础方法提取数据")

try:
    from scipy.io import loadmat
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("警告: SciPy未安装，无法读取.mat文件")

class EEGExtractor:
    """EEG数据提取器"""
    
    def __init__(self, data_path):
        """
        初始化EEG提取器
        
        Parameters:
        data_path (str): EEG数据文件路径
        """
        self.data_path = Path(data_path)
        self.eeg_data = None
        self.sampling_rate = None
        self.channel_names = None
        self.events = None
        self.metadata = {}
        
    def load_brainvision_data(self):
        """加载BrainVision格式的EEG数据"""
        if not MNE_AVAILABLE:
            raise ImportError("需要安装MNE-Python来读取BrainVision格式数据")
        
        # 查找.vhdr文件
        vhdr_files = list(self.data_path.glob("*.vhdr"))
        if not vhdr_files:
            raise FileNotFoundError(f"在{self.data_path}中未找到.vhdr文件")
        
        vhdr_file = vhdr_files[0]
        print(f"正在加载BrainVision数据: {vhdr_file}")
        
        # 使用MNE读取数据
        raw = mne.io.read_raw_brainvision(vhdr_file, preload=True)
        
        # 提取基本信息
        self.eeg_data = raw.get_data()  # 形状: (n_channels, n_samples)
        self.sampling_rate = raw.info['sfreq']
        self.channel_names = raw.ch_names
        
        # 提取事件信息
        events, event_id = mne.events_from_annotations(raw)
        if len(events) > 0:
            self.events = pd.DataFrame(events, columns=['sample', 'duration', 'event_id'])
            self.events['time'] = self.events['sample'] / self.sampling_rate
        
        print(f"数据加载完成:")
        print(f"  - 通道数: {len(self.channel_names)}")
        print(f"  - 采样率: {self.sampling_rate} Hz")
        print(f"  - 数据长度: {self.eeg_data.shape[1] / self.sampling_rate:.2f} 秒")
        print(f"  - 事件数: {len(events) if len(events) > 0 else 0}")
        
        return raw
    
    def load_bids_data(self):
        """加载BIDS格式的EEG数据"""
        # 查找JSON元数据文件
        json_files = list(self.data_path.glob("*_eeg.json"))
        if not json_files:
            raise FileNotFoundError(f"在{self.data_path}中未找到*_eeg.json文件")
        
        json_file = json_files[0]
        print(f"正在加载BIDS数据: {json_file}")
        
        # 读取元数据
        with open(json_file, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        self.sampling_rate = self.metadata.get('SamplingFrequency', 2500)
        
        # 读取通道信息
        channels_file = list(self.data_path.glob("*_channels.tsv"))
        if channels_file:
            self.channel_names = pd.read_csv(channels_file[0], sep='\t')['name'].tolist()
        
        # 读取事件信息
        events_file = list(self.data_path.glob("*_events.tsv"))
        if events_file:
            self.events = pd.read_csv(events_file[0], sep='\t')
            print(f"事件信息加载完成: {len(self.events)} 个事件")
        
        # 查找EEG数据文件
        eeg_files = list(self.data_path.glob("*.eeg"))
        if eeg_files:
            print("找到.eeg文件，尝试使用MNE读取...")
            if MNE_AVAILABLE:
                try:
                    # 查找对应的.vhdr文件
                    vhdr_files = list(self.data_path.glob("*.vhdr"))
                    if vhdr_files:
                        raw = mne.io.read_raw_brainvision(vhdr_files[0], preload=True)
                        self.eeg_data = raw.get_data()
                        self.channel_names = raw.ch_names
                        print("使用MNE成功读取EEG数据")
                        return raw
                except Exception as e:
                    print(f"MNE读取失败: {e}")
        
        print("警告: 无法直接读取EEG数据文件，请确保安装了MNE-Python")
        return None
    
    def extract_eeg_segments(self, event_type=None, pre_stim=0.1, post_stim=0.5):
        """
        提取特定事件的EEG片段
        
        Parameters:
        event_type: 事件类型（如果为None则提取所有事件）
        pre_stim: 刺激前时间（秒）
        post_stim: 刺激后时间（秒）
        
        Returns:
        segments: 提取的EEG片段数组
        """
        if self.eeg_data is None:
            raise ValueError("请先加载EEG数据")
        
        if self.events is None or len(self.events) == 0:
            raise ValueError("没有事件信息")
        
        # 计算样本数
        pre_samples = int(pre_stim * self.sampling_rate)
        post_samples = int(post_stim * self.sampling_rate)
        
        segments = []
        segment_info = []
        
        for idx, event in self.events.iterrows():
            if event_type is not None and 'event_id' in event and event['event_id'] != event_type:
                continue
            
            # 计算开始和结束样本
            start_sample = int(event['sample']) - pre_samples
            end_sample = int(event['sample']) + post_samples
            
            # 检查边界
            if start_sample < 0 or end_sample >= self.eeg_data.shape[1]:
                print(f"警告: 事件 {idx} 超出数据范围，跳过")
                continue
            
            # 提取片段
            segment = self.eeg_data[:, start_sample:end_sample]
            segments.append(segment)
            segment_info.append({
                'event_index': idx,
                'start_sample': start_sample,
                'end_sample': end_sample,
                'event_time': event.get('time', event['sample'] / self.sampling_rate)
            })
        
        print(f"提取了 {len(segments)} 个EEG片段")
        return np.array(segments), segment_info
    
    def save_data(self, output_dir, format='numpy'):
        """
        保存提取的EEG数据
        
        Parameters:
        output_dir: 输出目录
        format: 保存格式 ('numpy', 'csv', 'mat')
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.eeg_data is not None:
            if format == 'numpy':
                np.save(output_dir / 'eeg_data.npy', self.eeg_data)
                print(f"EEG数据已保存到: {output_dir / 'eeg_data.npy'}")
            
            elif format == 'csv':
                # 保存为CSV（注意：大文件可能很慢）
                df = pd.DataFrame(self.eeg_data.T, columns=self.channel_names)
                df.to_csv(output_dir / 'eeg_data.csv', index=False)
                print(f"EEG数据已保存到: {output_dir / 'eeg_data.csv'}")
            
            elif format == 'mat' and SCIPY_AVAILABLE:
                from scipy.io import savemat
                savemat(output_dir / 'eeg_data.mat', {
                    'eeg_data': self.eeg_data,
                    'channel_names': self.channel_names,
                    'sampling_rate': self.sampling_rate
                })
                print(f"EEG数据已保存到: {output_dir / 'eeg_data.mat'}")
        
        # 保存元数据
        if self.metadata:
            with open(output_dir / 'metadata.json', 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        
        # 保存事件信息
        if self.events is not None:
            self.events.to_csv(output_dir / 'events.csv', index=False)
            print(f"事件信息已保存到: {output_dir / 'events.csv'}")
        
        # 保存通道信息
        if self.channel_names:
            channel_df = pd.DataFrame({
                'channel_index': range(len(self.channel_names)),
                'channel_name': self.channel_names
            })
            channel_df.to_csv(output_dir / 'channels.csv', index=False)
            print(f"通道信息已保存到: {output_dir / 'channels.csv'}")
    
    def get_data_summary(self):
        """获取数据摘要信息"""
        summary = {
            'data_shape': self.eeg_data.shape if self.eeg_data is not None else None,
            'sampling_rate': self.sampling_rate,
            'duration_seconds': self.eeg_data.shape[1] / self.sampling_rate if self.eeg_data is not None else None,
            'n_channels': len(self.channel_names) if self.channel_names else 0,
            'channel_names': self.channel_names,
            'n_events': len(self.events) if self.events is not None else 0,
            'metadata': self.metadata
        }
        return summary

def main():
    """主函数 - 示例用法"""
    # 设置数据路径（请根据实际情况修改）
    data_path = r"D:\D\research\audioset下载\ds005429-main\ds005429-main\sub-01\ses-MMN\eeg"
    output_dir = "extracted_eeg_data"
    
    print("=== EEG数据提取工具 ===")
    print(f"数据路径: {data_path}")
    print(f"输出目录: {output_dir}")
    
    try:
        # 创建提取器
        extractor = EEGExtractor(data_path)
        
        # 尝试加载BIDS格式数据
        try:
            raw = extractor.load_bids_data()
            print("✓ 成功加载BIDS格式数据")
        except Exception as e:
            print(f"BIDS格式加载失败: {e}")
            # 尝试BrainVision格式
            try:
                raw = extractor.load_brainvision_data()
                print("✓ 成功加载BrainVision格式数据")
            except Exception as e2:
                print(f"BrainVision格式也加载失败: {e2}")
                return
        
        # 显示数据摘要
        summary = extractor.get_data_summary()
        print("\n=== 数据摘要 ===")
        print(f"数据形状: {summary['data_shape']}")
        print(f"采样率: {summary['sampling_rate']} Hz")
        print(f"时长: {summary['duration_seconds']:.2f} 秒")
        print(f"通道数: {summary['n_channels']}")
        print(f"事件数: {summary['n_events']}")
        
        # 保存数据
        print("\n=== 保存数据 ===")
        extractor.save_data(output_dir, format='numpy')
        
        # 如果有事件信息，提取一些片段作为示例
        if extractor.events is not None and len(extractor.events) > 0:
            print("\n=== 提取事件片段 ===")
            try:
                segments, segment_info = extractor.extract_eeg_segments(
                    pre_stim=0.1, post_stim=0.5
                )
                print(f"成功提取 {len(segments)} 个事件片段")
                
                # 保存片段
                np.save(os.path.join(output_dir, 'eeg_segments.npy'), segments)
                pd.DataFrame(segment_info).to_csv(
                    os.path.join(output_dir, 'segment_info.csv'), index=False
                )
                print("事件片段已保存")
            except Exception as e:
                print(f"提取事件片段失败: {e}")
        
        print("\n✓ 数据提取完成！")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
