#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EEG数据读取脚本
支持BrainVision格式(.eeg, .vhdr, .vmrk)和BIDS格式的EEG数据
"""

import os
import numpy as np
import pandas as pd
import json
from typing import Dict, List, Tuple, Optional, Union
import struct
import warnings

class EEGReader:
    """EEG数据读取器类"""
    
    def __init__(self, data_dir: str):
        """
        初始化EEG读取器
        
        Args:
            data_dir: EEG数据目录路径
        """
        self.data_dir = data_dir
        self.eeg_data = None
        self.channel_info = None
        self.events = None
        self.metadata = None
        
    def load_brainvision_data(self, subject_id: str = "sub-01", 
                            session: str = "ses-MMN", 
                            task: str = "Optimum1") -> Dict:
        """
        加载BrainVision格式的EEG数据
        
        Args:
            subject_id: 被试ID
            session: 会话名称
            task: 任务名称
            
        Returns:
            包含EEG数据的字典
        """
        # 构建文件路径
        base_name = f"{subject_id}_{session}_task-{task}_eeg"
        eeg_file = os.path.join(self.data_dir, f"{base_name}.eeg")
        vhdr_file = os.path.join(self.data_dir, f"{base_name}.vhdr")
        vmrk_file = os.path.join(self.data_dir, f"{base_name}.vmrk")
        json_file = os.path.join(self.data_dir, f"{base_name}.json")
        channels_file = os.path.join(self.data_dir, f"{base_name}_channels.tsv")
        events_file = os.path.join(self.data_dir, f"{base_name}_events.tsv")
        
        # 检查文件是否存在
        if not os.path.exists(eeg_file):
            raise FileNotFoundError(f"EEG数据文件不存在: {eeg_file}")
        if not os.path.exists(vhdr_file):
            raise FileNotFoundError(f"头文件不存在: {vhdr_file}")
            
        print(f"正在加载EEG数据: {subject_id} - {session} - {task}")
        
        # 读取头文件信息
        header_info = self._read_vhdr_file(vhdr_file)
        
        # 读取EEG数据
        eeg_data = self._read_eeg_file(eeg_file, header_info)
        
        # 读取通道信息
        if os.path.exists(channels_file):
            channel_info = self._read_channels_file(channels_file)
        else:
            channel_info = self._create_channel_info_from_header(header_info)
            
        # 读取事件信息
        if os.path.exists(events_file):
            events = self._read_events_file(events_file)
        else:
            events = self._read_vmrk_file(vmrk_file) if os.path.exists(vmrk_file) else None
            
        # 读取元数据
        if os.path.exists(json_file):
            with open(json_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        else:
            metadata = header_info
            
        # 存储数据
        self.eeg_data = eeg_data
        self.channel_info = channel_info
        self.events = events
        self.metadata = metadata
        
        return {
            'eeg_data': eeg_data,
            'channel_info': channel_info,
            'events': events,
            'metadata': metadata,
            'sampling_rate': header_info['sampling_rate'],
            'n_channels': header_info['n_channels'],
            'n_samples': eeg_data.shape[1]
        }
    
    def _read_vhdr_file(self, vhdr_file: str) -> Dict:
        """读取.vhdr头文件"""
        header_info = {}
        
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
    
    def _read_eeg_file(self, eeg_file: str, header_info: Dict) -> np.ndarray:
        """读取.eeg数据文件"""
        n_channels = header_info['n_channels']
        binary_format = header_info.get('binary_format', 'INT_16')
        
        # 根据二进制格式确定数据类型
        if binary_format == 'INT_16':
            dtype = np.int16
            bytes_per_sample = 2
        elif binary_format == 'FLOAT_32':
            dtype = np.float32
            bytes_per_sample = 4
        else:
            raise ValueError(f"不支持的二进制格式: {binary_format}")
            
        # 读取二进制数据
        with open(eeg_file, 'rb') as f:
            raw_data = f.read()
            
        # 计算样本数
        n_samples = len(raw_data) // (n_channels * bytes_per_sample)
        
        # 重塑数据
        if header_info.get('data_orientation', 'MULTIPLEXED') == 'MULTIPLEXED':
            # 多路复用格式: ch1_sample1, ch2_sample1, ..., ch1_sample2, ch2_sample2, ...
            data = np.frombuffer(raw_data, dtype=dtype)
            data = data.reshape(n_samples, n_channels).T
        else:
            # 向量化格式: ch1_all_samples, ch2_all_samples, ...
            data = np.frombuffer(raw_data, dtype=dtype)
            data = data.reshape(n_channels, n_samples)
            
        return data.astype(np.float64)
    
    def _read_channels_file(self, channels_file: str) -> pd.DataFrame:
        """读取channels.tsv文件"""
        return pd.read_csv(channels_file, sep='\t')
    
    def _create_channel_info_from_header(self, header_info: Dict) -> pd.DataFrame:
        """从头文件信息创建通道信息DataFrame"""
        channels = []
        for ch_num, ch_info in header_info.get('channels', {}).items():
            parts = ch_info.split(',')
            if len(parts) >= 4:
                name = parts[0]
                ref = parts[1] if parts[1] else ''
                resolution = float(parts[2]) if parts[2] else 1.0
                unit = parts[3] if parts[3] else 'µV'
                sampling_freq = header_info.get('sampling_rate', 2500)
                
                channels.append({
                    'name': name,
                    'type': 'EEG',
                    'units': unit,
                    'sampling_frequency': sampling_freq,
                    'reference': ref,
                    'resolution': resolution
                })
                
        return pd.DataFrame(channels)
    
    def _read_events_file(self, events_file: str) -> pd.DataFrame:
        """读取events.tsv文件"""
        return pd.read_csv(events_file, sep='\t')
    
    def _read_vmrk_file(self, vmrk_file: str) -> pd.DataFrame:
        """读取.vmrk标记文件"""
        events = []
        
        with open(vmrk_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            if line.startswith('Mk') and '=' in line:
                try:
                    parts = line.split('=', 1)[1].split(',')
                    if len(parts) >= 3:
                        event_type = parts[0].strip()
                        onset_str = parts[1].strip()
                        duration_str = parts[2].strip() if len(parts) > 2 else '0'
                        
                        # 检查字符串是否为空或无效
                        if onset_str and onset_str.isdigit():
                            onset_sample = int(onset_str)
                        else:
                            continue  # 跳过无效的行
                            
                        if duration_str and duration_str.isdigit():
                            duration_samples = int(duration_str)
                        else:
                            duration_samples = 0
                            
                        events.append({
                            'onset': onset_sample,
                            'duration': duration_samples,
                            'type': event_type
                        })
                except (ValueError, IndexError) as e:
                    # 跳过格式不正确的行
                    print(f"警告: 跳过格式不正确的标记行: {line[:50]}...")
                    continue
                    
        return pd.DataFrame(events)
    
    def get_channel_data(self, channel_name: str) -> np.ndarray:
        """获取指定通道的数据"""
        if self.eeg_data is None:
            raise ValueError("请先加载EEG数据")
            
        if self.channel_info is None:
            raise ValueError("通道信息未加载")
            
        # 查找通道索引
        channel_idx = None
        for idx, row in self.channel_info.iterrows():
            if row['name'] == channel_name:
                channel_idx = idx
                break
                
        if channel_idx is None:
            raise ValueError(f"未找到通道: {channel_name}")
            
        return self.eeg_data[channel_idx, :]
    
    def get_time_vector(self) -> np.ndarray:
        """获取时间向量"""
        if self.metadata is None:
            raise ValueError("元数据未加载")
            
        sampling_rate = self.metadata.get('SamplingFrequency', 2500)
        n_samples = self.eeg_data.shape[1]
        
        return np.arange(n_samples) / sampling_rate
    
    def get_events_in_time_range(self, start_time: float, end_time: float) -> pd.DataFrame:
        """获取指定时间范围内的事件"""
        if self.events is None:
            return pd.DataFrame()
            
        if 'onset' in self.events.columns:
            # 如果onset是时间（秒）
            if self.events['onset'].max() > 1000:  # 假设大于1000的是样本数
                sampling_rate = self.metadata.get('SamplingFrequency', 2500)
                onset_times = self.events['onset'] / sampling_rate
            else:
                onset_times = self.events['onset']
                
            mask = (onset_times >= start_time) & (onset_times <= end_time)
            return self.events[mask].copy()
        else:
            return pd.DataFrame()
    
    def plot_channel(self, channel_name: str, start_time: float = 0, 
                    duration: float = 10, ax=None):
        """绘制指定通道的数据"""
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 4))
            
        # 获取通道数据
        channel_data = self.get_channel_data(channel_name)
        time_vector = self.get_time_vector()
        
        # 计算时间范围
        sampling_rate = self.metadata.get('SamplingFrequency', 2500)
        start_sample = int(start_time * sampling_rate)
        end_sample = int((start_time + duration) * sampling_rate)
        
        # 确保不超出数据范围
        start_sample = max(0, start_sample)
        end_sample = min(len(channel_data), end_sample)
        
        # 绘制数据
        time_segment = time_vector[start_sample:end_sample]
        data_segment = channel_data[start_sample:end_sample]
        
        ax.plot(time_segment, data_segment)
        ax.set_xlabel('时间 (秒)')
        ax.set_ylabel('幅度 (µV)')
        ax.set_title(f'通道 {channel_name}')
        ax.grid(True)
        
        return ax
    
    def get_data_summary(self) -> Dict:
        """获取数据摘要信息"""
        if self.eeg_data is None:
            return {}
            
        summary = {
            '数据形状': self.eeg_data.shape,
            '采样率': self.metadata.get('SamplingFrequency', '未知'),
            '通道数': self.eeg_data.shape[0],
            '样本数': self.eeg_data.shape[1],
            '记录时长': f"{self.eeg_data.shape[1] / self.metadata.get('SamplingFrequency', 2500):.2f} 秒",
            '通道名称': list(self.channel_info['name']) if self.channel_info is not None else [],
            '事件数': len(self.events) if self.events is not None else 0
        }
        
        return summary


def main():
    """主函数 - 示例用法"""
    # EEG数据目录
    eeg_dir = r"D:\D\research\audioset下载\download_audioset-master\download_audioset-master\auditory oddball\sub-01\ses-MMN\eeg"
    
    # 检查目录是否存在
    if not os.path.exists(eeg_dir):
        print(f"错误: EEG数据目录不存在: {eeg_dir}")
        return
    
    # 创建EEG读取器
    reader = EEGReader(eeg_dir)
    
    try:
        # 加载数据
        print("正在加载EEG数据...")
        data = reader.load_brainvision_data()
        
        # 显示数据摘要
        print("\n=== 数据摘要 ===")
        summary = reader.get_data_summary()
        for key, value in summary.items():
            print(f"{key}: {value}")
        
        # 显示前几个通道的数据统计
        print("\n=== 通道数据统计 ===")
        for i in range(min(5, data['eeg_data'].shape[0])):
            channel_name = data['channel_info'].iloc[i]['name']
            channel_data = data['eeg_data'][i, :]
            print(f"通道 {channel_name}: 均值={np.mean(channel_data):.2f}µV, "
                  f"标准差={np.std(channel_data):.2f}µV, "
                  f"范围=[{np.min(channel_data):.2f}, {np.max(channel_data):.2f}]µV")
        
        # 显示事件信息
        if data['events'] is not None and len(data['events']) > 0:
            print(f"\n=== 事件信息 (前10个) ===")
            print(data['events'].head(10))
        else:
            print("\n=== 事件信息 ===")
            print("没有找到事件数据")
        
        # 绘制一个通道的数据示例
        print("\n正在绘制通道数据示例...")
        try:
            import matplotlib.pyplot as plt
            reader.plot_channel('Cz', start_time=0, duration=5)
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("matplotlib未安装，跳过绘图")
        except Exception as e:
            print(f"绘图时出错: {e}")
            
    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
    except Exception as e:
        print(f"加载数据时出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
