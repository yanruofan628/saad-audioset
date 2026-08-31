#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据MATLAB代码生成Optimum-1 MMN范式的音频刺激文件
重现GenerateTone.m和AttenuateSound.m的功能
"""

import numpy as np
import soundfile as sf
from pathlib import Path
import json

def generate_tone(sf, dur, freqs, amps=None, phases=None, itd=0, ild=0):
    """
    重现MATLAB的GenerateTone函数
    
    Parameters:
    sf: 采样频率 (Hz)
    dur: 持续时间 (ms)
    freqs: 频率列表 (Hz)
    amps: 振幅列表 (默认全为1)
    phases: 相位列表 (默认全为0)
    itd: 双耳时间差 (ms)
    ild: 双耳强度差 (dB)
    """
    
    if amps is None:
        amps = np.ones(len(freqs))
    if phases is None:
        phases = np.zeros(len(freqs))
    
    # 计算样本数
    number_of_samples = int(sf * dur / 1000)
    t = np.arange(number_of_samples) / sf
    
    if itd == 0 and ild == 0:
        # 单声道
        tone = np.zeros(len(t))
        for i in range(len(freqs)):
            f_component = amps[i] * np.sin(2 * np.pi * freqs[i] * t + phases[i])
            tone += f_component
        
        # 振幅归一化
        tone = tone / np.max(np.abs(tone))
        tone = tone * 0.999
        
        return tone
    else:
        # 立体声
        toneL = np.zeros(len(t))
        toneR = np.zeros(len(t))
        
        itd_sec = itd / 1000  # 转换为秒
        ipd = 2 * np.pi * freqs[0] * itd_sec
        phases_left = phases + ipd
        phases_right = phases
        
        for i in range(len(freqs)):
            f_component_L = amps[i] * np.sin(2 * np.pi * freqs[i] * t + phases_left[i])
            f_component_R = amps[i] * np.sin(2 * np.pi * freqs[i] * t + phases_right[i])
            toneL += f_component_L
            toneR += f_component_R
        
        # 应用ILD
        toneL = toneL * (10 ** (ild / 20))
        
        # 振幅归一化
        max_amp = max(np.max(np.abs(toneL)), np.max(np.abs(toneR)))
        toneL = toneL / max_amp
        toneR = toneR / max_amp
        
        tone = np.column_stack((toneL, toneR))
        tone = tone * 0.999
        
        return tone

def attenuate_sound(soundin, att_dB):
    """
    重现MATLAB的AttenuateSound函数
    
    Parameters:
    soundin: 输入音频
    att_dB: 衰减/放大 (dB)
    """
    
    if len(soundin.shape) == 2 and soundin.shape[1] == 2:
        # 立体声
        rms_sound_dB = max(
            np.linalg.norm(soundin[:, 0]) / np.sqrt(len(soundin[:, 0])),
            np.linalg.norm(soundin[:, 1]) / np.sqrt(len(soundin[:, 1]))
        )
    else:
        # 单声道
        rms_sound_dB = np.linalg.norm(soundin) / np.sqrt(len(soundin))
    
    ratio = (10 ** (att_dB / 20)) / rms_sound_dB
    soundout = ratio * soundin
    
    # 检查削波
    clipped_samples = np.sum(np.abs(soundout) > 1)
    if clipped_samples > 0:
        print(f"警告! 音频有 {clipped_samples} 个削波样本!")
    
    return soundout

def create_cosine_ramp(sf, duration_ms, ramp_ms=5):
    """
    创建余弦斜坡
    """
    dr = ramp_ms / 1000  # 转换为秒
    nr = int(sf * dr)
    main_duration_samples = int(sf * duration_ms / 1000)
    
    # 创建斜坡
    CSramp = np.sin(np.linspace(0, np.pi/2, nr))
    # 中间部分
    middle = np.ones(main_duration_samples - nr * 2)
    # 完整斜坡
    CSramp = np.concatenate([CSramp, middle, np.flip(CSramp)])
    
    return CSramp

def generate_optimum1_stimuli():
    """
    生成Optimum-1 MMN范式的所有刺激
    """
    
    print("=== 生成Optimum-1 MMN音频刺激 ===")
    
    # 参数设置（来自MATLAB代码）
    sf = 48000  # 采样频率
    sDuration = 50  # 标准持续时间 (ms)
    sFreqs = [523, 1046, 1569]  # 标准和弦频率
    sAmps = [1, 1/2, 1/4]  # 振幅调制
    
    # 创建余弦斜坡
    CSramp = create_cosine_ramp(sf, sDuration, 5)
    
    print("生成标准音调...")
    # 标准音调
    sTone = generate_tone(sf, sDuration, sFreqs, sAmps)
    sTone = sTone * CSramp
    sTone = np.column_stack((sTone, sTone))  # 双声道
    
    print("生成频率偏差...")
    # 频率偏差 - 高频
    FHdFreqs = [609, 609*2, 609*3]
    FHdTone = generate_tone(sf, sDuration, FHdFreqs, sAmps)
    FHdTone = FHdTone * CSramp
    FHdTone = np.column_stack((FHdTone, FHdTone))
    
    # 频率偏差 - 低频
    FLdFreqs = [450, 450*2, 450*3]
    FLdTone = generate_tone(sf, sDuration, FLdFreqs, sAmps)
    FLdTone = FLdTone * CSramp
    FLdTone = np.column_stack((FLdTone, FLdTone))
    
    print("生成强度偏差...")
    # 强度偏差
    IdTone = generate_tone(sf, sDuration, sFreqs, sAmps)
    IdTone = IdTone * CSramp
    IdTone = attenuate_sound(IdTone, -15)  # 衰减15dB
    IdTone = np.column_stack((IdTone, IdTone))
    
    print("生成持续时间偏差...")
    # 持续时间偏差
    DdDuration = sDuration - 27  # 短27ms
    DdTone = generate_tone(sf, DdDuration, sFreqs, sAmps)
    DdCSramp = create_cosine_ramp(sf, DdDuration, 5)
    DdTone = DdTone * DdCSramp
    DdTone = np.column_stack((DdTone, DdTone))
    
    print("生成位置偏差...")
    # 位置偏差 - 左声道
    LLdTone = generate_tone(sf, sDuration, sFreqs, sAmps, itd=-0.5, ild=-1)
    LLdTone = LLdTone * np.column_stack((CSramp, CSramp))
    
    # 位置偏差 - 右声道
    LRdTone = generate_tone(sf, sDuration, sFreqs, sAmps, itd=0.5, ild=1)
    LRdTone = LRdTone * np.column_stack((CSramp, CSramp))
    
    # 组织刺激
    stimuli = {
        'standard': sTone,
        'freq_high': FHdTone,
        'freq_low': FLdTone,
        'intensity': IdTone,
        'duration': DdTone,
        'location_left': LLdTone,
        'location_right': LRdTone
    }
    
    return stimuli, sf

def load_sequence_data():
    """
    加载序列数据和关键事件数据
    """
    print("加载序列数据和关键事件数据...")
    
    # 尝试读取MATLAB文件
    try:
        from scipy.io import loadmat
        
        # 读取刺激序列文件
        sequence_file = Path(r"D:\D\research\audioset下载\download_audioset-master\download_audioset-master\auditory oddball\sourcedata\sub-01\ses-MMN\aMMN_stimulation_SEQUENCE.mat")
        events_file = Path(r"D:\D\research\audioset下载\download_audioset-master\download_audioset-master\auditory oddball\sourcedata\sub-01\ses-MMN\aMMN_critical_events.mat")
        
        sequence = None
        rows = None
        events = None
        
        # 读取刺激序列文件
        if sequence_file.exists():
            print(f"读取刺激序列文件: {sequence_file}")
            seq_data = loadmat(sequence_file)
            print(f"刺激序列文件中的变量: {list(seq_data.keys())}")
            
            if 'SEQUENCE' in seq_data:
                sequence = seq_data['SEQUENCE'].flatten()
                print(f"SEQUENCE形状: {sequence.shape}")
                print(f"SEQUENCE前10个值: {sequence[:10]}")
            else:
                print("未找到SEQUENCE变量")
                
            if 'ROWS' in seq_data:
                rows = seq_data['ROWS'].flatten()
                print(f"ROWS形状: {rows.shape}")
                print(f"ROWS前10个值: {rows[:10]}")
            else:
                print("未找到ROWS变量")
        else:
            print("刺激序列文件不存在")
        
        # 读取关键事件文件
        if events_file.exists():
            print(f"读取关键事件文件: {events_file}")
            events_data = loadmat(events_file)
            print(f"关键事件文件中的变量: {list(events_data.keys())}")
            
            # 查找时间戳变量
            for key in events_data.keys():
                if not key.startswith('__'):
                    print(f"  {key}: {events_data[key].shape}")
                    if 't' in key.lower() or 'time' in key.lower():
                        events = events_data[key].flatten()
                        print(f"  使用 {key} 作为事件时间戳")
                        break
            
            if events is None:
                print("未找到时间戳变量")
        else:
            print("关键事件文件不存在")
        
        if sequence is not None:
            print(f"成功加载序列数据: {len(sequence)} 个试次")
            return sequence, rows, events
        else:
            print("序列数据加载失败，使用模拟数据")
            return None, None, None
            
    except Exception as e:
        print(f"读取MATLAB文件失败: {e}")
        print("使用模拟数据")
        return None, None, None

def generate_sequence_audio(sequence, rows, stimuli, sample_rate, output_dir="audio_stimuli"):
    """
    根据序列生成完整音频 - 按照MATLAB代码的精确结构
    结构：10个初始标准音 + 720对(偏差音+标准音) = 1450个声音
    时间：每个声音50ms，间隔500ms（总间隔550ms）
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"生成序列音频到 {output_dir}...")
    print(f"序列长度: {len(sequence)} 个偏差音")
    print(f"预期总声音数: 10 + 720*2 = 1450 个")
    print(f"时间安排: 每个声音50ms，间隔500ms")
    
    # 时间参数
    sound_duration_ms = 50  # 声音持续时间
    isi_ms = 500  # 声音间隔
    total_interval_ms = sound_duration_ms + isi_ms  # 总间隔550ms
    
    # 计算样本数
    sound_duration_samples = int(sample_rate * sound_duration_ms / 1000)
    isi_samples = int(sample_rate * isi_ms / 1000)
    
    print(f"声音持续时间: {sound_duration_ms}ms = {sound_duration_samples} 样本")
    print(f"间隔时间: {isi_ms}ms = {isi_samples} 样本")
    print(f"总间隔: {total_interval_ms}ms = {sound_duration_samples + isi_samples} 样本")
    
    # 刺激映射
    stim_map = {
        1: 'standard',
        2: 'freq_high',  # 频率偏差 - 高频
        3: 'intensity',  # 强度偏差
        4: 'duration',   # 持续时间偏差
        5: 'location_left'  # 位置偏差 - 左侧
    }
    
    all_audio_segments = []
    trial_info = []
    sound_count = 0
    current_time = 0.0
    
    # 创建静音段
    silence = np.zeros((isi_samples, 2))  # 立体声静音
    
    # 1. 10个初始标准音 (对应MATLAB代码的lines 197-211)
    print("生成10个初始标准音...")
    for i in range(10):
        standard_audio = stimuli['standard']
        
        # 确保声音长度正确（50ms）
        if len(standard_audio) != sound_duration_samples:
            # 如果长度不匹配，进行裁剪或填充
            if len(standard_audio) > sound_duration_samples:
                standard_audio = standard_audio[:sound_duration_samples]
            else:
                # 填充到正确长度
                padding = sound_duration_samples - len(standard_audio)
                if len(standard_audio.shape) == 2:
                    standard_audio = np.vstack([standard_audio, np.zeros((padding, 2))])
                else:
                    standard_audio = np.concatenate([standard_audio, np.zeros(padding)])
        
        all_audio_segments.append(standard_audio)
        
        trial_info.append({
            'sound_number': sound_count + 1,
            'type': 'initial_standard',
            'stimulus': 'standard',
            'time': current_time,
            'duration': sound_duration_ms / 1000.0,
            'samples': len(standard_audio)
        })
        
        current_time += total_interval_ms / 1000.0
        sound_count += 1
        
        # 添加间隔（除了最后一个声音）
        if i < 9:  # 前9个声音后添加间隔
            all_audio_segments.append(silence)
    
    # 2. 720对声音：偏差音 + 标准音 (对应MATLAB代码的lines 213-232)
    print("生成720对声音 (偏差音+标准音)...")
    for trl in range(720):
        # 选择偏差音类型
        stim_type = sequence[trl]
        
        # 根据ROWS选择具体的偏差音
        if stim_type == 2:  # 频率偏差
            if rows is not None and rows[trl] == 2:
                stim_name = 'freq_low'
            else:
                stim_name = 'freq_high'
        elif stim_type == 5:  # 位置偏差
            if rows is not None and rows[trl] == 2:
                stim_name = 'location_right'
            else:
                stim_name = 'location_left'
        else:
            stim_name = stim_map[stim_type]
        
        # 播放偏差音
        deviant_audio = stimuli[stim_name]
        
        # 确保声音长度正确
        if len(deviant_audio) != sound_duration_samples:
            if len(deviant_audio) > sound_duration_samples:
                deviant_audio = deviant_audio[:sound_duration_samples]
            else:
                padding = sound_duration_samples - len(deviant_audio)
                if len(deviant_audio.shape) == 2:
                    deviant_audio = np.vstack([deviant_audio, np.zeros((padding, 2))])
                else:
                    deviant_audio = np.concatenate([deviant_audio, np.zeros(padding)])
        
        all_audio_segments.append(deviant_audio)
        
        trial_info.append({
            'sound_number': sound_count + 1,
            'type': 'deviant',
            'stimulus': stim_name,
            'trial': trl + 1,
            'sequence_type': int(stim_type),
            'row': int(rows[trl]) if rows is not None else 1,
            'time': current_time,
            'duration': sound_duration_ms / 1000.0,
            'samples': len(deviant_audio)
        })
        
        current_time += total_interval_ms / 1000.0
        sound_count += 1
        
        # 添加间隔
        all_audio_segments.append(silence)
        
        # 播放标准音
        standard_audio = stimuli['standard']
        
        # 确保声音长度正确
        if len(standard_audio) != sound_duration_samples:
            if len(standard_audio) > sound_duration_samples:
                standard_audio = standard_audio[:sound_duration_samples]
            else:
                padding = sound_duration_samples - len(standard_audio)
                if len(standard_audio.shape) == 2:
                    standard_audio = np.vstack([standard_audio, np.zeros((padding, 2))])
                else:
                    standard_audio = np.concatenate([standard_audio, np.zeros(padding)])
        
        all_audio_segments.append(standard_audio)
        
        trial_info.append({
            'sound_number': sound_count + 1,
            'type': 'standard',
            'stimulus': 'standard',
            'trial': trl + 1,
            'time': current_time,
            'duration': sound_duration_ms / 1000.0,
            'samples': len(standard_audio)
        })
        
        current_time += total_interval_ms / 1000.0
        sound_count += 1
        
        # 添加间隔（除了最后一个声音）
        if trl < 719:  # 前719对后添加间隔
            all_audio_segments.append(silence)
    
    # 生成完整序列
    print("生成完整序列音频...")
    full_sequence = np.concatenate(all_audio_segments)
    
    # 验证声音数量
    expected_sounds = 10 + 720 * 2  # 1450
    actual_sounds = len([seg for seg in all_audio_segments if len(seg) == sound_duration_samples])
    print(f"实际生成声音数: {actual_sounds}")
    print(f"预期声音数: {expected_sounds}")
    
    if actual_sounds != expected_sounds:
        print(f"⚠️ 警告: 声音数量不匹配!")
    
    # 验证总时长
    expected_duration = expected_sounds * total_interval_ms / 1000.0
    actual_duration = len(full_sequence) / sample_rate
    print(f"预期总时长: {expected_duration:.2f} 秒")
    print(f"实际总时长: {actual_duration:.2f} 秒")
    
    # 保存音频文件
    sf.write(output_dir / "full_sequence.wav", full_sequence, sample_rate)
    
    # 保存试次信息
    with open(output_dir / "trial_info.json", 'w', encoding='utf-8') as f:
        json.dump(trial_info, f, indent=2, ensure_ascii=False)
    
    # 保存序列信息
    sequence_info = {
        'total_sounds': actual_sounds,
        'initial_standards': 10,
        'deviant_standard_pairs': 720,
        'sound_duration_ms': sound_duration_ms,
        'isi_ms': isi_ms,
        'total_interval_ms': total_interval_ms,
        'sequence': sequence.tolist() if hasattr(sequence, 'tolist') else list(sequence),
        'rows': rows.tolist() if rows is not None and hasattr(rows, 'tolist') else (list(rows) if rows is not None else None),
        'sample_rate': sample_rate,
        'total_duration_seconds': actual_duration
    }
    
    with open(output_dir / "sequence_info.json", 'w', encoding='utf-8') as f:
        json.dump(sequence_info, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 完整序列音频生成完成!")
    print(f"文件名: full_sequence.wav")
    print(f"总时长: {actual_duration / 60:.1f} 分钟")
    print(f"文件保存在: {output_dir}")
    
    return full_sequence

def main():
    """主函数"""
    print("=== Optimum-1 MMN 音频刺激生成器 ===")
    
    # 生成刺激
    stimuli, sf = generate_optimum1_stimuli()
    
    # 加载序列数据
    sequence, rows, events = load_sequence_data()
    
    # 如果没有序列数据，创建模拟序列
    if sequence is None:
        print("创建模拟序列...")
        # 按照Pakarinen et al. (2007)的规则生成序列
        # 1. 每个偏差类型在4个连续偏差中出现一次
        # 2. 两个连续偏差永远不相同
        
        sequence = np.zeros(720, dtype=int)
        sequence[:4] = np.random.permutation(4) + 2  # 2,3,4,5的随机排列
        
        for ar in range(4, 720, 4):
            ar_order = np.random.permutation(4) + 2  # 2,3,4,5的随机排列
            if sequence[ar-1] != ar_order[0]:
                sequence[ar:ar+4] = ar_order
            else:
                sequence[ar:ar+4] = ar_order[::-1]  # 翻转避免重复
        
        # 生成ROWS数组
        rows = np.ones(720, dtype=int)
        
        # 频率偏差：90个高频 + 90个低频
        freq_indices = sequence == 2
        freq_count = np.sum(freq_indices)
        high_low = np.concatenate([np.ones(90), np.ones(90) * 2])
        high_low = high_low[np.random.permutation(len(high_low))]
        rows[freq_indices] = high_low[:freq_count]
        
        # 位置偏差：90个左侧 + 90个右侧
        loc_indices = sequence == 5
        loc_count = np.sum(loc_indices)
        left_right = np.concatenate([np.ones(90), np.ones(90) * 2])
        left_right = left_right[np.random.permutation(len(left_right))]
        rows[loc_indices] = left_right[:loc_count]
        
        print(f"模拟序列生成完成:")
        print(f"  序列长度: {len(sequence)}")
        print(f"  偏差类型分布: {np.bincount(sequence)}")
        print(f"  频率偏差行分布: {np.bincount(rows[sequence == 2])}")
        print(f"  位置偏差行分布: {np.bincount(rows[sequence == 5])}")
    
    # 生成序列音频
    full_audio = generate_sequence_audio(sequence, rows, stimuli, sf)
    
    print(f"\n=== 生成完成 ===")
    print(f"刺激类型分布:")
    unique, counts = np.unique(sequence, return_counts=True)
    for stim_type, count in zip(unique, counts):
        print(f"  类型 {stim_type}: {count} 次")

if __name__ == "__main__":
    main()
