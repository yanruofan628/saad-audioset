#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频裁剪脚本：将视频裁剪成7秒片段
- 任务1：第四十回，从2分钟开始，切160个片段
- 任务2：第四十回，从21分钟开始，切160个片段
- 任务3：第二十九回，从2分钟开始，切160个片段
"""

import os
import subprocess
import sys


def check_ffmpeg():
    """检查ffmpeg是否可用"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, 
                              text=True,
                              encoding='utf-8',
                              errors='ignore',
                              timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def format_time(seconds):
    """将秒数转换为HH:MM:SS格式"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"


def get_video_info(input_video):
    """获取视频信息"""
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height',
            '-of', 'csv=s=x:p=0',
            input_video
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore',
            timeout=10
        )
        if result.returncode == 0:
            dimensions = result.stdout.strip().split('x')
            if len(dimensions) == 2:
                return int(dimensions[0]), int(dimensions[1])
    except:
        pass
    return None, None


def split_video_batch(input_video, output_dir, batch_name, start_offset_seconds, segment_duration=7, segments_count=160, scale_ratio=1.0, fps=25, max_width=1280, max_height=720):
    """
    使用ffmpeg将视频裁剪成指定数量的片段（快速版本）
    
    Args:
        input_video: 输入视频文件路径
        output_dir: 输出目录路径
        batch_name: 批次名称（用于显示）
        start_offset_seconds: 从视频的第几秒开始裁剪
        segment_duration: 每个片段的时长（秒），默认7秒
        segments_count: 要创建的片段数量，默认160个
    """
    print(f"\n{'='*60}")
    print(f"开始处理: {batch_name}")
    print(f"视频文件: {os.path.basename(input_video)}")
    print(f"起始时间: {start_offset_seconds}秒 ({start_offset_seconds/60:.1f}分钟)")
    print(f"片段时长: {segment_duration}秒")
    print(f"片段数量: {segments_count}个")
    print(f"总时长: {segments_count * segment_duration}秒 ({segments_count * segment_duration/60:.1f}分钟)")
    print(f"{'='*60}")
    
    # 检查输入文件是否存在
    if not os.path.exists(input_video):
        print(f"错误: 输入视频文件不存在: {input_video}")
        return False
    
    # 检查ffmpeg是否可用
    if not check_ffmpeg():
        print("错误: 未找到ffmpeg，请先安装ffmpeg")
        print("下载地址: https://ffmpeg.org/download.html")
        return False
    
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建输出目录: {output_dir}")
    
    # 获取视频尺寸信息（用于缩小和留白）
    video_width = None
    video_height = None
    video_width, video_height = get_video_info(input_video)
    
    if video_width and video_height:
        print(f"视频原始尺寸: {video_width}x{video_height}")
        
        # 计算目标分辨率（不超过720P，保持16:9）
        # 首先计算如果缩小到720P需要什么比例
        scale_to_720p = min(max_width / video_width, max_height / video_height)
        
        # 如果原始视频已经小于720P，不需要缩小到720P
        if scale_to_720p >= 1.0:
            target_width = video_width
            target_height = video_height
            print(f"原始视频已小于720P，保持原始尺寸")
        else:
            # 缩小到720P或更小，保持16:9比例
            target_width = int(video_width * scale_to_720p)
            target_height = int(video_height * scale_to_720p)
            # 确保是偶数（视频编码要求）
            target_width = target_width - (target_width % 2)
            target_height = target_height - (target_height % 2)
            print(f"缩小到720P以下: {target_width}x{target_height}")
        
        # 如果还需要进一步缩放（scale_ratio < 1.0）
        if scale_ratio < 1.0:
            scaled_width = int(target_width * scale_ratio)
            scaled_height = int(target_height * scale_ratio)
            # 确保是偶数
            scaled_width = scaled_width - (scaled_width % 2)
            scaled_height = scaled_height - (scaled_height % 2)
            print(f"进一步缩小后尺寸: {scaled_width}x{scaled_height} (保留目标尺寸的{scale_ratio*100:.0f}%)")
            print(f"四周留白: 视频居中显示，周围为黑边")
        else:
            scaled_width = target_width
            scaled_height = target_height
            print(f"最终输出尺寸: {scaled_width}x{scaled_height}")
    else:
        print("警告: 无法获取视频尺寸，将不进行缩放")
        scale_ratio = 1.0
        scaled_width = None
        scaled_height = None
    
    success_count = 0
    
    for i in range(segments_count):
        # 计算当前片段的起始时间（相对于整个视频）
        segment_start_time = start_offset_seconds + (i * segment_duration)
        
        # 文件命名：m001, m002, ..., m160
        output_file = os.path.join(output_dir, f"m{i + 1:03d}.mp4")
        
        # 构建ffmpeg命令
        # -ss: 起始时间
        # -i: 输入文件
        # -t: 持续时间
        # -an: 无音频
        # -y: 覆盖输出文件
        start_time_str = format_time(segment_start_time)
        duration_str = format_time(segment_duration)
        
        cmd = [
            'ffmpeg',
            '-ss', start_time_str,
            '-i', input_video,
            '-t', duration_str,
            '-an',  # 无音频
        ]
        
        # 总是需要重新编码（因为有分辨率限制和固定帧率）
        if video_width and video_height and scaled_width and scaled_height:
            # 构建滤镜链
            vf_parts = []
            
            # 1. 首先缩小到720P或更小（如果需要）
            scale_to_720p = min(max_width / video_width, max_height / video_height)
            if scale_to_720p < 1.0:
                # 需要缩小到720P
                temp_width = int(video_width * scale_to_720p)
                temp_height = int(video_height * scale_to_720p)
                temp_width = temp_width - (temp_width % 2)
                temp_height = temp_height - (temp_height % 2)
                vf_parts.append(f'scale={temp_width}:{temp_height}')
                current_width = temp_width
                current_height = temp_height
            else:
                current_width = video_width
                current_height = video_height
            
            # 2. 如果还需要进一步缩放（scale_ratio < 1.0）
            if scale_ratio < 1.0:
                vf_parts.append(f'scale={scaled_width}:{scaled_height}')
                # 添加留白（pad）
                pad_x = (current_width - scaled_width) // 2
                pad_y = (current_height - scaled_height) // 2
                vf_parts.append(f'pad={current_width}:{current_height}:{pad_x}:{pad_y}:black')
            
            # 3. 设置固定帧率
            if fps is not None:
                vf_parts.insert(0, f'fps={fps}')  # fps滤镜放在最前面
            
            # 应用滤镜
            if vf_parts:
                cmd.extend([
                    '-vf', ','.join(vf_parts)
                ])
            
            # 需要重新编码，使用最快的设置
            cmd.extend([
                '-c:v', 'libx264',  # 使用H.264编码
                '-preset', 'ultrafast',  # 最快编码预设
                '-tune', 'fastdecode',  # 优化解码速度
                '-crf', '23',  # 质量参数
                '-r', str(fps)  # 固定帧率
            ])
        else:
            # 如果无法获取视频信息，使用copy模式（但会保持原始分辨率）
            print("警告: 无法获取视频尺寸，使用copy模式（可能不满足分辨率要求）")
            cmd.extend([
                '-c', 'copy',
                '-r', str(fps)  # 尝试设置帧率（可能无效）
            ])
        
        cmd.extend([
            '-y',  # 覆盖输出文件
            output_file
        ])
        
        try:
            # 执行ffmpeg命令（隐藏输出）
            # 使用encoding='utf-8'和errors='ignore'避免编码错误
            result = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='ignore',
                timeout=30  # 每个片段最多30秒
            )
            
            if result.returncode == 0 and os.path.exists(output_file):
                if (i + 1) % 20 == 0:  # 每20个片段打印一次进度
                    print(f"  已完成: {i + 1}/{segments_count} 个片段")
                success_count += 1
            else:
                error_msg = result.stderr[:200] if result.stderr else "未知错误"
                print(f"  ✗ 处理 m{i + 1:03d}.mp4 失败: {error_msg}")
        
        except subprocess.TimeoutExpired:
            print(f"  ✗ 处理 m{i + 1:03d}.mp4 超时")
        except Exception as e:
            print(f"  ✗ 处理 m{i + 1:03d}.mp4 时出错: {str(e)}")
    
    print(f"\n{batch_name} 完成！成功创建 {success_count}/{segments_count} 个片段")
    print(f"输出目录: {output_dir}")
    return success_count > 0


def main():
    """主函数"""
    # 视频目录
    video_dir = r"A:\BaiduNetdiskDownload"
    
    # 视频文件名（如果找不到第四十二回，使用第四十回）
    video_42_name = None
    video_29_name = None
    
    # 查找视频文件
    if os.path.exists(video_dir):
        files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
        for f in files:
            if '42' in f or '四十二' in f or '第四十二' in f:
                video_42_name = f
            elif '40' in f or '四十' in f or '第四十' in f:
                if video_42_name is None:  # 如果没找到42回，用40回
                    video_42_name = f
            elif '29' in f or '二十九' in f or '第二十九' in f:
                video_29_name = f
    
    # 设置视频路径
    if video_42_name:
        video_42_path = os.path.join(video_dir, video_42_name)
        print(f"找到第四十/四十二回视频: {video_42_name}")
    else:
        print("错误: 未找到第四十/四十二回视频文件")
        return
    
    if video_29_name:
        video_29_path = os.path.join(video_dir, video_29_name)
        print(f"找到第二十九回视频: {video_29_name}")
    else:
        print("错误: 未找到第二十九回视频文件")
        return
    
    # 输出目录
    base_output_dir = r"A:\video_segments_7s"
    
    # 切片参数
    segment_duration = 7  # 7秒
    segments_count = 160  # 160个片段
    
    # 任务配置
    tasks = [
        {
            'name': '第四十回-2分钟',
            'video_path': video_42_path,
            'start_time': 2 * 60,  # 2分钟 = 120秒
            'output_dir': os.path.join(base_output_dir, 'batch_1_42回_2min')
        },
        {
            'name': '第四十回-21分钟',
            'video_path': video_42_path,
            'start_time': 21 * 60,  # 21分钟 = 1260秒
            'output_dir': os.path.join(base_output_dir, 'batch_2_42回_21min')
        },
        {
            'name': '第二十九回-2分钟',
            'video_path': video_29_path,
            'start_time': 2 * 60,  # 2分钟 = 120秒
            'output_dir': os.path.join(base_output_dir, 'batch_3_29回_2min')
        }
    ]
    
    print(f"\n{'='*60}")
    print("视频切片任务配置")
    print(f"{'='*60}")
    for i, task in enumerate(tasks, 1):
        print(f"\n任务{i}: {task['name']}")
        print(f"  视频: {os.path.basename(task['video_path'])}")
        print(f"  起始时间: {task['start_time']}秒 ({task['start_time']/60:.1f}分钟)")
        print(f"  片段时长: {segment_duration}秒")
        print(f"  片段数量: {segments_count}个")
        print(f"  输出目录: {task['output_dir']}")
    
    # 视频处理设置
    fps = 25  # 固定帧率：25 fps
    max_width = 1280  # 最大宽度：720P
    max_height = 720  # 最大高度：720P
    scale_ratio = 0.6  # 进一步缩小到目标尺寸的60%，四周留白
    
    print(f"\n视频处理设置:")
    print(f"  固定帧率: {fps} fps")
    print(f"  最大分辨率: {max_width}x{max_height} (720P)")
    print(f"  保持16:9比例: 是")
    if scale_ratio < 1.0:
        print(f"  进一步缩小比例: {scale_ratio*100:.0f}%")
        print(f"  四周留白: 视频居中显示，周围为黑边")
    print(f"  字幕保留: 是")
    print(f"  注意: 所有视频都会重新编码以满足分辨率和帧率要求")
    
    # 执行切片任务
    all_success = True
    for task in tasks:
        success = split_video_batch(
            input_video=task['video_path'],
            output_dir=task['output_dir'],
            batch_name=task['name'],
            start_offset_seconds=task['start_time'],
            segment_duration=segment_duration,
            segments_count=segments_count,
            scale_ratio=scale_ratio,
            fps=fps,
            max_width=max_width,
            max_height=max_height
        )
        if not success:
            all_success = False
            print(f"\n任务失败: {task['name']}")
    
    # 最终统计
    if all_success:
        print("\n" + "="*60)
        print("所有任务处理完成！")
        print("="*60)
        
        total_files = 0
        total_size = 0
        
        for task in tasks:
            if os.path.exists(task['output_dir']):
                output_files = [f for f in os.listdir(task['output_dir']) if f.endswith('.mp4')]
                batch_size = sum(os.path.getsize(os.path.join(task['output_dir'], f)) for f in output_files)
                
                print(f"\n{task['name']} ({task['output_dir']}):")
                print(f"  - 文件数量: {len(output_files)} 个")
                print(f"  - 文件大小: {batch_size / (1024 * 1024):.2f} MB")
                
                total_files += len(output_files)
                total_size += batch_size
        
        print(f"\n总计:")
        print(f"  - 总文件数: {total_files} 个")
        print(f"  - 总大小: {total_size / (1024 * 1024):.2f} MB")
    else:
        print("\n部分任务处理失败！")


if __name__ == "__main__":
    main()

