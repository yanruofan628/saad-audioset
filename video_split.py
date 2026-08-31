#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频裁剪脚本：将1小时视频裁剪成2批，每批144段12秒的无声视频
使用OpenCV处理，自动无声
"""

import os
import cv2
import sys


def create_output_directories():
    """创建输出目录"""
    base_dir = r"D:\D\research\audioset下载\video_segments"
    batch_dirs = []
    
    for batch_num in [1, 2]:
        batch_dir = os.path.join(base_dir, f"batch_{batch_num}")
        if not os.path.exists(batch_dir):
            os.makedirs(batch_dir)
            print(f"已创建输出目录: {batch_dir}")
        else:
            print(f"输出目录已存在: {batch_dir}")
        batch_dirs.append(batch_dir)
    
    return batch_dirs


def split_video_batch(input_video, output_dir, batch_num, start_offset_seconds):
    """
    使用OpenCV将视频裁剪成144段12秒的无声视频
    
    Args:
        input_video: 输入视频文件路径
        output_dir: 输出目录路径
        batch_num: 批次号（用于显示）
        start_offset_seconds: 从视频的第几秒开始裁剪
    """
    print(f"\n{'='*60}")
    print(f"开始处理批次 {batch_num}")
    print(f"从视频 {start_offset_seconds}秒 开始裁剪144段12秒的无声视频")
    print(f"{'='*60}")
    
    # 检查输入文件是否存在
    if not os.path.exists(input_video):
        print(f"错误: 输入视频文件不存在: {input_video}")
        return False
    
    # 打开视频文件
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件: {input_video}")
        return False
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = total_frames / fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"视频信息: FPS={fps:.2f}, 总时长={total_duration:.2f}秒, 分辨率={width}x{height}")
    
    # 每批144个12秒的视频片段
    segments_per_batch = 144
    segment_duration = 12  # 12秒
    frames_per_segment = int(fps * segment_duration)
    
    # 计算这批需要的总时长
    batch_duration = segments_per_batch * segment_duration
    end_offset_seconds = start_offset_seconds + batch_duration
    
    # 检查视频是否足够长
    if end_offset_seconds > total_duration:
        print(f"警告: 视频时长不足，从{start_offset_seconds}秒开始无法裁剪{segments_per_batch}个片段")
        available_duration = total_duration - start_offset_seconds
        segments_per_batch = int(available_duration // segment_duration)
        print(f"将创建 {segments_per_batch} 个片段")
    
    success_count = 0
    
    for i in range(segments_per_batch):
        # 计算当前片段的起始时间（相对于整个视频）
        segment_start_time = start_offset_seconds + (i * segment_duration)
        start_frame = int(segment_start_time * fps)
        
        # 文件命名：m001, m002, ..., m144
        output_file = os.path.join(output_dir, f"m{i + 1:03d}.mp4")
        
        # 设置视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
        
        if not out.isOpened():
            print(f"  ✗ 无法创建输出文件: {os.path.basename(output_file)}")
            continue
        
        try:
            # 跳转到开始位置
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # 读取并写入帧
            frames_written = 0
            for frame_idx in range(frames_per_segment):
                ret, frame = cap.read()
                if not ret:
                    break
                
                out.write(frame)
                frames_written += 1
            
            out.release()
            
            if frames_written == frames_per_segment:
                if (i + 1) % 10 == 0:  # 每10个片段打印一次进度
                    print(f"  已完成: {i + 1}/{segments_per_batch} 个片段")
                success_count += 1
            else:
                print(f"  ⚠ {os.path.basename(output_file)} 不完整 ({frames_written}/{frames_per_segment}帧)")
                success_count += 1
        
        except Exception as e:
            print(f"  ✗ 处理 m{i + 1:03d}.mp4 时出错: {str(e)}")
            out.release()
    
    cap.release()
    print(f"\n批次 {batch_num} 完成！成功创建 {success_count}/{segments_per_batch} 个片段")
    print(f"输出目录: {output_dir}")
    return success_count > 0


def main():
    """主函数"""
    input_video = r"C:\Users\Lenovo\Downloads\Video_ring.mp4"
    
    # 创建输出目录
    batch_dirs = create_output_directories()
    
    # 每批144个片段，每个12秒，共144*12=1728秒=28分48秒
    segment_duration = 12
    segments_per_batch = 144
    batch_duration = segment_duration * segments_per_batch  # 1728秒
    
    print(f"输入视频: {input_video}")
    print(f"每批参数: {segments_per_batch}个片段 × {segment_duration}秒 = {batch_duration}秒 ({batch_duration/60:.1f}分钟)")
    
    # 处理2批
    all_success = True
    for batch_num in range(1, 3):  # 批次1和批次2
        start_offset = (batch_num - 1) * batch_duration  # 批次1从0秒开始，批次2从1728秒开始
        output_dir = batch_dirs[batch_num - 1]
        
        success = split_video_batch(input_video, output_dir, batch_num, start_offset)
        if not success:
            all_success = False
            print(f"批次 {batch_num} 处理失败！")
    
    # 最终统计
    if all_success:
        print("\n" + "="*60)
        print("所有批次处理完成！")
        print("="*60)
        
        total_files = 0
        total_size = 0
        
        for batch_num, batch_dir in enumerate(batch_dirs, 1):
            output_files = [f for f in os.listdir(batch_dir) if f.endswith('.mp4')]
            batch_size = sum(os.path.getsize(os.path.join(batch_dir, f)) for f in output_files)
            
            print(f"\n批次 {batch_num} ({batch_dir}):")
            print(f"  - 文件数量: {len(output_files)} 个")
            print(f"  - 文件大小: {batch_size / (1024 * 1024):.2f} MB")
            
            total_files += len(output_files)
            total_size += batch_size
        
        print(f"\n总计:")
        print(f"  - 总文件数: {total_files} 个")
        print(f"  - 总大小: {total_size / (1024 * 1024):.2f} MB")
    else:
        print("\n部分批次处理失败！")


if __name__ == "__main__":
    main()
