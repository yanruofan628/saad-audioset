#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频裁剪脚本：裁剪 ring.mkv 视频
从40分钟开始，每10秒裁剪一次，140个片段放一个文件夹，共三个文件夹
使用OpenCV处理，自动无声
"""

import os
import cv2
import sys


def create_output_directories(base_dir, batch_count=3):
    """创建输出目录"""
    batch_dirs = []

    for batch_num in range(1, batch_count + 1):
        batch_dir = os.path.join(base_dir, f"batch_{batch_num}")
        if not os.path.exists(batch_dir):
            os.makedirs(batch_dir)
            print(f"已创建输出目录: {batch_dir}")
        else:
            print(f"输出目录已存在: {batch_dir}")
        batch_dirs.append(batch_dir)

    return batch_dirs


def split_video_batch(input_video, output_dir, batch_num, start_offset_seconds, segment_duration=7, segment_interval=10, segments_count=140):
    """
    使用OpenCV将视频裁剪成指定数量的片段

    Args:
        input_video: 输入视频文件路径
        output_dir: 输出目录路径
        batch_num: 批次号（用于显示）
        start_offset_seconds: 从视频的第几秒开始裁剪
        segment_duration: 每个片段的时长（秒），默认7秒
        segments_count: 要创建的片段数量，默认160个
    """
    print(f"\n{'='*60}")
    print(f"开始处理批次 {batch_num}")
    print(f"视频文件: {os.path.basename(input_video)}")
    print(f"从视频 {start_offset_seconds}秒 ({start_offset_seconds/60:.1f}分钟) 开始裁剪")
    print(f"每{segment_interval}秒裁剪一个{segment_duration}秒片段，共{segments_count}个")
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

    # 每批160个7秒的视频片段
    frames_per_segment = int(fps * segment_duration)

    # 计算这批需要的总时长（140个片段 × 10秒间隔 = 1400秒）
    batch_duration = segments_count * segment_interval
    end_offset_seconds = start_offset_seconds + batch_duration

    print(f"批次时长: {batch_duration}秒 ({batch_duration/60:.1f}分钟)")
    print(f"结束时间: {end_offset_seconds}秒 ({end_offset_seconds/60:.1f}分钟)")

    # 检查视频是否足够长
    if end_offset_seconds > total_duration:
        print(f"警告: 视频时长不足，从{start_offset_seconds}秒开始无法裁剪{segments_count}个片段")
        available_duration = total_duration - start_offset_seconds
        segments_count = int(available_duration // segment_interval)
        print(f"将创建 {segments_count} 个片段")
        batch_duration = segments_count * segment_interval

    success_count = 0

    for i in range(segments_count):
        # 计算当前片段的起始时间（相对于整个视频）
        # 每10秒裁剪一次，所以间隔是10秒
        segment_start_time = start_offset_seconds + (i * segment_interval)
        start_frame = int(segment_start_time * fps)

        # 文件命名：m001, m002, ..., m160
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
                if (i + 1) % 20 == 0:  # 每20个片段打印一次进度
                    print(f"  已完成: {i + 1}/{segments_count} 个片段")
                success_count += 1
            else:
                print(f"  ⚠ {os.path.basename(output_file)} 不完整 ({frames_written}/{frames_per_segment}帧)")
                success_count += 1

        except Exception as e:
            print(f"  ✗ 处理 m{i + 1:03d}.mp4 时出错: {str(e)}")
            out.release()

    cap.release()
    print(f"\n批次 {batch_num} 完成！成功创建 {success_count}/{segments_count} 个片段")
    print(f"输出目录: {output_dir}")
    return success_count > 0


def main():
    """主函数"""
    # 输入视频文件
    input_video = r"A:\BaiduNetdiskDownload\ring.mkv"

    # 输出目录
    base_output_dir = r"A:\video_segments_ring_40min_10s"

    # 裁剪参数
    segment_duration = 7  # 7秒
    segment_interval = 10  # 每10秒裁剪一次
    segments_per_batch = 140  # 每批140个片段

    # 从40分钟开始，每批之间连续（无间隔）
    start_time_minutes = 40  # 40分钟
    start_offset_seconds = start_time_minutes * 60  # 2400秒

    # 每批时长：140个片段 × 10秒间隔 = 1400秒 = 23分20秒
    batch_duration_seconds = segments_per_batch * segment_interval  # 1400秒

    # 创建输出目录
    batch_dirs = create_output_directories(base_output_dir, batch_count=3)

    print(f"\n输入视频: {input_video}")
    print(f"输出根目录: {base_output_dir}")
    print(f"起始时间: {start_time_minutes}分钟 ({start_offset_seconds}秒)")
    print(f"每批参数: {segments_per_batch}个片段 × {segment_interval}秒间隔 = {batch_duration_seconds}秒 ({batch_duration_seconds/60:.1f}分钟)")
    print(f"批次数量: 3个")
    print(f"总片段数: {segments_per_batch * 3}个")
    print(f"总时长: {batch_duration_seconds * 3}秒 ({batch_duration_seconds * 3 / 60:.1f}分钟)")

    # 处理3批
    all_success = True
    for batch_num in range(1, 4):  # 批次1, 2, 3
        # 每批的起始时间：批次1从4分钟开始，后续批次连续
        batch_start_offset = start_offset_seconds + (batch_num - 1) * batch_duration_seconds
        output_dir = batch_dirs[batch_num - 1]

        print(f"\n批次{batch_num}起始时间: {batch_start_offset}秒 ({batch_start_offset/60:.1f}分钟)")

        success = split_video_batch(
            input_video=input_video,
            output_dir=output_dir,
            batch_num=batch_num,
            start_offset_seconds=batch_start_offset,
            segment_duration=segment_duration,
            segment_interval=segment_interval,
            segments_count=segments_per_batch
        )
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
            if os.path.exists(batch_dir):
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
