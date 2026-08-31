#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频裁剪脚本：将视频裁剪成7秒片段 (OpenCV版本，使用MP4V编码)
- 任务1：第四十回，从2分钟开始，切160个片段
- 任务2：第四十回，从21分钟开始，切160个片段
- 任务3：第二十九回，从2分钟开始，切160个片段

使用OpenCV处理，保证与之前成功的video_split.py兼容
"""

import os
import cv2
import sys


def check_video_file(input_video):
    """检查视频文件是否存在"""
    if not os.path.exists(input_video):
        print(f"错误: 输入视频文件不存在: {input_video}")
        return False
    return True


def get_video_info(input_video):
    """获取视频信息"""
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件: {input_video}")
        return None, None, None

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()

    return fps, width, height, total_frames


def split_video_batch_opencv(input_video, output_dir, batch_name, start_offset_seconds, segment_duration=7, segments_count=160):
    """
    使用OpenCV将视频裁剪成指定数量的片段（MP4V编码）

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

    # 检查输入文件
    if not check_video_file(input_video):
        return False

    # 获取视频信息
    fps, width, height, total_frames = get_video_info(input_video)
    if fps is None:
        return False

    print(f"视频信息: {width}x{height}@{fps:.1f}fps, 总帧数: {total_frames}")

    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建输出目录: {output_dir}")

    # 计算每段的帧数
    frames_per_segment = int(fps * segment_duration)
    start_frame_offset = int(start_offset_seconds * fps)

    print(f"每段帧数: {frames_per_segment}")
    print(f"起始帧偏移: {start_frame_offset}")

    success_count = 0

    for i in range(segments_count):
        # 计算当前片段的起始帧
        segment_start_frame = start_frame_offset + (i * frames_per_segment)

        # 文件命名：m001, m002, ..., m160
        output_file = os.path.join(output_dir, f"m{i + 1:03d}.mp4")

        # 检查是否超出视频范围
        if segment_start_frame + frames_per_segment > total_frames:
            print(f"警告: 片段 {i + 1} 超出视频范围，停止处理")
            break

        # 设置视频写入器 (MP4V编码，与之前的脚本相同)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

        if not out.isOpened():
            print(f"  ✗ 无法创建输出文件: {os.path.basename(output_file)}")
            continue

        # 打开输入视频
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            print(f"  ✗ 无法重新打开输入视频")
            out.release()
            continue

        try:
            # 跳转到开始位置
            cap.set(cv2.CAP_PROP_POS_FRAMES, segment_start_frame)

            # 读取并写入帧
            frames_written = 0
            for frame_idx in range(frames_per_segment):
                ret, frame = cap.read()
                if not ret:
                    break

                out.write(frame)
                frames_written += 1

            out.release()
            cap.release()

            if frames_written == frames_per_segment:
                if (i + 1) % 20 == 0:  # 每20个片段打印一次进度
                    print(f"  已完成: {i + 1}/{segments_count} 个片段")
                success_count += 1
            else:
                print(f"  ⚠ {os.path.basename(output_file)} 不完整 ({frames_written}/{frames_per_segment}帧)")

        except Exception as e:
            print(f"  ✗ 处理 m{i + 1:03d}.mp4 时出错: {str(e)}")
            out.release()
            cap.release()

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
    base_output_dir = r"A:\video_segments_7s_opencv"

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
        print(f"  编码格式: MP4V (OpenCV)")

    print(f"\n视频处理设置:")
    print(f"  编码格式: MP4V (MPEG-4 Part 2)")
    print(f"  音频: 无 (OpenCV默认)")
    print(f"  帧率: 保持原始帧率")
    print(f"  分辨率: 保持原始分辨率")

    # 执行切片任务
    all_success = True
    for task in tasks:
        success = split_video_batch_opencv(
            input_video=task['video_path'],
            output_dir=task['output_dir'],
            batch_name=task['name'],
            start_offset_seconds=task['start_time'],
            segment_duration=segment_duration,
            segments_count=segments_count
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
