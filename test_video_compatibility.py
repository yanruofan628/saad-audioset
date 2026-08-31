#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 EPrime 2.0 对不同视频编码的兼容性
"""

import os
import cv2
import subprocess
import sys


def test_opencv_mp4v(input_video, output_file, duration=7):
    """使用 OpenCV 生成 MP4V 编码的视频（与之前的 video_split.py 相同）"""
    print("测试 OpenCV MP4V 编码...")

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("无法打开输入视频")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # MP4V 编码（与之前的脚本相同）
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    if not out.isOpened():
        print("无法创建 MP4V 输出文件")
        return False

    # 读取前7秒
    frames_to_read = int(fps * duration)
    frames_written = 0

    for i in range(frames_to_read):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        frames_written += 1

    cap.release()
    out.release()

    print(f"OpenCV MP4V 测试完成: {frames_written}帧写入")
    return os.path.exists(output_file)


def test_ffmpeg_h264(input_video, output_file, duration=7):
    """使用 FFmpeg 生成 H.264 编码的视频（与当前的 video_split_7s.py 相同）"""
    print("测试 FFmpeg H.264 编码...")

    cmd = [
        'ffmpeg',
        '-i', input_video,
        '-t', str(duration),
        '-an',  # 无音频
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-crf', '23',
        '-y',
        output_file
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30
        )
        success = result.returncode == 0 and os.path.exists(output_file)
        print(f"FFmpeg H.264 测试完成: {'成功' if success else '失败'}")
        return success
    except Exception as e:
        print(f"FFmpeg 测试出错: {e}")
        return False


def get_video_info(video_file):
    """获取视频基本信息"""
    try:
        cap = cv2.VideoCapture(video_file)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            return f"{width}x{height}@{fps:.1f}fps"
    except:
        pass
    return "未知"


def main():
    # 测试视频文件（需要您替换为实际路径）
    input_video = r"C:\Users\Lenovo\Downloads\Video_ring.mp4"

    if not os.path.exists(input_video):
        print(f"输入视频不存在: {input_video}")
        print("请修改脚本中的 input_video 路径")
        return

    print(f"输入视频: {input_video}")
    print(f"视频信息: {get_video_info(input_video)}")
    print()

    # 创建测试输出目录
    test_dir = "video_compatibility_test"
    os.makedirs(test_dir, exist_ok=True)

    # 测试两种编码格式
    opencv_file = os.path.join(test_dir, "test_opencv_mp4v.mp4")
    ffmpeg_file = os.path.join(test_dir, "test_ffmpeg_h264.mp4")

    opencv_success = test_opencv_mp4v(input_video, opencv_file)
    ffmpeg_success = test_ffmpeg_h264(input_video, ffmpeg_file)

    print("\n测试结果:")
    print(f"OpenCV MP4V: {'✓ 成功' if opencv_success else '✗ 失败'}")
    print(f"FFmpeg H.264: {'✓ 成功' if ffmpeg_success else '✗ 失败'}")

    if opencv_success and ffmpeg_success:
        print("\n两个格式都生成成功。")
        print("请将这两个测试文件导入 EPrime 2.0 进行播放测试。")
        print("如果 OpenCV 生成的视频能播放，而 FFmpeg 生成的不能，")
        print("那么 EPrime 2.0 可能不支持 H.264 编码。")

        print("\n测试文件位置:")
        print(f"OpenCV MP4V: {opencv_file}")
        print(f"FFmpeg H.264: {ffmpeg_file}")
    else:
        print("\n部分测试失败，请检查输入视频路径和 FFmpeg 安装。")


if __name__ == "__main__":
    main()
