#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频实验Web服务器启动脚本
"""

import subprocess
import sys
import os

def check_dependencies():
    """检查依赖是否安装"""
    required_packages = [
        'flask', 'flask-socketio', 'pygame', 'cv2', 'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            else:
                __import__(package)
            print(f"✓ {package} 已安装")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} 未安装")
    
    if missing_packages:
        print(f"\n缺少以下依赖包: {', '.join(missing_packages)}")
        print("请运行以下命令安装:")
        print("pip install -r requirements.txt")
        return False
    
    return True

def get_local_ip():
    """获取本机IP地址"""
    import socket
    try:
        # 连接到一个远程地址来获取本机IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

def main():
    print("=" * 60)
    print("音频实验 Web 服务器")
    print("=" * 60)
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    # 检查必要文件
    required_files = [
        'stilumi_web.py',
        'templates/experiment.html'
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            print(f"✗ 缺少必要文件: {file}")
            sys.exit(1)
        else:
            print(f"✓ {file} 存在")
    
    # 获取本机IP
    local_ip = get_local_ip()
    
    print("\n" + "=" * 60)
    print("启动信息:")
    print(f"本机访问地址: http://localhost:5000")
    print(f"局域网访问地址: http://{local_ip}:5000")
    print("=" * 60)
    print("使用说明:")
    print("1. 启动服务器后，在浏览器中打开上述地址")
    print("2. 点击页面上的按钮开始实验")
    print("3. 实验数据会自动保存到 experiment_output2 目录")
    print("4. 按 Ctrl+C 停止服务器")
    print("=" * 60)
    
    try:
        # 启动Web服务器
        subprocess.run([sys.executable, 'stilumi_web.py'])
    except KeyboardInterrupt:
        print("\n服务器已停止")
    except Exception as e:
        print(f"启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
