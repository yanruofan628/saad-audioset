#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用ngrok启动音频实验Web服务器
"""

import subprocess
import sys
import os
import time
import threading
import requests

def check_ngrok():
    """检查ngrok是否安装"""
    try:
        result = subprocess.run(['ngrok', 'version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✓ ngrok 已安装")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    print("✗ ngrok 未安装")
    print("请访问 https://ngrok.com 下载并安装ngrok")
    print("或者运行: pip install pyngrok")
    return False

def install_pyngrok():
    """安装pyngrok"""
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'pyngrok'], 
                      check=True)
        print("✓ pyngrok 安装成功")
        return True
    except subprocess.CalledProcessError:
        print("✗ pyngrok 安装失败")
        return False

def start_web_server():
    """启动Web服务器"""
    print("启动Web服务器...")
    try:
        subprocess.run([sys.executable, 'stilumi_web.py'])
    except KeyboardInterrupt:
        print("Web服务器已停止")

def start_ngrok_tunnel():
    """启动ngrok隧道"""
    try:
        from pyngrok import ngrok
        
        print("启动ngrok隧道...")
        # 创建HTTP隧道
        public_url = ngrok.connect(5000)
        
        print("=" * 60)
        print("🎉 ngrok隧道已启动！")
        print("=" * 60)
        print(f"公网访问地址: {public_url}")
        print("=" * 60)
        print("请将此地址发送给被试者:")
        print(f"📱 {public_url}")
        print("=" * 60)
        print("按 Ctrl+C 停止服务器和隧道")
        print("=" * 60)
        
        # 保持隧道运行
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n正在关闭ngrok隧道...")
            ngrok.disconnect(public_url)
            ngrok.kill()
            
    except ImportError:
        print("pyngrok 未安装，尝试安装...")
        if install_pyngrok():
            start_ngrok_tunnel()
        else:
            print("无法启动ngrok，请手动安装")
    except Exception as e:
        print(f"启动ngrok失败: {e}")
        print("请确保已安装ngrok并配置authtoken")

def main():
    print("=" * 60)
    print("音频实验 Web 服务器 (ngrok版本)")
    print("=" * 60)
    
    # 检查必要文件
    if not os.path.exists('stilumi_web.py'):
        print("✗ 缺少 stilumi_web.py 文件")
        sys.exit(1)
    
    print("正在启动服务器和ngrok隧道...")
    print("请稍等...")
    
    try:
        # 在新线程中启动Web服务器
        web_thread = threading.Thread(target=start_web_server, daemon=True)
        web_thread.start()
        
        # 等待Web服务器启动
        time.sleep(3)
        
        # 启动ngrok隧道
        start_ngrok_tunnel()
        
    except KeyboardInterrupt:
        print("\n服务器已停止")
    except Exception as e:
        print(f"启动失败: {e}")

if __name__ == "__main__":
    main()
