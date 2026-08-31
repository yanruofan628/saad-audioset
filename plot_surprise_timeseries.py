"""
绘制Wav2Vec2惊喜度时间序列图，用于检查数据质量
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

def load_wav2vec_results(output_dir):
    """加载wav2vec结果"""
    results = []
    possible_prefixes = ['nn_main', 'nn_sub', 'main', 'sub', 'nn', 'rn']
    
    for prefix in possible_prefixes:
        json_path = os.path.join(output_dir, f'wav2vec_surprise_raw_results_{prefix}.json')
        if os.path.exists(json_path):
            print(f"加载 {prefix} 结果: {json_path}")
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                results.extend(data)
                print(f"  加载了 {len(data)} 个结果")
    
    print(f"\n总共加载 {len(results)} 个音频对的结果")
    return results

def plot_surprise_timeseries(result, output_dir, n_samples=5):
    """
    绘制单个音频对的惊喜度时间序列
    
    参数:
        result: wav2vec结果字典
        output_dir: 输出目录
        n_samples: 要绘制的音频对数量
    """
    audio_file = result.get('audio_file', 'Unknown')
    methods_results = result.get('methods_results', {})
    
    if not methods_results:
        print(f"警告: {audio_file} 没有methods_results")
        return
    
    # 创建图形，每个方法一个子图
    methods = ['mse', 'cosine', 'representation_change', 'entropy']
    method_names = {
        'mse': 'MSE Loss',
        'cosine': 'Cosine Similarity',
        'representation_change': 'Representation Change',
        'entropy': 'Entropy'
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'惊喜度时间序列: {os.path.basename(audio_file)}', fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    frame_duration_ms = 25  # 假设每帧25ms
    methods_found = []
    
    for idx, method in enumerate(methods):
        if method not in methods_results:
            axes[idx].text(0.5, 0.5, f'{method_names.get(method, method)}\n数据缺失', 
                          ha='center', va='center', fontsize=14)
            axes[idx].set_title(method_names.get(method, method), fontsize=12)
            continue
        
        methods_found.append(method)
        method_result = methods_results[method]
        left_surprise = method_result.get('left_surprise', [])
        right_surprise = method_result.get('right_surprise', [])
        
        if not left_surprise or not right_surprise:
            axes[idx].text(0.5, 0.5, f'{method_names.get(method, method)}\n数据为空', 
                          ha='center', va='center', fontsize=14)
            axes[idx].set_title(method_names.get(method, method), fontsize=12)
            continue
        
        # 转换为numpy数组
        left_arr = np.array(left_surprise)
        right_arr = np.array(right_surprise)
        
        # 确保长度一致
        min_len = min(len(left_arr), len(right_arr))
        left_arr = left_arr[:min_len]
        right_arr = right_arr[:min_len]
        
        # 计算时间轴（秒）
        time_seconds = np.arange(min_len) * frame_duration_ms / 1000.0
        
        # 绘制左右声道
        axes[idx].plot(time_seconds, left_arr, label='左声道', color='blue', alpha=0.7, linewidth=1.5)
        axes[idx].plot(time_seconds, right_arr, label='右声道', color='red', alpha=0.7, linewidth=1.5)
        
        # 绘制差值（如果可能）
        diff = left_arr - right_arr
        axes2 = axes[idx].twinx()
        axes2.plot(time_seconds, diff, label='差值 (左-右)', color='green', alpha=0.5, linewidth=1, linestyle='--')
        axes2.set_ylabel('差值', fontsize=10, color='green')
        axes2.tick_params(axis='y', labelcolor='green')
        
        # 计算统计信息
        left_sum = np.sum(left_arr)
        right_sum = np.sum(right_arr)
        sum_diff = left_sum - right_sum
        left_max = np.max(left_arr)
        right_max = np.max(right_arr)
        max_diff = left_max - right_max
        
        # 添加统计信息到标题
        title = f'{method_names.get(method, method)}\n'
        title += f'Sum: L={left_sum:.2f}, R={right_sum:.2f}, Diff={sum_diff:.2f} | '
        title += f'Max: L={left_max:.2f}, R={right_max:.2f}, Diff={max_diff:.2f}'
        axes[idx].set_title(title, fontsize=10)
        
        axes[idx].set_xlabel('时间 (秒)', fontsize=11)
        axes[idx].set_ylabel('惊喜度', fontsize=11)
        axes[idx].legend(loc='upper left', fontsize=9)
        axes[idx].grid(True, alpha=0.3)
        axes2.legend(loc='upper right', fontsize=9)
    
    # 如果所有方法都缺失，跳过
    if not methods_found:
        plt.close(fig)
        return
    
    # 保存图片
    safe_filename = audio_file.replace('/', '_').replace('\\', '_').replace(':', '_')
    if len(safe_filename) > 100:
        safe_filename = safe_filename[:100]
    output_path = os.path.join(output_dir, f'surprise_timeseries_{safe_filename}.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [已保存] {output_path}")

def main():
    """主函数"""
    print("=== 绘制Wav2Vec2惊喜度时间序列 ===\n")
    
    # 设置路径
    wav2vec_output_dir = r"D:\D\research\audioset下载\wav2vec_surprise_heatmaps"
    output_dir = os.path.join(os.getcwd(), 'surprise_timeseries_plots')
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载结果
    results = load_wav2vec_results(wav2vec_output_dir)
    
    if not results:
        print("错误: 未找到任何wav2vec结果")
        print(f"请检查目录: {wav2vec_output_dir}")
        return
    
    # 选择前几个音频对进行可视化（或者随机选择）
    n_samples = 5  # 绘制前5个
    selected_results = results[:n_samples]
    
    print(f"\n开始绘制 {len(selected_results)} 个音频对的时间序列...")
    
    for i, result in enumerate(selected_results):
        audio_file = result.get('audio_file', 'Unknown')
        print(f"\n[{i+1}/{len(selected_results)}] 处理: {os.path.basename(audio_file)}")
        plot_surprise_timeseries(result, output_dir)
    
    print(f"\n=== 完成 ===")
    print(f"所有图片已保存到: {output_dir}")
    print(f"共绘制 {len(selected_results)} 个音频对的时间序列图")

if __name__ == '__main__':
    main()

