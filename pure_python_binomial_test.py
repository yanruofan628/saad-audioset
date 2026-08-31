import os
import csv
import math

class PurePythonBinomialTestAnalyzer:
    """纯Python二项假设检验分析器"""
    
    def __init__(self):
        """初始化分析器"""
        self.all_selections = []  # 存储所有选择数据
        self.participant_data = []  # 存储每个被试者的数据
        
    def read_csv_files(self, csv_folder):
        """
        读取CSV文件夹中的所有CSV文件
        
        参数:
        csv_folder: CSV文件夹路径
        
        返回:
        csv_data: CSV数据列表
        """
        print(f"正在读取CSV文件夹: {csv_folder}")
        
        csv_files = [
            "aiwenkai_1.csv", "aiwenkai2.csv", "lironghua.csv", "lironghua2.csv",
            "mayunmiao_1.csv", "mayunmiao2.csv", "ShangZiyang.csv", "ShangZiyang1.csv",
            "wjy1.csv", "wjy_2.csv"
        ]
        
        csv_data = []
        
        for filename in csv_files:
            file_path = os.path.join(csv_folder, filename)
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        rows = list(reader)
                    
                    print(f"成功读取: {filename}, 数据行数: {len(rows)}")
                    
                    # 提取选择数据
                    selections = [row['选择'] for row in rows]
                    participant_name = filename.replace('.csv', '')
                    
                    # 转换为数值：left=0, right=1
                    numeric_selections = [1 if s == 'right' else 0 for s in selections]
                    
                    csv_data.append({
                        'participant': participant_name,
                        'selections': numeric_selections,
                        'left_count': selections.count('left'),
                        'right_count': selections.count('right'),
                        'total': len(selections)
                    })
                    
                except Exception as e:
                    print(f"读取文件失败 {filename}: {e}")
            else:
                print(f"文件不存在: {filename}")
        
        print(f"成功读取 {len(csv_data)} 个CSV文件")
        return csv_data
    
    def parse_txt_file(self, file_path):
        """
        解析TXT文件，提取response数据
        
        参数:
        file_path: TXT文件路径
        
        返回:
        selections: 选择数据列表
        """
        print(f"正在解析TXT文件: {file_path}")
        
        selections = []
        # 尝试不同的编码方式
        encodings = ['utf-8', 'utf-16', 'gbk', 'gb2312', 'latin-1']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    lines = f.readlines()
                print(f"成功使用编码: {encoding}")
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError(f"无法使用任何编码读取文件: {file_path}")
        
        # 找到trial数据开始的位置
        for i, line in enumerate(lines):
            if line.strip().startswith('Level: 3') and '*** LogFrame Start ***' in lines[i+1]:
                # 这是一个trial的开始
                trial_data = {}
                
                # 解析trial信息
                for j in range(i+1, min(i+20, len(lines))):
                    current_line = lines[j].strip()
                    
                    if 'ImageDisplay1.RESP:' in current_line:
                        response = int(current_line.split('ImageDisplay1.RESP:')[1].strip())
                        selections.append(response)
                        break
                    elif '*** LogFrame End ***' in current_line:
                        break
        
        print(f"成功解析 {len(selections)} 个trial")
        return selections
    
    def read_txt_files(self, txt_files):
        """
        读取TXT文件列表
        
        参数:
        txt_files: TXT文件路径列表
        
        返回:
        txt_data: TXT数据列表
        """
        print(f"正在读取 {len(txt_files)} 个TXT文件")
        
        txt_data = []
        
        for file_path in txt_files:
            if os.path.exists(file_path):
                try:
                    selections = self.parse_txt_file(file_path)
                    
                    # 转换为数值：1=left=0, 2=right=1
                    numeric_selections = [s - 1 for s in selections]  # 1->0, 2->1
                    
                    participant_name = os.path.basename(os.path.dirname(file_path))
                    
                    txt_data.append({
                        'participant': participant_name,
                        'selections': numeric_selections,
                        'left_count': numeric_selections.count(0),
                        'right_count': numeric_selections.count(1),
                        'total': len(numeric_selections)
                    })
                    
                except Exception as e:
                    print(f"读取TXT文件失败 {file_path}: {e}")
            else:
                print(f"TXT文件不存在: {file_path}")
        
        print(f"成功读取 {len(txt_data)} 个TXT文件")
        return txt_data
    
    def combine_all_data(self, csv_data, txt_data):
        """
        合并所有数据
        
        参数:
        csv_data: CSV数据列表
        txt_data: TXT数据列表
        
        返回:
        combined_data: 合并后的数据
        """
        print("\n=== 合并所有数据 ===")
        
        all_data = csv_data + txt_data
        
        # 统计总体数据
        total_left = sum([d['left_count'] for d in all_data])
        total_right = sum([d['right_count'] for d in all_data])
        total_selections = sum([d['total'] for d in all_data])
        
        # 合并所有选择数据
        all_selections = []
        for data in all_data:
            all_selections.extend(data['selections'])
        
        combined_data = {
            'participants': all_data,
            'total_left': total_left,
            'total_right': total_right,
            'total_selections': total_selections,
            'all_selections': all_selections,
            'left_proportion': total_left / total_selections if total_selections > 0 else 0,
            'right_proportion': total_right / total_selections if total_selections > 0 else 0
        }
        
        print(f"总被试者数: {len(all_data)}")
        print(f"总选择次数: {total_selections}")
        print(f"选择left次数: {total_left}")
        print(f"选择right次数: {total_right}")
        print(f"left比例: {combined_data['left_proportion']:.4f}")
        print(f"right比例: {combined_data['right_proportion']:.4f}")
        
        return combined_data
    
    def log_binomial_coefficient(self, n, k):
        """计算二项式系数的对数 log(C(n,k))"""
        if k > n or k < 0:
            return float('-inf')
        if k == 0 or k == n:
            return 0.0
        
        # 使用斯特林公式的近似：log(C(n,k)) ≈ log(n!) - log(k!) - log((n-k)!)
        # 但为了避免计算大阶乘，我们使用更稳定的方法
        log_coeff = 0.0
        for i in range(k):
            log_coeff += math.log(n - i) - math.log(i + 1)
        return log_coeff
    
    def log_binomial_pmf(self, n, k, p):
        """计算二项分布概率质量函数的对数 log(P(X = k))"""
        if k < 0 or k > n:
            return float('-inf')
        
        # log(P(X = k)) = log(C(n,k)) + k*log(p) + (n-k)*log(1-p)
        log_coeff = self.log_binomial_coefficient(n, k)
        log_prob = log_coeff + k * math.log(p) + (n - k) * math.log(1 - p)
        return log_prob
    
    def binomial_pmf(self, n, k, p):
        """计算二项分布概率质量函数"""
        log_prob = self.log_binomial_pmf(n, k, p)
        if log_prob == float('-inf'):
            return 0.0
        return math.exp(log_prob)
    
    def binomial_cdf(self, n, k, p):
        """计算二项分布累积分布函数 P(X <= k)"""
        # 对于大样本，使用正态近似
        if n > 100:
            # 使用正态近似
            mean = n * p
            variance = n * p * (1 - p)
            std_dev = math.sqrt(variance)
            
            # 连续性修正
            z = (k + 0.5 - mean) / std_dev
            # 使用标准正态分布的近似
            if z < -6:
                return 0.0
            elif z > 6:
                return 1.0
            else:
                # 简单的正态分布CDF近似
                return 0.5 * (1 + math.erf(z / math.sqrt(2)))
        else:
            # 小样本时使用精确计算
            cdf = 0.0
            for i in range(k + 1):
                cdf += self.binomial_pmf(n, i, p)
            return cdf
    
    def binomial_test(self, n, k, p):
        """
        执行二项检验
        
        参数:
        n: 总试验次数
        k: 成功次数
        p: 假设概率
        
        返回:
        test_results: 检验结果
        """
        # 计算观察到的概率
        observed_p = k / n
        
        # 双尾检验：P(X <= k) + P(X >= n-k+1) 如果k <= n/2
        if k <= n / 2:
            p_value_two_tailed = 2 * self.binomial_cdf(n, k, p)
        else:
            p_value_two_tailed = 2 * (1 - self.binomial_cdf(n, k - 1, p))
        
        # 右尾检验：P(X >= k)
        p_value_right_tailed = 1 - self.binomial_cdf(n, k - 1, p)
        
        # 左尾检验：P(X <= k)
        p_value_left_tailed = self.binomial_cdf(n, k, p)
        
        # 计算置信区间（使用正态近似）
        alpha = 0.05
        z_alpha_half = 1.96  # 95%置信区间的临界值
        
        # Wilson置信区间
        p_hat = k / n
        n_total = n
        ci_lower = (p_hat + z_alpha_half**2/(2*n_total) - z_alpha_half * math.sqrt((p_hat*(1-p_hat) + z_alpha_half**2/(4*n_total))/n_total)) / (1 + z_alpha_half**2/n_total)
        ci_upper = (p_hat + z_alpha_half**2/(2*n_total) + z_alpha_half * math.sqrt((p_hat*(1-p_hat) + z_alpha_half**2/(4*n_total))/n_total)) / (1 + z_alpha_half**2/n_total)
        
        # 确保置信区间在[0,1]范围内
        ci_lower = max(0, ci_lower)
        ci_upper = min(1, ci_upper)
        
        test_results = {
            'n_trials': n,
            'n_right': k,
            'n_left': n - k,
            'observed_proportion': observed_p,
            'expected_proportion': p,
            'p_value_two_tailed': p_value_two_tailed,
            'p_value_right_tailed': p_value_right_tailed,
            'p_value_left_tailed': p_value_left_tailed,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'significant_two_tailed': p_value_two_tailed < 0.05,
            'significant_right_tailed': p_value_right_tailed < 0.05,
            'significant_left_tailed': p_value_left_tailed < 0.05
        }
        
        return test_results
    
    def perform_binomial_test(self, combined_data):
        """
        执行二项假设检验
        
        参数:
        combined_data: 合并后的数据
        
        返回:
        test_results: 检验结果
        """
        print("\n=== 执行二项假设检验 ===")
        
        n = combined_data['total_selections']  # 总试验次数
        k = combined_data['total_right']  # 成功次数（选择right）
        p = 0.5  # 假设概率
        
        # 执行二项检验
        test_results = self.binomial_test(n, k, p)
        
        print(f"二项检验结果:")
        print(f"  总试验次数: {n}")
        print(f"  选择right次数: {k}")
        print(f"  选择left次数: {n-k}")
        print(f"  观察到的right比例: {test_results['observed_proportion']:.4f}")
        print(f"  期望比例: {test_results['expected_proportion']:.4f}")
        print(f"  双尾检验p值: {test_results['p_value_two_tailed']:.6f}")
        print(f"  右尾检验p值: {test_results['p_value_right_tailed']:.6f}")
        print(f"  左尾检验p值: {test_results['p_value_left_tailed']:.6f}")
        print(f"  95%置信区间: [{test_results['ci_lower']:.4f}, {test_results['ci_upper']:.4f}]")
        
        if test_results['significant_two_tailed']:
            print(f"  结论: 拒绝H0，选择比例显著偏离0.5")
        else:
            print(f"  结论: 不拒绝H0，选择比例不显著偏离0.5")
        
        return test_results
    
    def save_results(self, combined_data, test_results, output_dir):
        """
        保存分析结果
        
        参数:
        combined_data: 合并后的数据
        test_results: 检验结果
        output_dir: 输出目录
        """
        print("\n=== 保存分析结果 ===")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存详细结果CSV
        csv_path = os.path.join(output_dir, 'binomial_test_detailed_results.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['participant', 'left_count', 'right_count', 'total', 'right_proportion'])
            
            for participant_data in combined_data['participants']:
                writer.writerow([
                    participant_data['participant'],
                    participant_data['left_count'],
                    participant_data['right_count'],
                    participant_data['total'],
                    participant_data['right_count'] / participant_data['total']
                ])
        
        # 保存汇总报告
        report_path = os.path.join(output_dir, 'binomial_test_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("二项假设检验分析报告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"总被试者数: {len(combined_data['participants'])}\n")
            f.write(f"总试验次数: {test_results['n_trials']}\n")
            f.write(f"Right选择次数: {test_results['n_right']}\n")
            f.write(f"Left选择次数: {test_results['n_left']}\n\n")
            
            f.write("统计检验结果:\n")
            f.write("-" * 30 + "\n")
            f.write(f"观察到的Right比例: {test_results['observed_proportion']:.4f}\n")
            f.write(f"期望比例: {test_results['expected_proportion']:.4f}\n")
            f.write(f"双尾检验p值: {test_results['p_value_two_tailed']:.6f}\n")
            f.write(f"右尾检验p值: {test_results['p_value_right_tailed']:.6f}\n")
            f.write(f"左尾检验p值: {test_results['p_value_left_tailed']:.6f}\n")
            f.write(f"95%置信区间: [{test_results['ci_lower']:.4f}, {test_results['ci_upper']:.4f}]\n\n")
            
            f.write("结论:\n")
            if test_results['significant_two_tailed']:
                f.write("拒绝H0，选择比例显著偏离0.5\n")
            else:
                f.write("不拒绝H0，选择比例不显著偏离0.5\n")
            
            f.write("\n详细被试者数据:\n")
            f.write("-" * 30 + "\n")
            for participant_data in combined_data['participants']:
                f.write(f"{participant_data['participant']}: Left={participant_data['left_count']}, Right={participant_data['right_count']}, 总计={participant_data['total']}, Right比例={participant_data['right_count']/participant_data['total']:.4f}\n")
        
        print(f"分析结果已保存:")
        print(f"  详细结果CSV: {csv_path}")
        print(f"  汇总报告: {report_path}")
    
    def analyze_all_data(self, csv_folder, txt_files, output_dir):
        """
        分析所有数据
        
        参数:
        csv_folder: CSV文件夹路径
        txt_files: TXT文件路径列表
        output_dir: 输出目录
        """
        print("=== 开始二项假设检验分析 ===")
        
        # 读取数据
        csv_data = self.read_csv_files(csv_folder)
        txt_data = self.read_txt_files(txt_files)
        
        # 合并数据
        combined_data = self.combine_all_data(csv_data, txt_data)
        
        # 执行二项检验
        test_results = self.perform_binomial_test(combined_data)
        
        # 保存结果
        self.save_results(combined_data, test_results, output_dir)
        
        print(f"\n分析完成！结果保存到: {output_dir}")
        
        return combined_data, test_results


def main():
    """主函数"""
    print("=== 二项假设检验分析工具 ===")
    
    # 设置文件路径
    csv_folder = r"D:\D\research\audioset下载\experiment_output2"
    
    txt_files = [
        r"D:\D\research\数据采集\liyanchen_1_20251010_090338.mff\benchmark_1_10-1-1.txt",
        r"D:\D\research\数据采集\liyanchen_2_20251010_094603.mff\benchmark_1_10-1-2.txt",
        r"D:\D\research\数据采集\shimin_s1_20251010_115112.mff\benchmark_1_10-1-1.txt",
        r"D:\D\research\数据采集\shimin_s2_20251010_110715.mff\benchmark_1_10-1-2.txt",
        r"D:\D\research\数据采集\jiachen1017_1_20251017_041306.mff\benchmark_1_10-1-2.txt",
        r"D:\D\research\数据采集\jiachen1017_2_20251017_050037.mff\benchmark_1_10-1-2.txt"
    ]
    
    output_dir = r"D:\D\research\audioset下载\binomial_test_results"
    
    # 创建分析器
    analyzer = PurePythonBinomialTestAnalyzer()
    
    # 分析所有数据
    try:
        combined_data, test_results = analyzer.analyze_all_data(csv_folder, txt_files, output_dir)
        print("\n分析成功完成！")
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
