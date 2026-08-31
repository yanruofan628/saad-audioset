"""
快速检查回归结果中的异常值
"""
import pandas as pd
import numpy as np

# 读取结果文件
results_file = r"D:\D\research\audioset下载\download_audioset-master\download_audioset-master\wav2vec_regression_results\wav2vec_model_statistics.csv"

try:
    df = pd.read_csv(results_file)
    
    print("=== 回归结果诊断 ===\n")
    print(f"总模型数: {len(df)}\n")
    
    # 检查R²异常
    print("R²值检查:")
    print(df[['feature_name', 'rsquared', 'rsquared_adj']])
    
    # 检查异常R²
    abnormal_r2 = df[df['rsquared'] > 1.0]
    if len(abnormal_r2) > 0:
        print(f"\n⚠️  发现异常R² > 1.0 的模型:")
        print(abnormal_r2[['feature_name', 'rsquared', 'rsquared_adj', 'coefficient', 'coefficient_pvalue']])
    
    # 检查特征值
    matched_file = r"D:\D\research\audioset下载\download_audioset-master\download_audioset-master\wav2vec_regression_results\matched_data.csv"
    matched_df = pd.read_csv(matched_file)
    
    print("\n=== 特征值统计 ===")
    for feature in ['mse_max_diff', 'representation_change_max_diff']:
        if feature in matched_df.columns:
            print(f"\n{feature}:")
            print(f"  均值: {matched_df[feature].mean():.6f}")
            print(f"  标准差: {matched_df[feature].std():.6f}")
            print(f"  最小值: {matched_df[feature].min():.6f}")
            print(f"  最大值: {matched_df[feature].max():.6f}")
            print(f"  缺失值: {matched_df[feature].isna().sum()}")
            
            # 检查是否有极端值
            q99 = matched_df[feature].quantile(0.99)
            q01 = matched_df[feature].quantile(0.01)
            print(f"  99%分位数: {q99:.6f}")
            print(f"  1%分位数: {q01:.6f}")
            
            extreme = matched_df[(matched_df[feature] > q99 * 10) | (matched_df[feature] < q01 * 10)]
            if len(extreme) > 0:
                print(f"  ⚠️  发现极端值: {len(extreme)} 个")
                print(extreme[[feature, 'probability']].head())

except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()

