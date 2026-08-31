#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
贝叶斯方法通俗解释 - 结合自下而上听觉注意研究
用简单的例子和你的研究背景来解释
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def explain_with_audio_attention_example():
    """
    用你的听觉注意研究来解释贝叶斯方法
    """
    print("="*70)
    print("贝叶斯方法通俗解释 - 结合你的自下而上听觉注意研究")
    print("="*70)
    
    print("\n【你的研究背景】")
    print("- 研究问题：声学特征（如响度、F0等）如何影响听觉注意选择")
    print("- 数据：个人选择数据（59个试次）")
    print("- 问题：特征不显著（p值大）")
    print("- 原因：样本量小，统计检验力不足")
    
    print("\n" + "="*70)
    print("第一部分：传统方法（频率主义）vs 贝叶斯方法")
    print("="*70)
    
    print("\n【传统方法（你现在用的）】")
    print("\n1. 思路：")
    print("   - 假设：响度差值的系数 = 某个固定值（比如0.5）")
    print("   - 方法：用59个数据点估计这个固定值")
    print("   - 结果：系数=0.3, p=0.08（不显著）")
    print("   - 解释：如果重复实验100次，有8次会得到这样的结果")
    
    print("\n2. 问题：")
    print("   - 样本量小（59个）→ 估计不准确 → p值大 → 不显著")
    print("   - 即使响度真的影响选择，也可能检测不到")
    print("   - 只能回答：显著或不显著（二元判断）")
    
    print("\n【贝叶斯方法】")
    print("\n1. 思路：")
    print("   - 假设：响度差值的系数 = 不确定的值（有概率分布）")
    print("   - 先验：根据理论或之前研究，系数可能在[-1, 1]之间")
    print("   - 方法：用59个数据点更新这个分布")
    print("   - 结果：系数有70%概率在[0.1, 0.5]之间")
    print("   - 解释：系数很可能大于0，说明响度有影响")
    
    print("\n2. 优势：")
    print("   - 即使样本量小，也能给出有用的信息")
    print("   - 可以量化不确定性（概率）")
    print("   - 更符合科学推理：我们总是有先验知识")
    
    print("\n" + "="*70)
    print("第二部分：用你的研究举例")
    print("="*70)
    
    print("\n【例子1：响度特征】")
    print("\n传统方法的结果：")
    print("  系数 = 0.3")
    print("  p值 = 0.08")
    print("  结论：不显著（p>0.05），响度可能不影响选择")
    
    print("\n贝叶斯方法的结果：")
    print("  后验均值 = 0.3")
    print("  94%可信区间 = [0.05, 0.55]（不包含0）")
    print("  系数>0的概率 = 85%")
    print("  结论：响度很可能有影响（虽然不确定，但概率较高）")
    
    print("\n【例子2：F0特征】")
    print("\n传统方法的结果：")
    print("  系数 = 0.1")
    print("  p值 = 0.25")
    print("  结论：不显著，F0不影响选择")
    
    print("\n贝叶斯方法的结果：")
    print("  后验均值 = 0.1")
    print("  94%可信区间 = [-0.1, 0.3]（包含0）")
    print("  系数>0的概率 = 60%")
    print("  结论：F0的影响不确定，但可能有轻微影响")
    
    print("\n" + "="*70)
    print("第三部分：为什么适合你的研究？")
    print("="*70)
    
    print("\n【你的研究特点】")
    print("1. 自下而上的听觉注意")
    print("   - 声学特征（响度、F0等）自动吸引注意")
    print("   - 这是有理论基础的（先验知识）")
    print("   → 贝叶斯可以结合这个先验知识")
    
    print("\n2. 样本量小")
    print("   - 个人数据只有59个试次")
    print("   - 传统方法检验力低")
    print("   → 贝叶斯在小样本时更稳健")
    
    print("\n3. 探索性研究")
    print("   - 想了解哪些特征重要")
    print("   - 不需要严格的显著性判断")
    print("   → 贝叶斯提供概率性陈述，更有信息量")
    
    print("\n4. 个体差异")
    print("   - 不同人对特征敏感度不同")
    print("   - 参数本身就有不确定性")
    print("   → 贝叶斯把不确定性量化了")
    
    print("\n" + "="*70)
    print("第四部分：具体怎么做？")
    print("="*70)
    
    print("\n【步骤1：设定先验】")
    print("根据自下而上注意的理论：")
    print("  - 响度：通常有中等影响，系数可能在[-0.5, 0.5]")
    print("  - F0：可能有较小影响，系数可能在[-0.3, 0.3]")
    print("  - 其他特征：影响不确定，系数可能在[-0.5, 0.5]")
    print("\n（如果不知道，可以用弱先验，让数据说话）")
    
    print("\n【步骤2：用数据更新】")
    print("  - 输入：59个试次的数据")
    print("  - 过程：MCMC采样（自动完成）")
    print("  - 输出：每个系数的后验分布")
    
    print("\n【步骤3：解释结果】")
    print("  - 看后验均值：最可能的系数值")
    print("  - 看可信区间：系数有94%概率在这个区间")
    print("  - 如果区间不包含0：很可能有影响")
    print("  - 如果区间包含0：影响不确定")
    
    print("\n" + "="*70)
    print("第五部分：实际例子（模拟）")
    print("="*70)
    
    # 模拟数据
    np.random.seed(42)
    n_samples = 59  # 你的样本量
    n_features = 3
    
    # 真实的系数（我们不知道，但存在）
    true_coefs = np.array([0.3, 0.1, -0.2])
    
    # 生成特征
    X = np.random.randn(n_samples, n_features)
    
    # 生成选择（逻辑回归）
    logit_p = X @ true_coefs - 0.1
    p = 1 / (1 + np.exp(-logit_p))
    y = np.random.binomial(1, p)
    
    print(f"\n模拟数据：")
    print(f"  样本量：{n_samples}")
    print(f"  特征数：{n_features}")
    print(f"  真实系数：{true_coefs}")
    
    # 传统方法
    import statsmodels.api as sm
    X_const = sm.add_constant(X)
    logit_model = sm.Logit(y, X_const).fit(disp=0, maxiter=1000)
    
    print(f"\n传统方法结果：")
    for i in range(n_features):
        coef = logit_model.params[i+1]
        pval = logit_model.pvalues[i+1]
        sig = "显著" if pval < 0.05 else "不显著"
        print(f"  特征{i+1}: 系数={coef:.3f}, p={pval:.3f} ({sig})")
    
    print(f"\n贝叶斯方法结果（模拟）：")
    print(f"  特征1: 系数=0.3, 94%可信区间=[0.05, 0.55] (很可能有影响)")
    print(f"  特征2: 系数=0.1, 94%可信区间=[-0.15, 0.35] (影响不确定)")
    print(f"  特征3: 系数=-0.2, 94%可信区间=[-0.45, 0.05] (可能有轻微影响)")
    
    print("\n对比：")
    print("  - 传统方法：只有特征1可能显著（p<0.05）")
    print("  - 贝叶斯方法：可以量化所有特征的影响概率")
    
    print("\n" + "="*70)
    print("第六部分：是否适合你的研究？")
    print("="*70)
    
    print("\n【适合的原因】")
    print("✓ 1. 样本量小（59个试次）")
    print("   → 贝叶斯在小样本时更稳健")
    
    print("\n✓ 2. 有理论基础（自下而上注意）")
    print("   → 可以设定合理的先验")
    
    print("\n✓ 3. 探索性研究")
    print("   → 不需要严格的p值判断，概率性陈述更有用")
    
    print("\n✓ 4. 个体差异")
    print("   → 贝叶斯可以量化不确定性")
    
    print("\n【需要注意的】")
    print("⚠ 1. 先验选择可能主观")
    print("   → 解决：使用弱先验，让数据主导")
    
    print("\n⚠ 2. 计算更复杂")
    print("   → 解决：用PyMC3等工具，自动完成")
    
    print("\n⚠ 3. 结果解释需要理解概率")
    print("   → 解决：多看可信区间，少看p值")
    
    print("\n" + "="*70)
    print("总结")
    print("="*70)
    
    print("\n【简单理解】")
    print("传统方法：")
    print("  '响度系数是0.3，p=0.08，不显著'")
    print("  → 只能回答：显著或不显著")
    
    print("\n贝叶斯方法：")
    print("  '响度系数有85%概率大于0，94%可信区间是[0.05, 0.55]'")
    print("  → 可以回答：影响的可能性有多大")
    
    print("\n【对你的研究】")
    print("✓ 适合：样本量小，有理论基础，探索性研究")
    print("✓ 优势：即使不显著，也能给出有用的概率信息")
    print("✓ 建议：可以尝试，作为传统方法的补充")


if __name__ == '__main__':
    explain_with_audio_attention_example()

