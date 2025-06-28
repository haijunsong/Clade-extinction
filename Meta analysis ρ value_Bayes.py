import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as stats
import os
import pymc as pm
import arviz as az

# Specify the TkAgg backend to avoid issues with displaying plots
# Ensure that 'tkinter' is installed in your environment
matplotlib.use('TkAgg')

# Configure Matplotlib to support English fonts and handle negative numbers
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False


def fisher_z_transform(rho):
    """Perform Fisher's Z transformation on correlation coefficients."""
    # Clip rho to avoid values exactly at -1 or 1
    rho = np.clip(rho, -0.999999, 0.999999)
    return np.arctanh(rho)


def fisher_z_inverse(z):
    """Inverse Fisher's Z transformation to obtain correlation coefficients."""
    # Clip z to avoid numerical overflow
    z = np.clip(z, -20, 20)
    return (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)


def derSimonian_Laird(z, se_z):
    """
    Estimate tau² using the DerSimonian and Laird method for random-effects models.

    Parameters:
    - z: Transformed correlation coefficients (Fisher's Z)
    - se_z: Standard errors of the transformed correlation coefficients

    Returns:
    - tau2: Estimated tau squared
    - Q: Heterogeneity statistic
    - df: Degrees of freedom
    """
    k = len(z)
    w_fixed = 1 / (se_z ** 2)
    z_fixed = np.sum(w_fixed * z) / np.sum(w_fixed)
    Q = np.sum(w_fixed * (z - z_fixed) ** 2)
    df = k - 1
    C = np.sum(w_fixed) - (np.sum(w_fixed ** 2) / np.sum(w_fixed))
    tau2 = max(0, (Q - df) / C)
    return tau2, Q, df


def create_forest_plot(variable, study_labels, rho_values, ci_low_individual, ci_high_individual,
                       rho_random, ci_low_rho, ci_high_rho, rho_bayes, ci_low_bayes, ci_high_bayes,
                       output_path):
    """
    Create and save a forest plot.

    Parameters:
    - variable: Name of the variable being analyzed
    - study_labels: List of study names
    - rho_values: List of individual rho values
    - ci_low_individual: List of lower bounds for individual studies
    - ci_high_individual: List of upper bounds for individual studies
    - rho_random: Combined rho from random-effects meta-analysis
    - ci_low_rho: Lower bound of the combined rho
    - ci_high_rho: Upper bound of the combined rho
    - rho_bayes: Combined rho from Bayesian meta-analysis
    - ci_low_bayes: Lower bound of the Bayesian combined rho
    - ci_high_bayes: Upper bound of the Bayesian combined rho
    - output_path: Path to save the forest plot
    """
    plt.figure(figsize=(8, max(6, len(study_labels) * 0.4)))

    # Y positions
    y_positions = np.arange(len(study_labels))

    # Plot individual studies
    plt.errorbar(rho_values, y_positions, xerr=[rho_values - ci_low_individual, ci_high_individual - rho_values],
                 fmt='o', color='black', ecolor='gray', elinewidth=1, capsize=3, label='Individual Studies')

    # Plot random-effects combined estimate
    plt.errorbar(rho_random, -1, xerr=[[rho_random - ci_low_rho], [ci_high_rho - rho_random]],
                 fmt='s', color='blue', ecolor='blue', elinewidth=2, capsize=5, label='Random Effects')

    # Plot Bayesian combined estimate
    plt.errorbar(rho_bayes, -2, xerr=[[rho_bayes - ci_low_bayes], [ci_high_bayes - rho_bayes]],
                 fmt='D', color='green', ecolor='green', elinewidth=2, capsize=5, label='Bayesian Meta-Analysis')

    # Scatter points for combined estimates
    plt.scatter(rho_random, -1, color='blue', marker='s')
    plt.scatter(rho_bayes, -2, color='green', marker='D')

    # Annotate studies
    plt.yticks(list(y_positions) + [-1, -2], list(study_labels) + ['Random Effects', 'Bayesian Meta-Analysis'])
    plt.axvline(0, color='red', linestyle='--')

    # Set limits
    plt.xlim(-1.1, 1.1)

    plt.xlabel("Correlation Coefficient (rho)")
    plt.title(f"Forest Plot for {variable}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def random_effects_meta_analysis(rho_values, n_values):
    """
    Perform a random-effects meta-analysis.

    Parameters:
    - rho_values: Array of Spearman's rho coefficients
    - n_values: Array of sample sizes

    Returns:
    - Dictionary containing meta-analysis results
    """
    # Apply Fisher's Z transformation
    z = fisher_z_transform(rho_values)
    # Calculate standard errors
    se_z = 1 / np.sqrt(n_values - 3)
    # Estimate tau squared
    tau2, Q, df = derSimonian_Laird(z, se_z)
    # Calculate weights
    w_random = 1 / (se_z ** 2 + tau2)
    # Combined effect
    z_random = np.sum(w_random * z) / np.sum(w_random)
    se_random = np.sqrt(1 / np.sum(w_random))
    # Confidence intervals
    ci_low_random = z_random - 1.96 * se_random
    ci_high_random = z_random + 1.96 * se_random
    # Inverse Fisher Z transformation
    rho_random = fisher_z_inverse(z_random)
    ci_low_rho = fisher_z_inverse(ci_low_random)
    ci_high_rho = fisher_z_inverse(ci_high_random)
    # p-value
    z_stat = z_random / se_random
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    # I^2 calculation
    I2 = max(0, ((Q - df) / Q)) * 100 if Q > df else 0

    results = {
        'rho_random': rho_random,
        'ci_low_rho': ci_low_rho,
        'ci_high_rho': ci_high_rho,
        'p_value': p_value,
        'Q': Q,
        'df': df,
        'Tau2': tau2,
        'I2': I2
    }

    return results


def get_hdi_bounds(mu_hdi):
    """
    Extract the lower and upper bounds from the HDI dataset.

    Parameters:
    - mu_hdi: xarray.Dataset containing HDI bounds

    Returns:
    - Tuple of (ci_low, ci_high)
    """
    lower = mu_hdi['mu'].sel(hdi='lower').item()
    # Attempt to get 'upper', if not available, try 'higher'
    if 'upper' in mu_hdi['mu']['hdi'].values:
        high = mu_hdi['mu'].sel(hdi='upper').item()
    elif 'higher' in mu_hdi['mu']['hdi'].values:
        high = mu_hdi['mu'].sel(hdi='higher').item()
    else:
        raise KeyError("HDI upper bound not found. Available bounds: " + ", ".join(mu_hdi['mu']['hdi'].values))
    return lower, high


def bayesian_meta_analysis(z, se_z, n_values):
    """
    Perform a Bayesian meta-analysis using PyMC.

    Parameters:
    - z: Transformed correlation coefficients (Fisher's Z)
    - se_z: Standard errors of the transformed correlation coefficients
    - n_values: Array of sample sizes

    Returns:
    - Dictionary containing Bayesian meta-analysis results
    - Trace object from PyMC sampling
    """
    with pm.Model() as model:
        # Hyperpriors
        mu = pm.Normal('mu', mu=0, sigma=10)
        tau = pm.HalfNormal('tau', sigma=10)
        # Study-specific effects
        theta = pm.Normal('theta', mu=mu, sigma=tau, shape=len(z))
        # Likelihood
        y = pm.Normal('y', mu=theta, sigma=se_z, observed=z)
        # Sampling with increased target_accept to reduce divergences
        trace = pm.sample(2000, tune=1000, target_accept=0.99, return_inferencedata=True)

    # 打印 trace.posterior 的结构以调试
    print("Trace posterior structure:")
    print(trace.posterior)

    # Posterior summaries for 'mu'
    mu_posterior_mean = trace.posterior['mu'].mean().item()
    print(f"Posterior mean of mu: {mu_posterior_mean}")

    # 计算 mu 的 95% 高密度区间（HDI）
    mu_hdi = az.hdi(trace.posterior['mu'], hdi_prob=0.95)
    print("mu_hdi structure:")
    print(mu_hdi)

    # 提取 HDI 的上下边界
    try:
        ci_low_bayes, ci_high_bayes = get_hdi_bounds(mu_hdi)
    except KeyError as e:
        print(f"HDI 提取时发生 KeyError: {e}")
        raise
    except AttributeError as e:
        print(f"HDI 提取时发生 AttributeError: {e}")
        raise

    # 将 Fisher 的 Z 转换回 Pearson 的 r
    rho_bayes = fisher_z_inverse(mu_posterior_mean)
    rho_bayes_low = fisher_z_inverse(ci_low_bayes)
    rho_bayes_high = fisher_z_inverse(ci_high_bayes)

    # I^2 calculation for Bayesian
    tau_posterior = trace.posterior['tau'].values.flatten()  # Shape (8000,)
    tau2_posterior = tau_posterior ** 2  # Shape (8000,)
    variance_within = 1 / (n_values - 3)  # Shape (12,)

    # 扩展 tau2_posterior 和 variance_within 的形状以匹配
    I2_per_sample = tau2_posterior[:, None] / (tau2_posterior[:, None] + variance_within[None, :])  # Shape (8000,12)

    # 对每个样本的所有研究取平均
    I2_mean_per_sample = I2_per_sample.mean(axis=1)  # Shape (8000,)

    # 取所有样本的平均值，转换为百分比
    I2_bayes = I2_mean_per_sample.mean() * 100  # Scalar

    # p-value (two-tailed: probability that mu is different from zero)
    prob_mu_gt_0 = (trace.posterior['mu'] > 0).mean().item()
    prob_mu_lt_0 = (trace.posterior['mu'] < 0).mean().item()
    p_value_bayes = 2 * min(prob_mu_gt_0, prob_mu_lt_0)

    results = {
        'rho_bayes': rho_bayes,
        'ci_low_bayes': rho_bayes_low,
        'ci_high_bayes': rho_bayes_high,
        'p_value_bayes': p_value_bayes,
        'tau_bayes': tau_posterior.mean(),
        'I2_bayesian': I2_bayes
    }

    return results, trace


def main():
    # 假设数据保存在当前目录的 Excel 文件中
    # 修改为您的实际文件路径
    excel_path = r"F:\1-投稿\2022-clades灭绝\Pyrate\12 clades 2025-3\output1myr\Spearman correlation_results pyrate v0.3 for meta.xlsx"

    # 读取 Excel 文件中的数据
    # 假设文件中有三个表格：'Correlation Coefficients', 'P Values', 'Number'
    # 请确保工作表名称与 Excel 文件中的实际名称一致
    try:
        rho_df = pd.read_excel(excel_path, sheet_name='Correlation Coefficients')
        p_df = pd.read_excel(excel_path, sheet_name='P Values')
        n_df = pd.read_excel(excel_path, sheet_name='Number')
    except ValueError as ve:
        print(f"读取工作表时发生错误: {ve}")
        print("请确认工作表名称是否正确，并存在于 Excel 文件中。")
        return
    except FileNotFoundError:
        print(f"无法找到文件: {excel_path}")
        return
    except Exception as e:
        print(f"读取 Excel 文件时发生未预期的错误: {e}")
        return

    # 提取变量名（假设第一列为研究名称）
    variables = rho_df.columns[1:]
    study_names = rho_df.iloc[:, 0].tolist()

    meta_analysis_results = []

    # 创建保存森林图的目录
    plots_dir = os.path.join(os.path.dirname(excel_path), "plots")
    os.makedirs(plots_dir, exist_ok=True)

    for variable in variables:
        rho_values = rho_df[variable].values
        p_values = p_df[variable].values
        n_values = n_df[variable].values

        # 数据有效性检查
        valid_indices = (~np.isnan(rho_values)) & (~np.isnan(n_values)) & (np.abs(rho_values) < 1)
        valid_rho = rho_values[valid_indices]
        valid_n = n_values[valid_indices]
        valid_study_names = np.array(study_names)[valid_indices]

        if len(valid_rho) < 2:
            print(f"Variable '{variable}' has insufficient valid studies for meta-analysis. Skipping.")
            continue

        # Random-effects meta-analysis
        meta_results = random_effects_meta_analysis(valid_rho, valid_n)

        # Bayesian meta-analysis
        z = fisher_z_transform(valid_rho)
        se_z = 1 / np.sqrt(valid_n - 3)
        bayes_results, trace = bayesian_meta_analysis(z, se_z, valid_n)

        # Collect results
        result = {
            'Variable': variable,
            'Combined_rho_Frequentist': round(meta_results['rho_random'], 3),
            'CI_Lower_rho_Frequentist': round(meta_results['ci_low_rho'], 3),
            'CI_Upper_rho_Frequentist': round(meta_results['ci_high_rho'], 3),
            'p_value_Frequentist': round(meta_results['p_value'], 3),
            'Combined_rho_Bayesian': round(bayes_results['rho_bayes'], 3),
            'CI_Lower_rho_Bayesian': round(bayes_results['ci_low_bayes'], 3),
            'CI_Upper_rho_Bayesian': round(bayes_results['ci_high_bayes'], 3),
            'p_value_Bayesian': round(bayes_results['p_value_bayes'], 3),
            'Q': round(meta_results['Q'], 3),
            'df': meta_results['df'],
            'Tau2_Frequentist': round(meta_results['Tau2'], 3),
            'I2': round(meta_results['I2'], 2),
            'Tau_Bayesian': round(bayes_results['tau_bayes'], 3),
            'I2_Bayesian': round(bayes_results['I2_bayesian'], 2)
        }
        meta_analysis_results.append(result)

        # 创建森林图
        # 计算置信区间确保在 [-1, 1]
        ci_low_individual_z = z - 1.96 * se_z
        ci_high_individual_z = z + 1.96 * se_z
        ci_low_individual = fisher_z_inverse(ci_low_individual_z)
        ci_high_individual = fisher_z_inverse(ci_high_individual_z)
        # Clip to [-1, 1] to avoid invalid correlation coefficients
        ci_low_individual = np.clip(ci_low_individual, -1, 1)
        ci_high_individual = np.clip(ci_high_individual, -1, 1)

        create_forest_plot(
            variable=variable,
            study_labels=valid_study_names,
            rho_values=valid_rho,
            ci_low_individual=ci_low_individual,
            ci_high_individual=ci_high_individual,
            rho_random=meta_results['rho_random'],
            ci_low_rho=meta_results['ci_low_rho'],
            ci_high_rho=meta_results['ci_high_rho'],
            rho_bayes=bayes_results['rho_bayes'],
            ci_low_bayes=bayes_results['ci_low_bayes'],
            ci_high_bayes=bayes_results['ci_high_bayes'],
            output_path=os.path.join(plots_dir, f'forest_plot_{variable.replace(" ", "_")}.pdf')
        )
        print(f"Forest plot for '{variable}' saved.")

        # 可选：绘制和保存采样诊断图，以检查模型的收敛性
        # az.plot_trace(trace)
        # plt.savefig(os.path.join(plots_dir, f'trace_plot_{variable.replace(" ", "_")}.pdf'))
        # plt.close()
        #
        # az.plot_energy(trace)
        # plt.savefig(os.path.join(plots_dir, f'energy_plot_{variable.replace(" ", "_")}.pdf'))
        # plt.close()

    # 保存所有元分析结果到 Excel 文件
    if meta_analysis_results:
        results_df = pd.DataFrame(meta_analysis_results)
        # 重排列顺序
        results_df = results_df[[
            'Variable',
            'Combined_rho_Frequentist',
            'CI_Lower_rho_Frequentist',
            'CI_Upper_rho_Frequentist',
            'p_value_Frequentist',
            'Combined_rho_Bayesian',
            'CI_Lower_rho_Bayesian',
            'CI_Upper_rho_Bayesian',
            'p_value_Bayesian',
            'Q',
            'df',
            'Tau2_Frequentist',
            'I2',
            'Tau_Bayesian',
            'I2_Bayesian'
        ]]
        results_output_path = os.path.join(os.path.dirname(excel_path), "Meta_Analysis_Results.xlsx")
        results_df.to_excel(results_output_path, index=False)
        print(f"\nMeta-analysis results have been saved to: {results_output_path}")
    else:
        print("\nNo meta-analysis was performed.")


if __name__ == "__main__":
    main()