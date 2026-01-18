import streamlit as st
import numpy as np
import pandas as pd
import hashlib
from statsmodels.stats.proportion import proportions_ztest, proportion_effectsize
from statsmodels.stats.power import NormalIndPower
import math

# --- 页面设置 ---
st.set_page_config(page_title="AB实验全功能教学沙盘", layout="wide")


# --- 核心哈希分流算法 ---
def get_group(user_id, layer_name, salt):
    input_str = f"{user_id}_{layer_name}_{salt}"
    hash_val = int(hashlib.md5(input_str.encode()).hexdigest(), 16)
    return 'A' if (hash_val % 100) < 50 else 'B'


# --- 核心算法：贝叶斯分析 ---
def run_bayesian_analysis(c_clicks, c_n, t_clicks, t_n):
    """
    使用 Beta 分布进行贝叶斯后验采样
    计算 B 组优于 A 组的概率以及期望损失
    """
    # 采用无信息先验 Beta(1,1)
    alpha_a, beta_a = 1 + c_clicks, 1 + c_n - c_clicks
    alpha_b, beta_b = 1 + t_clicks, 1 + t_n - t_clicks

    # 抽取 20,000 个样本模拟后验分布
    samples = 20000
    a_samples = np.random.beta(alpha_a, beta_a, samples)
    b_samples = np.random.beta(alpha_b, beta_b, samples)

    # 计算 B > A 的频率作为概率
    prob_b_better = (b_samples > a_samples).mean()
    # 计算期望损失：如果 B 实际比 A 差，选 B 平均会损失多少转化率
    expected_loss = np.maximum(a_samples - b_samples, 0).mean()

    return prob_b_better, expected_loss


# --- 侧边栏：上帝视角控制 ---
with st.sidebar:
    st.header("🛠️ 实验环境配置")
    dau = st.number_input("日活流量 (DAU)", value=1000, step=100)
    base_p = st.slider("基础转化率 (Baseline)", 0.01, 0.50, 0.10)

    st.divider()
    st.header("🏗️ 正交架构")
    salt_1 = st.text_input("Layer 1 盐值", "UI_EXP")
    salt_2 = st.text_input("Layer 2 盐值", "ALG_EXP")

    st.divider()
    st.header("👁️ 上帝视角 (真相)")
    true_lift = st.slider("设定的真实提升", -0.20, 0.20, 0.05, help="这是只有上帝知道的真相。")

# --- 头部教学引入 ---
st.title("🧪 A/B 实验全功能沙盘：从 P 值到贝叶斯")
st.markdown("""
本沙盘融合了**频率派显著性**与**贝叶斯决策信心**。
你可以观察在不同样本量下，两种统计流派如何对同一组数据给出不同的解读。
""")

# --- 第一阶段：排期预测 (保留原逻辑) ---
st.header("📅 第一阶段：排期预测 (Planning)")
mde_target = st.slider("目标灵敏度 (MDE %)", 0.01, 0.15, 0.05)

obj = NormalIndPower()
try:
    p2_mde = base_p * (1 + mde_target)
    es = proportion_effectsize(p2_mde, base_p)
    req_n_per_group = obj.solve_power(effect_size=es, alpha=0.05, power=0.8, ratio=1.0)
    req_n = math.ceil(req_n_per_group * 2)
    est_days = math.ceil(req_n / dau)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("所需总样本量", f"{req_n:,}")
        st.write(f"**教学点：** 为了看得清 {mde_target:.1%} 的微小改动，你需要积累足够的‘光线’（样本）。")
    with col2:
        st.metric("建议实验时长", f"{est_days} 天")
        st.write("**教学点：** 提前结束实验会大幅增加‘第一类错误’（虚假显著）的风险。")
except:
    st.error("计算失败，请调整参数。")

# --- 第二阶段：AA 实验自检 (保留原逻辑) ---
st.divider()
st.header("🛡️ 第二阶段：AA 实验 (System Check)")
if st.button("运行 AA 实验自检"):
    aa_n = min(req_n, 5000)
    aa_users = [{"ID": f"u_{i}", "G": get_group(f"u_{i}", "L1", salt_1)} for i in range(aa_n)]
    df_aa = pd.DataFrame(aa_users)
    df_aa['C'] = df_aa['G'].apply(lambda x: np.random.binomial(1, base_p))
    aa_res = df_aa.groupby('G')['C'].agg(['count', 'sum'])
    z_aa, p_aa = proportions_ztest(aa_res['sum'][::-1], aa_res['count'][::-1])

    if p_aa < 0.05:
        st.error(f"🚨 AA失败 (P={p_aa:.4f})：分流器不公平！此时结论不可信。")
    else:
        st.success(f"✅ AA通过 (P={p_aa:.4f})：分流公平，实验环境纯净。")

# --- 第三阶段：实时运行与审计 ---
st.divider()
st.header("📊 第三阶段：实时运行与双引擎决策")
days_run = st.slider("实验已运行天数", 1, max(30, est_days + 7), min(7, est_days))
current_n = dau * days_run

# 数据模拟
user_data = []
for i in range(current_n):
    uid = f"user_{i}"
    g1 = get_group(uid, "L1", salt_1)
    user_data.append({"ID": uid, "G": g1})
df_ab = pd.DataFrame(user_data)
df_ab['Click'] = df_ab['G'].apply(lambda x: np.random.binomial(1, base_p * (1 + true_lift) if x == 'B' else base_p))

res = df_ab.groupby('G')['Click'].agg(['count', 'sum'])
ca_n, ca_s = res.loc['A', 'count'], res.loc['A', 'sum']
cb_n, cb_s = res.loc['B', 'count'], res.loc['B', 'sum']

# --- 核心：频率派 vs 贝叶斯 对比面板 ---
st.subheader("⚖️ 决策博弈：谁更可信？")
col_freq, col_bayes = st.columns(2)

# 频率派计算
z, p_val = proportions_ztest([cb_s, ca_s], [cb_n, ca_n])
with col_freq:
    st.info("### 频率派 (Frequentist)")
    st.metric("P-value", f"{p_val:.4f}")
    if p_val < 0.05:
        st.success("✅ 结论显著！可以拒绝原假设。")
    else:
        st.error("❌ 不显著。差异可能来自随机波动。")
    st.write("**教学点：** P值回答的是‘意外程度’。")

# 贝叶斯派计算
prob_b_wins, exp_loss = run_bayesian_analysis(ca_s, ca_n, cb_s, cb_n)
with col_bayes:
    st.info("### 贝叶斯派 (Bayesian)")
    st.metric("B 组胜出概率", f"{prob_b_wins:.2%}")
    st.metric("选择 B 的风险 (期望损失)", f"{exp_loss:.6f}")
    if prob_b_wins > 0.95:
        st.success("✅ 信心充足！B 组大概率真的更好。")
    else:
        st.warning("⚠️ 信心尚早。虽然可能占优，但仍有风险。")
    st.write("**教学点：** 贝叶斯回答的是‘下注信心’。")

# --- 核心：审计评分 (保留并融入贝叶斯概念) ---
st.divider()
st.subheader("🛡️ 实验可信度质量审计")
try:
    raw_p = obj.power(effect_size=es, nobs1=current_n / 2, alpha=0.05, ratio=1.0)
    curr_power = float(raw_p.power) if hasattr(raw_p, 'power') else float(raw_p)
except:
    curr_power = 0.0

score = 0
audit_log = []
if current_n >= req_n:
    score += 40
    audit_log.append("✅ **样本充足**：已跑满排期，规避了‘偷看问题’。")
else:
    audit_log.append(f"❌ **样本不足**：仅完成预测量的 {current_n / req_n:.1%}。")

if curr_power >= 0.8:
    score += 30
    audit_log.append("✅ **功效充足**：探测器功率达标，观察值趋于真相。")
else:
    audit_log.append(f"❌ **低功效警告**：Power 仅 {curr_power:.1%}。小心‘赢家诅咒’。")

if p_val < 0.05:
    score += 30
    audit_log.append("✅ **频率派显著**：证据强度达标。")

aud1, aud2 = st.columns([1, 2])
with aud1:
    st.metric("实验可信度总分", f"{score}/100")
    if score == 100:
        st.success("💎 极高可信度：直接下结论")
    elif score >= 70:
        st.warning("⚠️ 中度可信：建议结合业务风险决策")
    else:
        st.error("🚫 低可信度：拒绝决策")

with aud2:
    for log in audit_log: st.write(log)

# 指标看板
st.divider()
c1, c2, c3 = st.columns(3)
obs_lift = (cb_s / cb_n) / (ca_s / ca_n) - 1
c1.metric("观察到的提升", f"{obs_lift:.2%}", delta=f"{(obs_lift - true_lift):.2%} (偏离真值)")
c2.metric("当前样本总量", f"{current_n:,}")
c3.metric("统计功效 (Power)", f"{curr_power:.2%}")

# --- 教学笔记 (保留并新增) ---
st.divider()
with st.expander("📖 实验背后的科学逻辑 (教学必读)"):
    st.markdown("""
    1. **第一/二类错误**：Alpha 是“咋呼”（没效说有效），Beta 是“木讷”（有效没测出来）。
    2. **赢家诅咒**：在 Power 极低时，如果刚好显著，你看到的提升往往是被随机误差夸大后的“虚假繁荣”。
    3. **频率派 vs 贝叶斯**：
        - **频率派**更严谨，只有证据确凿（P < 0.05）才说话。
        - **贝叶斯**更直观，它告诉你“如果你选 B，有多少概率会赢，如果输了会损失多少”。在小样本决策时，贝叶斯的“期望损失”比 P 值更有业务意义。
    4. **正交性**：依靠哈希盐值（Salt）重新洗牌。你可以试着修改左侧的 Layer 2 盐值，观察它是否干扰了 Layer 1 的结果。
    """)