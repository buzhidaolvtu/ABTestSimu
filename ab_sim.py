import streamlit as st
import numpy as np
import pandas as pd
import hashlib
from statsmodels.stats.proportion import proportions_ztest, proportion_effectsize
from statsmodels.stats.power import NormalIndPower
from scipy.stats import chi2_contingency
import math

# --- 页面设置 ---
st.set_page_config(page_title="AB实验科学沙盘", layout="wide")


# --- 核心哈希分流算法 ---
def get_group(user_id, layer_name, salt):
    input_str = f"{user_id}_{layer_name}_{salt}"
    hash_val = int(hashlib.md5(input_str.encode()).hexdigest(), 16)
    return 'A' if (hash_val % 100) < 50 else 'B'


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
    true_lift = st.slider("设定的真实提升", -0.20, 0.20, 0.05, help="这是只有上帝知道的真相，实验的目的是捕捉它。")

# --- 头部教学引入 ---
st.title("🧪 A/B 实验：从“撞大运”到“科学决策”")
st.markdown("""
本沙盘旨在演示：**为什么你不能只看 P 值就做决定？** 我们将模拟一个完整的实验周期，揭示样本量、统计功效与信任度之间的隐秘关系。
""")

# --- 第一阶段：MDE 与排期 ---
st.header("📅 第一阶段：排期预测 (Planning)")
mde_target = st.slider("目标灵敏度 (MDE %)", 0.01, 0.15, 0.05, help="你想检测出多小的提升？越小越难。")

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

# --- 第二阶段：AA 实验自检 ---
st.divider()
st.header("🛡️ 第二阶段：AA 实验 (System Check)")
st.write("在相信结论前，先验证分流器是否公平。")

if st.button("运行 AA 实验自检"):
    aa_n = min(req_n, 5000)
    aa_users = [{"ID": f"u_{i}", "G": get_group(f"u_{i}", "L1", salt_1)} for i in range(aa_n)]
    df_aa = pd.DataFrame(aa_users)
    df_aa['C'] = df_aa['G'].apply(lambda x: np.random.binomial(1, base_p))
    aa_res = df_aa.groupby('G')['C'].agg(['count', 'sum'])
    z_aa, p_aa = proportions_ztest(aa_res['sum'][::-1], aa_res['count'][::-1])

    if p_aa < 0.05:
        st.error(f"🚨 AA失败 (P={p_aa:.4f})：两组天生就有差异！此时的AB结论不可信。")
    else:
        st.success(f"✅ AA通过 (P={p_aa:.4f})：分流公平，实验环境纯净。")

# --- 第三阶段：AB 实验模拟与审计 ---
st.divider()
st.header("📊 第三阶段：实时运行与可信度审计")
days_run = st.slider("实验已运行天数", 1, max(30, est_days + 7), min(7, est_days))
current_n = dau * days_run

# 模拟 AB 数据
user_data = []
for i in range(current_n):
    uid = f"user_{i}"
    g1 = get_group(uid, "L1", salt_1)
    g2 = get_group(uid, "L2", salt_2)
    user_data.append({"ID": uid, "L1": g1, "L2": g2})
df_ab = pd.DataFrame(user_data)
df_ab['Click'] = df_ab['L1'].apply(lambda x: np.random.binomial(1, base_p * (1 + true_lift) if x == 'B' else base_p))

# 统计分析
res = df_ab.groupby('L1')['Click'].agg(['count', 'sum'])
ca, cb = res.loc['A', 'sum'] / res.loc['A', 'count'], res.loc['B', 'sum'] / res.loc['B', 'count']
obs_lift = (cb / ca - 1) if ca > 0 else 0
z, p_val = proportions_ztest([res.loc['B', 'sum'], res.loc['A', 'sum']], [res.loc['B', 'count'], res.loc['A', 'count']])

# Power 计算
try:
    raw_p = obj.power(effect_size=es, nobs1=current_n / 2, alpha=0.05, ratio=1.0)
    curr_power = float(raw_p.power) if hasattr(raw_p, 'power') else float(raw_p)
except:
    curr_power = 0.0

# --- 核心：审计逻辑 ---
score = 0
audit_log = []

# 审计 1: 样本充足性
if current_n >= req_n:
    score += 40
    audit_log.append("✅ **样本充足**：已达到排期要求，规避了‘偷看问题’。")
else:
    audit_log.append(f"❌ **样本不足**：仅完成 {current_n / req_n:.1%}。此时的显著性可能只是随机波动。")

# 审计 2: 统计功效
if curr_power >= 0.8:
    score += 30
    audit_log.append("✅ **功效充足**：探测器功率达标，观察到的提升值相对稳健。")
else:
    audit_log.append(f"❌ **低功效警告**：Power 仅 {curr_power:.1%}。小心‘赢家诅咒’（虚高收益）。")

# 审计 3: 显著性
if p_val < 0.05:
    score += 30
    audit_log.append("✅ **结果显著**：P 值低于 0.05，拒绝零假设。")
else:
    audit_log.append("❌ **不显著**：数据差异在随机误差范围内。")

# 展示审计结果
aud1, aud2 = st.columns([1, 2])
with aud1:
    st.subheader("🛡️ 可信度评分")
    st.metric("得分", f"{score}/100")
    if score == 100:
        st.success("💎 高可信度：建议采纳结论")
    elif score >= 70:
        st.warning("⚠️ 中可信度：谨慎参考，建议补跑")
    else:
        st.error("🚫 低可信度：结论无效，禁止决策")

with aud2:
    st.subheader("📑 审计详情")
    for log in audit_log: st.write(log)

# 指标仪表盘
st.divider()
c1, c2, c3 = st.columns(3)
c1.metric("观察提升", f"{obs_lift:.2%}", delta=f"{(obs_lift - true_lift):.2%} (偏离真值)")
c2.metric("P-value", f"{p_val:.4f}")
c3.metric("当前 Power", f"{curr_power:.2%}")

# --- 底部教学笔记 ---
st.divider()
with st.expander("📖 实验背后的科学逻辑 (教学必读)"):
    st.write("""
    1. **第一类错误 (Alpha)**：误报。就像没病被诊断出有病。我们通过 P < 0.05 将其控制在 5%。
    2. **第二类错误 (Beta)**：漏报。有病没查出来。我们通过 Power > 80% 来降低这个风险。
    3. **赢家诅咒 (Winner's Curse)**：在样本不足、功效极低时，只有当运气极好、波动极大时才会显著。所以低功效下的显著，看到的收益通常是虚高的。
    4. **正交性 (Orthogonality)**：利用哈希洗牌，让不同实验层互不干扰。但在小样本下，洗牌会洗不匀。
    """)