"""
FOF 基金评价打分引擎 —— 全局配置

考核基准：70% * 国证成长(399370) + 30% * 国证价值(399371)
本模块只负责"选谁"（基金评价打分），不负责组合优化/择时。
所有可调参数集中在此，便于回测与调参。
"""
from __future__ import annotations
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# 基准与风格因子
# ---------------------------------------------------------------------------
# RBSA 把基金收益约束回归到下列风格因子上（权重>=0、和为1）。
# code 为各数据源通用的指数代码；akshare 东财接口用带市场前缀的形式（见 provider）。
STYLE_FACTORS: dict[str, str] = {
    "growth": "399370",   # 国证成长
    "value": "399371",    # 国证价值
}

# 考核基准的风格权重（用于计算基金相对“考核基准”的风格偏离与超额）
BENCHMARK_WEIGHTS: dict[str, float] = {
    "growth": 0.7,
    "value": 0.3,
}

# 当前FOF组合规模（亿元）。用于把产品容量、ETF成交额参与率转换为优化器可执行的权重上限。
PORTFOLIO_AUM_YI: float = 14.0

# 年化无风险利率（用于夏普/超额计算的基线）。可改为按期取国债收益率。
RISK_FREE_ANNUAL: float = 0.018


# ---------------------------------------------------------------------------
# 收益频率与窗口
# ---------------------------------------------------------------------------
# 收益频率：'W'(周) 或 'D'(日)。RBSA 习惯用周频降噪。
RETURN_FREQ: str = "W"
PERIODS_PER_YEAR: dict[str, int] = {"W": 52, "D": 252}

# 评价窗口（按 RETURN_FREQ 计的期数）。默认近 3 年周频 ≈ 156 期。
EVAL_WINDOW: int = 156
# 滚动 RBSA 子窗口（用于看风格漂移与 alpha 一致性）。
ROLLING_WINDOW: int = 52
# 一只基金进入评价所需的最少有效期数（数据太短不评）。
MIN_OBS: int = 52


# ---------------------------------------------------------------------------
# 打分维度与权重
# ---------------------------------------------------------------------------
# 每个指标：方向 +1 表示越大越好，-1 表示越小越好（如回撤）。
# 综合分 = Σ weight * zscore(指标, 同组内)。权重无需归一，引擎内部会归一化。
@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    direction: int          # +1 越大越好 / -1 越小越好
    weight: float


SCORE_METRICS: list[MetricSpec] = [
    MetricSpec("style_alpha_ann", "风格调整后alpha(年化)", +1, 0.30),
    MetricSpec("info_ratio",      "信息比率IR",            +1, 0.25),
    MetricSpec("excess_win_rate", "超额胜率",              +1, 0.10),
    MetricSpec("alpha_consistency","alpha一致性",          +1, 0.10),
    MetricSpec("max_drawdown",    "最大回撤",              -1, 0.10),
    MetricSpec("calmar",          "Calmar",                +1, 0.05),
    MetricSpec("style_r2",        "风格拟合R²",            +1, 0.05),
    MetricSpec("size_score",      "规模适中度",            +1, 0.025),
    MetricSpec("tenure_years",    "现任经理任期(年)",      +1, 0.025),
]

# 规模适中度：偏好规模落在 [下限, 上限] 区间，过小有清盘/流动性风险，过大有容量约束。
SIZE_SWEET_SPOT_YI: tuple[float, float] = (2.0, 80.0)   # 单位：亿元


# ---------------------------------------------------------------------------
# 基金筛选（进入打分池的硬条件）
# ---------------------------------------------------------------------------
@dataclass
class UniverseFilter:
    # 允许的基金类型关键词（主动权益 + 股票型 ETF/指数增强）
    allowed_type_keywords: list[str] = field(
        default_factory=lambda: ["股票型", "偏股", "混合型-偏股", "灵活配置", "指数增强", "增强指数型股票", "ETF"]
    )
    # 投资范围硬约束：主动权益基金满一年且规模≥2亿；非QDII股票ETF规模≥5亿。
    # 仅在带PIT元数据的 strict_eligibility 模式下强制执行，避免把缺字段的实时清单误当合格。
    min_active_equity_size_yi: float = 2.0
    min_stock_etf_size_yi: float = 5.0
    min_track_record_years: float = 1.0
    # 主动基金现任基金经理须在本基金任职满该年限；缺失任职日不放行。
    min_manager_tenure_years: float = 1.0
    require_manager_tenure: bool = True
    # 实盘建仓只允许最新PIT状态为“开放申购”（或“开放大额申购”）的主动基金；
    # 暂停、限大额、限额/限购/限制、封闭及状态缺失均不放行。ETF为交易所交易产品，不适用本字段。
    require_open_subscription: bool = True
    exclude_keywords: list[str] = field(
        default_factory=lambda: ["持有", "联接", "QDII", "FOF"]  # 范围排除词；QDII同时检查名称与类型
    )
    # “被动指数型股票基金”可能包含ETF，也可能是普通场外指数基金；只有明确ETF标签的
    # 才能作为ETF放行，未标ETF的被动产品不能以“主动权益基金”身份混入。
    passive_fund_type_keywords: list[str] = field(default_factory=lambda: ["被动指数"])
    # 仅按名称尾部识别份额类别，不能用裸字母 E/C 做子串匹配，否则会误杀 ETF。
    exclude_share_class_suffixes: list[str] = field(default_factory=lambda: ["C", "E", "C类", "E类"])
    # 允许的管理人（基金公司）短名白名单；strict 模式下产品 management_company 须匹配其一。
    # 空列表表示不限制。匹配规则见 engine.universe.management_company_allowed。
    allowed_management_companies: list[str] = field(default_factory=lambda: [
        "易方达基金", "华宝基金", "华夏基金", "华商基金", "广发基金", "富国基金", "万家基金",
        "南方基金", "中银国际证券", "嘉实基金", "中信保诚基金", "景顺长城基金", "国泰基金",
        "招商基金", "鹏华基金", "华安基金", "工银瑞信基金", "华泰柏瑞基金", "永赢基金",
        "中欧基金", "中银基金", "平安基金", "大成基金",
    ])


DEFAULT_FILTER = UniverseFilter()


# ---------------------------------------------------------------------------
# 第③层：风格择时（成长 vs 价值，相对考核基准的主动偏离）
# ---------------------------------------------------------------------------
# 每个信号产出一个"观点" view ∈ [-1, 1]：>0 偏成长、<0 偏价值。
# 合成观点 = Σ w_i·view_i（在有数据的信号上归一），再映射到目标风格权重：
#     w_growth = benchmark_growth + MAX_TILT · composite_view
#     w_value = 1 - w_growth
@dataclass
class StyleTimingConfig:
    # 三个信号的相对权重（无需归一，引擎内部按"可用信号"归一）
    signal_weights: dict[str, float] = field(default_factory=lambda: {
        "valuation": 0.40,   # 估值价差（均值回归，逆向）——需 iFinD 估值，缺则自动跳过
        "momentum": 0.40,    # 风格动量（趋势）——仅需价格
        "vol_regime": 0.20,  # 波动 regime（风险偏好，逆向）——仅需价格
    })
    # 相对考核基准的最大单边主动偏离。默认只偏离 10%，避免风格轮动吞噬选基 alpha。
    # 这是本层唯一的主动风险旋钮：=0 即完全中性；越大越激进。
    max_tilt: float = 0.10
    # 信号参数
    momentum_lookback: int = 120     # 交易日，约半年相对强弱
    vol_window: int = 20             # 交易日，短期已实现波动
    zscore_window: int = 504         # 交易日，约两年，用于信号标准化（因果滚动）
    # 调仓频率：'M' 月度 / 'W' 周度
    rebalance: str = "M"
    # 估值价差用的指标
    valuation_metric: str = "pe_ttm"


STYLE_TIMING = StyleTimingConfig()


# ---------------------------------------------------------------------------
# 第④层：组合优化器（把选基 alpha + 风格目标落到基金/ETF 权重）
# ---------------------------------------------------------------------------
@dataclass
class OptimizerConfig:
    # 进入优化的主动基金候选数。容量约束下，小规模低成长基金未必能承担足够权重，
    # 因此保留更宽候选集，避免风格目标因容量被动失配。
    n_candidates: int = 20
    # 全池候选中至少保留的“风格补全”主动基金数：当期目标低于候选主风格时，
    # 从成长载荷显著更低的一侧按综合分补入，避免被迫使用低流动性ETF补风格。
    style_complement_candidates: int = 6
    style_complement_gap: float = 0.15
    # 单只基金权重上限（分散度约束）
    max_weight_fund: float = 0.10
    # 单笔主动基金申购金额不得超过该基金最新规模的比例。该约束与单基金权重上限
    # 共同生效；例如14亿元FOF买入1.4亿元时，标的基金规模至少应为7亿元。
    max_order_to_fund_aum: float = 0.20
    # 单只基金权重下限（>0 时为"要么不买、要么至少买这么多"，这里用 0 简化）
    min_weight_fund: float = 0.0
    # 风险厌恶系数 γ：目标 max αᵀx − γ·xᵀΣx。越大越保守（越压低特质风险）。
    risk_aversion: float = 8.0
    # 是否启用风格 ETF 补全资产（纯成长/纯价值），保证风格目标总能命中
    use_style_etf: bool = True
    etf_total_cap: float = 0.25        # ETF 合计权重上限（优先用主动基金赚 alpha；ETF只作补全兜底）
    # 期望 alpha 的收缩系数（向 0 收缩，降低对历史 alpha 点估计噪声的敏感）
    alpha_shrink: float = 0.5
    # 协方差对角收缩（向对角阵收缩，提升小样本稳定性）0~1
    cov_shrink: float = 0.2
    # 可选：年化特质跟踪误差上限（None 表示不约束；与用户"先不约束"一致默认 None）
    te_budget_annual: float | None = None


OPTIMIZER = OptimizerConfig()


# ---------------------------------------------------------------------------
# 第⑥层：滚动回测与交易成本
# ---------------------------------------------------------------------------
@dataclass
class BacktestConfig:
    # 首次建仓前需要的周频历史。默认完整使用 EVAL_WINDOW，避免短样本 alpha 参与决策。
    warmup_obs: int = EVAL_WINDOW
    # 单边成本：基金申赎费、ETF 佣金/冲击成本的保守简化假设。
    # 实盘和正式回测应按产品费率、持有期及账户佣金覆写，绝不能把 0 成本当成默认真相。
    fund_one_way_cost: float = 0.0015
    etf_one_way_cost: float = 0.0003
    # 与生产对齐的容量与评分参数（回测 CLI 可覆写）
    portfolio_aum_yi: float = PORTFOLIO_AUM_YI
    max_order_to_fund_aum: float = OPTIMIZER.max_order_to_fund_aum
    etf_participation_rate: float = 0.10
    etf_adv_lookback: int = 20
    enforce_active_fund_capacity: bool = False
    enforce_etf_capacity: bool = False
    strict_eligibility: bool = False
    eval_start: str = "2019-01-01"
    score_cache_dir: str = ""
    full_universe_scoring: bool = True
    # 调仓频率：'QE' 季度末（与 PIT 季度快照对齐，推荐）；'ME' 月末
    rebalance_freq: str = "QE"
    # 是否预取全区间 PIT 并集净值（极慢）；默认按需加载持仓基金
    prefetch_all_pit_codes: bool = False
    # 回测加速：跳过滚动 RBSA（alpha_consistency），每只基金可快 ~100 倍
    backtest_skip_rolling: bool = True
    # 全池轻量 RBSA 后仅保留 top-N 进入综合打分（0=不截断）
    backtest_preselect_pool: int = 400
    # 评分并行进程数
    backtest_workers: int = 4


BACKTEST = BacktestConfig()

# 风格 ETF 补全资产（代码仅作标识；优化只用其风格载荷，alpha=0）
STYLE_ETF_ASSETS = {
    "ETF_GROWTH": {"name": "国证成长ETF(补全)", "growth_load": 1.0, "value_load": 0.0},
    "ETF_VALUE": {"name": "国证价值ETF(补全)", "growth_load": 0.0, "value_load": 1.0},
}
