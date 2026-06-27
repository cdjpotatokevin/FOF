# FOF 基金评价打分引擎

为"相对收益考核（基准 = 70% 国证成长 399370 + 30% 国证价值 399371）"设计的 FOF 选基模块。
目标：选出能**剥离成长/价值 beta 之后仍持续贡献超额**的主动权益基金与股票型 ETF/指增，
作为战胜该基准的第一块拼图。

## 为什么这样设计

考核基准是**偏成长的大中盘 beta**。简单买宽基或按绝对收益排名选基，赢的往往是
"风格踩对"的运气，不可复制。本引擎的核心是用 **RBSA（Sharpe 收益法风格分析）**
把每只基金分解为"等价的成长/价值组合 + 选股超额"：

```
基金收益_t = w_growth · 国证成长_t + w_value · 国证价值_t + 主动收益_t
             └────────── 风格 beta（运气）──────────┘   └─ 选股 alpha（能力）─┘
   s.t. w >= 0, Σw = 1
```

`主动收益` 的年化均值就是**风格调整后 alpha**——这才是 FOF 真正要为之付费的东西。

## 目录结构

```
fof_system/
  config.py            基准/因子、打分维度与权重、窗口、筛选条件（调参集中在此）
  pipeline.py          端到端：数据 → 基准 → RBSA → 指标 → 综合分
  run_score.py         命令行入口
  data/                可插拔数据源
    base.py            DataProvider 抽象接口 + 净值→收益工具 + 可选指数估值
    pit.py             PIT数据仓：双时间轴、manifest、基金/ETF快照与行情读取
    akshare_provider.py   免费、无需 token（已可直接跑；指数取数东财/新浪双路兜底）
    ifind_client.py       封装 ifind-finance-data skill 的 call.py（子进程，MCP通道）
    ifind_http.py         iFinD HTTP access-token + 专题报表(data_pool)适配器
    ifind_provider.py     同花顺 iFinD 数据源（指数估值/宏观/持仓的独特来源）
    mock_provider.py      合成数据，离线验证用，含已知真值
  engine/
    rbsa.py            Sharpe 约束风格回归 + 滚动风格漂移      （第②层）
    metrics.py         alpha/IR/胜率/回撤/Calmar/一致性/规模/任期（第②层）
    scoring.py         组内稳健 z-score 加权合成综合分(0~100)  （第②层）
    universe.py        候选池初筛                              （第②层）
    signals.py         风格择时三信号：估值价差/动量/波动regime（第③层）
    style_timing.py    信号合成→有界 tilt 目标权重 + 当期建议  （第③层）
    style_backtest.py  tilt vs 配置基准回测（防前视）            （第③层）
    risk_model.py      主动收益协方差 + 期望alpha + TE/漂移监控 （第④/⑥层）
    optimizer.py       特征targeting均值方差优化（SLSQP）      （第④层）
    attribution.py     收益法风格/选股归因（恒等拆分）         （第⑤层）
  pipeline.py          第②层端到端
  portfolio.py         第④层编排：串 ②选基/③风格目标/④优化
  run_score.py         第②层 CLI（基金打分）
  run_data.py          第①层 CLI（PIT基金/ETF快照和ETF行情入库）
  run_style.py         第③层 CLI（风格择时）
  run_portfolio.py     第④层 CLI（组合优化）
  run_attribution.py   第⑤层 CLI（组合归因）
  run_backtest.py      第⑥层 CLI（整链滚动回测，含交易成本）
  run_monitor.py       第⑥层 CLI（调仓前TE、集中度与风格漂移）
  tests/               RBSA/打分/择时/优化/归因/回测（pytest）
```

## 数据源说明

| 数据源 | 用途 | 状态 |
|---|---|---|
| **akshare** | 指数/基金长历史价格净值（稳、免费、全序列） | 已可直接跑 |
| **iFinD**（经 skill） | **指数估值PE/PB**、基金资料核验、基金季报持仓——akshare 拿不全的 | 需在 skill 的 `mcp_config.json` 填真实 `auth_token` |
| **iFinD HTTP** | THS_DS指数PE/PB、THS_HQ ETF成交额 | 仅在进程环境设置 `IFIND_REFRESH_TOKEN` |
| mock | 离线确定性合成数据，单测与无网验证 | 已可直接跑 |

iFinD 经 `ifind-finance-data` skill 的 `call.py`（MCP）接入，是自然语言 `query` 接口、
长区间会截断——故**长历史价格用 akshare，iFinD 专用于估值、基金资料核验和持仓**这类定向查询。
当前服务端基金工具集未开放全市场 `search_funds`；但已验证 iFinD HTTP API 鉴权可用。HTTP
全市场筛选需在 iFinD 终端/超级命令先生成专题报表 `reportname`，再经项目的
`fetch-ifind-data-pool` 写入 PIT。没有报表编号时，使用 iFinD/内部导出CSV经
`run_data import-universe` 落库，不能逐只资料查询后拼凑历史基金池。
若 skill 目录路径变化，设环境变量 `IFIND_SKILL_DIR` 指向含 call.py 的目录即可。

## 快速开始

```bash
pip install -r requirements.txt

# === 第①层 PIT 数据底座 ===
# 0) 将“今天可见”的基金/ETF主数据保存为有 available_at 的快照
#    注意：它不能倒灌进历史回测；历史PIT需要导入当期留存的清单/公告。
python -m fof_system.run_data --root /data/fof_pit snapshot-universe \
    --source akshare --asof 2026-06-23
# 0b) 接入实际股票ETF行情（复权收盘、成交量、成交额、换手）；默认交易日后一天可用
python -m fof_system.run_data --root /data/fof_pit ingest-etf --source akshare \
    --codes 510300,159915 --start 2018-01-01 --end 2026-06-23
# 0c) 导入供应商/自有留存的历史主数据。CSV 可包含 aum_yi、manager、manager_start、
#     subscription_status 等字段；available_at 必须填写真实披露/可用日期。
python -m fof_system.run_data --root /data/fof_pit import-universe \
    --csv 2023Q4_universe.csv --source ifind --effective-date 2023-12-31 --available-at 2024-01-25

# 0d) 若已在 iFinD 超级命令生成基金专题报表，直接执行并PIT化（凭据只在当前终端环境变量）
export IFIND_REFRESH_TOKEN='在此仅临时设置，不要写入文件'
python -m fof_system.run_data --root /data/fof_pit fetch-ifind-data-pool \
    --reportname pXXXXX --functionpara '{"日期":"2024-12-31"}' \
    --outputpara 'pXXXXX_f001:Y,pXXXXX_f002:Y,pXXXXX_f003:Y' \
    --field-map '{"code":"pXXXXX_f001","name":"pXXXXX_f002","aum_yi":"pXXXXX_f003","fund_type":"pXXXXX_f004","inception":"pXXXXX_f005"}' \
    --asset-type fund --aum-unit yi --effective-date 2024-12-31 --available-at 2025-01-25

# 0d-1) iFinD“基金业绩回报”报表 p04955：自动换算AUM = 最新规模(亿份) × 单位净值。
#       股票ETF口径：被动指数型股票基金、名称含ETF且不含联接；名称含QDII+ETF者为QDII ETF并排除。
python -m fof_system.run_data --root /data/fof_pit fetch-ifind-p04955 \
    --edate 20260623 --p0 0 --jjlb 051001004 --user-sectorid 'your-user-sectorid|' \
    --effective-date 2026-06-23 --available-at 2026-06-24
#       基金代码会强制规范为六码字符串（如 1234 → 001234）。
# 0d-2) 历史PIT基金池回补：默认季度末，并在入库前验证成立日/净值日期不晚于截点。
#       available_at 默认滞后两个工作日；真实披露时间可用时应替换为真实日期。
python -m fof_system.run_data --root /data/fof_pit backfill-ifind-p04955 \
    --start 2021-01-01 --end 2026-06-23 --frequency QE --jjlb 051001004 \
    --user-sectorid 'your-user-sectorid|'
# 所有后续iFinD HTTP响应会保存到 <root>/raw/ifind_http，并自动按请求参数缓存去重。
# 审计供应商dataVol、请求指纹和原始响应清单：
python -m fof_system.run_data --root /data/fof_pit ifind-cache-audit
# 0e) 导入基金季报持仓。report-period 是报告期末，available-at 是实际披露日，二者不可混用。
python -m fof_system.run_data --root /data/fof_pit import-holdings \
    --csv 2023Q4_holdings.csv --source ifind --report-period 2023-12-31 --available-at 2024-01-20

> **投资范围硬约束**：PIT模式下，主动权益基金必须成立满一年且AUM不低于2亿元；
> 股票ETF必须有明确 `is_stock_etf=true`、`is_qdii=false` 标签且AUM不低于5亿元。
> 数据缺失时系统不会猜测并放行，因此导入历史主数据时请包含 `aum_yi`、`inception`，
> ETF还需包含 `asset_type`、`is_stock_etf` 和 `is_qdii`。
> 主动基金还必须为开放申购状态；`暂停大额申购`、`限额/限购/限制申购`及供应商显式给出的
> `daily_subscription_limit_yi` 或 `has_daily_subscription_limit=true` 均会被严格排除。
> 若供应商未给出数值限额，交易日仍须以渠道回传的实际限额复核，不能把空字段解释为“无限额”。

# === 第②层 基金打分 ===
# 1) 离线自检（合成数据，验证整条链路与 RBSA 还原能力）
python -m fof_system.run_score --source mock --top 5
# 2) akshare 评价指定基金
python -m fof_system.run_score --source akshare \
    --codes 005827,110011,163406 --start 2021-01-01 --out scores.csv
# 3) 按最新PIT严格池评价全体可投基金（不按规模截断；耗时较长）
python -m fof_system.run_score --source akshare --pit-root /data/fof_pit \
    --universe-asof 2026-06-23 --end 2026-06-23 --out full_universe_scores.csv
# 如仅用于连通性测试，才显式使用 --limit；它不能作为正式筛选口径。
python -m fof_system.run_score --source akshare --limit 30 -v
# 3b) 有PIT主数据时按指定历史时点选择候选池，并使用其AUM/经理元数据
python -m fof_system.run_score --source akshare --pit-root /data/fof_pit \
    --universe-asof 2024-01-31 --end 2024-01-31 --limit 30

# === 第③层 风格择时 ===
# 4) 离线自检
python -m fof_system.run_style --source mock
# 5) akshare 价格 + iFinD 估值（三信号齐全；估值会分块抓取并缓存）
python -m fof_system.run_style --source akshare --val-source ifind --start 2018-01-01
# 也可用HTTP超级命令协议（THS_DS）获取估值；不写入refresh token
python -m fof_system.run_style --source akshare --val-source ifind_http --start 2018-01-01

# === 第④层 组合优化 ===
# 6) 离线自检（整条 ②③④ 链路）
python -m fof_system.run_portfolio --source mock
# 7) 真实：候选基金 + 风格目标自动来自第③层
python -m fof_system.run_portfolio --source akshare --val-source ifind \
    --codes 005827,163406,110011,161005,260108
# 8) 手动指定风格目标（成长暴露75%）、单票上限、风险厌恶
python -m fof_system.run_portfolio --source akshare --target-growth 0.75 \
    --codes 005827,163406,110011,161005,260108 --max-weight 0.12 --gamma 8
# 8b) 正式全池筛选：对PIT严格合规池评分，不按规模前N名预截断；
#     优化器使用“风格调整后alpha × RBSA R²”再收缩后的期望alpha，并从全池
#     补入高分低成长主动基金，且仅允许PIT状态为开放申购、无暂停大额申购/单日限额的主动基金；
#     单基金上限10%，且单笔主动基金申购额不超过该基金规模的20%；默认保留20只主动候选、
#     至少6只低成长侧补足候选，同时给出5只未持有备选。
python -m fof_system.run_portfolio --source akshare --pit-root /data/fof_pit \
    --universe-asof 2026-06-23 --end 2026-06-23 --all-eligible \
    --score-out full_universe_scores.csv \
    --portfolio-aum-yi 14 --max-order-to-fund-aum 0.20 --capacity-asof 2026-06-24 --backup-out backup_funds.csv
#     后续同一数据截面重算权重时，复用评分文件，无需再次对全池拉取/评分：
python -m fof_system.run_portfolio --source akshare --pit-root /data/fof_pit \
    --universe-asof 2026-06-23 --end 2026-06-23 \
    --scores-in full_universe_scores.csv --portfolio-aum-yi 14 --capacity-asof 2026-06-24 \
    --portfolio-out target_portfolio.csv --summary-out target_portfolio_summary.json

# === 第⑤层 组合归因 ===
# 9) 离线自检（整条 ②③④⑤）
python -m fof_system.run_attribution --source mock
# 10) 真实：把超额拆成 选基(②) vs 风格择时(③)
python -m fof_system.run_attribution --source akshare --val-source ifind \
    --codes 005827,163406,110011,161005,260108 --start 2019-01-01

# === 第⑥层 整链滚动回测 ===
# 11) 离线自检：选基、择时、优化、换仓成本在同一条无前视链路中检验
python -m fof_system.run_backtest --source mock
# 12) 真实回测：候选池必须按历史可得范围准备，避免幸存者偏差
python -m fof_system.run_backtest --source akshare --val-source ifind \
    --codes 005827,163406,110011,161005,260108 --start 2016-01-01 \
    --fund-cost 0.0015 --etf-cost 0.0005 --pit-root /data/fof_pit --out-dir backtest_output
# 13) 调仓前：检查目标组合的事前TE、目标可达性和基金风格漂移
python -m fof_system.run_monitor --source akshare --val-source ifind \
    --codes 005827,163406,110011,161005,260108

# 14) 参数/交易成本稳健性：同一walk-forward顺序下比较基础、高成本、保守、宽松情景
python -m fof_system.run_robustness --source mock --out robustness.csv

# 15) ETF容量：先写入ETF成交额PIT行情，再按目标ETF权重反推可承载的组合规模
python -m fof_system.run_capacity --pit-root /data/fof_pit --asof 2026-06-24 \
    --weights 159915=0.25 --participation-rate 0.10
# 用iFinD THS_HQ写入ETF成交额（代码需带交易所后缀）
python -m fof_system.run_data --root /data/fof_pit ingest-etf --source ifind_http \
    --codes 589990.SH --start 2025-06-24 --end 2026-06-24

# 0g) 批量补齐股票ETF全池成交额PIT：自动从PIT主数据中筛选
#     AUM≥5亿元、非QDII、明确股票ETF标签的产品；支持 AkShare 兜底或 iFinD HTTP。
python -m fof_system.run_data --root /data/fof_pit ingest-etf-pool --source akshare \
    --universe-asof 2026-06-23 --start 2025-06-24 --end 2026-06-24 \
    --skip-existing --codes-out stock_etf_pool.csv

# 0h) 用渠道/供应商CSV补主数据字段，例如 manager_start、单日申购限额字段等。
#     该命令不会改写旧快照，而是在 available-at 日期生成一份新的PIT补丁快照。
python -m fof_system.run_data --root /data/fof_pit patch-universe-fields \
    --asof 2026-06-23 --patch-csv manager_start_patch.csv \
    --source channel:manager-tenure --available-at 2026-06-25

# 0i) 交易日前校验：正式下单前优先使用渠道/TA当日CSV；缺少限额字段默认失败。
python -m fof_system.run_pretrade_check \
    --portfolio-csv run_outputs/2026-06-25/wanneng2011_overlay_full.csv \
    --backup-csv run_outputs/2026-06-25/wanneng2011_backups.csv \
    --status-source csv --status-csv channel_pretrade_status.csv \
    --asof 2026-06-25 --report-out pretrade_report.csv --summary-out pretrade_summary.json
```

> **配置 iFinD 密钥**：编辑 `ifind-finance-data` skill 目录下的 `mcp_config.json`，
> 把 `auth_token` 从占位符改成你的真实密钥（MCP官网→个人中心→密钥）。
> 配好后第③层的**估值价差信号**会自动启用——这通常是风格择时里最有效的信号。

## 打分维度（可在 config.SCORE_METRICS 调权重）

| 维度 | 含义 | 默认权重 |
|---|---|---|
| 风格调整后 alpha（年化） | 剥离成长/价值后的选股超额，**核心** | 0.30 |
| 信息比率 IR | alpha / 主动波动，稳定性 | 0.25 |
| 超额胜率 | 主动收益为正的期数占比 | 0.10 |
| alpha 一致性 | 滚动窗口里 alpha>0 的比例（非一次性运气） | 0.10 |
| 最大回撤 | 越小越好 | 0.10 |
| Calmar | 年化收益/回撤 | 0.05 |
| 风格拟合 R² | 载荷可信度 | 0.05 |
| 规模适中度 | 偏好 2–80 亿（避清盘/容量） | 0.025 |
| 经理任期 | 现任经理稳定性 | 0.025 |

综合分在**同业组内**做稳健 z-score（中位数/MAD，抗异常值）后加权，再映射到 0~100。

## 重要提醒（方法论局限）

- **RBSA 的 alpha 点估计在短窗口上有噪声**：单只基金 3 年周频的年化 alpha 标准误约
  2–5%（取决于基金特质波动）。因此别只看 alpha 排名，要结合 **IR、alpha 一致性、t 值**
  共同判断；窗口尽量拉长（默认 156 周，可在 config.EVAL_WINDOW 调）。
- **基金持仓季度披露且滞后**：本引擎以净值回归（RBSA）为主、不依赖持仓，规避了披露滞后；
  若接入持仓穿透可作为风格载荷的交叉验证。
- 风格因子目前为成长/价值二维；如需更细（大小盘、行业、动量/质量），在
  `config.STYLE_FACTORS` 增列即可，RBSA 与打分会自动适配。
- **PIT数据底座是生产门槛**：`data.pit.PITDataStore` 的读取同时约束 `effective_date`
  和 `available_at`，不会把尚未披露的基金状态带回历史。当前AkShare命令可记录现在开始
  的快照与ETF行情；完整历史PIT基金池、历史AUM/经理状态和持仓披露日仍需由iFinD或自有
  数据库批量导入，不能把今天的全市场清单伪装成历史事实。
- **持仓穿透同样有披露滞后**：持仓数据以 `report_period` 与 `available_at` 双字段入库；
  `read_holdings_asof()` 只给出当时已披露的最新一季，适合作为RBSA风格判断的交叉验证，
  不应被误读为实时持仓。

## 路线图

六层系统进度：

- 🟡 **①数据底座**：已具备PIT双时间轴、manifest、ETF实际行情合同、主数据快照和入库CLI；
  历史基金池/经理/AUM/持仓的供应商级PIT回填仍待接入。
- ✅ **②基金评价**：RBSA 风格调整 alpha 打分（`run_score`）。
- ✅ **③风格择时**：成长/价值有界 tilt（估值价差+动量+波动regime，`run_style`）。
  - 注：估值价差信号需 iFinD 估值数据；未配密钥时自动降级为"动量+波动"两信号。
- ✅ **④组合优化器**：把②的选基 alpha 和③的风格目标落到**真实基金/ETF 权重**（`run_portfolio`）。
- ✅ **⑤风险与归因**：收益法风格/选股归因，把超额恒等拆成"风格择时 vs 选股"（`run_attribution`）。
- ✅ **⑥回测与监控**：无前视整链滚动回测、毛/净收益与换仓成本、ex-ante 特质TE、
  风格目标可达性诊断及滚动 RBSA 风格漂移函数（`run_backtest` / `risk_model`）。

### 第⑤层方法论

FOF 持基金而非个股，经典 Brinson 分桶不适用；改用 RBSA 载荷做**收益法恒等拆分**：

```
每期：R_p − R_b = 风格择时 + 选股   （恒等，无期内残差）
  风格择时 = (W^p_g−0.7)·R_g + (W^p_v−0.3)·R_v       ← 第③层的功劳
  选股     = Σ_i x_i·(r_i − 风格复制_i)               ← 第②层的功劳
```

- 跨期把逐期项几何链接成累计超额，差额记为"链接残差"（来自复利与载荷时变，通常很小）。
- 输出逐基金选股贡献，直接看哪只基金在赚/亏选股超额——可操作。
- 行业层归因（Brinson 行业版）需 iFinD 持仓穿透到个股 + 行业收益，列为后续扩展。

### 第④层方法论

特征 targeting 的均值-方差优化：

```
max_x  αᵀx − γ·xᵀΣx        α=clip(R²,0,1)×各基金风格调整后alpha，再向0收缩；Σ=主动收益协方差(收缩)
s.t.   Σx = 1, x ≥ 0, x_i ≤ w_max
       Σ x_i·成长载荷_i ≈ 第③层目标成长暴露   （二次惩罚，避免共线求解失败）
       Σ x_etf ≤ etf_cap                      （ETF 仅作风格补全兜底）
       x_i ≤ 参与率×近20日ADV/组合规模          （真实ETF的单日容量上限；无成交额即为0）
```

- **关键拆分**：风格 beta 风险以第③层目标为中心管理；优化器真正控制的是**剥离风格后的
  特质风险**（选基风险），故 Σ 用 RBSA 残差(主动收益)而非总收益。目标偏离与可达性会单独报告。
- α 先按RBSA的R²折减（低解释度残差不应获得同等置信度），再向0收缩；Σ同样做收缩，降低对历史点估计噪声的敏感。
- 纯风格 ETF 补全资产（载荷 1/0、alpha=0）用于补全风格；在单基金上限、ETF上限和
  候选数共同限制下，系统会先计算可达成长区间。目标不可达会明确告警，不能视作已命中。
- `te_budget_annual` 可选特质 TE 上限；默认 None（与"先不约束"一致）。
- 候选数×单票上限需留出优化自由度（如 15×15%）：太紧会把权重全顶到上限、失去区分度。

### 第③层方法论与现实

- 信号均**因果计算**、月末定权重次日生效，回测无前视；估值(iFinD)与价格(akshare)
  自动对齐到同一交易日历。
- `max_tilt`（默认 ±10%）是本层唯一主动风险旋钮：=0 即完全贴住70/30基准，越大越激进。
- 旧的50/50风格择时实验不能直接外推到70/30考核。应使用 `run_backtest` 在同一基准、
  同一成本假设、同一候选池口径下重新检验；特别要检查估值逆向信号是否与趋势信号冲突。
