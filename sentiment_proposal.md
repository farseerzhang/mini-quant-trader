# YouTube 博主观点因子 — 方案（草案）

状态：草案，待确认。在你确认之前不写代码。
日期：2026-10-12

---

# 1. 目标

抓取几位财经 YouTube 博主（中英文各若干）对美股个股的点评，从字幕中提取结构化观点，按 **日期 × 股票** 汇总出谁看多、谁看空，并在积累足够数据后检验每个博主的历史命中率，判断它能否作为情绪因子接入现有系统。

和 `proposal.md` 一样，这是研究工具：**博主观点不是交易信号**，在命中率检验通过之前，它不进入 `generate-signals`。

## 不做（第一版）

- 不转写没有字幕的视频（Whisper 放到 S5）。
- 不做调度；和主系统一样，按需运行 CLI 命令。
- 不改动 `quant/` 里现有的任何模块和表，也不碰当前未提交的改动。新代码都在 `quant/sentiment/` 和新表里。
- 不做 Streamlit 页面；先用命令行和 Markdown 报告。

---

# 2. 流程

```
config/settings.yaml (sentiment.channels)
        │
        ▼
 ① 拉视频列表   yt-dlp（不需要 key）        → sentiment_videos
        │
        ▼
 ② 拉字幕       youtube-transcript-api       → sentiment_transcripts
        │       （无字幕 → status = NO_TRANSCRIPT，跳过）
        ▼
 ③ 提取观点     Claude API，结构化输出       → sentiment_opinions
        │       （每条观点带原句和时间戳，并校验原句确实在字幕里）
        ▼
 ④ 汇总报告     日期 × 股票 × 看多/看空      → reports/sentiment/<date>.md
        │
        ▼
 ⑤ 命中率       观点之后 5/20/60 个交易日的超额收益（对 SPY）
                → reports/sentiment/scorecard.md
```

每一步都是幂等的：以 `video_id` 为键，重复运行只处理新的或失败的视频，与主系统的约定（proposal 第 30 节）一致。

---

# 3. 数据来源

## 3.1 视频列表：`yt-dlp`

- 用 `yt-dlp --flat-playlist` 读取频道的 `/videos` 页，拿到 `video_id`、标题、时长；再对每个新视频读一次元数据，拿到精确的发布时间。
- 好处是不需要 Google Cloud key。坏处是它和 `yfinance` 一样是非官方工具，可能失效。
- 因此和行情数据一样放在一个接口后面（`VideoSource`），以后可以换成 YouTube Data API v3，而不影响下游。
- 过滤：跳过 Shorts（时长 < 3 分钟）和直播回放（可配置），只保留回看窗口内的视频（默认 14 天）。

## 3.2 字幕：`youtube-transcript-api`

- 语言优先级：人工字幕优先于自动字幕；英文频道取 `en`，中文频道依次取 `zh-Hans`、`zh-Hant`、`zh`。
- 记录字幕是人工的还是自动生成的（`is_generated`）。自动字幕错字多，报告里会标出来，命中率也可以分开统计。
- 繁体统一转成简体后再提取（`opencc`），方便做别名映射。
- 原始字幕（带每段的起始秒数）完整存进数据库。这是"原始数据是唯一事实来源"（proposal 4.3）：以后换提示词或换模型，可以直接重新提取，不用重新抓。
- 限速：请求之间随机等待几秒，失败重试一次后标为 `FAILED`，下次运行再试。只在本机或家庭网络运行。

---

# 4. 观点提取

## 4.1 模型

- 用 Anthropic Python SDK，模型名写在配置里（`sentiment.extraction.model`）。
- S2 阶段会先人工标注约 20 个视频作为小金标集，再在它上面比较 Haiku 5.5（`claude-haiku-5-5`）和 Sonnet 5.5（`claude-sonnet-5-5`）的准确率和成本，然后选定一个。在此之前不预设。
- 用工具调用 / 结构化输出强制返回 JSON，固定的 system prompt 和股票别名表放在前面，开启 prompt caching。
- 长视频按约 10 分钟切块，块之间保留少量重叠，每块独立提取，最后按股票合并去重。

## 4.2 每条观点的结构

| 字段 | 说明 |
|---|---|
| `ticker` | 美股代码，必须能在别名表或代码清单里找到，否则标为 `UNRESOLVED` |
| `stance` | `BULLISH` / `BEARISH` / `NEUTRAL` / `CONDITIONAL` |
| `horizon` | `SHORT`（数周内）/ `MEDIUM`（数月）/ `LONG`（一年以上）/ `UNSPECIFIED` |
| `condition` | 条件式观点的条件，例如"跌到 120 以下再买"；否则为空 |
| `conviction` | 1–3，博主语气的强弱（不是模型的自信度） |
| `action` | 博主自己是否说了买入/加仓/卖出/持有：`BOUGHT` / `SOLD` / `HOLDING` / `NONE` |
| `quote` | 原句（字幕原文，不改写） |
| `start_seconds` | 原句在视频中的时间点，可拼成 `youtu.be/<id>?t=<秒>` 直接跳转核对 |
| `reasoning` | 一句话说明博主的理由 |

规则（写进提示词）：

- 只记录博主**自己的**观点。转述分析师评级、新闻或观众提问不算，除非博主明确表态。
- "短期看空、长期看多"拆成两条，分别带不同的 `horizon`。
- 只是顺带提到、没有表态的股票不记录。
- ETF 和指数（SPY、QQQ）照常记录，标为 `asset_type = ETF/INDEX`，汇总时与个股分开。

## 4.3 防止编造

- **原句校验**：提取后检查 `quote` 是否在对应字幕块中出现（做空白和标点归一化后的模糊匹配，阈值可配）。对不上的观点标为 `UNVERIFIED`，不进入汇总，但保留下来用于检查。
- **代码校验**：模型给出的代码必须在别名表或 SEC 公司代码清单（`company_tickers.json`，本地缓存）里能找到。
- 每条观点都记录 `prompt_version` 和 `model`。换提示词后重新提取会生成新版本，旧版本保留，便于比较。

## 4.4 股票别名表

`config/sentiment_aliases.yaml`，人工维护，并允许模型提出补充建议：

```yaml
NVDA: [英伟达, 辉达, 老黄, Nvidia]
AAPL: [苹果, Apple]
GOOGL: [谷歌, 谷歌A, Alphabet, Google]
TSLA: [特斯拉, Tesla]
BRK-B: [伯克希尔, 巴菲特的公司, Berkshire]
```

中文自动字幕常见的错写（例如把"英伟达"识别成"英伟大"）也加进别名表。模型无法确定的提法输出为 `UNRESOLVED`，并在报告末尾列出来，由你决定是否加入别名表。

---

# 5. 时间约定（避免未来函数）

- 存储视频的发布时间 `published_at`（UTC）。
- 观点的**生效时间** = 发布时间之后的第一个 NYSE 交易日开盘，沿用 proposal 4.5 的 `next_open` 约定。盘中发布的视频，同样从下一个交易日开盘才算。
- 命中率一律从生效时间开始计算，所以不会用到博主发布时还看不到的价格。
- 字幕可能比视频晚几小时才出现；我们记录的是视频发布时间，不是抓取时间，所以对回测没有影响。但如果以后要做实时信号，就必须使用抓取时间。

---

# 6. 数据库

新增 4 张表，放在同一个 SQLite 文件里，定义写在 `quant/sentiment/tables.py`，共用现有的 `metadata`，这样 `quant/persistence/tables.py` 不需要改动。每次运行仍写一行 `pipeline_runs`（command = `sentiment-*`）。

开发库可以随时删掉重建，因此在 S4 之前不做迁移。

### sentiment_channels
`channel_id` (PK), `handle`, `name`, `language` (en/zh), `style_note`, `added_at`。由配置同步，配置是唯一事实来源（与 `symbols` 表的做法相同）。

### sentiment_videos
`video_id` (PK), `channel_id`, `title`, `published_at`, `duration_seconds`, `transcript_status` (`PENDING` / `OK` / `NO_TRANSCRIPT` / `FAILED` / `SKIPPED`), `fetched_at`。

### sentiment_transcripts
`video_id` (PK), `language`, `is_generated`, `segments_json`（`[{start, duration, text}]`）, `char_count`, `fetched_at`。

### sentiment_opinions
`opinion_id` (PK), `video_id`, `channel_id`, `published_at`, `effective_date`, `ticker`, `asset_type`, `stance`, `horizon`, `condition`, `conviction`, `action`, `quote`, `start_seconds`, `reasoning`, `verification` (`VERIFIED` / `UNVERIFIED` / `UNRESOLVED`), `model`, `prompt_version`, `extracted_at`。

唯一约束：`(video_id, ticker, horizon, prompt_version)`。

---

# 7. CLI

挂在现有的 `quant` 命令下，作为一组子命令：

```bash
quant sentiment fetch      [--days 14] [--channel <handle>]   # ①②：视频列表 + 字幕
quant sentiment extract    [--limit N] [--reextract]          # ③：只处理还没有当前 prompt_version 观点的视频
quant sentiment report     [--from 2026-10-01] [--to ...]     # ④：生成日期 × 股票汇总
quant sentiment scorecard                                     # ⑤：命中率
quant sentiment run                                           # ①–④ 依次运行
```

`extract` 运行前会打印要处理的视频数和预估 token 数；超过 `sentiment.extraction.max_videos_per_run` 时停下，需要加 `--yes` 才继续。

---

# 8. 输出

## 8.1 每日汇总（`reports/sentiment/<date>.md`）

```
2026-10-09（生效日 2026-10-10）

| 股票 | 看多 | 看空 | 条件 | 净值 | 博主 |
|------|------|------|------|------|------|
| NVDA | 3    | 0    | 1    | +3   | Joseph Carlson ▲, 美投讲美股 ▲, Chip Stock Investor ▲, 视野环球财经 ◆(跌到 160 再买) |
| TSLA | 0    | 2    | 0    | -2   | ...  |

各观点原句与跳转链接见附录。
未识别的提法：英伟大(×2)、"那个做数据中心的"(×1)
```

净值 = 看多数 − 看空数，`CONDITIONAL` 和 `NEUTRAL` 不计入。同一博主在同一天对同一只股票只算一次。

## 8.2 博主成绩单（`reports/sentiment/scorecard.md`）

对每个博主 × 时间范围，统计生效后 5 / 20 / 60 个交易日的收益：

- 看多观点：股票相对 SPY 的超额收益是否为正。
- 看空观点：超额收益是否为负。
- 报告样本数、命中率、平均超额收益，以及置换检验的 p 值。基准是同一博主、同一批股票随机打乱看多/看空标签后的结果。
- 样本数少于 30 的格子只显示数字，不下结论。

价格从现有的 `daily_prices` 读取。被提到但不在股票池里的代码，会通过同一个 `YFinanceProvider` 拉取，写入一张单独的 `sentiment_prices` 表，不污染主股票池。

---

# 9. 和量化系统的衔接（S6，之后再做）

只有满足下面的条件，才考虑把它做成特征：

1. 至少积累 3 个月的观点。
2. 在**事先登记**的检验中，至少有一个博主（或博主组合）的 20 日超额命中率显著高于随机，做法与 proposal 13.1 / 18.1 的预登记相同。
3. 这些观点覆盖了股票池中足够多的股票。

满足后的候选特征：`sentiment_net_5d`（近 5 个交易日内的净看多数）、`sentiment_consensus`（一致性）、`sentiment_contrarian`（命中率显著为负的博主取反）。它们作为新列加入 `features`，需要升级 `features.version`，并按照 Phase 5 之后的规则重新注册模型。

**限制**：YouTube 上的历史视频可以回补，所以可以做回测，但回补的历史存在幸存者偏差：我们是今天挑出这几个博主的，而他们能活到今天，可能正是因为过去说得准。因此回补数据只用于探索；真正的证据来自从今天开始的前向记录，与 proposal 第 15.5 节的思路一致。

---

# 10. 配置（新增到 `config/settings.yaml`）

```yaml
sentiment:
  lookback_days: 14
  min_duration_seconds: 180          # 跳过 Shorts
  include_live: false
  request_delay_seconds: [2, 5]      # 随机等待区间
  channels:
    - {handle: "@JosephCarlsonShow",    language: en, style: 长线持仓}
    - {handle: "@FinancialEducation",   language: en, style: 激进成长}
    - {handle: "@meitou",               language: zh, style: 综合}
    - {handle: "@shiyehuanqiucaijing",  language: zh, style: 保守}
  extraction:
    model: claude-sonnet-5-5         # S2 比较后确定
    prompt_version: 1
    chunk_minutes: 10
    quote_match_threshold: 0.85
    max_videos_per_run: 40
  evaluation:
    horizons: [5, 20, 60]
    benchmark: SPY
    min_samples: 30
```

频道 handle 是占位符，S0 会核实真实的 `channel_id`。

API key 只从环境变量 `ANTHROPIC_API_KEY` 读取（由你写进 `.env`，并确认 `.env` 已在 `.gitignore` 里）。第一版不需要 YouTube key。

---

# 11. 代码结构

```
quant/sentiment/
  __init__.py
  tables.py         # 4 张新表
  sources.py        # VideoSource 接口 + YtDlpSource
  transcripts.py    # 字幕抓取、繁简转换、切块
  aliases.py        # 别名表 + SEC 代码清单校验
  extract.py        # Claude 调用、结构化输出、原句校验
  prompts.py        # system prompt（带 prompt_version）
  aggregate.py      # 日期 × 股票汇总
  scorecard.py      # 命中率
  report.py         # Markdown 报告
  cli.py            # 子命令，在 quant/cli.py 中注册（只加一行）
config/sentiment_aliases.yaml
tests/sentiment/
  fixtures/         # 保存好的真实字幕片段（中英各几段）
  test_*.py         # 不联网：模拟 VideoSource 和 Anthropic 客户端
```

新依赖放在可选组里，不影响主系统的安装：

```toml
[project.optional-dependencies]
sentiment = ["yt-dlp", "youtube-transcript-api", "anthropic", "opencc-python-reimplemented", "python-dotenv"]
```

`quant/cli.py` 当前有未提交的改动。注册子命令需要在那里加一行，我会等你提交之后再加。

---

# 12. 阶段

| 阶段 | 内容 | 完成标准 |
|---|---|---|
| **S0 探路** | 用 4 个候选频道核实 `channel_id`；统计最近 14 天的视频数、字幕覆盖率（人工 / 自动 / 无）、平均时长 | 一张表交给你，据此确定最终频道名单 |
| **S1 抓取** | 表、`fetch` 命令、幂等、限速、测试 | 重复运行不重复写入；断网后再跑能补齐 |
| **S2 提取** | 提示词、结构化输出、原句校验、别名表、约 20 个视频的人工金标 | 在金标集上，股票识别和立场的准确率都 ≥ 85%；模型选定 |
| **S3 报告** | 每日汇总 + 附录原句链接 | 你看过一周的报告，认可质量 |
| **S4 成绩单** | 历史回补（例如 6 个月）+ 命中率 + 置换检验 | 报告生成；结论按 9 节的标准写，不夸大 |
| **S5 Whisper** | 无字幕视频的本地转写（可选） | 视 S0 的字幕覆盖率决定是否需要 |
| **S6 因子** | 按第 9 节预登记后，再接入 `features` | 另写预登记 |

---

# 13. 风险

| 风险 | 应对 |
|---|---|
| `yt-dlp` 或字幕接口失效、IP 被限 | 放在接口后面；限速；本机运行；失败的视频标为 `FAILED` 以便下次重试 |
| YouTube 条款上的灰色地带 | 只抓少量频道，只供个人研究，不再分发字幕原文 |
| 中文自动字幕错字多 | 别名表收录常见错写；记录 `is_generated`；命中率可以分开统计 |
| 模型编造观点或代码 | 原句校验 + 代码校验，对不上的不进入汇总 |
| 观点模糊，难以归类 | 设 `CONDITIONAL` / `NEUTRAL`，保留原句供人工核对 |
| 样本小，命中率不可靠 | 设最小样本数和置换检验；回补数据只用于探索 |
| API 成本失控 | 每次运行前预估 token；设每次运行的上限；字幕只抓一次，可重复提取 |

---

# 14. 需要你确认

1. **频道**：第一版就用 Joseph Carlson、Financial Education、美投讲美股、视野环球财经这 4 个吗？
2. **回补**：S4 回补多久的历史？我建议 6 个月。
3. **股票范围**：只统计 `settings.yaml` 股票池里的股票，还是提到的美股都统计？我建议都统计，汇总时把股票池内的单独标出来。
4. **报告语言**：报告用中文写吗？
