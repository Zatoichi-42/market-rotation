---
name: data_source_disclaimers
description: Always show all available data sources with disclaimers when lagging — format live data (alt data - source/disclaimer) [lagging data - source/disclaimer]
type: feedback
---

Always use ALL available data sources in summaries, with clear disclaimers for any that lag.

Format: live data (alt data — source/disclaimer) [lagging data — source/disclaimer]

**Why:** The user wants full transparency on what's driving each signal and how fresh each data point is. Never silently omit a source — show it with its lag caveat so the user can weigh confidence accordingly.

**How to apply:** Every explanation string, dashboard caption, and console summary that references a signal should cite all sources feeding it. If FRED OAS is available alongside HYG/LQD, show both. Tag real-time sources as live, tag FRED/macro as lagging with business-day delay noted.
