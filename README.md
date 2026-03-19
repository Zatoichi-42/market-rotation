# market-rotation
  cd ~/Projects/market-rotation
  source .venv/bin/activate
  streamlit run dashboard/app.py
  Opens in your browser at http://localhost:8501.


  Or use the daily script (tests + compute + dashboard):

  python scripts/run_daily.py --test

  Console-only (no browser):

  python scripts/run_daily.py --no-dash

  ---
  How to Visually Verify (tab by tab)

  Tab 1: Regime Gate

  ┌─────────────────────────────┬──────────────────────────────────────────────────┐
  │        What you see         │                  How to verify                   │
  ├─────────────────────────────┼──────────────────────────────────────────────────┤
  │ VIX number +                │                                                  │
  │ NORMAL/FRAGILE/HOSTILE      │ Google "VIX index" — should match within ~0.5    │
  │ badge                       │                                                  │
  ├─────────────────────────────┼──────────────────────────────────────────────────┤
  │ VIX/VIX3M chart with 20/30  │ TradingView: CBOE:VIX overlay with CBOE:VIX3M.   │
  │ threshold lines             │ If VIX is below VIX3M (contango), chart lines    │
  │                             │ shouldn't cross — matches NORMAL                 │
  ├─────────────────────────────┼──────────────────────────────────────────────────┤
  │ Signal breakdown cards (4   │ Each shows raw value + color. VIX <20 = green,   │
  │ signals)                    │ 20-30 = orange, >30 = red. Cross-check term      │
  │                             │ structure with TradingView CBOE:VIX/CBOE:VIX3M   │
  └─────────────────────────────┴──────────────────────────────────────────────────┘

  Tab 2: Sector Rankings

  ┌────────────────────┬───────────────────────────────────────────────────────────┐
  │    What you see    │                       How to verify                       │
  ├────────────────────┼───────────────────────────────────────────────────────────┤
  │ Ranked table with  │ Go to finviz.com → Groups → S&P Sectors → sort by "Perf   │
  │ RS 5d/20d/60d      │ Month" column. Top/bottom should roughly match your ranks │
  │                    │  1 and 11                                                 │
  ├────────────────────┼───────────────────────────────────────────────────────────┤
  │ Sparklines (60-day │ TradingView: chart XLE/SPY (or whichever sector) on       │
  │  RS trend)         │ daily. Uptrend = positive RS, downtrend = negative RS     │
  ├────────────────────┼───────────────────────────────────────────────────────────┤
  │ Composite bar      │ Tallest bar = strongest sector. Should match Finviz sort  │
  │ chart              │ order                                                     │
  └────────────────────┴───────────────────────────────────────────────────────────┘

  Tab 3: Breadth

  ┌──────────────────────┬─────────────────────────────────────────────────────────┐
  │     What you see     │                      How to verify                      │
  ├──────────────────────┼─────────────────────────────────────────────────────────┤
  │ RSP/SPY ratio + 20d  │ TradingView: chart AMEX:RSP/AMEX:SPY. Rising line =     │
  │ change + z-score     │ HEALTHY, falling = NARROWING/DIVERGING                  │
  ├──────────────────────┼─────────────────────────────────────────────────────────┤
  │ SPY vs RSP indexed   │ If SPY is pulling away from RSP, that's breadth         │
  │ chart (bottom panel) │ divergence — the system should say NARROWING or         │
  │                      │ DIVERGING                                               │
  └──────────────────────┴─────────────────────────────────────────────────────────┘

  Tab 4: Replay

  ┌─────────────────┬──────────────────────────────────────────────────────────────┐
  │  What you see   │                        How to verify                         │
  ├─────────────────┼──────────────────────────────────────────────────────────────┤
  │ Date slider     │ Slide to a date you remember (e.g., a sell-off). Does the    │
  │ over 693        │ regime say HOSTILE/FRAGILE?                                  │
  │ snapshots       │                                                              │
  ├─────────────────┼──────────────────────────────────────────────────────────────┤
  │ "What happened  │ The system shows what the top-ranked sectors did 10/20 days  │
  │ AFTER" forward  │ later. If it called XLE #1 and XLE outperformed → system was │
  │ returns         │  right. If it underperformed → that's the kind of pattern    │
  │                 │ you'll tune for                                              │
  └─────────────────┴──────────────────────────────────────────────────────────────┘

  Tab 5: Debug

  ┌───────────────────────────────┬────────────────────────────────────────────────┐
  │         What you see          │                 How to verify                  │
  ├───────────────────────────────┼────────────────────────────────────────────────┤
  │ Raw JSON for every signal,    │ This is your ground truth. If something looks  │
  │ every RS value, every state   │ wrong on another tab, check here for the exact │
  │ explanation                   │  numbers                                       │
  ├───────────────────────────────┼────────────────────────────────────────────────┤
  │ VIX Data table (last 10 days) │ Compare the ratio column to TradingView        │
  │                               │ CBOE:VIX/CBOE:VIX3M                            │
  └───────────────────────────────┴────────────────────────────────────────────────┘

  Quick sanity checks (should take ~2 minutes):

  1. VIX: Dashboard says 25.1 → Google says ~25 → pass
  2. Top sector: Dashboard says XLE rank 1 → Finviz "Energy" is top 1-month performer →
   pass
  3. Bottom sector: Dashboard says XLB rank 11 → Finviz "Basic Materials" is bottom →
  pass
  4. Breadth: Dashboard says NARROWING → TradingView RSP/SPY trending down → pass
  5. Replay: Slide to a known crash date → regime should be HOSTILE → pass
