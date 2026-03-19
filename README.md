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
cd ~/Projects/market-rotation
  source .venv/bin/activate
                                                                                       
  # Kill any old Streamlit                                                             
  pkill -f streamlit 2>/dev/null                                                       
                                                                                       
  # Clear stale bytecode                                                               
  find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null                             
                                                                                       
  # Launch                                                        
  streamlit run dashboard/app.py                                                       
                                                                  
  Opens at http://localhost:8501                                                       
   
  Or console-only (no browser):                                                        
  python scripts/run_daily.py --no-dash                           
                                                                                       
  Or with tests first:                                                                 
  python scripts/run_daily.py --test                                                   
                                     
