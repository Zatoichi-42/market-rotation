   pkill -f streamlit; find . -name __pycache__ -exec rm -rf {} + 2>/dev/null
  source .venv/bin/activate && streamlit run dashboard/app.py 
