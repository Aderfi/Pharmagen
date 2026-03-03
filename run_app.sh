#!/bin/bash
echo "💊 Launching Pharmagen Web Interface..."
uv run streamlit run src/interface/app.py --server.address 0.0.0.0
