import os
from pathlib import Path
import sys

def create_desktop_launcher():
    print("=" * 60)
    print("AR-BOT: WINDOWS DESKTOP LAUNCHER GENERATOR")
    print("=" * 60)

    # 1. Dynamically locate the Desktop folder
    home = Path.home()
    desktop = home / "Desktop"
    
    if not desktop.exists():
        desktop = home / "OneDrive" / "Desktop"
        
    if not desktop.exists():
        # Fallback to home directory if desktop cannot be found
        desktop = home
        
    launcher_path = desktop / "ar-bot.bat"
    
    # 2. Get absolute path of this workspace
    workspace_path = Path(__file__).parent.absolute()
    
    # 3. Create the batch script content
    batch_content = f"""@echo off
title AR-BOT Meta-Trader Live Platform
color 0E
echo ============================================================
echo   🤖 LAUNCHING AR-BOT NEURAL NETWORK TRADING ECOSYSTEM 🤖
echo ============================================================
echo.
echo   Current Workspace: {workspace_path}
echo   Desktop Launcher:  {launcher_path}
echo.
echo   [1/3] Spinning up Live Streamlit GUI Dashboard...
cd /d "{workspace_path}"
start /b cmd /c "streamlit run streamlit_app.py --server.port 8501" > nul 2>&1

echo   [2/3] Starting Multi-Strategy Trading Bot Engine (Simulated WebSocket)...
:: We run train_model to ensure ML coefficients exist, then boot the dashboard app.
start /b cmd /c "python train_model.py --train --backtest" > nul 2>&1

echo   [3/3] Opening Premium HTML Landing Page & Dashboard GUI...
timeout /t 3 /nobreak > nul
start "" "http://localhost:8501"
start "" "{workspace_path}\\landing_page\\index.html"

echo.
echo ============================================================
echo   ✨ AR-BOT PLATFORM IS NOW ONLINE 24-7 READY! ✨
echo   Streamlit Dashboard: http://localhost:8501
echo   Landing Page Info:   {workspace_path}\\landing_page\\index.html
echo ============================================================
echo.
echo   Leave this terminal window open to keep the processes running.
echo   To shut down, press CTRL+C or close this window.
echo.
pause
"""
    
    # 4. Write the launcher batch file
    try:
        with open(launcher_path, "w", encoding="utf-8") as f:
            f.write(batch_content)
        print(f"[SUCCESS] Desktop launcher 'ar-bot.bat' created successfully!")
        print(f"[PATH] Saved at: {launcher_path}")
        print("\nYou can now double click this file on your Desktop to run the entire project!")
    except Exception as e:
        print(f"[ERROR] Failed to write file to Desktop: {e}")
        print("Saving launcher in the local workspace directory instead...")
        local_launcher = workspace_path / "ar-bot.bat"
        with open(local_launcher, "w", encoding="utf-8") as f:
            f.write(batch_content)
        print(f"[SUCCESS] Launcher saved locally in workspace at: {local_launcher}")
    print("=" * 60)

if __name__ == "__main__":
    create_desktop_launcher()
