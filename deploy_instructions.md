# ☁️ How to Deploy Your AR-Bot Dashboard 24/7 for FREE

Our advanced **SQLite database auto-fallback layer** makes the entire platform extremely portable and lightweight. It requires no complex cloud setup or PostgreSQL databases, allowing you to host the live Streamlit GUI and bot completely free 24-7.

Here are the two easiest ways to deploy:

---

## Option 1: Streamlit Community Cloud (Recommended)
Streamlit Community Cloud is 100% free, runs Streamlit apps directly from GitHub, and provides 24-7 availability with automatic wakeup.

### Steps:
1. **Upload to GitHub:**
   - Create a new public or private repository on GitHub named `trading-bot`.
   - Push your code files (excluding `.env`, `__pycache__`, and `models/` weights) to GitHub:
     ```bash
     git init
     git add .
     git commit -m "Initial commit - Neural Net Bot"
     git remote add origin https://github.com/YOUR_USERNAME/trading-bot.git
     git branch -M main
     git push -u origin main
     ```
2. **Connect to Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io/) and log in using your GitHub account.
   - Click the **"New app"** button.
3. **Configure Deployment:**
   - **Repository:** Choose your `trading-bot` repository.
   - **Branch:** Select `main`.
   - **Main file path:** Type `streamlit_app.py`.
   - Click **"Deploy!"**
4. **Platform Live!**
   - Streamlit will automatically read `requirements.txt`, install all libraries, spin up the local SQLite database fallback, and host your dynamic trading GUI under a unique custom link (e.g. `https://ar-bot.streamlit.app`).

---

## Option 2: Hugging Face Spaces
Hugging Face offers free 24-7 server spaces with standard Streamlit integration. It is fully sandboxed, stable, and highly reliable.

### Steps:
1. **Create Space:**
   - Log in or sign up at [huggingface.co](https://huggingface.co/).
   - Click on your profile picture in the top-right and select **"New Space"**.
   - **Space Name:** Type `ar-bot`.
   - **SDK:** Choose **"Streamlit"**.
   - **Space License:** Choose `Apache-2.0` or leave default.
   - **Visibility:** Public or Private.
   - Click **"Create Space"**.
2. **Upload Files:**
   - Clone the space repository locally or upload files directly through the Hugging Face web interface.
   - Simply upload `streamlit_app.py`, `requirements.txt`, the `config/`, `data/`, `indicators/`, `ml/`, `utils/`, and `landing_page/` folders.
3. **Automated Boot:**
   - Hugging Face will automatically read the `requirements.txt`, compile all elements, and host your terminal under a free 24-7 URL!

---

## ⚡ Active Trading & WebSocket Integration
Because free cloud servers have execution timeouts, if you want your bot to run trading processes continuously in the background, you can:
- Put your Telegram API credentials in the environment variables (secrets) on the platform settings.
- Trigger model retraining directly from the live dashboard panel whenever you want to update neuron weights based on new market pricing!
