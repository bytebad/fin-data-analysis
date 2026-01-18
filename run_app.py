#!/usr/bin/env python3
"""
Script to run the Streamlit Document Q&A Assistant
"""

import subprocess
import sys
import os

# Disable uvloop to prevent asyncio compatibility issues
os.environ['UVLOOP_DISABLE'] = '1'

def main():
    """Run the Streamlit application"""
    print("🚀 Starting Document Q&A Assistant...")

    # Check if required environment variables are set
    required_env_vars = ["GEMINI_API_KEY", "LLAMA_API_KEY"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]

    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please create a .env file with these variables or set them in your environment.")
        print("Example .env file:")
        print("GEMINI_API_KEY=your_gemini_api_key_here")
        print("LLAMA_API_KEY=your_llama_api_key_here")
        sys.exit(1)

    # Check if streamlit is available
    try:
        import streamlit
        print("✅ Streamlit found")
    except ImportError:
        print("❌ Streamlit is not installed. Please run: pip install -r requirements.txt")
        sys.exit(1)

    # Run the Streamlit app
    try:
        print("🌐 Opening Streamlit app in browser...")
        print("📝 If the browser doesn't open automatically, visit: http://localhost:8501")

        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.headless", "true",
            "--server.port", "8501"
        ], check=True)

    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Streamlit app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()