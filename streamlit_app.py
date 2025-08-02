"""
Streamlit Cloud entry point for Nomad AI - Lifestyle Discovery Assistant.
"""

import sys
import os

# Add src directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

# Import and run the main app
try:
    # Import the main app from src/web_app/app.py
    sys.path.append('src/web_app')
    from app import main
    
    if __name__ == "__main__":
        main()
        
except ImportError as e:
    import streamlit as st
    st.error(f"Import error: {e}")
    st.info("Please check that all dependencies are installed correctly.")
    
except Exception as e:
    import streamlit as st
    st.error(f"Application error: {e}")
    st.info("Please check the application logs for more details.")
