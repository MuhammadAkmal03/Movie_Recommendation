import streamlit as st
import os

def render_agent_tab(models):
    """
    Render the AI Agent tab content
    """
    st.markdown("### 🤖 AI Movie Assistant")
    st.markdown("Chat with AI to find your perfect movie! Ask questions in natural language.")
    
    # Try to load API key from .env
    api_key_from_env = os.getenv("GEMINI_API_KEY")
    
    # API Key input
    if api_key_from_env:
        api_key = api_key_from_env
    else:
        st.info("💡 **Tip**: Add `GEMINI_API_KEY=your_key` to `.env` file to avoid entering it every time!")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            api_key = st.text_input(
                "Enter your Google Gemini API Key:",
                type="password",
                help="Get your free API key from https://aistudio.google.com/app/apikey"
            )
        with col2:
            st.markdown("[Get API Key](https://aistudio.google.com/app/apikey)")
    
    if not api_key:
        st.info("👆 Enter your Gemini API key above to start chatting!")
        
        # Show example questions
        st.markdown("### 💡 What you can ask:")
        st.markdown("""
        - "Show me movies like Inception"
        - "I want to watch Nolan movies"
        - "Recommend something funny but not stupid"
        - "Tell me about The Dark Knight"
        - "Find sci-fi movies with time travel"
        """)
        return
    
    # Initialize agent
    if 'agent' not in st.session_state or st.session_state.get('api_key') != api_key:
        try:
            # Import here to avoid app crash if dependency is missing
            from src.agent.movie_agent import MovieAgent
            
            with st.spinner("Initializing AI Agent..."):
                st.session_state.agent = MovieAgent(api_key, models)
                st.session_state.api_key = api_key
                st.success("✅ AI Assistant ready!")
                
        except ImportError:
            st.error("❌ `google-generativeai` package not found.")
            st.info("Please run: `pip install google-generativeai`")
            st.session_state.agent = None
            return
        except Exception as e:
            st.error(f"❌ Failed to initialize agent: {str(e)}")
            st.session_state.agent = None
            return
    
    # Only show chat if agent initialized successfully
    if st.session_state.get('agent') is None:
        st.warning("⚠️ AI Assistant not available. Please fix the error above.")
        return

    # Initialize chat history
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    
    # Quick action buttons
    st.markdown("### 🎬 Quick Actions")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("💎 Hidden Gems", use_column_width=True):
            st.session_state.quick_prompt = "Recommend a highly rated underrated movie that I might have missed"
    with col2:
        if st.button("🍿 Blockbusters", use_column_width=True):
            st.session_state.quick_prompt = "What are the top grossing movies of the last decade?"
    with col3:
        if st.button("😂 Comedy Night", use_column_width=True):
            st.session_state.quick_prompt = "Suggest some hilarious comedy movies for a movie night"
    with col4:
        if st.button("🤯 Mind Blowers", use_column_width=True):
            st.session_state.quick_prompt = "Show me movies with crazy plot twists like Inception or Shutter Island"
    
    
    st.markdown("---")
    
    # Display chat messages
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Handle quick prompts
    if 'quick_prompt' in st.session_state:
        prompt = st.session_state.quick_prompt
        del st.session_state.quick_prompt
        _process_message(prompt)
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about movies..."):
        _process_message(prompt)
    
    # Clear chat button at bottom
    if st.session_state.chat_messages:  # Only show if there are messages
        if st.button("🗑️ Clear Chat History", use_column_width=True, type="secondary"):
            st.session_state.chat_messages = []
            st.session_state.agent.reset_conversation()
            st.rerun()


def _process_message(prompt):
    """Process a user message and generate response"""
    # Add user message
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.agent.chat(prompt)
                st.markdown(response)
                st.session_state.chat_messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_msg = f"I apologize, but I encountered an error: {str(e)}"
                st.error(error_msg)
                st.session_state.chat_messages.append({"role": "assistant", "content": error_msg})
    
    # Force rerun to update chat history display if needed
    # st.rerun() # Optional, usually not needed with chat_message
