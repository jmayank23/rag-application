import streamlit as st

def display_landing_page():
    """
    Display the landing page for setting up a new chat session
    """
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 30px;">
            <h1>Welcome to Langchain RAG Chatbot</h1>
            <p>Set up your chat environment and start a new conversation</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Create a card-like container for settings
    with st.container():
        st.markdown(
            """
            <div style="padding: 10px; margin-bottom: 20px; border-radius: 10px; background-color: #1e2130; border: 1px solid #4e5d95;">
                <h2 style="text-align: center; color: #ffffff;">Configure Your Chat Session</h2>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Display settings in columns for better organization
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Settings")
            
            # LLM Model Selection
            model_options = ["gpt-4o", "gpt-4o-mini"]
            st.selectbox(
                "Select LLM Model", 
                options=model_options, 
                index=model_options.index(st.session_state.get("model", "gpt-4o-mini")),
                key="model"
            )
            
            # Add a description about models
            st.markdown(
                """
                <div style="font-size: 0.8em; color: #adb5bd;">
                <p><strong>gpt-4o:</strong> More powerful but slower</p>
                <p><strong>gpt-4o-mini:</strong> Faster but less capable</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col2:
            st.subheader("Vector Database Settings")
            
            # Vector Database Selection
            vector_db_options = ["chromadb", "pinecone"]
            st.selectbox(
                "Select Vector Database", 
                options=vector_db_options, 
                index=vector_db_options.index(st.session_state.get("vector_db", "chromadb")),
                key="vector_db"
            )
            
            # Embedding Model Selection
            embedding_model_options = ["openai", "huggingface"]
            st.selectbox(
                "Select Embedding Model", 
                options=embedding_model_options, 
                index=embedding_model_options.index(st.session_state.get("embedding_model", "openai")),
                key="embedding_model"
            )
            
            # Add a description about vector stores and embeddings
            st.markdown(
                """
                <div style="font-size: 0.8em; color: #adb5bd;">
                <p><strong>Vector Database:</strong> Where document embeddings are stored</p>
                <p><strong>Embedding Model:</strong> Converts text to numerical representations</p>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    # Store the selected values in session state
    st.session_state.previous_vector_db = st.session_state.vector_db
    st.session_state.previous_embedding_model = st.session_state.embedding_model
    
    # Center the start button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Create a prominent start button
        if st.button("Start New Chat", key="start_new_chat_landing", use_container_width=True):
            # Set flag to indicate we're now in an active chat
            st.session_state.in_chat_session = True
            # Reset chat-specific state
            st.session_state.messages = []
            st.session_state.session_id = None
            st.session_state.current_session_id = None
            st.rerun()
    
    # Show existing sessions if there are any
    if st.session_state.sessions and len(st.session_state.sessions) > 0:
        st.markdown(
            """
            <div style="padding: 10px; margin-top: 30px; margin-bottom: 20px; border-radius: 10px; background-color: #1e2130; border: 1px solid #4e5d95;">
                <h2 style="text-align: center; color: #ffffff;">Previous Chat Sessions</h2>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Create a table of previous sessions
        session_data = []
        for session_id, session in st.session_state.sessions.items():
            session_data.append({
                "id": session_id,
                "name": session.get("name", "Unnamed Session"),
                "timestamp": session.get("timestamp", "Unknown time"),
                "model": session.get("model", "Unknown model"),
                "messages": len(session.get("messages", [])) // 2  # Divide by 2 to get conversation turns
            })
        
        # Sort sessions by timestamp (newest first)
        import datetime
        session_data.sort(key=lambda x: x["timestamp"], reverse=True)
        
        # Show sessions in a more visual way
        for i, session in enumerate(session_data):
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f"**{session['name']}**")
                timestamp = datetime.datetime.fromisoformat(session["timestamp"]).strftime("%Y-%m-%d %H:%M") if isinstance(session["timestamp"], str) else "Unknown time"
                st.markdown(f"<small>{timestamp} | {session['messages']} messages</small>", unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"<small>Model: {session['model']}</small>", unsafe_allow_html=True)
            
            with col3:
                if st.button("Load", key=f"load_session_{i}"):
                    # Load the selected session
                    from sidebar import load_session
                    load_session(session["id"])
                    st.session_state.in_chat_session = True
                    st.rerun()
            
            st.markdown("---")