import streamlit as st
import google.generativeai as genai
import re
from prompts import SYSTEM_PROMPTS

# 1. Page Config
st.set_page_config(layout="wide", page_title="CORN AX Mentoring Service")

# 2. Sidebar
with st.sidebar:
    st.title("🌽 CORN AX Mentoring")
    
    # API Key Input
    api_key = st.text_input("Google API Key", type="password")
    
    # Model Selection
    available_models = []
    if api_key:
        genai.configure(api_key=api_key)
        try:
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    available_models.append(m.name)
        except Exception as e:
            st.error(f"API Key Error: {e}")
    
    selected_model_name = st.selectbox(
        "사용할 모델 선택",
        available_models if available_models else ["gemini-pro"],
        index=0 if available_models else 0
    )
    
    # Navigation
    st.markdown("---")
    st.subheader("Mentoring Step")
    selected_step = st.radio(
        "단계를 선택하세요",
        ["1. 시장조사", "2. 문제정의", "3. 해결책"],
        index=0
    )
    
    # Map selection to internal keys
    step_map = {
        "1. 시장조사": "market",
        "2. 문제정의": "problem",
        "3. 해결책": "solution"
    }
    current_step_key = step_map[selected_step]
    
    st.markdown("---")
    if st.button("대화 초기화"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()

# 3. Session State Initialization
if "history" not in st.session_state:
    st.session_state["history"] = {
        "market": [],
        "problem": [],
        "solution": []
    }

if "output" not in st.session_state:
    st.session_state["output"] = {
        "market": "",
        "problem": "",
        "solution": ""
    }

# 4. Main Layout
if not api_key:
    st.warning("왼쪽 사이드바에 Google API Key를 입력해주세요.")
    st.stop()

left_col, right_col = st.columns(2)

# Helper function to extract output
def parse_response(text):
    pattern = r"\[\[OUTPUT\]\](.*?)\[\[/OUTPUT\]\]"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip(), re.sub(pattern, "", text, flags=re.DOTALL).strip()
    return None, text

# Helper function to get context
def get_context(current_step):
    context = ""
    if current_step == "problem":
        market_out = st.session_state["output"]["market"]
        if market_out:
            context += f"\n[이전 단계(시장조사) 결과]\n{market_out}\n"
    elif current_step == "solution":
        market_out = st.session_state["output"]["market"]
        problem_out = st.session_state["output"]["problem"]
        if market_out:
            context += f"\n[이전 단계(시장조사) 결과]\n{market_out}\n"
        if problem_out:
            context += f"\n[이전 단계(문제정의) 결과]\n{problem_out}\n"
    return context

# 5. Chat Interface (Left Panel)
with left_col:
    st.header(f"💬 {selected_step} Chat")
    
    # Display chat history
    for msg in st.session_state["history"][current_step_key]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    # Chat Input
    if prompt := st.chat_input("내용을 입력하세요..."):
        # User message
        st.session_state["history"][current_step_key].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # AI Response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                # Prepare model and history with safety settings
                safety_settings = [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                ]
                
                model = genai.GenerativeModel(
                    selected_model_name,
                    safety_settings=safety_settings
                )
                
                # Construct history for API
                api_history = []
                # Add system prompt
                api_history.append({"role": "user", "parts": [SYSTEM_PROMPTS[current_step_key]]})
                api_history.append({"role": "model", "parts": ["네, 알겠습니다. 주어진 역할과 지시사항에 따라 멘토링을 진행하겠습니다."]})
                
                # Add context if exists (as a system injection or pre-prompt)
                context = get_context(current_step_key)
                if context:
                     api_history.append({"role": "user", "parts": [f"참고할 이전 단계 데이터입니다:\n{context}"]})
                     api_history.append({"role": "model", "parts": ["네, 이전 단계 데이터를 참고하여 답변하겠습니다."]})

                # Add conversation history
                for msg in st.session_state["history"][current_step_key]:
                    role = "user" if msg["role"] == "user" else "model"
                    api_history.append({"role": role, "parts": [msg["content"]]})
                
                # Generate response
                chat = model.start_chat(history=api_history)
                response = chat.send_message(prompt, stream=True)
                
                for chunk in response:
                    # Check if chunk has text before accessing it
                    if hasattr(chunk, 'text') and chunk.text:
                        full_response += chunk.text
                        message_placeholder.markdown(full_response + "▌")
                
                # If no response was generated, show a message
                if not full_response:
                    full_response = "죄송합니다. 응답을 생성하는 중 문제가 발생했습니다. 질문을 다시 입력해 주시거나, 다른 방식으로 표현해 주세요."
                
                message_placeholder.markdown(full_response)
                
                # Parse output
                output_content, chat_content = parse_response(full_response)
                
                # Update session state
                st.session_state["history"][current_step_key].append({"role": "assistant", "content": chat_content})
                if output_content:
                    st.session_state["output"][current_step_key] = output_content
                    
                # Force refresh to show output in right col
                st.rerun()
                
            except Exception as e:
                error_msg = str(e)
                if "finish_reason" in error_msg or "valid Part" in error_msg:
                    st.error("⚠️ AI 응답이 안전 필터에 의해 차단되었거나 생성되지 않았습니다. 질문을 다르게 표현해 주세요.")
                else:
                    st.error(f"Error: {error_msg}")

# 6. Output Viewer (Right Panel)
with right_col:
    st.header(f"📝 {selected_step} 산출물")
    st.markdown("---")
    
    current_output = st.session_state["output"][current_step_key]
    if current_output:
        # Display the output
        st.markdown(current_output)
        
        # Add buttons at the bottom
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📋 문서 복사하기", key=f"copy_{current_step_key}"):
                # Streamlit doesn't support direct clipboard access, so we'll use a text area
                st.session_state[f"show_copy_{current_step_key}"] = True
        
        with col2:
            st.link_button("🚀 구글문서로 바로가기", "https://docs.google.com/document/create?usp=docs_web")
        
        # Show copyable text area if copy button was clicked
        if st.session_state.get(f"show_copy_{current_step_key}", False):
            st.text_area(
                "아래 내용을 복사하세요 (Ctrl+A, Ctrl+C):",
                current_output,
                height=200,
                key=f"copy_area_{current_step_key}"
            )
            if st.button("닫기", key=f"close_copy_{current_step_key}"):
                st.session_state[f"show_copy_{current_step_key}"] = False
                st.rerun()
    else:
        st.info("아직 생성된 산출물이 없습니다. 왼쪽 채팅창에서 대화를 시작하세요.")

