"""
Chatbot UI Component - Giao diện chatbox tích hợp vào dashboard.
"""

import streamlit as st
from scripts.chatbot_service import PortfolioChatbot, create_quick_question_buttons


def get_welcome_message():
    """Trả về tin nhắn chào mừng"""
    return """Xin chào! 👋

Tôi là trợ lý AI của Portfolio Dashboard. Bạn cần giúp gì hôm nay? Hãy thử bắt đầu bằng việc:"""

def _clear_chat_history():
    """Xóa hoàn toàn lịch sử chat hiện tại."""
    st.session_state.chat_messages = []
    chatbot_instance = st.session_state.get("chatbot")
    if chatbot_instance is not None:
        chatbot_instance.clear_history()
    st.session_state.show_quick_questions = True
    st.session_state.is_thinking = False

def reset_chat_with_welcome():
    """Reset chat history và thêm lại welcome message"""
    _clear_chat_history()
    st.session_state.chat_messages.append({
        "role": "assistant",
        "content": get_welcome_message()
    })

def initialize_chatbot_session():
    """Khởi tạo session state cho chatbot"""
    if 'chatbot' not in st.session_state:
        try:
            from scripts.config import GEMINI_API_KEY
            
            # Kiểm tra API key có hợp lệ không
            if not GEMINI_API_KEY or GEMINI_API_KEY == "your-gemini-api-key-here":
                st.session_state.chatbot = None
                st.session_state.chatbot_error = "Chưa cấu hình GEMINI_API_KEY trong config.py"
            else:
                st.session_state.chatbot = PortfolioChatbot(GEMINI_API_KEY)
                
        except (ImportError, AttributeError) as e:
            st.session_state.chatbot = None
            st.session_state.chatbot_error = f"Lỗi import: {str(e)}"
        except Exception as e:
            st.session_state.chatbot = None
            st.session_state.chatbot_error = f"Lỗi khởi tạo chatbot: {str(e)}"
    
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
        # Thêm tin nhắn chào mở đầu từ chatbot
        if st.session_state.chatbot is not None:
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": get_welcome_message()
            })
    
    if 'show_quick_questions' not in st.session_state:
        st.session_state.show_quick_questions = True
    
    if 'is_thinking' not in st.session_state:
        st.session_state.is_thinking = False


def render_chat_controls(container, key_prefix="sidebar"):
    """Hiển thị các nút thao tác chatbot trong container được cung cấp."""
    is_thinking = st.session_state.get('is_thinking', False)
    col1, col2 = container.columns(2)
    if col1.button(
        "Xóa lịch sử",
        key=f"{key_prefix}_clear_btn",
        use_container_width=True,
        disabled=is_thinking
    ):
        _clear_chat_history()
        st.rerun()

    if col2.button(
        "Cuộc trò chuyện mới",
        key=f"{key_prefix}_new_btn",
        use_container_width=True,
        disabled=is_thinking
    ):
        reset_chat_with_welcome()
        st.rerun()


def render_chatbot_sidebar(portfolio_context=None):
    """
    Render giao diện chatbot trong sidebar
    
    Args:
        portfolio_context (dict): Context về danh mục đầu tư hiện tại (tùy chọn, tự động lấy nếu không có)
    """
    # Khởi tạo chatbot session
    initialize_chatbot_session()
    
    # Tự động lấy portfolio context từ session state nếu không được truyền vào
    if portfolio_context is None:
        portfolio_context = get_current_portfolio_context()
    
    # Kiểm tra lỗi cấu hình - hiển thị trước expander
    if st.session_state.chatbot is None:
        with st.sidebar:
            st.warning(st.session_state.get('chatbot_error', 'Lỗi khởi tạo chatbot'))
            st.info("Vui lòng thêm `GEMINI_API_KEY = 'your-api-key'` vào file scripts/config.py\n\nLấy API key miễn phí tại: https://makersuite.google.com/app/apikey")
        return
    
    # Tạo expander cho chatbot với các nút hành động bên trong
    chat_section = st.sidebar.expander("Trợ lý AI", expanded=False)

    with chat_section:
        st.markdown("Đặt câu hỏi về đầu tư, chiến lược, hoặc các chỉ số tài chính")
        
        # Các nút hành động ở đầu expander
        render_chat_controls(chat_section, key_prefix="sidebar")
        
        st.markdown("---")
        
        # Container với chiều cao cố định và thanh cuộn
        chat_container = st.container(height=400)
        with chat_container:
            # Hiển thị lịch sử chat
            for message in st.session_state.chat_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Hiển thị trạng thái đang suy nghĩ với animation
            if st.session_state.is_thinking:
                with st.chat_message("assistant"):
                    st.markdown("_Đang suy nghĩ..._")
            
            # Hiển thị các câu hỏi gợi ý nếu có ít hơn 2 tin nhắn (chỉ có welcome message)
            if len(st.session_state.chat_messages) <= 1 and st.session_state.show_quick_questions and not st.session_state.is_thinking:
                quick_questions = create_quick_question_buttons()

                for idx in range(0, len(quick_questions), 2):
                    row_questions = quick_questions[idx:idx + 2]
                    cols = st.columns(len(row_questions))
                    for col_idx, question in enumerate(row_questions):
                        with cols[col_idx]:
                            if st.button(
                                question,
                                key=f"quick_q_{idx + col_idx}",
                                use_container_width=True,
                                disabled=st.session_state.is_thinking,
                                type="secondary"
                            ):
                                handle_user_message(question, portfolio_context, chat_parent=chat_container)
                                st.session_state.show_quick_questions = False
                                st.rerun()
        
        # Input chat
        user_input = st.chat_input("Nhập câu hỏi của bạn...", disabled=st.session_state.is_thinking)
        
        if user_input and not st.session_state.is_thinking:
            handle_user_message(user_input, portfolio_context, chat_parent=chat_container)
            st.session_state.show_quick_questions = False
            st.rerun()



def handle_user_message(user_message, portfolio_context=None, chat_parent=None):
    """
    Xử lý tin nhắn từ người dùng
    
    Args:
        user_message (str): Tin nhắn từ người dùng
        portfolio_context (dict): Context về danh mục
    """
    # Thêm tin nhắn user vào chat
    st.session_state.chat_messages.append({
        "role": "user",
        "content": user_message
    })

    # Đánh dấu đang xử lý để disable input và hiển thị tiến trình
    st.session_state.is_thinking = True

    # Xác định container hiển thị chat hiện tại
    chat_context = chat_parent if chat_parent is not None else st

    # Hiển thị lại tin nhắn người dùng ngay lập tức trong khung chat hiện tại
    with chat_context.chat_message("user"):
        st.markdown(user_message)

    try:
        # Lấy context text nếu có
        context_text = None
        if portfolio_context:
            context_text = st.session_state.chatbot.get_portfolio_context(
                selected_stocks=portfolio_context.get('selected_stocks'),
                optimization_result=portfolio_context.get('optimization_result')
            )

        # Hiển thị spinner "đang suy nghĩ" trong khung chat của bot
        with chat_context.chat_message("assistant"):
            with st.spinner("Trợ lý đang suy nghĩ..."):
                response = st.session_state.chatbot.generate_response(
                    user_message,
                    context_text
                )
            st.markdown(response)

        # Lưu response vào lịch sử chat để hiển thị ở lượt rerun tiếp theo
        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": response
        })
    except Exception as e:
        error_message = f"Xin lỗi, đã có lỗi xảy ra: {str(e)}"
        with chat_context.chat_message("assistant"):
            st.error(error_message)
        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": error_message
        })
    finally:
        st.session_state.is_thinking = False


def get_current_portfolio_context():
    """
    Lấy context về danh mục đầu tư hiện tại từ session state
    
    Returns:
        dict: Context về danh mục
    """
    context = {}
    
    # Lấy danh sách cổ phiếu đã chọn từ dashboard
    if 'selected_stocks' in st.session_state and st.session_state.selected_stocks:
        context['selected_stocks'] = st.session_state.selected_stocks
    
    # Lấy kết quả tối ưu hóa nếu có
    if 'optimization_result' in st.session_state and st.session_state.optimization_result:
        context['optimization_result'] = st.session_state.optimization_result
    
    return context if context else None


def render_chatbot_page():
    """
    Render trang chatbot đầy đủ (không phải sidebar)
    """
    # Khởi tạo chatbot session
    initialize_chatbot_session()
    
    # CSS cho giao diện chatbot
    st.markdown("""
        <style>
        /* Animation cho thinking dots */
        @keyframes blink {
            0%, 100% { opacity: 0.3; }
            50% { opacity: 1; }
        }
        
        /* Styling cho tin nhắn người dùng */
        [data-testid="stChatMessage"][data-testid*="user"] {
            background-color: #E3F2FD;
            border-radius: 18px;
            padding: 12px 16px;
            margin-left: 20%;
            margin-bottom: 12px;
        }
        
        /* Styling cho tin nhắn bot */
        [data-testid="stChatMessage"][data-testid*="assistant"] {
            background-color: #F5F5F5;
            border-radius: 18px;
            padding: 12px 16px;
            margin-right: 20%;
            margin-bottom: 12px;
        }
        
        /* Thinking indicator */
        .thinking-indicator {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            padding: 10px 15px;
            background-color: #F5F5F5;
            border-radius: 15px;
            margin-left: 10px;
        }
        
        .thinking-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: #666;
            animation: blink 1.4s infinite;
        }
        
        .thinking-dot:nth-child(2) {
            animation-delay: 0.2s;
        }
        
        .thinking-dot:nth-child(3) {
            animation-delay: 0.4s;
        }
        
        /* Input bar styling với focus effect */
        .stChatInputContainer {
            border: 2px solid #E0E0E0;
            border-radius: 25px;
            padding: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            transition: all 0.3s ease;
        }
        
        .stChatInputContainer:focus-within {
            border-color: #2196F3;
            box-shadow: 0 4px 12px rgba(33,150,243,0.2);
            transform: translateY(-1px);
        }
        
        /* Styling cho các nút gợi ý */
        .stButton > button[kind="secondary"] {
            border-radius: 20px;
            border: 1px solid #E0E0E0;
            background: white;
            padding: 10px 20px;
            transition: all 0.3s ease;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        .stButton > button[kind="secondary"]:hover {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-color: transparent;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(102,126,234,0.3);
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header với title và các nút hành động
    col_title = st.columns([1])[0]
    with col_title:
        st.title("Trợ lý AI - Tư vấn đầu tư")
        st.markdown("Đặt câu hỏi về đầu tư, chiến lược, hoặc các chỉ số tài chính")
    
    st.markdown("---")
    
    # Kiểm tra lỗi cấu hình
    if st.session_state.chatbot is None:
        st.error(st.session_state.get('chatbot_error', 'Lỗi khởi tạo chatbot'))
        st.info("Vui lòng thêm GEMINI_API_KEY vào file scripts/config.py. Lấy API key miễn phí tại: https://makersuite.google.com/app/apikey")
        return
    
    # Container cho chat với chiều cao lớn hơn
    chat_container = st.container(height=500)
    with chat_container:
        # Hiển thị lịch sử chat
        for idx, message in enumerate(st.session_state.chat_messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Hiển thị trạng thái đang suy nghĩ với animation
        if st.session_state.is_thinking:
            with st.chat_message("assistant"):
                st.markdown("""
                <div class="thinking-indicator">
                    <div class="thinking-dot"></div>
                    <div class="thinking-dot"></div>
                    <div class="thinking-dot"></div>
                </div>
                """, unsafe_allow_html=True)
        
        # Hiển thị các câu hỏi gợi ý dưới dạng chips có thể click
        if len(st.session_state.chat_messages) <= 1 and st.session_state.show_quick_questions and not st.session_state.is_thinking:
            quick_questions = create_quick_question_buttons()

            for idx in range(0, len(quick_questions), 2):
                row_questions = quick_questions[idx:idx + 2]
                cols = st.columns(len(row_questions))
                for col_idx, question in enumerate(row_questions):
                    with cols[col_idx]:
                        if st.button(
                            question,
                            key=f"page_quick_q_{idx + col_idx}",
                            use_container_width=True,
                            disabled=st.session_state.is_thinking,
                            type="secondary"
                        ):
                            portfolio_context = get_current_portfolio_context()
                            handle_user_message(question, portfolio_context, chat_parent=chat_container)
                            st.session_state.show_quick_questions = False
                            st.rerun()
    
    # Input chat ở ngoài container để không bị cuộn
    user_input = st.chat_input("Nhập câu hỏi của bạn...", disabled=st.session_state.is_thinking)
    
    if user_input and not st.session_state.is_thinking:
        portfolio_context = get_current_portfolio_context()
        handle_user_message(user_input, portfolio_context, chat_parent=chat_container)
        st.session_state.show_quick_questions = False
        st.rerun()

