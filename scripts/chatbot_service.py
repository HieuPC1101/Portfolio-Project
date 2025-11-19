"""
Chatbot Service - Xử lý logic AI chatbot cho ứng dụng danh mục đầu tư.
Sử dụng Google Gemini API (miễn phí)
"""

import os
from datetime import datetime
import google.generativeai as genai
import pandas as pd
import streamlit as st


def load_gemini_api_key() -> str:
    """Ưu tiên sử dụng biến môi trường rồi tới tệp bí mật cục bộ."""
    env_key = os.getenv("GEMINI_API_KEY")
    if env_key:
        return env_key

    try:
        from .secret_config import GEMINI_API_KEY as secret_key  # type: ignore

        return secret_key
    except ImportError:
        return ""


class PortfolioChatbot:
    """Lớp xử lý chatbot hỗ trợ tư vấn danh mục đầu tư"""
    
    def __init__(self, api_key):
        """
        Khởi tạo chatbot với Google Gemini API key
        
        Args:
            api_key (str): Google Gemini API key
        """
        self.api_key = api_key
        genai.configure(api_key=api_key)
        
        # Cấu hình safety settings
        self.safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            }
        ]
        
        # Sử dụng Gemini Flash Latest với safety settings
        self.model = genai.GenerativeModel(
            'gemini-flash-latest',
            safety_settings=self.safety_settings
        )
        self.conversation_history = []
        
    def get_system_prompt(self, portfolio_data=None):
        """
        Tạo system prompt cho chatbot với thông tin về danh mục
        
        Args:
            portfolio_data (dict): Dữ liệu danh mục đầu tư hiện tại
            
        Returns:
            str: System prompt
        """
        base_prompt = (
    "Bạn là một trợ lý AI chuyên về tư vấn đầu tư chứng khoán tại thị trường Việt Nam.\n\n"
    "**Nhiệm vụ chính:**\n"
    "- Tư vấn chiến lược đầu tư (ngắn hạn, trung hạn, dài hạn).\n"
    "- Giải thích các chỉ số tài chính và phương pháp định giá doanh nghiệp.\n"
    "- Phân tích các rủi ro (thị trường, ngành, cổ phiếu cụ thể).\n"
    "- Tối ưu hóa danh mục đầu tư dựa trên các tiêu chí người dùng cung cấp như khẩu vị rủi ro, mục tiêu lợi nhuận, và thời gian đầu tư.\n\n"
    "**Nguyên tắc hoạt động:**\n"
    "1.  **Dựa trên dữ liệu:** Luôn phân tích dựa trên dữ liệu thực tế, cập nhật. Ưu tiên sử dụng thông tin từ báo cáo tài chính của công ty, dữ liệu giao dịch và các nguồn tin tài chính uy tín tại Việt Nam.\n"
    "2.  **Cụ thể và rõ ràng:** Đưa ra lời khuyên cụ thể, có luận điểm. Giải thích các thuật ngữ và phương pháp phân tích một cách đơn giản, dễ hiểu.\n"
    "3.  **Cá nhân hóa:** Nếu người dùng cung cấp thông tin về danh mục đầu tư hiện tại, hãy sử dụng thông tin đó làm cơ sở để đưa ra các đề xuất tối ưu và phù hợp nhất.\n\n"
    "**Định dạng trả lời:**\n"
    "- Sử dụng tiếng Việt, văn phong ngắn gọn, đi thẳng vào vấn đề, không bỏ sót câu trả lời.\n"
    "- Ưu tiên các giải pháp và hành động mang tính thực tiễn.\n\n"
)

        if portfolio_data:
            base_prompt += f"\n\nThông tin danh mục:\n{portfolio_data}"
            
        return base_prompt
    
    def add_message_to_history(self, role, content):
        """
        Thêm tin nhắn vào lịch sử hội thoại
        
        Args:
            role (str): 'user' hoặc 'assistant'
            content (str): Nội dung tin nhắn
        """
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now()
        })
        
        # Giới hạn lịch sử chỉ giữ 10 tin nhắn gần nhất để tránh vượt token limit
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
    
    def generate_response(self, user_message, portfolio_context=None):
        """
        Sinh câu trả lời từ chatbot sử dụng Google Gemini
        
        Args:
            user_message (str): Câu hỏi từ người dùng
            portfolio_context (dict): Thông tin context về danh mục
            
        Returns:
            str: Câu trả lời từ chatbot
        """
        try:
            # Thêm tin nhắn của user vào lịch sử
            self.add_message_to_history("user", user_message)
            
            # Tạo prompt đầy đủ với system prompt và context
            full_prompt = self.get_system_prompt(portfolio_context) + "\n\n"
            
            # Thêm lịch sử hội thoại
            for msg in self.conversation_history[-6:]:  # Chỉ lấy 6 tin nhắn gần nhất
                role_label = "Người dùng" if msg["role"] == "user" else "Trợ lý"
                full_prompt += f"{role_label}: {msg['content']}\n\n"
            
            # Debug: In ra để kiểm tra
            print(f"[DEBUG] Sending prompt to Gemini API...")
            
            # Gọi Google Gemini API
            response = self.model.generate_content(
                full_prompt,
                generation_config={
                    'temperature': 0.7,
                    'top_p': 0.95,
                    'top_k': 40,
                    'max_output_tokens': 2048,
                }
            )
            
            print(f"[DEBUG] Received response from Gemini API")
            
            # Kiểm tra response có bị block không
            if hasattr(response, 'prompt_feedback') and hasattr(response.prompt_feedback, 'block_reason'):
                if response.prompt_feedback.block_reason:
                    error_msg = f"Câu hỏi bị chặn do: {response.prompt_feedback.block_reason}. Vui lòng thử câu hỏi khác."
                    print(f"[DEBUG] Blocked: {error_msg}")
                    return error_msg
            
            # Kiểm tra response có parts không
            if not response or not hasattr(response, 'parts') or not response.parts:
                error_msg = "Xin lỗi, tôi không nhận được phản hồi từ AI. Vui lòng thử lại."
                print(f"[DEBUG] No parts in response")
                return error_msg
            
            # Lấy text từ response
            try:
                assistant_message = response.text
                print(f"[DEBUG] Got response text: {len(assistant_message)} chars")
            except Exception as text_error:
                error_msg = f"Xin lỗi, không thể đọc phản hồi: {str(text_error)}. Vui lòng thử lại."
                print(f"[DEBUG] Error getting text: {text_error}")
                return error_msg
            
            # Kiểm tra message có nội dung không
            if not assistant_message or assistant_message.strip() == "":
                error_msg = "Xin lỗi, câu trả lời trống. Vui lòng thử câu hỏi khác."
                print(f"[DEBUG] Empty response")
                return error_msg
            
            # Thêm response vào lịch sử
            self.add_message_to_history("assistant", assistant_message)
            
            return assistant_message
            
        except AttributeError as e:
            # Lỗi khi response không có text
            return "Xin lỗi, không thể lấy câu trả lời. Vui lòng thử câu hỏi khác."
        except Exception as e:
            error_message = f"Xin lỗi, đã có lỗi xảy ra: {str(e)}"
            if "API" in str(e) or "key" in str(e).lower():
                error_message = "Vui lòng kiểm tra GEMINI_API_KEY trong file config.py"
            return error_message
    
    def clear_history(self):
        """Xóa lịch sử hội thoại"""
        self.conversation_history = []
    
    def get_portfolio_context(self, selected_stocks=None, optimization_result=None):
        """
        Tạo context về danh mục đầu tư hiện tại
        
        Args:
            selected_stocks (list): Danh sách mã cổ phiếu đã chọn
            optimization_result (dict): Kết quả tối ưu hóa danh mục
            
        Returns:
            str: Thông tin context dạng text
        """
        context_parts = []
        
        if selected_stocks:
            context_parts.append(f"Các cổ phiếu đang xem xét: {', '.join(selected_stocks)}")
        
        if optimization_result:
            if "Trọng số danh mục" in optimization_result:
                weights_str = ", ".join([f"{k}: {v:.2%}" for k, v in optimization_result["Trọng số danh mục"].items()])
                context_parts.append(f"Trọng số phân bổ: {weights_str}")
            
            if "Lợi nhuận kỳ vọng" in optimization_result:
                context_parts.append(f"Lợi nhuận kỳ vọng: {optimization_result['Lợi nhuận kỳ vọng']:.2%}")
            
            if "Rủi ro (Độ lệch chuẩn)" in optimization_result:
                context_parts.append(f"Rủi ro: {optimization_result['Rủi ro (Độ lệch chuẩn)']:.2%}")
            
            if "Tỷ lệ Sharpe" in optimization_result:
                context_parts.append(f"Sharpe Ratio: {optimization_result['Tỷ lệ Sharpe']:.4f}")
        
        return "\n".join(context_parts) if context_parts else None


def create_quick_question_buttons():
    """
    Tạo các nút câu hỏi nhanh cho người dùng
    
    Returns:
        list: Danh sách các câu hỏi nhanh
    """
    quick_questions = [
        "Giải thích mô hình Markowitz",
        "Làm sao để giảm rủi ro?",
        "Các chỉ số tài chính quan trọng",
        "Nên đa dạng hóa bao nhiêu cổ phiếu?",
    ]
    return quick_questions
