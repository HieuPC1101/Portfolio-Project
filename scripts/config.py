from datetime import datetime, timedelta

from .chatbot_service import load_gemini_api_key

# Cấu hình thời gian phân tích
ANALYSIS_END_DATE = datetime.now().strftime("%Y-%m-%d")  # Hôm nay
ANALYSIS_START_DATE = (datetime.now() - timedelta(days=365*3)).strftime("%Y-%m-%d")  # Hôm nay trừ 3 năm

# Các thông số khác
DEFAULT_MARKET = "HOSE"  # Sàn giao dịch mặc định
CACHE_EXPIRY_HOURS = 24  # Thời gian hết hạn cache (giờ)
DEFAULT_INVESTMENT_AMOUNT = 1000000  # Số tiền đầu tư mặc định (VND)

GEMINI_API_KEY = load_gemini_api_key()
