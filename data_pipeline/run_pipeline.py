"""
Quick Start Script
Chạy pipeline một cách nhanh chóng với các tùy chọn phổ biến
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from pipeline_orchestrator import run_test_pipeline, run_pipeline
from db_config import DATA_CONFIG, print_config


def main():
    """Main function"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║         PORTFOLIO DATA PIPELINE - QUICK START                ║
╚══════════════════════════════════════════════════════════════╝

Chọn chế độ chạy:
    
1. 🧪 TEST MODE   - Chạy với 10 cổ phiếu, dữ liệu 1 tháng (nhanh)
2. 📊 SMALL MODE  - Chạy với 50 cổ phiếu, dữ liệu 3 tháng
3. 🚀 FULL MODE   - Chạy với tất cả cổ phiếu, dữ liệu từ .env (lâu)
4. ⚙️  SHOW CONFIG - Hiển thị cấu hình hiện tại từ .env
5. ❌ EXIT        - Thoát

""")
    
    choice = input("Nhập lựa chọn của bạn (1-5): ").strip()
    
    if choice == '1':
        print("\n🧪 Chạy TEST MODE...")
        print("="*60)
        success = run_test_pipeline(num_stocks=10)
        
    elif choice == '2':
        print("\n📊 Chạy SMALL MODE...")
        print("="*60)
        success = run_pipeline(
            csv_file=DATA_CONFIG['csv_file_path'],
            start_date='2024-09-01',
            end_date='2024-11-30',
            max_symbols=50
        )
        
    elif choice == '3':
        print("\n🚀 Chạy FULL MODE...")
        print("="*60)
        confirm = input("⚠️  Cảnh báo: Chế độ này sẽ mất nhiều thời gian (có thể > 30 phút). Tiếp tục? (y/n): ").strip().lower()
        
        if confirm == 'y':
            success = run_pipeline(
                csv_file=DATA_CONFIG['csv_file_path'],
                start_date=DATA_CONFIG['start_date'],
                end_date=DATA_CONFIG['end_date'],
                max_symbols=None  # All stocks
            )
        else:
            print("\n❌ Đã hủy")
            return
    
    elif choice == '4':
        print("\n⚙️  CẤU HÌNH HIỆN TẠI (từ .env):")
        print_config()
        return
            
    elif choice == '5':
        print("\n👋 Tạm biệt!")
        return
        
    else:
        print("\n❌ Lựa chọn không hợp lệ!")
        return
    
    # Print result
    print("\n" + "="*60)
    if success:
        print("✅ PIPELINE HOÀN THÀNH THÀNH CÔNG!")
        print("\n📊 Bước tiếp theo:")
        print("   Chạy dashboard: streamlit run scripts/dashboard.py")
    else:
        print("❌ PIPELINE THẤT BẠI!")
        print("\n🔍 Kiểm tra:")
        print("   1. PostgreSQL đang chạy?")
        print("   2. File CSV tồn tại?")
        print("   3. Kết nối internet ổn định?")
        print("   4. Cấu hình trong .env đúng chưa?")
    print("="*60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline bị gián đoạn bởi người dùng")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Lỗi: {e}")
        sys.exit(1)
