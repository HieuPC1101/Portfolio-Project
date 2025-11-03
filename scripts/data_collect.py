import pandas as pd
import schedule
import time
from vnstock import Vnstock
import os
import logging
import argparse

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Đường dẫn đến thư mục 'data'
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(data_dir, exist_ok=True)
file_path = os.path.join(data_dir, "company_info.csv")

# Hàm lưu DataFrame vào file CSV
def save_to_csv(df, file_path):
    try:
        # Luôn ghi đè dữ liệu mới lên dữ liệu cũ
        df.to_csv(file_path, mode='w', header=True, index=False)
        logging.info(f"Dữ liệu đã được ghi đè lên file {file_path}!")
    except Exception as e:
        logging.error(f"Lỗi khi lưu dữ liệu vào CSV: {e}")

# Hàm chính để thu thập dữ liệu và lưu vào CSV
def run_task():
    logging.info("Bắt đầu thu thập dữ liệu...")
    try:
        stock = Vnstock().stock(symbol='VN30F1M', source='VCI')
        logging.info("Đã khởi tạo đối tượng Vnstock.")
        stock_icb = stock.listing.symbols_by_industries()[['symbol', 'organ_name', 'icb_code1']]
        logging.info(f"Đã lấy danh sách ngành và mã chứng khoán theo ICB: {len(stock_icb)} dòng.")
        icb_all = stock.listing.industries_icb()[['icb_name', 'icb_code']]
        logging.info(f"Đã lấy dữ liệu các ngành ICB: {len(icb_all)} dòng.")
        result2 = pd.merge(stock_icb, icb_all, left_on='icb_code1', right_on='icb_code', how='inner')[['symbol', 'organ_name', 'icb_name']]
        logging.info(f"Đã tạo bảng kết quả ngành: {len(result2)} dòng.")
        def get_stocks_by_exchange(stock, exchange):
            symbols = stock.listing.symbols_by_group(exchange).tolist()
            logging.info(f"Đã lấy {len(symbols)} mã từ sàn {exchange}.")
            return pd.DataFrame({"symbol": symbols, "exchange": exchange})
        def get_all_stocks(stock):
            exchanges = ['HOSE', 'HNX', 'UPCOM']
            stock_dfs = [get_stocks_by_exchange(stock, exchange) for exchange in exchanges]
            return pd.concat(stock_dfs, ignore_index=True)
        all_stocks_df = get_all_stocks(stock)
        logging.info(f"Tổng số mã chứng khoán từ các sàn: {len(all_stocks_df)}.")
        result3 = pd.merge(result2, all_stocks_df, on='symbol', how='inner')
        logging.info(f"Kết quả cuối cùng sau khi ghép: {len(result3)} dòng.")
        save_to_csv(result3, file_path)
        logging.info("Công việc đã hoàn thành và dữ liệu được cập nhật!")
    except Exception as e:
        logging.error(f"Lỗi trong quá trình thu thập dữ liệu: {e}")

# Chế độ chạy: ngay hoặc schedule
def main():
    parser = argparse.ArgumentParser(description="Thu thập dữ liệu chứng khoán.")
    parser.add_argument('--mode', choices=['now', 'schedule'], default='now', help='Chọn chế độ chạy: now (ngay) hoặc schedule (lập lịch)')
    args = parser.parse_args()
    if args.mode == 'now':
        run_task()
    else:
        logging.info("Thiết lập lịch chạy hàng ngày lúc 08:00...")
        schedule.every().day.at("08:00").do(run_task)
        while True:
            schedule.run_pending()
            time.sleep(1)

if __name__ == "__main__":
    main()
