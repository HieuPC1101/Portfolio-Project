"""
Module news_tab.py
Tạo tab tổng hợp tin tức từ nhiều nguồn để hỗ trợ quyết định đầu tư.
"""
import streamlit as st
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import feedparser
import json
import time
import math
import numbers
import re
from email.utils import parsedate_to_datetime

VN_STOCK_KEYWORDS = [
    "chứng khoán",
    "thị trường việt nam",
    "thị trường chứng khoán",
    "vn-index",
    "vnindex",
    "vn30",
    "vni",
    "hose",
    "hnx",
    "upcom",
    "vietstock",
    "doanh nghiệp niêm yết",
    "cổ phiếu",
    "ssi",
    "vcb",
    "vic",
    "vnm"
]

EXCLUDED_TOPIC_KEYWORDS = [
    "crypto",
    "bitcoin",
    "ethereum",
    "blockchain",
    "forex",
    "fed",
    "nasdaq",
    "dow jones",
    "s&p",
    "us market",
    "wall street",
    "goldman sachs",
    "chứng khoán mỹ",
    "trái phiếu mỹ",
    "tiền ảo",
    "tiền điện tử"
]

VIETSTOCK_RSS_FEEDS = [
    "https://vietstock.vn/830/chung-khoan/co-phieu.rss",
    "https://vietstock.vn/739/chung-khoan/giao-dich-noi-bo.rss",
    "https://vietstock.vn/741/chung-khoan/niem-yet.rss"
]

VNECONOMY_ARTICLE_SLUG = re.compile(r"^/[\w\-/]+-e\d+\.htm$")
POSITIVE_NEWS_KEYWORDS = ["tăng", "hồi phục", "lãi", "tang", "hoi phuc", "lai"]
NEGATIVE_NEWS_KEYWORDS = ["giảm", "bán tháo", "lỗ", "giam", "ban thao", "lo"]

# ======================================================
# 🔧 HÀM PHỤ TRỢ
# ======================================================
def convert_relative_date(relative_date):
    """Chuyển đổi thời gian tương đối thành thời gian thực"""
    try:
        if "minute" in relative_date:
            minutes = int(relative_date.split()[0])
            return datetime.now() - timedelta(minutes=minutes)
        elif "hour" in relative_date:
            hours = int(relative_date.split()[0])
            return datetime.now() - timedelta(hours=hours)
        elif "day" in relative_date:
            days = int(relative_date.split()[0])
            return datetime.now() - timedelta(days=days)
        else:
            return datetime.now()
    except Exception as e:
        st.warning(f"Error parsing date: {e}")
        return datetime.now()


def parse_cafebiz_datetime(raw_text: str) -> datetime:
    """Parse CafeBiz timestamp strings like '07:57 23/11/2025'."""
    if not raw_text:
        return datetime.now()

    cleaned = raw_text.replace("|", " ").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    patterns = [
        "%H:%M %d/%m/%Y",
        "%H:%M %d-%m-%Y",
        "%d/%m/%Y %H:%M",
        "%d-%m-%Y %H:%M",
        "%I:%M %p %d/%m/%Y",
        "%d/%m/%Y %I:%M %p",
    ]

    for pattern in patterns:
        try:
            return datetime.strptime(cleaned, pattern)
        except ValueError:
            continue

    return datetime.now()


def is_vietnam_stock_article(title: str, content: str) -> bool:
    """Kiểm tra bài viết có liên quan đến thị trường chứng khoán Việt Nam."""
    combined_text = f"{title or ''} {content or ''}".lower()
    if any(excluded in combined_text for excluded in EXCLUDED_TOPIC_KEYWORDS):
        return False
    return any(keyword in combined_text for keyword in VN_STOCK_KEYWORDS)


def format_display_date(date_value):
    """Định dạng thời gian thành chuỗi thân thiện DD/MM/YYYY - HH:MM"""
    try:
        if isinstance(date_value, datetime):
            dt = date_value
        elif isinstance(date_value, numbers.Number):
            timestamp = float(date_value)
            if timestamp > 1e12:
                timestamp /= 1000  # vnstock trả về millisecond
            dt = datetime.fromtimestamp(timestamp)
        elif isinstance(date_value, time.struct_time):
            dt = datetime.fromtimestamp(time.mktime(date_value))
        elif isinstance(date_value, str):
            stripped_value = date_value.strip()
            if stripped_value.isdigit():
                timestamp = float(stripped_value)
                if timestamp > 1e12:
                    timestamp /= 1000
                dt = datetime.fromtimestamp(timestamp)
            else:
                dt = parsedate_to_datetime(stripped_value)
        else:
            dt = datetime.now()

        if dt.tzinfo is not None:
            dt = dt.astimezone().replace(tzinfo=None)

        return dt.strftime("%d/%m/%Y - %H:%M")
    except Exception:
        if isinstance(date_value, str) and date_value:
            return date_value
        return datetime.now().strftime("%d/%m/%Y - %H:%M")


def get_news_sentiment_styles(title: str, content: str):
    """Determine sentiment style configuration based on simple keyword scan."""
    text = f"{title or ''} {content or ''}".lower()
    sentiment = "neutral"

    if any(keyword in text for keyword in POSITIVE_NEWS_KEYWORDS):
        sentiment = "positive"
    elif any(keyword in text for keyword in NEGATIVE_NEWS_KEYWORDS):
        sentiment = "negative"

    styles = {
        "positive": {
            "border": "#22c55e",
            "background": "linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%)",
            "label": "Tin tích cực"
        },
        "negative": {
            "border": "#ef4444",
            "background": "linear-gradient(135deg, #fee2e2 0%, #fecaca 100%)",
            "label": "Tin tiêu cực"
        },
        "neutral": {
            "border": "#d97706",
            "background": "linear-gradient(135deg, #fef3c7 0%, #fde68a 100%)",
            "label": "Tin trung lập"
        }
    }
    return styles[sentiment]


@st.cache_data(ttl=300, show_spinner=False)
def fetch_rss_news(source="vnexpress", max_articles=5):
    """Lấy tin từ RSS Feed - Phương pháp đáng tin cậy hơn"""
    
    # Special handling for vnEconomy - use web scraping instead
    if source == "vnEconomy":
        return scrape_vneconomy_news(max_articles)
    if source == "cafebiz":
        return scrape_cafebiz_news(max_articles)
    
    rss_urls = {
        "vnexpress": "https://vnexpress.net/rss/kinh-doanh.rss",
        "cafef": "https://cafef.vn/thi-truong-chung-khoan.rss",
        "vietstock": VIETSTOCK_RSS_FEEDS
    }
    
    if source not in rss_urls:
        return []
    
    urls = rss_urls[source]
    if not isinstance(urls, list):
        urls = [urls]

    aggregated_news = []
    last_warning = None

    # Try each URL and accumulate until we have enough articles
    for url_index, url in enumerate(urls):
        try:
            # Enhanced headers to avoid blocking
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'application/rss+xml, application/xml, text/xml, */*',
                'Accept-Language': 'vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache'
            }
            
            # Dùng requests để lấy RSS với timeout ngắn
            response = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
            response.raise_for_status()
            
            # Parse RSS
            feed = feedparser.parse(response.content)
            
            # Check if feed has entries
            if not feed.entries:
                last_warning = f"⚠️ Không tìm thấy bài viết từ {source}"
                continue
            
            for entry in feed.entries:
                if len(aggregated_news) >= max_articles:
                    break
                try:
                    title = entry.title if hasattr(entry, 'title') else "No Title"
                    link = entry.link if hasattr(entry, 'link') else ""
                    
                    # Parse date
                    published_struct = getattr(entry, 'published_parsed', None)
                    updated_struct = getattr(entry, 'updated_parsed', None)
                    if published_struct:
                        date = format_display_date(published_struct)
                    elif updated_struct:
                        date = format_display_date(updated_struct)
                    elif hasattr(entry, 'published'):
                        date = format_display_date(entry.published)
                    elif hasattr(entry, 'updated'):
                        date = format_display_date(entry.updated)
                    else:
                        date = format_display_date(datetime.now())
                    
                    # Get content
                    content = ""
                    if hasattr(entry, 'summary'):
                        content = BeautifulSoup(entry.summary, 'html.parser').get_text(strip=True)
                    elif hasattr(entry, 'description'):
                        content = BeautifulSoup(entry.description, 'html.parser').get_text(strip=True)
                    else:
                        content = "Nội dung đang được cập nhật..."
                    
                    normalized_content = content[:500] + "..." if len(content) > 500 else content
                    if not is_vietnam_stock_article(title, normalized_content):
                        continue

                    aggregated_news.append({
                        "title": title,
                        "date": date,
                        "content": normalized_content,
                        "link": link,
                        "source": source.upper()
                    })
                except Exception:
                    continue
            
            if len(aggregated_news) >= max_articles:
                break
                
        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP {e.response.status_code}"
            last_warning = f"⚠️ Không thể tải RSS từ {source}: {error_msg}"
            continue
        except requests.exceptions.Timeout:
            last_warning = f"⚠️ Timeout khi tải RSS từ {source}"
            continue
        except requests.exceptions.ConnectionError:
            last_warning = f"⚠️ Lỗi kết nối đến {source}"
            continue
        except Exception as e:
            last_warning = f"⚠️ Không thể tải RSS từ {source}: {str(e)[:80]}"
            continue

    if aggregated_news:
        return aggregated_news[:max_articles]

    if last_warning:
        st.warning(last_warning)
    else:
        st.warning(f"⚠️ Không thể tải RSS từ {source}")
    return []


@st.cache_data(ttl=300, show_spinner=False)
def scrape_vneconomy_news(max_articles=5):
    """
    Web scraping cho vnEconomy khi RSS không hoạt động
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7',
            'Connection': 'keep-alive'
        }
        
        base_section = "https://vneconomy.vn/chung-khoan.htm"
        max_section_pages = 5  # crawl deeper pages to get đủ bài liên quan chứng khoán
        urls_to_try = []

        for page in range(1, max_section_pages + 1):
            if page == 1:
                urls_to_try.append(base_section)
            else:
                urls_to_try.append(f"{base_section}?p={page}")

        # Fallback pages bổ sung thêm bối cảnh kinh tế Việt Nam nếu trang chính thiếu bài
        urls_to_try.extend([
            "https://vneconomy.vn/kinh-te.htm",
            "https://vneconomy.vn"
        ])
        
        collected_news = []
        seen_links = set()

        for base_url in urls_to_try:
            try:
                response = requests.get(base_url, headers=headers, timeout=15)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
                
                page_news = []
                
                # Find article containers - vnEconomy uses different classes
                # Try multiple possible selectors
                article_selectors = [
                    'div.story',
                    'div.story-item',
                    'article.story',
                    'div.news-item',
                    'div.item-news'
                ]
                
                articles = []
                for selector in article_selectors:
                    articles = soup.select(selector)
                    if articles:
                        break
                
                if not articles:
                    # Fallback: find any links that look like articles
                    articles = soup.find_all('a', href=True)
                    articles = [a for a in articles if '/tin-tuc/' in a.get('href', '') or '/kinh-te/' in a.get('href', '')][:max_articles * 2]
                
                for article in articles[:max_articles * 3]:
                    if len(collected_news) >= max_articles:
                        break
                    
                    try:
                        # Extract title
                        title_elem = article.find('h3') or article.find('h2') or article.find('a')
                        if not title_elem:
                            continue
                        
                        title = title_elem.get_text(strip=True)
                        if not title or len(title) < 10:
                            continue
                        
                        # Extract link
                        link_elem = article.find('a') if article.name != 'a' else article
                        link = link_elem.get('href', '') if link_elem else ''
                        if link and not link.startswith('http'):
                            link = f"https://vneconomy.vn{link}"
                        
                        # Extract date
                        time_elem = article.find('time') or article.find('span', class_=['time', 'date', 'published'])
                        raw_date = time_elem.get_text(strip=True) if time_elem else datetime.now()
                        date = format_display_date(raw_date) if raw_date else format_display_date(datetime.now())
                        
                        # Extract description
                        desc_elem = article.find('p') or article.find('div', class_=['description', 'desc', 'summary'])
                        content = desc_elem.get_text(strip=True) if desc_elem else "Đọc thêm tại vneconomy.vn"
                        
                        if len(content) < 20:
                            content = f"{title[:100]}... Đọc thêm tại vneconomy.vn"
                        
                        normalized_content = content[:500] + "..." if len(content) > 500 else content

                        passes_filter = is_vietnam_stock_article(title, normalized_content)
                        lower_text = f"{title} {normalized_content}".lower()
                        if not passes_filter:
                            if (link.startswith("https://vneconomy.vn/chung-khoan") or link.startswith("/chung-khoan") or "chung-khoan" in base_url.lower()) and not any(excluded in lower_text for excluded in EXCLUDED_TOPIC_KEYWORDS):
                                passes_filter = True
                        if not passes_filter:
                            continue

                        unique_key = link or title
                        if unique_key in seen_links:
                            continue
                        seen_links.add(unique_key)

                        page_news.append({
                            "title": title,
                            "date": date,
                            "content": normalized_content,
                            "link": link,
                            "source": "VNECONOMY "
                        })
                    except Exception:
                        continue
                
                if len(collected_news) + len(page_news) < max_articles:
                    for anchor in soup.find_all('a', href=True):
                        if len(collected_news) + len(page_news) >= max_articles:
                            break
                        raw_href = anchor.get('href', '')
                        if not raw_href or raw_href.startswith('javascript') or raw_href.startswith('#'):
                            continue
                        if not VNECONOMY_ARTICLE_SLUG.match(raw_href):
                            continue
                        anchor_title = anchor.get_text(strip=True)
                        if not anchor_title or len(anchor_title) < 10:
                            continue
                        link = raw_href if raw_href.startswith('http') else f"https://vneconomy.vn{raw_href}"
                        if link in seen_links:
                            continue

                        placeholder_content = f"Tin nhanh VnEconomy: {anchor_title}. Đọc nội dung chi tiết trên trang gốc."
                        passes_filter = is_vietnam_stock_article(anchor_title, placeholder_content)
                        if not passes_filter:
                            lower_text = anchor_title.lower()
                            if (link.startswith("https://vneconomy.vn/chung-khoan") or raw_href.startswith("/chung-khoan") or "chung-khoan" in base_url.lower()) and not any(excluded in lower_text for excluded in EXCLUDED_TOPIC_KEYWORDS):
                                passes_filter = True
                        if not passes_filter:
                            continue

                        seen_links.add(link)
                        page_news.append({
                            "title": anchor_title,
                            "date": format_display_date(datetime.now()),
                            "content": placeholder_content,
                            "link": link,
                            "source": "VNECONOMY "
                        })

                if page_news:
                    collected_news.extend(page_news)
                    if len(collected_news) >= max_articles:
                        return collected_news[:max_articles]
                    
            except Exception:
                continue
        
        return collected_news
        
    except Exception as e:
        st.warning(f"⚠️ Không thể scrape vnEconomy: {str(e)[:80]}")
        return []


@st.cache_data(ttl=300, show_spinner=False)
def scrape_cafebiz_news(max_articles=5):
    """Scrape CafeBiz category page to avoid dead RSS feed."""
    section_url = "https://cafebiz.vn/cau-chuyen-kinh-doanh/chung-khoan.chn"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7',
        'Connection': 'keep-alive'
    }

    def fetch_article_body(url: str) -> str:
        try:
            detail_response = requests.get(url, headers=headers, timeout=15)
            detail_response.raise_for_status()
            detail_soup = BeautifulSoup(detail_response.content, 'html.parser')
            content_div = detail_soup.find('div', class_='newscontent')
            if content_div:
                return content_div.get_text(separator=' ', strip=True)
        except Exception:
            return ""
        return ""

    try:
        response = requests.get(section_url, headers=headers, timeout=15)
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        st.warning(f"⚠️ Không thể tải CafeBiz: {str(exc)[:80]}")
        return []

    soup = BeautifulSoup(response.content, 'html.parser')
    article_boxes = soup.select('div.cfbiznews_box')
    if not article_boxes:
        st.warning("⚠️ Không tìm thấy bài viết CafeBiz phù hợp")
        return []

    collected_news = []
    seen_links = set()

    for box in article_boxes:
        if len(collected_news) >= max_articles:
            break

        title_elem = box.select_one('.cfbiznews_title')
        if not title_elem:
            continue
        title = title_elem.get_text(strip=True)
        if not title or len(title) < 10:
            continue

        link = title_elem.get('href') or ''
        if not link:
            continue
        if link.startswith('//'):
            link = f"https:{link}"
        elif link.startswith('/'):
            link = f"https://cafebiz.vn{link}"
        if link in seen_links:
            continue
        seen_links.add(link)

        timestamp_elem = box.select_one('.cfbiznews_tt')
        raw_timestamp = timestamp_elem.get_text(strip=True) if timestamp_elem else ''
        published_dt = parse_cafebiz_datetime(raw_timestamp)
        display_date = format_display_date(published_dt)

        summary_elem = box.select_one('.cfbiznews_des') or box.select_one('.cfbiznews_sapo')
        summary_text = summary_elem.get_text(strip=True) if summary_elem else ''

        article_text = summary_text
        if len(article_text) < 120:
            detailed = fetch_article_body(link)
            article_text = detailed or summary_text or "Đọc thêm trên CafeBiz.vn"

        normalized_content = article_text[:500] + "..." if len(article_text) > 500 else article_text
        if not is_vietnam_stock_article(title, normalized_content):
            continue

        collected_news.append({
            "title": title,
            "date": display_date,
            "content": normalized_content,
            "link": link,
            "source": "CAFEBIZ"
        })

    if not collected_news:
        st.warning("⚠️ CafeBiz không có bài viết phù hợp trong thời điểm này")

    return collected_news[:max_articles]


@st.cache_data(ttl=300, show_spinner=False)  # Cache 5 phút
def scrape_investing_news(page_num, max_articles=5):
    """
    Scrape tin tức từ Investing.com
    
    Args:
        page_num: Số trang cần crawl
        max_articles: Số bài viết tối đa cần lấy
    
    Returns:
        List[dict]: Danh sách tin tức
    """
    # URL đúng cho Investing.com stock market news
    if page_num == 1:
        url = "https://www.investing.com/news/stock-market-news"
    else:
        url = f"https://www.investing.com/news/stock-market-news/{page_num}"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Cache-Control": "max-age=0"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=15, verify=True)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        st.error(f"⚠️ Không thể kết nối đến Investing.com: {str(e)[:100]}")
        st.info("💡 Có thể do: (1) Mạng bị chặn, (2) Website đang bảo trì, (3) Cần VPN")
        return []

    soup = BeautifulSoup(response.content, 'html.parser')
    articles = soup.find_all('div', class_='news-analysis-v2_content__z0iLP w-full text-xs sm:flex-1')

    news_data = []
    for article in articles:
        if len(news_data) >= max_articles:
            break
            
        try:
            # Lấy tiêu đề
            title_elem = article.find(
                'a',
                class_='text-inv-blue-500 hover:text-inv-blue-500 hover:underline focus:text-inv-blue-500 focus:underline whitespace-normal text-sm font-bold leading-5 !text-[#181C21] sm:text-base sm:leading-6 lg:text-lg lg:leading-7'
            )
            if not title_elem:
                continue
            title = title_elem.get_text(strip=True)

            # Lấy thời gian
            time_elem = article.find('time')
            if time_elem:
                date_text = time_elem.get_text(strip=True)
                if "ago" in date_text:
                    date = format_display_date(convert_relative_date(date_text))
                else:
                    date = format_display_date(date_text)
            else:
                date = format_display_date(datetime.now())

            # Lấy liên kết bài viết chi tiết
            link = title_elem.get('href', '')
            if link.startswith("http"):
                full_link = link
            else:
                full_link = f"https://www.investing.com{link}"

            # Lấy nội dung bài viết chi tiết
            content = "Loading..."
            try:
                detail_response = requests.get(full_link, headers=headers, timeout=10)
                detail_response.raise_for_status()
                detail_soup = BeautifulSoup(detail_response.content, 'html.parser')
                content_div = detail_soup.find('div', class_='article_WYSIWYG__O0uhw article_articlePage__UMz3q text-[18px] leading-8')
                content = content_div.get_text(strip=True) if content_div else "No Content Available"
            except requests.exceptions.RequestException as e:
                content = f"Error retrieving content: {e}"

            if not is_vietnam_stock_article(title, content):
                continue

            news_data.append({
                "title": title,
                "date": date,
                "content": content,
                "link": full_link
            })
            
        except Exception as e:
            st.warning(f"⚠️ Error processing article: {e}")
            continue

    return news_data


def render_pagination_controls(total_pages):
    """Hiển thị điều hướng trang ở cuối tab"""
    st.divider()
    spacer_left, control_col, spacer_right = st.columns([1, 2, 1])

    with control_col:
        prev_col, info_col, next_col = st.columns([1, 1, 1], gap="small")

        prev_disabled = st.session_state.news_current_page <= 1
        next_disabled = st.session_state.news_current_page >= total_pages

        if prev_col.button("⬅️", use_container_width=True, disabled=prev_disabled, key="news_prev_btn"):
            st.session_state.news_current_page -= 1
            st.rerun()

        info_col.markdown(
            f"<div style='text-align:center; font-size:16px; font-weight:600;'>Trang {st.session_state.news_current_page} / {total_pages}</div>",
            unsafe_allow_html=True
        )

        if next_col.button("➡️", use_container_width=True, disabled=next_disabled, key="news_next_btn"):
            st.session_state.news_current_page += 1
            st.rerun()


# ======================================================
# 📰 RENDER TAB NEWS
# ======================================================
def render(ticker: str = None):
    """Hiển thị tab tin tức từ nhiều nguồn"""
    
    st.header("📰 Tin tức Thị trường Chứng khoán Việt Nam")
    
    # Chọn nguồn tin
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("""
        <p style='color:#94a3b8'>
        Tin tức mới nhất về thị trường chứng khoán Việt Nam từ nhiều nguồn tin uy tín.
        </p>
        """, unsafe_allow_html=True)
    
    with col2:
        news_source = st.selectbox(
            "📡 Chọn nguồn:",
            ["vnexpress", "cafef", "cafebiz", "vietstock", "vnEconomy"],
            format_func=lambda x: {
                "vnexpress": "VnExpress",
                "cafef": "CafeF", 
                "cafebiz": "CafeBiz",
                "vietstock": "VietStock",
                "vnEconomy": "VnEconomy"
            }.get(x, x)
        )
    
    # Khởi tạo session state cho số trang
    if 'news_current_page' not in st.session_state:
        st.session_state.news_current_page = 1
    
    per_page = 5
    
    # ======================================================
    # 📊 LẤY VÀ HIỂN THỊ TIN TỨC
    # ======================================================
    # Lấy nhiều tin tức để phân trang
    with st.spinner(f"🔍 Đang tải tin tức từ {news_source.upper()}..."):
        news = fetch_rss_news(news_source, max_articles=50)
    
    if not news:
        st.error(f"❌ Không thể tải tin tức từ nguồn {news_source.upper()}")
        
        # Hiển thị hướng dẫn khắc phục
        st.markdown("""
        ### 🔧 Nguyên nhân có thể:
        
        1. **🌐 Kết nối mạng**: Kiểm tra internet của bạn
        2. **🚫 Website chặn**: Nguồn tin có thể chặn request tự động
        3. **🔒 Firewall/Antivirus**: Có thể đang chặn kết nối
        4. **⏱️ Timeout**: Server phản hồi quá chậm
        
        ### 💡 Giải pháp:
        
        - **Thử nguồn khác**: Chọn nguồn tin khác trong dropdown ở trên
        - Refresh lại trang sau vài giây
        - Kiểm tra kết nối internet
        """)
        
        return  # Dừng execution nếu không có tin tức
    else:
        total_pages = max(1, math.ceil(len(news) / per_page))
        current_page = min(st.session_state.news_current_page, total_pages)
        if current_page != st.session_state.news_current_page:
            st.session_state.news_current_page = current_page
            st.rerun()
        start_idx = (current_page - 1) * per_page
        page_news = news[start_idx:start_idx + per_page]
        if not page_news and current_page > 1:
            st.session_state.news_current_page = 1
            st.rerun()

        # Hiển thị từng bài viết
        for index, item in enumerate(page_news, start=start_idx + 1):
            sentiment_styles = get_news_sentiment_styles(item['title'], item['content'])
            border_color = sentiment_styles['border']
            background_style = sentiment_styles['background']
            sentiment_label = sentiment_styles['label']
            title_link = f"<a href='{item['link']}' target='_blank' style='color:#0f172a; text-decoration:none;'>{item['title']}</a>"

            with st.container():
                st.markdown(f"""
                <div style='
                    background: {background_style};
                    border-left: 4px solid {border_color};
                    padding: 15px;
                    border-radius: 8px;
                    margin-bottom: 15px;
                '>
                    <div style='display: flex; justify-content: space-between; align-items: center; gap: 16px;'>
                        <h4 style='color: #0f172a; margin: 0 0 10px 0; flex: 1;'>📰 {title_link}</h4>
                        <span style='font-size:12px; font-weight:600; color:{border_color}; padding:4px 10px; border:1px solid {border_color}; border-radius:999px;'>
                            {sentiment_label}
                        </span>
                    </div>
                    <p style='color: #6b7280; font-size: 14px; margin: 0;'>
                        📅 <b>Đăng lúc:</b> {item['date']}
                    </p>
                </div>
                """, unsafe_allow_html=True)

                st.write(item['content'])
                st.markdown("<br>", unsafe_allow_html=True)
    
    render_pagination_controls(total_pages if 'total_pages' in locals() else 1)
