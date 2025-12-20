"""
UI Styling for Optimization Comparison Tab
Custom CSS and helper functions for enhanced visual design
"""

def get_custom_css():
    """
    Returns custom CSS for the optimization comparison tab with gradient cards and modern styling.
    """
    return """
    <style>
    /* Metric Cards with Gradients */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s;
        margin-bottom: 20px;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
    }
    .metric-value {
        font-size: 36px;
        font-weight: bold;
        margin: 10px 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-label {
        font-size: 13px;
        opacity: 0.9;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    
    /* Section Headers */
    .section-header {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        padding: 18px 25px;
        border-radius: 10px;
        color: white;
        font-size: 22px;
        font-weight: 700;
        margin: 25px 0 20px 0;
        box-shadow: 0 3px 10px rgba(79, 172, 254, 0.3);
        border-left: 5px solid #0077b6;
    }
    
    /* Info Boxes */
    .info-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-left: 5px solid #4285f4;
        padding: 20px;
        border-radius: 8px;
        margin: 20px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .info-box h3 {
        color: #1a73e8;
        margin-top: 0;
    }
    .info-box ul, .info-box ol {
        margin-left: 20px;
    }
    .info-box li {
        margin: 8px 0;
        line-height: 1.6;
    }
    
    /* Legend Styling */
    .legend-card {
        background-color: #f8f9fa;
        padding: 18px;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        height: 100%;
    }
    .legend-card p {
        margin: 10px 0;
        display: flex;
        align-items: center;
        font-size: 14px;
    }
    .legend-icon {
        margin-right: 10px;
        font-size: 16px;
    }
    
    /* Highlight Badge */
    .highlight-badge {
        background: linear-gradient(135deg, #90EE90 0%, #7FD67F 100%);
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 11px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        display: inline-block;
        margin-right: 8px;
    }
    
    /* Tab Styling Enhancement */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 12px 24px;
        font-weight: 600;
    }
    
    /* DataFrame Styling */
    .dataframe {
        font-size: 13px;
    }
    
    /* Download Button Styling */
    .stDownloadButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        font-weight: 600;
    }
    </style>
    """


def create_metric_card(label, value, gradient="purple"):
    """
    Create an HTML metric card with gradient background.
    
    Args:
        label (str): Metric label
        value (str/float): Metric value to display
        gradient (str): Gradient type ('purple', 'pink', 'blue', 'green')
    
    Returns:
        str: HTML string for the metric card
    """
    gradients = {
        "purple": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
        "pink": "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)",
        "blue": "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)",
        "green": "linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)",
        "orange": "linear-gradient(135deg, #fa709a 0%, #fee140 100%)"
    }
    
    bg_gradient = gradients.get(gradient, gradients["purple"])
    
    return f"""
    <div class="metric-card" style="background: {bg_gradient};">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """


def create_section_header(title, icon="📊"):
    """
    Create a styled section header.
    
    Args:
        title (str): Section title
        icon (str): Emoji icon
    
    Returns:
        str: HTML string for the section header
    """
    return f'<div class="section-header">{icon} {title}</div>'


def create_legend_box(items):
    """
    Create a legend box with items.
    
    Args:
        items (list): List of tuples (icon, label, description)
    
    Returns:
        str: HTML string for legend box
    """
    html = '<div class="legend-card">'
    for icon, label, desc in items:
        html += f'<p><span class="legend-icon">{icon}</span><strong>{label}:</strong> {desc}</p>'
    html += '</div>'
    return html
