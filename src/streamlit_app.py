import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time
import requests
import json

# -----------------------------------------------------------------------------
# 1. CẤU HÌNH & STYLE PREMIUM (LIGHT THEME)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Retention Intelligence AI",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Premium Light" Experience
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    :root {
        --bg-color: #f8fafc;
        --card-bg: rgba(255, 255, 255, 0.8);
        --card-border: rgba(226, 232, 240, 0.8);
        --accent-emerald: #059669;
        --accent-blue: #2563eb;
        --accent-red: #dc2626;
        --text-main: #0f172a;
        --text-muted: #64748b;
        --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: var(--text-main);
    }

    .stApp {
        background-color: var(--bg-color);
    }

    /* Premium Card Base */
    .glass-card {
        background: var(--card-bg);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border: 1px solid var(--card-border);
        border-radius: 20px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: var(--shadow-md);
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg);
        border-color: rgba(37, 99, 235, 0.2);
    }

    /* Target Streamlit Plotly Containers */
    [data-testid="stPlotlyChart"] {
        background: white;
        border: 1px solid var(--card-border);
        border-radius: 20px;
        padding: 10px;
        box-shadow: var(--shadow-sm);
    }

    /* Custom Metric Component */
    .metric-box {
        text-align: center;
    }
    .metric-label {
        color: var(--text-muted);
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: var(--text-main);
    }
    .metric-unit {
        font-size: 0.85rem;
        color: var(--text-muted);
        margin-left: 2px;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: white;
        border-right: 1px solid var(--card-border);
    }
    .sidebar-title {
        font-size: 1.4rem;
        font-weight: 800;
        color: var(--accent-blue);
        text-align: center;
    }

    /* Action Button */
    .stButton>button {
        background: var(--accent-blue);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 700;
        width: 100%;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background: #1d4ed8;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
    }

    /* Custom Alerts */
    .premium-alert {
        padding: 16px;
        border-radius: 12px;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 12px;
        border: 1px solid transparent;
        width: 100%; /* Ensure it fills the container */
        box-sizing: border-box;
    }
    .alert-critical {
        background: #fef2f2;
        border-color: #fee2e2;
        color: #991b1b;
    }
    .alert-warning {
        background: #fffbeb;
        border-color: #fef3c7;
        color: #92400e;
    }
    .alert-success {
        background: #f0fdf4;
        border-color: #dcfce7;
        color: #166534;
    }
    .alert-info {
        background: #eff6ff;
        border-color: #dbeafe;
        color: #1e40af;
    }

    .chart-title {
        text-align: center;
        color: var(--text-muted);
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 1.2px;
        margin-bottom: 8px;
        text-transform: uppercase;
    }

    /* Hide default Streamlit junk */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. BACKEND CONNECTION (FastAPI)
# -----------------------------------------------------------------------------
API_URL = "http://localhost:8000"

def get_analysis(customer_id):
    try:
        response = requests.get(f"{API_URL}/predict/{customer_id}", timeout=8)
        if response.status_code == 200:
            data = response.json()
            return data["features"], data["probability"], data["is_churn"], data.get("shap_values", {})
        elif response.status_code == 404:
            st.warning(f"🔍 Không tìm thấy khách hàng ID #{customer_id}.")
            return None, None, None, None
        else:
            st.error(f"❌ Lỗi Server: {response.text}")
            return None, None, None, None
    except Exception as e:
        st.error(f"📡 Lỗi kết nối Inference Server. Vui lòng kiểm tra backend.")
        return None, None, None, None

def check_server_health():
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.json()
    except:
        return None

# -----------------------------------------------------------------------------
# 3. VISUALIZATIONS (PREMIUM PLOTLY)
# -----------------------------------------------------------------------------
def create_gauge_chart(probability):
    """Create a simple, clean gauge chart."""
    color = "#10b981" if probability < 0.3 else "#f59e0b" if probability < 0.7 else "#ef4444"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability * 100,
        number = {'suffix': "%", 'font': {'size': 56, 'color': '#1e293b', 'family': 'Inter'}},
        domain = {'x': [0, 1], 'y': [0.1, 1]},
        gauge = {
            'axis': {'range': [0, 100], 'visible': False},
            'bar': {'color': color, 'thickness': 0.5},
            'bgcolor': "#e2e8f0",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 30], 'color': '#d1fae5'},
                {'range': [30, 70], 'color': '#fef3c7'},
                {'range': [70, 100], 'color': '#fecaca'}
            ]
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        height=350,
        margin=dict(l=30, r=30, t=30, b=40)
    )
    return fig

def create_radar_chart(features):
    """Create a clear, readable radar chart."""
    categories = ['Gắn bó', 'Tần suất', 'Hỗ trợ', 'Chi tiêu', 'Trễ hạn']
    max_vals = {'Tenure': 60, 'Usage Frequency': 25, 'Support Calls': 8, 'Total Spend': 800, 'Payment Delay': 15}
    
    def norm(v, k): 
        return min((v / max_vals.get(k, 100)) * 100, 100)

    cust = [
        norm(features.get('Tenure', 0), 'Tenure'),
        norm(features.get('Usage Frequency', 0), 'Usage Frequency'),
        norm(features.get('Support Calls', 0), 'Support Calls'),
        norm(features.get('Total Spend', 0), 'Total Spend'),
        norm(features.get('Payment Delay', 0), 'Payment Delay')
    ]
    avg = [50, 40, 30, 50, 20]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=avg + [avg[0]], theta=categories + [categories[0]], 
        fill='toself', name='Trung bình',
        line=dict(color='#94a3b8', width=2), fillcolor='rgba(148, 163, 184, 0.15)'
    ))
    fig.add_trace(go.Scatterpolar(
        r=cust + [cust[0]], theta=categories + [categories[0]], 
        fill='toself', name='Khách hàng',
        line=dict(color='#2563eb', width=3), fillcolor='rgba(37, 99, 235, 0.25)',
        marker=dict(size=10, color='#2563eb')
    ))

    fig.update_layout(
        polar=dict(
            bgcolor='white',
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="#e2e8f0", 
                           tickfont={'size': 10, 'color': '#64748b'}, tickvals=[0, 50, 100]),
            angularaxis=dict(gridcolor="#e2e8f0", tickfont={'size': 12, 'color': '#334155'})
        ),
        showlegend=True,
        paper_bgcolor='rgba(0,0,0,0)',
        height=350,
        margin=dict(l=60, r=60, t=30, b=60),
        legend=dict(font={'size': 12}, orientation="h", yanchor="top", y=-0.05, x=0.5, xanchor="center")
    )
    return fig

def create_shap_chart(shap_values):
    """Create a clear SHAP importance bar chart."""
    vn_map = {
        "Age": "Tuổi", "Gender": "Giới tính", "Tenure": "Gắn bó",
        "Usage Frequency": "Tần suất", "Support Calls": "Hỗ trợ",
        "Payment Delay": "Trễ hạn", "Subscription Type": "Gói cước",
        "Contract Length": "Hợp đồng", "Total Spend": "Chi tiêu",
        "Last Interaction": "Tương tác"
    }
    
    sorted_items = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    features = [vn_map.get(x[0], x[0]) for x in sorted_items]
    values = [x[1] for x in sorted_items]
    colors = ['#ef4444' if v > 0 else '#22c55e' for v in values]

    fig = go.Figure(go.Bar(
        y=features,
        x=values,
        orientation='h',
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{'+' if v > 0 else ''}{v:.2f}" for v in values],
        textposition='outside',
        textfont=dict(color='#334155', size=12)
    ))

    max_v = max(abs(v) for v in values) * 1.5 if values else 1
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=250,
        margin=dict(l=90, r=70, t=10, b=10),
        xaxis=dict(showgrid=True, gridcolor="#f1f5f9", zerolinecolor="#cbd5e1", 
                  tickfont={'size': 11}, range=[-max_v, max_v], zeroline=True),
        yaxis=dict(autorange="reversed", tickfont={'size': 13, 'color': '#1e293b'}),
        font={'family': "Inter"}
    )
    return fig

# -----------------------------------------------------------------------------
# 4. UI COMPONENTS
# -----------------------------------------------------------------------------
def render_metric(label, value, unit="", color=None):
    color_style = f'style="color: {color};"' if color else ""
    st.markdown(f"""
    <div class="glass-card" style="padding: 20px; margin-bottom: 15px;">
        <div class="metric-box">
            <div class="metric-label" style="font-size: 0.65rem;">{label}</div>
            <div class="metric-value" style="font-size: 1.8rem;" {color_style}>{value}<span class="metric-unit" style="font-size: 0.8rem;">{unit}</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_alert(type, title, message):
    icon_map = {"critical": "🚨", "warning": "⚠️", "success": "✅", "info": "💡"}
    st.markdown(f"""
    <div class="premium-alert alert-{type}" style="padding: 15px; margin-bottom: 12px;">
        <div style="font-size: 1.2rem;">{icon_map.get(type, '●')}</div>
        <div>
            <div style="font-weight: 700; font-size: 0.95rem; margin-bottom: 2px;">{title}</div>
            <div style="font-size: 0.85rem; opacity: 0.9;">{message}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. MAIN LAYOUT
# -----------------------------------------------------------------------------

# Sidebar
with st.sidebar:
    st.markdown('<div style="padding: 10px 0;"><div class="sidebar-title" style="font-size: 1.4rem;">RETENTION AI</div></div>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #64748b; font-size: 0.8rem; margin-top: -10px;'>Hệ thống phân tích rủi ro</p>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("👤 Tra cứu khách hàng")
    customer_id = st.number_input("Nhập ID khách hàng", min_value=1, value=71967, step=1)
    analyze_btn = st.button("CHẠY PHÂN TÍCH")
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🌐 Hệ thống")
    health = check_server_health()
    if health:
        st.markdown(f"<div style='font-size: 0.85rem;'>🟢 <b>Server</b>: Online</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size: 0.85rem;'>🔵 <b>SHAP</b>: {'Sẵn sàng' if health.get('explainer_ready') else 'Lỗi'}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='font-size: 0.85rem;'>🔴 <b>Server</b>: Offline</div>", unsafe_allow_html=True)

# Main Content
if analyze_btn:
    with st.spinner("Đang thực hiện phân tích..."):
        features, prob, is_churn, shap_values = get_analysis(customer_id)
        
        if features:
            st.markdown(f"<h2 style='font-weight: 800; margin-bottom: 20px;'>Báo cáo khách hàng #{customer_id}</h2>", unsafe_allow_html=True)
            
            # Row 1: Key Metrics
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                render_metric("Thời gian gắn bó", f"{features.get('Tenure', 0):.0f}", "tháng")
            with m2:
                render_metric("Tổng chi tiêu", f"{features.get('Total Spend', 0):.0f}", "$")
            with m3:
                calls = features.get('Support Calls', 0)
                render_metric("Số cuộc gọi hỗ trợ", f"{calls:.0f}", "lần", color="#ef4444" if calls > 5 else None)
            with m4:
                contract = "Năm" if features.get('Contract Length', 0) == 1 else "Tháng"
                render_metric("Loại hợp đồng", contract, "", color="#3b82f6")
            
            # Charts Section - 2 Row Layout for better visibility
            st.markdown("<h4 style='margin-top: 10px; margin-bottom: 15px; color: #64748b;'>📊 Phân tích định lượng & Giải thích</h4>", unsafe_allow_html=True)
            
            # Row 2a: Gauge + Radar (2 columns)
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.markdown('<div class="chart-title">XÁC SUẤT RỜI BỎ</div>', unsafe_allow_html=True)
                st.plotly_chart(create_gauge_chart(prob), use_container_width=True, config={'staticPlot': True})
                
            with col_right:
                st.markdown('<div class="chart-title">HÀNH VI KHÁCH HÀNG</div>', unsafe_allow_html=True)
                st.plotly_chart(create_radar_chart(features), use_container_width=True, config={'staticPlot': True})
            
            # Row 2b: SHAP (full width)
            st.markdown('<div class="chart-title" style="margin-top: 20px;">CÁC YẾU TỐ ẢNH HƯỞNG CHÍNH (SHAP)</div>', unsafe_allow_html=True)
            if shap_values:
                st.plotly_chart(create_shap_chart(shap_values), use_container_width=True, config={'staticPlot': True})
            else:
                st.info("Đang tải SHAP...")
                
            # Row 3: Strategic Insights
            st.markdown("<h4 style='margin-top: 10px; margin-bottom: 15px; color: #94a3b8;'>💡 Đề xuất & Hành động</h4>", unsafe_allow_html=True)
            
            # Use a container instead of raw HTML div to avoid empty frame issue
            with st.container():
                if shap_values:
                    top_risk = sorted(shap_values.items(), key=lambda x: x[1], reverse=True)[0]
                    if top_risk[1] > 0.05:
                        render_alert("info", "Phân tích nguyên nhân", f"Yếu tố rủi ro lớn nhất là <b>{top_risk[0]}</b> (<b>+{top_risk[1]:.2f}</b>).")

                if prob > 0.7:
                    render_alert("critical", "NGUY CƠ RỜI BỎ CỰC CAO", "Cần kích hoạt quy trình giữ chân khẩn cấp.")
                elif prob > 0.3:
                    render_alert("warning", "RỦI RO TRUNG BÌNH", "Gửi các chương trình ưu đãi chủ động.")
                else:
                    render_alert("success", "TRẠNG THÁI ỔN ĐỊNH", "Duy trì chăm sóc định kỳ.")
                
                delay = features.get('Payment Delay', 0)
                if delay > 3:
                    render_alert("critical", "RỦI RO THANH TOÁN", f"Trễ hạn {delay:.0f} lần. Cần nhắc nợ khéo léo.")

else:
    # Welcome Screen
    st.markdown("""
    <div style='text-align: center; padding: 120px 0;'>
        <h1 style='font-size: 4.5rem; font-weight: 900; color: var(--accent-blue); margin-bottom: 20px;'>
            Retention Intelligence
        </h1>
        <p style='font-size: 1.4rem; color: var(--text-muted); max-width: 800px; margin: 0 auto; line-height: 1.8;'>
            Nền tảng phân tích rủi ro khách hàng dựa trên AI. Tích hợp <b>Feast Feature Store</b>, 
            mô hình <b>XGBoost</b> và giải thích <b>SHAP</b> để đưa ra những quyết định giữ chân khách hàng chính xác nhất.
        </p>
        <div style='margin-top: 40px; color: var(--accent-blue); font-weight: 600;'>
            ← Nhập ID khách hàng ở thanh bên để bắt đầu
        </div>
    </div>
    """, unsafe_allow_html=True)
