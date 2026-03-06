"""
AI Deal Hunter - Professional E-commerce Marketplace Prototype
Streamlit app for comparing electronics deals across stores.
Uses Gemini API for AI-generated deal insights (see deal_hunter.py for API key).
"""
import html
import json
import hashlib

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from deal_hunter import model as gemini_model
except Exception:
    gemini_model = None

# =============================================================================
# CONFIG & CONSTANTS
# =============================================================================
st.set_page_config(layout="wide", page_title="AI Deal Hunter", initial_sidebar_state="expanded")

PRIMARY_COLOR = "#E30613"
PLACEHOLDER_IMG = "https://via.placeholder.com/300"

# Product image URLs (product_id -> URL). Use placeholder if missing.
PRODUCT_IMAGES = {
    "iphone_13": "images/iphone_13.jpg",
    "iphone_14": "images/iphone_14.jpeg",
    "samsung_s23": "images/Samsung_s23.jpg",

    "macbook_air_m1": "images/Macbook_Air_m1.jpg",
    "macbook_air_m2": "images/macbook_air_m2.jpeg",
    "dell_xps_13": "images/dell_xps-13.jpg",
    "hp_spectre": "images/hp_spectre.png",
    "lenovo_legion": "images/lenovo_legion.png",

    "airpods_pro": "images/airpods_pro.jpg",
    "logitech_mouse": "images/logitech_mouse.png",
    "logitech_keyboard": "images/logitech_keyboard.jpg",
    "anker_charger": "images/anker_charger.jpg",

    "apple_watch": "images/apple_watch.jpeg",
    "ipad_air": "images/ipad_air.jpg",

    "playstation5": "images/playstation_5.jpg",
    "xbox_series_x": "images/xbox_series_x.png"
}

CATEGORY_KEYS = ["📱 Smartphones", "💻 Laptops", "🎮 Gaming", "🎧 Audio", "📟 Tablets"]
CATEGORIES = {
    "📱 Smartphones": ["iphone_13", "iphone_14", "samsung_s23"],
    "💻 Laptops": ["macbook_air_m1", "macbook_air_m2", "dell_xps_13", "hp_spectre", "lenovo_legion"],
    "🎮 Gaming": ["playstation5", "xbox_series_x"],
    "🎧 Audio": ["airpods_pro", "logitech_mouse", "logitech_keyboard", "anker_charger"],
    "📟 Tablets": ["ipad_air", "apple_watch", "samsung_tv_55", "lg_tv_55", "ssd_1tb", "external_hdd"],
}

PRODUCT_DISPLAY_NAMES = {
    "iphone_13": "iPhone 13", "iphone_14": "iPhone 14", "samsung_s23": "Samsung S23",
    "macbook_air_m1": "MacBook Air M1", "macbook_air_m2": "MacBook Air M2",
    "airpods_pro": "AirPods Pro", "apple_watch": "Apple Watch", "ipad_air": "iPad Air",
    "playstation5": "PlayStation 5", "xbox_series_x": "Xbox Series X",
    "samsung_tv_55": "Samsung TV 55\"", "lg_tv_55": "LG TV 55\"",
    "dell_xps_13": "Dell XPS 13", "hp_spectre": "HP Spectre", "lenovo_legion": "Lenovo Legion",
    "logitech_mouse": "Logitech Mouse", "logitech_keyboard": "Logitech Keyboard",
    "anker_charger": "Anker Charger", "ssd_1tb": "SSD 1TB", "external_hdd": "External HDD",
}

STORE_DISPLAY_NAMES = {
    "KontaktHome": "Kontakt Home",
    "Irshad": "Irshad",
    "BakuElectronics": "Baku Electronics",
}


# =============================================================================
# DATA LOADING
# =============================================================================
def load_stores_data():
    """Load all store data from data/stores.jsonl. Returns (stores_data dict, all_store_names list)."""
    stores_data = {}
    all_store_names = []
    try:
        with open("data/stores.jsonl", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                store, product = item["store"], item["product"]
                if store not in stores_data:
                    stores_data[store] = {}
                    all_store_names.append(store)
                stores_data[store][product] = item
    except FileNotFoundError:
        st.error("data/stores.jsonl not found.")
        st.stop()
    return stores_data, all_store_names


def get_cheapest_offer_for_product(product_id, stores_data, all_store_names):
    """Return (store_name, item_dict) for the store with lowest new_price for this product."""
    best_store, best_data, best_price = None, None, None
    for store in all_store_names:
        products = stores_data.get(store, {})
        if product_id not in products:
            continue
        data = products[product_id]
        price = data["new_price"]
        if best_price is None or price < best_price:
            best_price, best_store, best_data = price, store, data
    return best_store, best_data


def get_all_offers_for_product(product_id, stores_data, all_store_names):
    """Return list of (store, new_price, item_dict) for all stores that have this product."""
    out = []
    for store in all_store_names:
        products = stores_data.get(store, {})
        if product_id not in products:
            continue
        data = products[product_id]
        out.append((store, data["new_price"], data))
    return sorted(out, key=lambda x: x[1])


def get_store_badge(store_name):
    return STORE_DISPLAY_NAMES.get(store_name, store_name)


def get_product_image_url(product_id):
    """Return image URL for product; fallback to placeholder if missing."""
    return PRODUCT_IMAGES.get(product_id, PLACEHOLDER_IMG)


def stock_indicator(product_id):
    """Deterministic 'stock left' number for UI (3-8) based on product id."""
    n = int(hashlib.md5(product_id.encode()).hexdigest()[:4], 16) % 6 + 3
    return n


def _fallback_recommendation(product, store, old_price, new_price, discount_pct, savings):
    """Simple template when AI API is unavailable."""
    store_label = get_store_badge(store)
    return (
        f"Strong deal on {product}: {discount_pct:.0f}% off saves you {savings:.0f} AZN. "
        f"Best price at {store_label} compared to typical market prices."
    )


@st.cache_data(ttl=3600)
def generate_ai_recommendation(product, store, old_price, new_price, rating):
    """
    Call Gemini to generate a short professional deal insight.
    Cached so the same deal does not trigger repeated API calls.
    On API failure, returns a simple template message.
    """
    discount = ((old_price - new_price) / old_price) * 100 if old_price else 0
    savings = old_price - new_price
    store_label = get_store_badge(store)

    prompt = f"""You are an expert e-commerce pricing analyst.

Analyze this deal and write a short professional insight explaining why it is a good deal.

Product: {product}
Store: {store_label}
Original price: {old_price} AZN
New price: {new_price} AZN
Discount: {discount:.1f}%
Rating: {rating}

Explain the value of this deal in 2-3 sentences."""

    if gemini_model is None:
        return _fallback_recommendation(product, store, old_price, new_price, discount, savings)

    try:
        response = gemini_model.generate_content(prompt)
        if response and response.text:
            return response.text.strip()
    except Exception:
        pass
    return _fallback_recommendation(product, store, old_price, new_price, discount, savings)


# =============================================================================
# CSS – PROFESSIONAL E-COMMERCE STYLING (high contrast, white marketplace)
# =============================================================================
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* === GLOBAL: white background + dark text === */
    * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; }
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] { background-color: #FFFFFF !important; }
    .block-container { padding: 0.4rem 2rem 2rem !important; max-width: 100% !important; padding-top: 0.4rem !important; background: #FFFFFF !important; }
    header[data-testid="stHeader"] { background: transparent !important; height: 0 !important; padding: 0 !important; }
    header[data-testid="stHeader"] > div { display: none !important; }
    div[data-testid="stToolbar"] { display: none !important; }

    /* === Streamlit default overrides: ensure all text is visible === */
    p, span, label, div[class^="st"], .stMarkdown, .stMarkdown p { color: #111111 !important; }
    p { color: #222222 !important; }
    label { color: #111111 !important; }
    h1, h2, h3, h4, h5, h6 { color: #111111 !important; font-weight: 700 !important; }
    h1 { font-size: 2rem !important; color: #111111 !important; }
    h2 { font-size: 1.5rem !important; color: #111111 !important; }
    h3 { font-size: 1.25rem !important; color: #111111 !important; }
    h4 { font-size: 1.1rem !important; color: #111111 !important; }

    /* === Sidebar: light grey background + dark text === */
    [data-testid="stSidebar"] { background: #F5F5F5 !important; border-right: 1px solid #e0e0e0 !important; }
    [data-testid="stSidebar"] *, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label { color: #111111 !important; }
    [data-testid="stSidebar"] .stMarkdown { color: #111111 !important; }
    [data-testid="stSidebar"] .stMarkdown p { color: #222222 !important; }
    [data-testid="stSidebar"] .stRadio label {
        padding: 12px 14px !important; margin: 4px 8px !important; border-radius: 8px !important;
        font-weight: 500 !important; font-size: 14px !important; color: #111111 !important;
    }
    [data-testid="stSidebar"] .stRadio label:hover { background: #fff0f0 !important; color: #E30613 !important; }
    [data-testid="stSidebar"] .stRadio label:has(input:checked) { background: #fff0f0 !important; color: #E30613 !important; }
    [data-testid="stSidebar"] .stSelectbox label { color: #111111 !important; }

    /* === Input: search and all text inputs === */
    input { color: #111111 !important; background-color: #FFFFFF !important; }
    input::placeholder { color: #666666 !important; opacity: 1; }
    [data-testid="stTextInput"] input { color: #111111 !important; background: #FFFFFF !important; border: 1px solid #ddd !important; }
    [data-testid="stTextInput"] input::placeholder { color: #666666 !important; }
    [data-testid="stTextInput"] label { color: #111111 !important; }

    /* === Tables: headers and cells visible === */
    table, th, td { color: #111111 !important; }
    th { background-color: #F5F5F5 !important; color: #111111 !important; font-weight: 600 !important; }
    td { color: #222222 !important; }
    [data-testid="stDataFrame"] th, [data-testid="stDataFrame"] td { color: #111111 !important; }

    /* === Buttons and metrics === */
    [data-testid="stMetricLabel"] { color: #111111 !important; }
    [data-testid="stMetricValue"] { color: #111111 !important; }

    /* === Header bar === */
    .header-bar {
        background: #FFFFFF !important;
        border-bottom: 2px solid #E30613;
        padding: 12px 2rem;
        margin: -0.4rem -2rem 0 -2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .header-logo { font-size: 22px; font-weight: 700; color: #E30613 !important; }
    .header-nav { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }
    .header-nav span { color: #222222 !important; font-size: 13px; font-weight: 500; padding: 6px 12px; border-radius: 8px; }
    .header-nav span.active { background: #E30613 !important; color: #FFFFFF !important; }
    .basket-counter {
        background: #E30613; color: #FFFFFF !important; padding: 6px 12px; border-radius: 20px;
        font-size: 13px; font-weight: 700;
    }

    /* === Product cards: all text dark and visible === */
    .product-card {
        background: #FFFFFF !important; border-radius: 14px; padding: 0; border: 1px solid #e0e0e0;
        margin-bottom: 20px; overflow: hidden; position: relative;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    }
    .product-card:hover { border-color: #E30613; box-shadow: 0 8px 24px rgba(227,6,19,0.12); transform: translateY(-2px); }
    .product-card.selected { border: 2px solid #E30613; background: #fffbfb !important; }
    .product-card .card-img { width: 100%; height: 180px; object-fit: cover; background: #F5F5F5; }
    .product-card-image-wrap { border-radius: 14px 14px 0 0; overflow: hidden; border: 1px solid #e0e0e0; border-bottom: none; }
    .product-card-body-wrap { border: 1px solid #e0e0e0; border-top: none; border-radius: 0 0 14px 14px; padding: 0; background: #FFFFFF !important; margin-top: 0; position: relative; }
    .product-card .discount-badge {
        position: absolute; top: 10px; right: 10px; background: #E30613; color: #FFFFFF !important;
        padding: 5px 10px; border-radius: 6px; font-size: 12px; font-weight: 700; z-index: 1;
    }
    div[data-testid="column"] div[data-testid="stImage"] { border-radius: 14px 14px 0 0; overflow: hidden; border: 1px solid #e0e0e0; border-bottom: none; }
    .product-card .card-body { padding: 14px 16px; background: #FFFFFF !important; }
    .product-card .card-title { margin: 0 0 6px 0; font-size: 15px; font-weight: 600; color: #111111 !important; }
    .product-card .stars { color: #c2410c; font-size: 12px; margin: 4px 0; }
    .product-card .old-price { text-decoration: line-through; color: #555555 !important; font-size: 13px; margin: 0; }
    .product-card .new-price { color: #E30613 !important; font-size: 18px; font-weight: 700; margin: 4px 0 6px 0; }
    .product-card .store-badge {
        display: inline-block; background: #F5F5F5 !important; color: #111111 !important;
        padding: 4px 8px; border-radius: 6px; font-size: 11px; font-weight: 600; margin-bottom: 8px;
    }
    .product-card .stock { font-size: 11px; color: #E30613 !important; margin-bottom: 8px; }
    .price-table { font-size: 12px; margin-top: 8px; border-collapse: collapse; width: 100%; color: #111111 !important; }
    .price-table th, .price-table td { padding: 4px 8px; text-align: left; border-bottom: 1px solid #e0e0e0; color: #111111 !important; }
    .price-table th { color: #111111 !important; font-weight: 600 !important; background: #F5F5F5 !important; }
    .price-table td { color: #222222 !important; }

    /* === Super Deals panel === */
    .super-deals-wrap { padding: 18px; background: #F5F5F5 !important; border-radius: 14px; border: 1px solid #e0e0e0; margin-bottom: 16px; }
    .super-deals-wrap h3 { color: #E30613 !important; font-size: 16px; margin: 0 0 12px 0; font-weight: 700; }
    .super-deal-card {
        background: #FFFFFF !important; border-radius: 10px; padding: 12px 14px; margin-bottom: 10px;
        border: 1px solid #e0e0e0; font-size: 13px; box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        color: #111111 !important;
    }
    .super-deal-card .name { font-weight: 600; color: #111111 !important; }
    .super-deal-card .new { color: #E30613 !important; font-weight: 700; }
    .super-deal-card .deal-badge { background: #E30613; color: #FFFFFF !important; padding: 2px 6px; border-radius: 4px; font-size: 11px; font-weight: 700; margin-left: 6px; }
    .super-deal-card .ai-tiny { font-size: 11px; color: #222222 !important; margin-top: 6px; }

    /* === Deal insights & metrics === */
    .insight-card { background: #F5F5F5 !important; border-left: 4px solid #E30613; padding: 14px 16px; border-radius: 0 10px 10px 0; margin: 12px 0; font-size: 14px; color: #111111 !important; }
    .metric-card { background: #FFFFFF !important; border: 1px solid #e0e0e0; border-radius: 12px; padding: 16px; text-align: center; }
    .metric-card .value { font-size: 24px; font-weight: 700; color: #E30613 !important; }
    .metric-card .label { font-size: 12px; color: #222222 !important; margin-top: 4px; }

    /* === Caption and small text === */
    .stCaption, [data-testid="stCaptionContainer"] { color: #222222 !important; }
    small { color: #222222 !important; }

    /* === Main content: section titles and markdown === */
    [data-testid="stMarkdown"] { color: #111111 !important; }
    [data-testid="stMarkdown"] p, [data-testid="stMarkdown"] h1, [data-testid="stMarkdown"] h2,
    [data-testid="stMarkdown"] h3, [data-testid="stMarkdown"] h4 { color: #111111 !important; }
    .element-container { color: inherit; }
    div[data-testid="column"] { color: #111111 !important; }
    </style>
    """, unsafe_allow_html=True)


# =============================================================================
# SESSION STATE
# =============================================================================
if "selected_products" not in st.session_state:
    st.session_state.selected_products = set()
if "current_category" not in st.session_state:
    st.session_state.current_category = CATEGORY_KEYS[0]

# Load data once
stores_data, all_store_names = load_stores_data()
inject_css()

# =============================================================================
# HEADER: Logo, Nav, Search, Basket Counter
# =============================================================================
def render_header(basket_count):
    active = st.session_state.current_category
    nav_html = "".join(
        f'<span class="{"active" if c == active else ""}">{c}</span>' for c in CATEGORY_KEYS
    )
    st.markdown(f"""
    <div class="header-bar">
        <div class="header-logo">🛒 AI Deal Hunter</div>
        <nav class="header-nav">{nav_html}</nav>
        <div style="display:flex;align-items:center;gap:12px;">
            <span class="basket-counter">🛒 {basket_count}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    # Nav buttons (sync category with sidebar)
    cols = st.columns(5)
    for idx, cat in enumerate(CATEGORY_KEYS):
        with cols[idx]:
            if st.button(cat, key=f"nav_{idx}", use_container_width=True):
                st.session_state.current_category = cat
                st.rerun()


basket_count = len(st.session_state.selected_products)
render_header(basket_count)

# Search in main area (below header) – placeholder and label styled via CSS for visibility
search_query = st.text_input(
    "🔍 Search products",
    placeholder="Search by product name...",
    key="search",
    label_visibility="visible",
)

# =============================================================================
# SIDEBAR: Categories + Basket Preview
# =============================================================================
st.sidebar.markdown("### 📦 Categories")
category = st.sidebar.radio(
    "Category",
    CATEGORY_KEYS,
    index=CATEGORY_KEYS.index(st.session_state.current_category) if st.session_state.current_category in CATEGORY_KEYS else 0,
    label_visibility="collapsed",
)
st.session_state.current_category = category
products_in_category = CATEGORIES.get(category, [])

# Build full product options with rating and all store prices
product_options = []
for pid in products_in_category:
    store, data = get_cheapest_offer_for_product(pid, stores_data, all_store_names)
    if not data:
        continue
    discount = round((data["old_price"] - data["new_price"]) / data["old_price"] * 100, 0)
    name = PRODUCT_DISPLAY_NAMES.get(pid, pid)
    rating = data.get("rating", 4.5)
    all_offers = get_all_offers_for_product(pid, stores_data, all_store_names)
    product_options.append({
        "id": pid,
        "name": name,
        "old_price": data["old_price"],
        "new_price": data["new_price"],
        "discount": discount,
        "store": store,
        "rating": rating,
        "all_offers": all_offers,
    })

# Filter by search
if search_query and search_query.strip():
    q = search_query.strip().lower()
    product_options = [p for p in product_options if q in p["name"].lower()]

# Sort options
sort_by = st.sidebar.selectbox(
    "Sort by",
    ["Highest discount", "Lowest price", "Highest rating"],
    index=0,
)
if sort_by == "Highest discount":
    product_options = sorted(product_options, key=lambda x: -x["discount"])
elif sort_by == "Lowest price":
    product_options = sorted(product_options, key=lambda x: x["new_price"])
else:
    product_options = sorted(product_options, key=lambda x: -x["rating"])

# Basket preview in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 🛒 Basket preview")
if st.session_state.selected_products:
    for pid in sorted(st.session_state.selected_products):
        name = PRODUCT_DISPLAY_NAMES.get(pid, pid)
        st.sidebar.caption(f"• {name}")
else:
    st.sidebar.caption("_No items yet_")
st.sidebar.markdown(f"**Items:** {len(st.session_state.selected_products)}")

# =============================================================================
# MAIN: Product Grid (3 columns) + Right Panel (Super Deals)
# =============================================================================
main_col, right_col = st.columns([3, 1])

with main_col:
    st.markdown("#### 📋 Products")
    card_cols = st.columns(3)
    for i, opt in enumerate(product_options):
        col = card_cols[i % 3]
        with col:
            in_basket = opt["id"] in st.session_state.selected_products
            card_class = "product-card selected" if in_basket else "product-card"
            store_label = get_store_badge(opt["store"])
            stock = stock_indicator(opt["id"])
            stars = "★" * int(opt["rating"]) + "☆" * (5 - int(opt["rating"]))
            img_url = get_product_image_url(opt["id"])
            # Price comparison table rows
            table_rows = "".join(
                f'<tr><td>{get_store_badge(s)}</td><td>{p} AZN</td></tr>'
                for s, p, _ in opt["all_offers"]
            )
            # Product image at top (scale to column width; fallback handled by dict + PLACEHOLDER_IMG)
            st.image(img_url, use_container_width=True)
            # Card body: discount badge, name, prices, store, stock, table
            st.markdown(f"""
            <div class="{card_class} product-card-body-wrap">
                <div class="discount-badge">-{int(opt["discount"])}%</div>
                <div class="card-body">
                    <h4 class="card-title">{opt["name"]}</h4>
                    <p class="stars">{stars} {opt["rating"]}</p>
                    <p class="old-price">{opt["old_price"]} AZN</p>
                    <p class="new-price">{opt["new_price"]} AZN</p>
                    <span class="store-badge">{store_label}</span>
                    <p class="stock">Only {stock} left in stock!</p>
                    <table class="price-table"><thead><tr><th>Store</th><th>Price</th></tr></thead><tbody>{table_rows}</tbody></table>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if in_basket:
                if st.button("Remove from Basket", key=f"b_{opt['id']}", use_container_width=True):
                    st.session_state.selected_products.discard(opt["id"])
                    st.rerun()
            else:
                if st.button("Add to Basket", key=f"b_{opt['id']}", use_container_width=True):
                    st.session_state.selected_products.add(opt["id"])
                    st.rerun()

    st.markdown("---")
    calculate = st.button("🔥 **Calculate Best Deal**", use_container_width=True, type="primary")

# Right panel: Super Deals (discount >= 40% AND rating > 4; fallback: top by discount)
with right_col:
    all_deals = []
    for store in all_store_names:
        for pid, data in stores_data[store].items():
            discount = (data["old_price"] - data["new_price"]) / data["old_price"] * 100
            rating = data.get("rating", 4.0)
            all_deals.append({
                "product": pid,
                "name": PRODUCT_DISPLAY_NAMES.get(pid, pid),
                "store": store,
                "old_price": data["old_price"],
                "new_price": data["new_price"],
                "discount": round(discount, 0),
                "rating": rating,
            })
    super_deals = [d for d in all_deals if d["discount"] >= 40 and d["rating"] > 4]
    if not super_deals:
        super_deals = sorted(all_deals, key=lambda x: -x["discount"])[:8]
    else:
        super_deals.sort(key=lambda x: -x["discount"])
    st.markdown("<div class='super-deals-wrap'><h3>🔥 Super Deals</h3>", unsafe_allow_html=True)
    for d in super_deals[:6]:
        ai = generate_ai_recommendation(
            d["name"], d["store"], d["old_price"], d["new_price"], d.get("rating", 4.5)
        )
        ai_preview = (ai[:80] + "…") if len(ai) > 80 else ai
        st.markdown(f"""
        <div class="super-deal-card">
            <span class="name">{html.escape(d["name"])}</span><br>
            <span class="new">{d["new_price"]} AZN</span> <span class="deal-badge">-{int(d["discount"])}%</span><br>
            <span style="font-size:11px;color:#666;">{html.escape(get_store_badge(d["store"]))}</span>
            <p class="ai-tiny">{html.escape(ai_preview)}</p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Trending deals (top by discount)
    st.markdown("### 📈 Trending deals")
    trending = sorted(all_deals, key=lambda x: -x["discount"])[:3]
    for t in trending:
        st.caption(f"**{t['name']}** — {t['new_price']} AZN (−{int(t['discount'])}%)")

# =============================================================================
# RESULTS: After "Calculate Best Deal"
# =============================================================================
selected_list = sorted(st.session_state.selected_products)
if calculate and selected_list:
    results = []
    total_price = 0
    total_original = 0
    for product in selected_list:
        cheapest_store, best_data = get_cheapest_offer_for_product(product, stores_data, all_store_names)
        if not cheapest_store or not best_data:
            continue
        discount = (best_data["old_price"] - best_data["new_price"]) / best_data["old_price"] * 100
        total_price += best_data["new_price"]
        total_original += best_data["old_price"]
        results.append({
            "product": product,
            "name": PRODUCT_DISPLAY_NAMES.get(product, product),
            "store": cheapest_store,
            "old_price": best_data["old_price"],
            "new_price": best_data["new_price"],
            "discount": round(discount, 2),
            "rating": best_data.get("rating", 4.5),
        })

    total_savings = total_original - total_price
    best_discount = max((r["discount"] for r in results), default=0)

    st.markdown("---")
    st.markdown("## 📊 Your Best Deals")

    # Metrics row
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Total Basket Price", f"{total_price:.2f} AZN", None)
    with m2:
        st.metric("Total Savings", f"{total_savings:.2f} AZN", f"vs {total_original:.2f} AZN")
    with m3:
        st.metric("Best Discount Found", f"{best_discount:.1f}%", None)

    # Price comparison bar chart (per product, stores)
    chart_data = []
    for r in results:
        for store in all_store_names:
            products = stores_data.get(store, {})
            if r["product"] in products:
                chart_data.append({
                    "Product": r["name"],
                    "Store": get_store_badge(store),
                    "Price (AZN)": products[r["product"]]["new_price"],
                })
    if chart_data:
        df_chart = pd.DataFrame(chart_data)
        fig = px.bar(df_chart, x="Product", y="Price (AZN)", color="Store", barmode="group",
                     color_discrete_sequence=["#E30613", "#333", "#666"])
        fig.update_layout(margin=dict(t=20, b=60), xaxis_tickangle=-35, legend_title="Store")
        st.plotly_chart(fig, use_container_width=True)

    # Discount % bar chart
    df_disc = pd.DataFrame([{"Product": r["name"], "Discount %": r["discount"]} for r in results])
    fig2 = px.bar(df_disc, x="Product", y="Discount %", color="Discount %",
                  color_continuous_scale=["#ffcccc", "#E30613"], text_auto=".1f")
    fig2.update_layout(showlegend=False, margin=dict(t=20, b=60), xaxis_tickangle=-35)
    fig2.update_traces(textposition="outside")
    st.plotly_chart(fig2, use_container_width=True)

    # Deal Insights: AI-generated recommendations (cached per deal)
    st.markdown("---")
    st.markdown("## 🔥 Deal Insights")
    for item in results:
        store_label = get_store_badge(item["store"])
        ai_insight = generate_ai_recommendation(
            item["name"],
            item["store"],
            item["old_price"],
            item["new_price"],
            item.get("rating", 4.5),
        )
        st.markdown(f"""
        <div class="insight-card">
            <strong>{html.escape(item["name"])}</strong> — Best at {html.escape(store_label)} • <strong>{item["discount"]}% off</strong><br>
            {html.escape(ai_insight)}
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f'<div class="metric-card" style="margin-top:16px;"><span class="value">💰 {total_price:.2f} AZN</span><br><span class="label">Total basket price</span></div>', unsafe_allow_html=True)

elif calculate and not selected_list:
    st.warning("Please add at least one product to the basket, then click **Calculate Best Deal**.")
