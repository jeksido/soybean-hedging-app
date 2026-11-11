import streamlit as st

st.set_page_config(
    page_title="大豆期貨避險決策支援系統",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

import sqlite3
from pathlib import Path
from types import SimpleNamespace
import yfinance as yf
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from datetime import datetime, date
import math
import os
import requests
import streamlit.components.v1 as components


# =====================================================
# 系統常數
# =====================================================
SOYBEAN_TICKER = "ZS=F"
CONTRACT_BUSHELS = 5000
BUSHELS_PER_METRIC_TON = 36.7437
CONTRACT_METRIC_TONS = CONTRACT_BUSHELS / BUSHELS_PER_METRIC_TON


# =====================================================
# 完全離線的採購資料：只使用本機 SQLite，絕不連線 Supabase
# 注意：Streamlit Community Cloud 的本機檔案可能在重新部署時消失。
# =====================================================
DB_PATH = Path(__file__).resolve().parent / "soybean_local.db"
ALLOWED_COLUMNS = {
    "user_id", "purchase_order_no", "commodity", "quantity", "quantity_unit",
    "expected_purchase_date", "futures_contract", "hedge_ratio", "status",
    "entry_futures_price", "contracts_count", "actual_hedge_ratio",
}


def open_local_db():
    connection = sqlite3.connect(str(DB_PATH), timeout=15)
    connection.row_factory = sqlite3.Row
    return connection


def init_local_db():
    with open_local_db() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS soybean_exposures (
                exposure_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER DEFAULT 1,
                purchase_order_no TEXT NOT NULL,
                commodity TEXT NOT NULL DEFAULT 'Soybean',
                quantity REAL NOT NULL,
                quantity_unit TEXT NOT NULL,
                expected_purchase_date TEXT NOT NULL,
                futures_contract TEXT,
                hedge_ratio REAL NOT NULL DEFAULT 0,
                actual_hedge_ratio REAL,
                entry_futures_price REAL,
                contracts_count INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'Unhedged',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)


class LocalTable:
    """最小化的本機資料操作介面，保留既有六個模組的呼叫方式。"""
    def __init__(self):
        self.action = "select"
        self.payload = None
        self.filter_id = None
        self.order_column = None

    def select(self, columns="*"):
        if columns != "*":
            raise ValueError("本機模式僅支援選取全部欄位")
        self.action = "select"
        return self

    def order(self, column):
        if column not in ALLOWED_COLUMNS:
            raise ValueError("不支援的排序欄位")
        self.order_column = column
        return self

    def insert(self, payload):
        self.action, self.payload = "insert", payload
        return self

    def update(self, payload):
        self.action, self.payload = "update", payload
        return self

    def delete(self):
        self.action = "delete"
        return self

    def eq(self, column, value):
        if column != "exposure_id":
            raise ValueError("僅支援依紀錄編號更新或刪除")
        self.filter_id = int(value)
        return self

    def execute(self):
        with open_local_db() as connection:
            if self.action == "select":
                sql = "SELECT * FROM soybean_exposures"
                if self.order_column:
                    sql += f" ORDER BY {self.order_column}"
                rows = connection.execute(sql).fetchall()
                return SimpleNamespace(data=[dict(row) for row in rows])
            if self.action in ("insert", "update"):
                if not self.payload or set(self.payload) - ALLOWED_COLUMNS:
                    raise ValueError("包含不支援的資料欄位")
                keys = list(self.payload)
                values = [self.payload[k] for k in keys]
                if self.action == "insert":
                    placeholders = ", ".join("?" for _ in keys)
                    sql = f"INSERT INTO soybean_exposures ({', '.join(keys)}) VALUES ({placeholders})"
                    connection.execute(sql, values)
                else:
                    if self.filter_id is None:
                        raise ValueError("更新必須指定紀錄編號")
                    assignments = ", ".join(f"{k} = ?" for k in keys)
                    connection.execute(
                        f"UPDATE soybean_exposures SET {assignments}, updated_at = CURRENT_TIMESTAMP WHERE exposure_id = ?",
                        values + [self.filter_id],
                    )
            elif self.action == "delete":
                if self.filter_id is None:
                    raise ValueError("刪除必須指定紀錄編號")
                connection.execute("DELETE FROM soybean_exposures WHERE exposure_id = ?", (self.filter_id,))
        return SimpleNamespace(data=[])


class LocalDatabase:
    def table(self, name):
        if name != "soybean_exposures":
            raise ValueError("本機模式僅支援 soybean_exposures")
        return LocalTable()


init_local_db()
local_db = LocalDatabase()
LOCAL_DB_READY = True

# =====================================================
# 大豆行情資料
# =====================================================
@st.cache_data(ttl=900, show_spinner=False)
def fetch_soybean_futures(period: str, interval: str) -> pd.DataFrame:
    """從 Yahoo Finance 下載 CBOT 大豆期貨代理行情。"""
    data = yf.download(
        SOYBEAN_TICKER,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if data is None or data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()
    date_col = "Datetime" if "Datetime" in data.columns else "Date"
    data = data.rename(columns={date_col: "日期"})
    data["日期"] = pd.to_datetime(data["日期"], errors="coerce")

    numeric_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    return data.dropna(subset=["日期", "Close"]).sort_values("日期")


def cents_per_bushel_to_usd_per_ton(price_cents: float) -> float:
    """將美分／蒲式耳換算為美元／公噸。"""
    return (price_cents / 100.0) * BUSHELS_PER_METRIC_TON


def annualized_volatility(close: pd.Series, periods_per_year: int = 252) -> float:
    """使用對數報酬率估算年化歷史波動率。"""
    returns = np.log(close / close.shift(1)).dropna()
    if len(returns) < 2:
        return float("nan")
    return float(returns.std() * np.sqrt(periods_per_year))


def show_database_status(compact: bool = False):
    """顯示資料庫連線狀態。"""
    if LOCAL_DB_READY:
        (st.sidebar if compact else st).success("本機 SQLite：可使用（未連線 Supabase）")
    elif compact:
        st.sidebar.warning("本機資料庫尚未就緒")
    else:
        st.info("本機資料庫尚未就緒，行情功能仍可使用。")


# =====================================================
# 側邊欄導覽
# =====================================================
st.sidebar.title("🌱 系統功能選單")
st.sidebar.caption("大豆期貨避險決策支援系統｜完整展示版")

menu = st.sidebar.radio(
    "請選擇模組：",
    [
        "0. 系統首頁",
        "1. 大豆期貨行情儀表板",
        "2. 大豆採購曝險管理",
        "3. 期貨避險試算器",
        "4. 情境分析",
        "5. 歷史回測",
    ],
)

# 選單切換後，嘗試透過 Streamlit 內建收合按鈕自動收起側欄。
# 不強制隱藏側欄，以便使用者隨時再次開啟。
if "previous_menu" not in st.session_state:
    st.session_state["previous_menu"] = menu
elif st.session_state["previous_menu"] != menu:
    st.session_state["previous_menu"] = menu
    components.html("""
    <script>
    try {
      const doc = window.parent.document;
      const sidebar = doc.querySelector('[data-testid="stSidebar"]');
      const expanded = sidebar && sidebar.getAttribute('aria-expanded') !== 'false'
          && sidebar.getBoundingClientRect().width > 100;
      if (expanded) {
        const btn = sidebar.querySelector('[data-testid="stSidebarCollapseButton"] button')
          || sidebar.querySelector('button[aria-label*="Close"]')
          || sidebar.querySelector('button[aria-label*="Collapse"]');
        if (btn) btn.click();
      }
    } catch(e) { console.debug('Sidebar auto-collapse unavailable', e); }
    </script>
    """, height=0)

st.sidebar.markdown("---")
show_database_status(compact=True)
st.sidebar.info("採購紀錄僅存於本機；雲端重新部署可能清除資料。")
st.sidebar.caption("本機 SQLite 示範模式｜不連線 Supabase｜行情可能延遲。")
st.sidebar.caption("目前為單一使用者示範版；未建立登入與資料隔離前，請勿開放他人使用採購資料。")


# USDA NASS 月均農民收取價格（非特定交貨點每日現貨報價）
MONTHS = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,
          "JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_usda_spot(api_key, first_year, geo):
    params = {
        "key": api_key, "source_desc":"SURVEY", "commodity_desc":"SOYBEANS",
        "statisticcat_desc":"PRICE RECEIVED", "freq_desc":"MONTHLY",
        "agg_level_desc":geo, "year__GE":str(first_year), "format":"JSON",
    }
    response = requests.get("https://quickstats.nass.usda.gov/api/api_GET/",
                            params=params, timeout=35)
    response.raise_for_status()
    payload = response.json()
    if "error" in payload:
        raise ValueError(str(payload["error"]))
    data = pd.DataFrame(payload.get("data", []))
    if data.empty:
        raise ValueError("USDA 沒有回傳符合條件的月度現貨價格。")
    data = data[(data["unit_desc"].astype(str).str.upper().str.replace(" ", "") == "$/BU") &
                (data["reference_period_desc"].isin(MONTHS))].copy()
    if "domain_desc" in data.columns:
        data = data[data["domain_desc"] == "TOTAL"]
    if geo == "STATE":
        data = data[data["state_name"] == "IOWA"]
    data["月份"] = data["reference_period_desc"].map(MONTHS)
    data["日期"] = pd.to_datetime(dict(year=pd.to_numeric(data["year"], errors="coerce"),
                                       month=data["月份"], day=1), errors="coerce")
    data["現貨價格"] = pd.to_numeric(data["Value"].astype(str).str.replace(",", "", regex=False), errors="coerce") * 100
    data = data.dropna(subset=["日期", "現貨價格"])
    data = data[data["現貨價格"] > 0].sort_values("日期")
    data = data.drop_duplicates("日期", keep="last")
    if data.empty:
        raise ValueError("USDA 回傳資料缺少有效的月均價格（美元／蒲式耳）。")
    return data[["日期", "現貨價格"]].reset_index(drop=True)



# =====================================================
# 模組 0：系統首頁
# =====================================================
if menu == "0. 系統首頁":
    st.title("🌱 大豆期貨避險決策支援系統")
    st.subheader("以美國大豆採購企業之價格風險為例")

    st.info(
        "本系統整合大豆期貨行情、採購曝險管理、避險口數試算、價格情境分析與歷史回測，協助企業比較不同期貨避險策略。"
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("研究標的", "CBOT 大豆期貨")
    with col2:
        st.metric("核心避險方式", "多頭期貨避險")
    with col3:
        st.metric("標準合約規模", f"{CONTRACT_BUSHELS:,} 蒲式耳")

    st.markdown(
        f"""
        ### 專題情境
        大豆採購企業預計在未來買入大豆，若擔心價格上漲，可先買進大豆期貨，
        並在實際採購時賣出期貨平倉，以期貨損益抵銷部分現貨採購成本變動。

        ### 系統功能流程
        1. 查詢大豆期貨與後續現貨資料
        2. 建立大豆採購曝險
        3. 設定避險比例與計算期貨口數
        4. 比較不避險、部分避險與完全避險
        5. 進行情境分析、基差分析與歷史回測

        ### 單位基準
        - 期貨報價：美分／蒲式耳
        - 每口合約：{CONTRACT_BUSHELS:,} 蒲式耳
        - 約當公噸：{CONTRACT_METRIC_TONS:,.2f} 公噸／口
        """
    )

    st.warning("本系統為教學與專題研究用途，不構成交易或投資建議。")


# =====================================================
# 模組 1：大豆期貨行情儀表板
# =====================================================
elif menu == "1. 大豆期貨行情儀表板":
    st.header("📈 大豆期貨行情儀表板")
    st.caption("第一階段先使用 Yahoo Finance 的 ZS=F 作為 CBOT 大豆期貨行情代理資料。")

    c1, c2 = st.columns(2)
    with c1:
        period_label = st.selectbox(
            "查詢期間",
            ["3 個月", "6 個月", "1 年", "2 年", "5 年", "最長期間"],
            index=2,
        )
    with c2:
        chart_type = st.selectbox("圖表類型", ["K 線圖", "收盤價折線圖"])

    period_map = {
        "3 個月": "3mo",
        "6 個月": "6mo",
        "1 年": "1y",
        "2 年": "2y",
        "5 年": "5y",
        "最長期間": "max",
    }

    try:
        with st.spinner("正在取得大豆期貨行情..."):
            soybean_df = fetch_soybean_futures(period_map[period_label], "1d")
    except Exception as exc:
        soybean_df = pd.DataFrame()
        st.error(f"行情下載失敗：{exc}")

    if soybean_df.empty:
        st.warning(
            "目前未取得 ZS=F 行情。請稍後重試並確認部署環境可連線至 Yahoo Finance。"
        )
    else:
        latest = soybean_df.iloc[-1]
        previous = soybean_df.iloc[-2] if len(soybean_df) >= 2 else latest
        latest_close = float(latest["Close"])
        previous_close = float(previous["Close"])
        change = latest_close - previous_close
        change_pct = change / previous_close * 100 if previous_close else 0.0
        usd_per_ton = cents_per_bushel_to_usd_per_ton(latest_close)
        vol = annualized_volatility(soybean_df["Close"])

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("最新收盤價", f"{latest_close:,.2f} 美分/蒲式耳", f"{change_pct:+.2f}%")
        m2.metric("換算價格", f"{usd_per_ton:,.2f} 美元/公噸")
        m3.metric("年化歷史波動率", "--" if np.isnan(vol) else f"{vol:.2%}")
        m4.metric("資料日期", latest["日期"].strftime("%Y-%m-%d"))

        if chart_type == "K 線圖":
            fig = go.Figure(
                data=[
                    go.Candlestick(
                        x=soybean_df["日期"],
                        open=soybean_df["Open"],
                        high=soybean_df["High"],
                        low=soybean_df["Low"],
                        close=soybean_df["Close"],
                        name="大豆期貨",
                    )
                ]
            )
        else:
            fig = go.Figure(
                data=[
                    go.Scatter(
                        x=soybean_df["日期"],
                        y=soybean_df["Close"],
                        mode="lines",
                        name="收盤價",
                    )
                ]
            )

        fig.update_layout(
            title="CBOT 大豆期貨歷史行情（ZS=F）",
            xaxis_title="日期",
            yaxis_title="美分／蒲式耳",
            height=560,
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
        )
        st.plotly_chart(fig, width="stretch")

        with st.expander("查看資料與單位說明"):
            st.write(
                f"一口標準合約為 {CONTRACT_BUSHELS:,} 蒲式耳，約 {CONTRACT_METRIC_TONS:,.2f} 公噸。"
            )
            st.write("美元／公噸 = 美分／蒲式耳 ÷ 100 × 36.7437。")
            display_cols = [c for c in ["日期", "Open", "High", "Low", "Close", "Volume"] if c in soybean_df.columns]
            st.dataframe(soybean_df[display_cols].tail(20), width="stretch", hide_index=True)

        st.caption(
            f"頁面檢視時間：{datetime.now().strftime('%Y-%m-%d %H:%M')}｜"
            "最新行情資料日期以上方『資料日期』為準；ZS=F 為代理資料，回測仍須確認現貨、合約月份及換月方式。"
        )


# =====================================================
# 模組 2：大豆採購曝險管理
# =====================================================
elif menu == "2. 大豆採購曝險管理":
    st.header("📝 大豆採購曝險管理")
    st.caption("建立未來大豆採購需求，追蹤採購期限、避險比例與避險狀態。")
    show_database_status()

    if not LOCAL_DB_READY:
        st.warning("請確認程式所在資料夾有寫入權限，以建立本機 SQLite 檔案。")
        st.stop()

    status_map = {
        "Unhedged": "尚未避險",
        "Partially Hedged": "部分避險",
        "Futures Hedged": "期貨避險",
        "Closed": "已結清",
        "Expired": "已到期",
    }
    reverse_status_map = {v: k for k, v in status_map.items()}

    st.warning("本版本只使用本機 SQLite，不連線 Supabase。若公開部署，所有訪客可能共用同一份測試資料；請勿填入真實或敏感資料。")

    st.subheader("新增大豆採購需求")
    with st.form("add_soybean_exposure_form", clear_on_submit=True):
        row1_col1, row1_col2, row1_col3 = st.columns(3)
        with row1_col1:
            purchase_order_no = st.text_input("採購單號", placeholder="例如：SOY-2026-001")
        with row1_col2:
            quantity = st.number_input("採購數量", min_value=0.0, step=10.0, format="%.2f")
        with row1_col3:
            quantity_unit = st.selectbox("數量單位", ["公噸", "蒲式耳"])

        row2_col1, row2_col2, row2_col3 = st.columns(3)
        with row2_col1:
            expected_purchase_date = st.date_input("預計採購日", value=date.today())
        with row2_col2:
            futures_contract = st.text_input("預計期貨合約", placeholder="例如：ZSX26，可暫時留白")
        with row2_col3:
            hedge_ratio_pct = st.number_input(
                "預計避險比例（%）", min_value=0.0, max_value=100.0, value=0.0, step=5.0
            )

        submitted = st.form_submit_button("💾 新增採購需求", type="primary")

        if submitted:
            if not purchase_order_no.strip():
                st.warning("請輸入採購單號。")
            elif quantity <= 0:
                st.warning("採購數量必須大於 0。")
            else:
                try:
                    payload = {
                        "user_id": 1,
                        "purchase_order_no": purchase_order_no.strip(),
                        "commodity": "Soybean",
                        "quantity": float(quantity),
                        "quantity_unit": quantity_unit,
                        "expected_purchase_date": str(expected_purchase_date),
                        "futures_contract": futures_contract.strip() or None,
                        "hedge_ratio": float(hedge_ratio_pct) / 100.0,
                        "status": "Unhedged" if hedge_ratio_pct == 0 else "Partially Hedged",
                    }
                    local_db.table("soybean_exposures").insert(payload).execute()
                    st.success(f"已新增採購需求：{purchase_order_no.strip()}")
                    st.rerun()
                except Exception as exc:
                    st.error(f"新增失敗：{exc}")
                    st.info("若無法新增，請檢查本機資料夾是否可寫入。")

    st.divider()
    st.subheader("目前大豆採購曝險總表")

    try:
        response = (
            local_db.table("soybean_exposures")
            .select("*")
            .order("expected_purchase_date")
            .execute()
        )
        records = response.data or []

        if not records:
            st.info("目前尚無大豆採購需求。請先新增第一筆資料。")
        else:
            df_raw = pd.DataFrame(records)
            df_raw["expected_purchase_date"] = pd.to_datetime(
                df_raw["expected_purchase_date"], errors="coerce"
            )
            today_ts = pd.Timestamp(date.today())
            df_raw["days_to_purchase"] = (
                df_raw["expected_purchase_date"].dt.normalize() - today_ts
            ).dt.days

            def maturity_bucket(days):
                if pd.isna(days):
                    return "日期異常"
                if days < 0:
                    return "已逾期"
                if days <= 30:
                    return "短期（30天內）"
                if days <= 90:
                    return "中期（31～90天）"
                return "長期（91天以上）"

            df_raw["maturity_bucket"] = df_raw["days_to_purchase"].apply(maturity_bucket)
            df_raw["hedge_ratio"] = pd.to_numeric(df_raw.get("hedge_ratio", 0), errors="coerce").fillna(0)
            # 新欄位存在時，以真實口數換算的覆蓋率呈現，允許 >100%。
            if "actual_hedge_ratio" in df_raw.columns:
                df_raw["display_hedge_ratio"] = pd.to_numeric(
                    df_raw["actual_hedge_ratio"], errors="coerce"
                ).fillna(df_raw["hedge_ratio"])
            else:
                df_raw["display_hedge_ratio"] = df_raw["hedge_ratio"]
            df_raw["quantity"] = pd.to_numeric(df_raw["quantity"], errors="coerce").fillna(0)
            df_raw["hedged_quantity"] = df_raw["quantity"] * df_raw["display_hedge_ratio"]
            df_raw["unhedged_quantity"] = df_raw["quantity"] - df_raw["hedged_quantity"]

            metric1, metric2, metric3, metric4 = st.columns(4)
            metric1.metric("採購需求筆數", f"{len(df_raw):,}")
            metric2.metric("採購總量（公噸換算）", f"{df_raw.apply(lambda row: row['quantity'] if row['quantity_unit'] == '公噸' else row['quantity'] / BUSHELS_PER_METRIC_TON, axis=1).sum():,.2f}")
            metric3.metric("平均避險比例", f"{df_raw['display_hedge_ratio'].mean():.1%}")
            metric4.metric("30天內需求", f"{((df_raw['days_to_purchase'] >= 0) & (df_raw['days_to_purchase'] <= 30)).sum():,}")

            df_display = df_raw.copy()
            df_display.insert(0, "序號", range(1, len(df_display) + 1))
            df_display["避險比例"] = df_display["display_hedge_ratio"].map(lambda x: f"{x:.1%}")
            df_display["狀態"] = df_display["status"].map(status_map).fillna(df_display["status"])
            df_display["預計採購日"] = df_display["expected_purchase_date"].dt.strftime("%Y-%m-%d")
            df_display = df_display.rename(columns={
                "purchase_order_no": "採購單號",
                "commodity": "商品",
                "quantity": "採購數量",
                "quantity_unit": "單位",
                "futures_contract": "期貨合約",
                "days_to_purchase": "剩餘天數",
                "maturity_bucket": "期限分類",
                "hedged_quantity": "已避險數量",
                "unhedged_quantity": "未避險數量",
            })
            visible_cols = [
                "序號", "採購單號", "商品", "採購數量", "單位", "預計採購日",
                "剩餘天數", "期限分類", "期貨合約", "避險比例", "已避險數量",
                "未避險數量", "狀態",
            ]
            for quantity_col in ["採購數量", "已避險數量", "未避險數量"]:
                df_display[quantity_col] = pd.to_numeric(
                    df_display[quantity_col], errors="coerce"
                ).round(2)
            st.dataframe(
                df_display[visible_cols],
                width="stretch",
                hide_index=True,
                column_config={
                    "採購數量": st.column_config.NumberColumn(format="%.2f"),
                    "已避險數量": st.column_config.NumberColumn(format="%.2f"),
                    "未避險數量": st.column_config.NumberColumn(format="%.2f"),
                },
            )

            st.divider()
            action_col1, action_col2 = st.columns(2)

            options = []
            option_to_id = {}
            for row in df_raw.itertuples():
                label = f"{row.purchase_order_no}｜{row.quantity:,.2f} {row.quantity_unit}｜{row.expected_purchase_date:%Y-%m-%d}"
                options.append(label)
                option_to_id[label] = row.exposure_id

            with action_col1:
                st.subheader("更新避險狀態")
                update_item = st.selectbox("選擇採購需求", options, key="update_soybean_item")
                update_selected_id = option_to_id[update_item]
                update_selected_row = df_raw.loc[
                    df_raw["exposure_id"] == update_selected_id
                ].iloc[0]
                current_status_cn = status_map.get(
                    update_selected_row["status"], "尚未避險"
                )
                status_options = list(reverse_status_map.keys())
                update_status_cn = st.selectbox(
                    "新狀態",
                    status_options,
                    index=status_options.index(current_status_cn),
                    key=f"update_status_{update_selected_id}",
                )
                update_ratio_pct = st.number_input(
                    "實際／目前避險比例（%）",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(update_selected_row["hedge_ratio"] * 100),
                    step=0.1,
                    key=f"update_ratio_{update_selected_id}",
                )
                if st.button("更新狀態與比例", type="primary"):
                    try:
                        selected_id = update_selected_id
                        update_payload = {
                            "status": reverse_status_map[update_status_cn],
                            "hedge_ratio": float(update_ratio_pct) / 100.0,
                        }
                        (
                            local_db.table("soybean_exposures")
                            .update(update_payload)
                            .eq("exposure_id", selected_id)
                            .execute()
                        )
                        st.success("採購需求已更新。")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"更新失敗：{exc}")

            with action_col2:
                st.subheader("帶入試算與刪除")
                selected_item = st.selectbox("選擇資料", options, key="select_soybean_item")
                selected_id = option_to_id[selected_item]
                selected_row = df_raw.loc[df_raw["exposure_id"] == selected_id].iloc[0]

                if st.button("帶入期貨避險試算"):
                    st.session_state["selected_soybean_exposure"] = {
                        "exposure_id": int(selected_row["exposure_id"]),
                        "purchase_order_no": selected_row["purchase_order_no"],
                        "quantity": float(selected_row["quantity"]),
                        "quantity_unit": selected_row["quantity_unit"],
                        "expected_purchase_date": selected_row["expected_purchase_date"].strftime("%Y-%m-%d"),
                        "days_to_purchase": int(selected_row["days_to_purchase"]),
                        "futures_contract": selected_row.get("futures_contract") or "",
                        "hedge_ratio": float(selected_row["hedge_ratio"]),
                    }
                    st.success(f"已將 {selected_row['purchase_order_no']} 帶入模組3。")

                st.warning("刪除後無法復原。")
                confirm_delete = st.checkbox("我確認刪除此筆採購需求（無法復原）", key=f"confirm_delete_{selected_id}")
                if st.button("刪除此筆採購需求", disabled=not confirm_delete):
                    try:
                        (
                            local_db.table("soybean_exposures")
                            .delete()
                            .eq("exposure_id", selected_id)
                            .execute()
                        )
                        st.success("資料已刪除。")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"刪除失敗：{exc}")

    except Exception as exc:
        st.error(f"讀取 soybean_exposures 時發生錯誤：{exc}")
        st.info("請確認本機 SQLite 檔案可正常讀取。")


# =====================================================
# 模組 3：大豆期貨避險試算器
# =====================================================
elif menu == "3. 期貨避險試算器":
    st.header("🧮 大豆期貨避險試算器")
    st.caption("針對未來大豆採購需求，計算多頭期貨避險口數、實際覆蓋率與名目價值。")

    selected = st.session_state.get("selected_soybean_exposure")
    if selected:
        st.success(f"已載入採購單：{selected['purchase_order_no']}")

    try:
        latest_df = fetch_soybean_futures("1mo", "1d")
        market_price = float(latest_df.iloc[-1]["Close"]) if not latest_df.empty else 0.0
        market_date = latest_df.iloc[-1]["日期"].strftime("%Y-%m-%d") if not latest_df.empty else "無資料"
    except Exception:
        market_price = 0.0
        market_date = "無資料"

    default_quantity = float(selected["quantity"]) if selected else 500.0
    default_unit = selected["quantity_unit"] if selected else "公噸"
    default_ratio = float(selected["hedge_ratio"] * 100) if selected else 50.0
    default_contract = selected.get("futures_contract", "") if selected else ""

    with st.form("soybean_futures_hedge_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            quantity = st.number_input("採購數量", min_value=0.01, value=default_quantity, step=10.0)
            quantity_unit = st.selectbox(
                "數量單位", ["公噸", "蒲式耳"], index=0 if default_unit == "公噸" else 1
            )
        with c2:
            hedge_ratio_pct = st.number_input(
                "目標避險比例（%）", min_value=0.0, max_value=100.0,
                value=default_ratio, step=5.0
            )
            rounding_method = st.selectbox("期貨口數取整方式", ["四捨五入", "向下取整", "向上取整"])
        with c3:
            futures_price = st.number_input(
                "期貨建倉價格（美分／蒲式耳）",
                min_value=0.01,
                value=market_price if market_price > 0 else 1000.0,
                step=1.0,
            )
            futures_contract = st.text_input("期貨合約代碼", value=default_contract, placeholder="例如：ZSX26")

        commission_per_contract = st.number_input(
            "每口單邊交易成本（美元，可填0）", min_value=0.0, value=0.0, step=1.0
        )
        calculated = st.form_submit_button("開始計算", type="primary")

    if calculated:
        quantity_bushels = quantity * BUSHELS_PER_METRIC_TON if quantity_unit == "公噸" else quantity
        quantity_metric_tons = quantity if quantity_unit == "公噸" else quantity / BUSHELS_PER_METRIC_TON
        target_ratio = hedge_ratio_pct / 100.0
        target_bushels = quantity_bushels * target_ratio
        theoretical_contracts = target_bushels / CONTRACT_BUSHELS

        if rounding_method == "向下取整":
            actual_contracts = int(np.floor(theoretical_contracts))
        elif rounding_method == "向上取整":
            actual_contracts = int(np.ceil(theoretical_contracts))
        else:
            actual_contracts = int(np.floor(theoretical_contracts + 0.5))

        actual_hedged_bushels = actual_contracts * CONTRACT_BUSHELS
        actual_hedged_tons = actual_hedged_bushels / BUSHELS_PER_METRIC_TON
        actual_ratio = actual_hedged_bushels / quantity_bushels if quantity_bushels else 0.0
        unhedged_bushels = quantity_bushels - actual_hedged_bushels
        unhedged_tons = unhedged_bushels / BUSHELS_PER_METRIC_TON
        nominal_value = actual_hedged_bushels * futures_price / 100.0
        round_trip_cost = actual_contracts * commission_per_contract * 2

        st.divider()
        st.subheader("試算結果")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("理論期貨口數", f"{theoretical_contracts:.2f} 口")
        r2.metric("實際期貨口數", f"{actual_contracts} 口")
        r3.metric(
            "實際避險覆蓋率",
            f"{actual_ratio:.2%}",
            f"與目標相差 {actual_ratio - target_ratio:+.2%}",
            delta_color="off",
        )
        r4.metric("期貨部位名目價值", f"US$ {nominal_value:,.2f}")

        detail = pd.DataFrame({
            "項目": [
                "原始採購數量", "換算採購數量", "目標避險數量", "實際避險數量",
                "未避險數量", "每口合約規模", "建倉價格", "估計來回交易成本"
            ],
            "結果": [
                f"{quantity:,.2f} {quantity_unit}",
                f"{quantity_bushels:,.2f} 蒲式耳（{quantity_metric_tons:,.2f} 公噸）",
                f"{target_bushels:,.2f} 蒲式耳",
                f"{actual_hedged_bushels:,.2f} 蒲式耳（{actual_hedged_tons:,.2f} 公噸）",
                f"{unhedged_bushels:,.2f} 蒲式耳（{unhedged_tons:,.2f} 公噸）",
                f"{CONTRACT_BUSHELS:,} 蒲式耳（約 {CONTRACT_METRIC_TONS:,.2f} 公噸）",
                f"{futures_price:,.2f} 美分／蒲式耳",
                f"US$ {round_trip_cost:,.2f}",
            ]
        })
        st.dataframe(detail, width="stretch", hide_index=True)

        compare_rows = []
        for method in ["向下取整", "四捨五入", "向上取整"]:
            if method == "向下取整":
                contracts = int(np.floor(theoretical_contracts))
            elif method == "向上取整":
                contracts = int(np.ceil(theoretical_contracts))
            else:
                contracts = int(np.floor(theoretical_contracts + 0.5))
            covered = contracts * CONTRACT_BUSHELS
            ratio = covered / quantity_bushels if quantity_bushels else 0
            compare_rows.append({
                "取整方式": method,
                "期貨口數": contracts,
                "覆蓋蒲式耳": covered,
                "實際覆蓋率": f"{ratio:.2%}",
                "與目標差距": f"{ratio-target_ratio:+.2%}",
            })
        st.subheader("不同取整方式比較")
        st.dataframe(pd.DataFrame(compare_rows), width="stretch", hide_index=True)

        if actual_contracts == 0 and target_ratio > 0:
            st.warning("採購規模或避險比例不足一口標準期貨，取整後為0口。可考慮調整比例或研究微型合約。")
        elif actual_ratio > 1:
            st.warning("實際覆蓋率超過100%，屬於超額避險；價格反向變動時可能增加額外風險。")
        elif actual_ratio > target_ratio:
            st.info("實際口數略高於目標避險比例，存在部分超額覆蓋。")
        elif actual_ratio < target_ratio:
            st.info("實際口數低於目標避險比例，仍保留部分未避險曝險。")

        st.markdown(
            f"""
            ### 操作解讀
            大豆採購方擔心未來價格上漲，因此採用**多頭避險**：
            1. 現在買進 **{actual_contracts} 口**大豆期貨。
            2. 實際採購大豆時，在現貨市場買入大豆。
            3. 同時賣出期貨平倉，以期貨損益抵銷部分現貨成本變動。

            行情參考日期：**{market_date}**。期貨避險主要目的為降低採購成本波動，不保證產生獲利。
            """
        )

        st.session_state["soybean_hedge_result"] = {
            "exposure_id": selected.get("exposure_id") if selected else None,
            "purchase_order_no": selected.get("purchase_order_no") if selected else "手動試算",
            "quantity": float(quantity),
            "quantity_unit": quantity_unit,
            "target_hedge_ratio": float(target_ratio),
            "actual_hedge_ratio": float(actual_ratio),
            "futures_contract": futures_contract,
            "entry_futures_price": float(futures_price),
            "contracts_count": int(actual_contracts),
            "nominal_value_usd": float(nominal_value),
            "actual_hedged_bushels": float(actual_hedged_bushels),
            "actual_hedged_tons": float(actual_hedged_tons),
            "unhedged_bushels": float(unhedged_bushels),
            "unhedged_tons": float(unhedged_tons),
        }

        # -----------------------------------------------------
# 將已保存的試算結果寫回本機 SQLite
# 此區塊必須放在 if calculated 外面
# -----------------------------------------------------
saved_result = st.session_state.get("soybean_hedge_result")

if menu == "3. 期貨避險試算器" and saved_result:
    st.divider()
    st.subheader("💾 儲存避險試算結果")

    saved_order = saved_result.get(
        "purchase_order_no",
        "手動試算"
    )
    saved_contracts = saved_result.get(
        "contracts_count",
        0
    )
    saved_ratio = saved_result.get(
        "actual_hedge_ratio",
        0.0
    )

    st.info(
        f"目前待儲存結果：{saved_order}｜"
        f"{saved_contracts} 口期貨｜"
        f"實際覆蓋率 {saved_ratio:.2%}"
    )

    exposure_id = saved_result.get("exposure_id")

    if exposure_id is None:
        st.warning(
            "這是手動試算結果，沒有對應的採購需求，"
            "因此無法寫回資料庫。"
        )

    elif not LOCAL_DB_READY:
        st.warning(
            "本機 SQLite 尚未就緒，無法儲存資料。"
        )

    else:
        if st.button(
            "💾 將試算結果寫回採購需求",
            type="primary",
            key="save_soybean_hedge_result"
        ):
            try:
                actual_ratio = float(
                    saved_result["actual_hedge_ratio"]
                )
                contracts_count = int(
                    saved_result["contracts_count"]
                )

                if contracts_count <= 0:
                    new_status = "Unhedged"
                elif actual_ratio >= 0.999:
                    new_status = "Futures Hedged"
                else:
                    new_status = "Partially Hedged"

                # 本機資料庫可額外保存實際覆蓋率（可能超過 100%）。
                update_payload = {
                    "hedge_ratio": min(actual_ratio, 1.0),
                    "actual_hedge_ratio": actual_ratio,
                    "futures_contract": (
                        saved_result.get(
                            "futures_contract"
                        )
                        or None
                    ),
                    "entry_futures_price": float(
                        saved_result[
                            "entry_futures_price"
                        ]
                    ),
                    "contracts_count": contracts_count,
                    "status": new_status,
                }

                (
                    local_db.table(
                        "soybean_exposures"
                    )
                    .update(update_payload)
                    .eq(
                        "exposure_id",
                        exposure_id
                    )
                    .execute()
                )

                if (
                    "selected_soybean_exposure"
                    in st.session_state
                ):
                    st.session_state[
                        "selected_soybean_exposure"
                    ]["hedge_ratio"] = min(actual_ratio, 1.0)

                st.success(
                    "✅ 試算結果已成功寫回 "
                    "本機 soybean_exposures。"
                )
                if actual_ratio > 1.0:
                    st.warning("實際覆蓋率超過 100%，本機資料庫已另外保留完整覆蓋率。")

                st.info(
                    "請回到「2. 大豆採購曝險管理」"
                    "查看更新後的比例、數量與狀態。"
                )

            except Exception as exc:
                st.error(
                    f"❌ 試算結果寫回失敗：{exc}"
                )
                st.info("請確認本機 SQLite 檔案可寫入。")


# =====================================================
# 模組 4～5：下一階段逐一改造
# =====================================================
elif menu == "4. 情境分析":
    st.header("📊 大豆價格情境分析")
    st.caption(
        "模擬未來大豆現貨與期貨價格變動，"
        "比較不避險及不同期貨避險比例下的有效採購成本。"
    )

    saved_result = st.session_state.get(
        "soybean_hedge_result"
    )
    selected_exposure = st.session_state.get(
        "selected_soybean_exposure"
    )

    # -------------------------------------------------
    # 預設值
    # -------------------------------------------------
    if saved_result:
        default_quantity = float(
            saved_result.get("quantity", 500.0)
        )
        default_unit = saved_result.get(
            "quantity_unit",
            "公噸"
        )
        default_entry_futures_price = float(
            saved_result.get(
                "entry_futures_price",
                1200.0
            )
        )
        default_contracts = int(
            saved_result.get(
                "contracts_count",
                0
            )
        )
        loaded_order_no = saved_result.get(
            "purchase_order_no",
            "手動試算"
        )

        st.success(
            f"已載入模組3結果：{loaded_order_no}｜"
            f"{default_contracts}口期貨"
        )

    elif selected_exposure:
        default_quantity = float(
            selected_exposure.get(
                "quantity",
                500.0
            )
        )
        default_unit = selected_exposure.get(
            "quantity_unit",
            "公噸"
        )
        default_entry_futures_price = 1200.0
        default_contracts = 0

        st.info(
            "目前已載入模組2的採購資料，"
            "但尚未取得模組3完整試算結果。"
        )

    else:
        default_quantity = 500.0
        default_unit = "公噸"
        default_entry_futures_price = 1200.0
        default_contracts = 2

        st.info(
            "目前沒有模組2或模組3的資料，"
            "可以使用下方欄位進行手動情境分析。"
        )

    # -------------------------------------------------
    # 情境分析輸入
    # -------------------------------------------------
    with st.form("soybean_scenario_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            scenario_quantity = st.number_input(
                "採購數量",
                min_value=0.01,
                value=default_quantity,
                step=10.0
            )

            scenario_unit = st.selectbox(
                "數量單位",
                ["公噸", "蒲式耳"],
                index=(
                    0
                    if default_unit == "公噸"
                    else 1
                )
            )

        with col2:
            entry_spot_price = st.number_input(
                "目前現貨價格（美分／蒲式耳）",
                min_value=0.01,
                value=default_entry_futures_price,
                step=1.0
            )

            entry_futures_price = st.number_input(
                "期貨建倉價格（美分／蒲式耳）",
                min_value=0.01,
                value=default_entry_futures_price,
                step=1.0
            )

        with col3:
            selected_contracts = st.number_input(
                "實際期貨口數",
                min_value=0,
                value=default_contracts,
                step=1
            )

            transaction_cost = st.number_input(
                "每口來回交易成本（美元）",
                min_value=0.0,
                value=0.0,
                step=1.0
            )

        st.markdown("#### 基差假設")

        basis_col1, basis_col2 = st.columns(2)

        with basis_col1:
            entry_basis = entry_spot_price - entry_futures_price
            st.metric("建倉時基差（由現貨－期貨計算）", f"{entry_basis:+.2f} 美分／蒲式耳")

        with basis_col2:
            ending_basis = st.number_input(
                "採購時基差（現貨－期貨）",
                value=(
                    entry_spot_price
                    - entry_futures_price
                ),
                step=1.0
            )

        scenario_changes_text = st.text_input(
            "價格變動情境（%，以逗號分隔）",
            value="-20,-10,0,10,20"
        )

        run_scenario = st.form_submit_button(
            "開始情境分析",
            type="primary"
        )

    # -------------------------------------------------
    # 執行情境分析
    # -------------------------------------------------
    if run_scenario:
        try:
            scenario_changes = [
                float(value.strip()) / 100
                for value in scenario_changes_text.split(",")
                if value.strip()
            ]

            if any(not math.isfinite(x) or x <= -1 for x in scenario_changes):
                raise ValueError("價格變動率必須大於 -100%，且為有限數字。")

            if not scenario_changes:
                raise ValueError(
                    "請至少輸入一個價格變動情境。"
                )

            if scenario_unit == "公噸":
                total_bushels = (
                    scenario_quantity
                    * BUSHELS_PER_METRIC_TON
                )
            else:
                total_bushels = scenario_quantity

            actual_contracts = int(
                selected_contracts
            )

            hedged_bushels = (
                actual_contracts
                * CONTRACT_BUSHELS
            )

            actual_hedge_ratio = (
                hedged_bushels / total_bushels
                if total_bushels > 0
                else 0
            )

            total_transaction_cost = (
                actual_contracts
                * transaction_cost
            )

            results = []

            for price_change in scenario_changes:
                # 未來現貨價格
                future_spot_price = (
                    entry_spot_price
                    * (1 + price_change)
                )

                # 依照期末基差推算期貨平倉價格
                future_futures_price = (
                    future_spot_price
                    - ending_basis
                )

                # 未避險的現貨採購成本
                unhedged_cash_cost = (
                    total_bushels
                    * future_spot_price
                    / 100
                )

                # 多頭期貨損益
                futures_profit_loss = (
                    hedged_bushels
                    * (
                        future_futures_price
                        - entry_futures_price
                    )
                    / 100
                )

                # 避險後有效採購成本
                hedged_effective_cost = (
                    unhedged_cash_cost
                    - futures_profit_loss
                    + total_transaction_cost
                )

                # 每蒲式耳有效採購價格
                effective_price_per_bushel = (
                    hedged_effective_cost
                    / total_bushels
                    if total_bushels > 0
                    else 0
                )

                # 避險相對不避險的成本差異
                cost_difference = (
                    hedged_effective_cost
                    - unhedged_cash_cost
                )

                results.append({
                    "價格變動率": price_change,
                    "未來現貨價格": future_spot_price,
                    "期貨平倉價格": future_futures_price,
                    "不避險採購成本": unhedged_cash_cost,
                    "期貨損益": futures_profit_loss,
                    "交易成本": total_transaction_cost,
                    "避險後有效成本": hedged_effective_cost,
                    "每蒲式耳有效價格（美元）":
                        effective_price_per_bushel,
                    "相對成本變化": cost_difference
                })

            scenario_df = pd.DataFrame(results)

            st.divider()
            st.subheader("情境分析結果")

            metric1, metric2, metric3, metric4 = (
                st.columns(4)
            )

            metric1.metric(
                "總採購量",
                f"{total_bushels:,.2f} 蒲式耳"
            )

            metric2.metric(
                "期貨口數",
                f"{actual_contracts} 口"
            )

            metric3.metric(
                "實際避險覆蓋率",
                f"{actual_hedge_ratio:.2%}"
            )

            metric4.metric(
                "基差變動",
                f"{ending_basis - entry_basis:+.2f}"
            )

            # -----------------------------------------
            # 顯示結果表格
            # -----------------------------------------
            display_df = scenario_df.copy()

            display_df["價格變動率"] = (
                display_df["價格變動率"]
                .map(lambda value: f"{value:+.0%}")
            )

            money_columns = [
                "不避險採購成本",
                "期貨損益",
                "交易成本",
                "避險後有效成本",
                "相對成本變化"
            ]

            for column in money_columns:
                display_df[column] = (
                    display_df[column]
                    .map(
                        lambda value:
                        f"US$ {value:,.2f}"
                    )
                )

            price_columns = [
                "未來現貨價格",
                "期貨平倉價格",
                "每蒲式耳有效價格（美元）"
            ]

            for column in price_columns:
                display_df[column] = (
                    display_df[column]
                    .map(
                        lambda value:
                        f"{value:,.2f}"
                    )
                )

            st.dataframe(
                display_df,
                width="stretch",
                hide_index=True
            )
            st.caption(
                "相對成本變化 = 避險後有效成本－不避險採購成本；"
                "負值代表避險後成本較低，正值代表避險後成本較高。"
            )

            # -----------------------------------------
            # 成本比較圖
            # -----------------------------------------
            cost_fig = go.Figure()

            cost_fig.add_trace(
                go.Scatter(
                    x=(
                        scenario_df["價格變動率"]
                        * 100
                    ),
                    y=scenario_df[
                        "不避險採購成本"
                    ],
                    mode="lines+markers",
                    name="不避險採購成本"
                )
            )

            cost_fig.add_trace(
                go.Scatter(
                    x=(
                        scenario_df["價格變動率"]
                        * 100
                    ),
                    y=scenario_df[
                        "避險後有效成本"
                    ],
                    mode="lines+markers",
                    name="期貨避險後有效成本"
                )
            )

            cost_fig.update_layout(
                title="不同價格情境下的採購成本比較",
                xaxis_title="大豆現貨價格變動率（%）",
                yaxis_title="採購成本（美元）",
                height=520,
                hovermode="x unified"
            )

            st.plotly_chart(
                cost_fig,
                width="stretch"
            )

            # -----------------------------------------
            # 期貨損益圖
            # -----------------------------------------
            pnl_colors = [
                (
                    "#2E8B57"
                    if value >= 0
                    else "#C0392B"
                )
                for value in scenario_df["期貨損益"]
            ]

            pnl_fig = go.Figure(
                data=[
                    go.Bar(
                        x=(
                            scenario_df[
                                "價格變動率"
                            ]
                            * 100
                        ),
                        y=scenario_df[
                            "期貨損益"
                        ],
                        marker_color=pnl_colors,
                        name="期貨損益"
                    )
                ]
            )

            pnl_fig.update_layout(
                title="不同價格情境下的期貨損益",
                xaxis_title="大豆現貨價格變動率（%）",
                yaxis_title="期貨損益（美元）",
                height=460
            )

            st.plotly_chart(
                pnl_fig,
                width="stretch"
            )

            # -----------------------------------------
            # 結果解讀
            # -----------------------------------------
            st.subheader("情境解讀")

            increase_results = scenario_df[
                scenario_df["價格變動率"] > 0
            ]

            decrease_results = scenario_df[
                scenario_df["價格變動率"] < 0
            ]

            if not increase_results.empty:
                average_positive_futures_pnl = (
                    increase_results[
                        "期貨損益"
                    ].mean()
                )

                st.success(
                    "當大豆價格上漲時，多頭期貨部位"
                    f"平均產生 US$ "
                    f"{average_positive_futures_pnl:,.2f} "
                    "的期貨損益，可抵銷部分現貨"
                    "採購成本增加。"
                )

            if not decrease_results.empty:
                average_negative_futures_pnl = (
                    decrease_results[
                        "期貨損益"
                    ].mean()
                )

                st.info(
                    "當大豆價格下跌時，現貨採購成本"
                    "下降，但多頭期貨部位可能產生"
                    f"損失；目前下跌情境的平均期貨"
                    f"損益為 US$ "
                    f"{average_negative_futures_pnl:,.2f}。"
                )

            if ending_basis != entry_basis:
                st.warning(
                    "建倉與採購時的基差不同，"
                    "因此現貨價格與期貨價格不會"
                    "完全同步變動，仍存在基差風險。"
                )
            else:
                st.info(
                    "目前假設建倉與採購時基差相同。"
                    "在這項簡化假設下，期貨避險效果"
                    "主要受避險比例與整數口數影響。"
                )

            st.markdown(
                """
                ### 計算邏輯

                ```text
                不避險採購成本
                = 未來現貨價格 × 採購數量

                多頭期貨損益
                =（期貨平倉價格－期貨建倉價格）
                  × 實際避險數量

                避險後有效成本
                = 現貨採購成本
                  －期貨損益
                  ＋交易成本
                ```

                期貨避險的目的不是保證成本最低，
                而是降低大豆採購成本受市場價格變動
                影響的程度。
                """
            )

            st.session_state[
                "soybean_scenario_result"
            ] = scenario_df

        except ValueError as exc:
            st.error(f"情境設定錯誤：{exc}")

        except Exception as exc:
            st.error(f"情境分析失敗：{exc}")

    st.warning(
        "本情境分析採用簡化模型，尚未納入滑價、"
        "保證金、換月成本、每日結算及實際現貨"
        "品質或地區價差。"
    )

elif menu == "5. 歷史回測":
    st.header("⏳ USDA 現貨與 CBOT 期貨：真實基差歷史回測")
    st.caption("USDA NASS 每月農民收取價格 × Yahoo Finance ZS=F 連續期貨代理行情")
    st.info("USDA NASS 為月度平均生產者收取價格，不是某一天、某交貨點的可成交現貨報價；"
            "本回測將同月期貨每日收盤價取月均值進行配對，以避免把月均價當作每日即時價。")
    saved = st.session_state.get("soybean_hedge_result") or st.session_state.get("selected_soybean_exposure") or {}
    try:
        default_key = st.secrets.get("USDA_NASS_API_KEY", os.getenv("USDA_NASS_API_KEY", ""))
    except Exception:
        default_key = os.getenv("USDA_NASS_API_KEY", "")
    with st.form("usda_basis_backtest"):
        a,b,c = st.columns(3)
        with a:
            api_key = st.text_input("USDA NASS API Key", value=default_key, type="password",
                                    help="至 https://quickstats.nass.usda.gov/api 申請免費 Key")
            geo_label = st.selectbox("現貨價格地理範圍", ["美國全國平均", "愛荷華州（Iowa）"])
            history_years = st.selectbox("回測期間", [1,2,3,5,10], index=3)
        with b:
            horizon = st.selectbox("提前避險月數", [1,2,3,6], index=2)
            quantity = st.number_input("每次採購數量", min_value=0.01,
                                       value=float(saved.get("quantity",500.0)), step=10.0)
            unit = st.selectbox("數量單位", ["公噸","蒲式耳"],
                                index=0 if saved.get("quantity_unit","公噸")=="公噸" else 1)
        with c:
            ratio_text = st.text_input("比較避險比例（%）", "0,50,80,100")
            transaction_cost = st.number_input("每口來回交易成本（美元）", min_value=0.0, value=0.0)
            frequency = st.selectbox("採購頻率", ["每月一次", "每季一次"])
        run = st.form_submit_button("取得 USDA 價格並執行基差回測", type="primary")
    if run:
        try:
            if not api_key.strip():
                raise ValueError("請先輸入 USDA NASS API Key；不會使用虛構現貨價格替代。")
            ratios = sorted({float(x.strip())/100 for x in ratio_text.split(",") if x.strip()})
            if not ratios or any(not math.isfinite(x) or x<0 or x>1 for x in ratios):
                raise ValueError("避險比例須為 0 至 100 之間的數字。")
            with st.spinner("正在下載 USDA 月均現貨及 CBOT 期貨行情..."):
                geo = "NATIONAL" if geo_label == "美國全國平均" else "STATE"
                spot = fetch_usda_spot(api_key.strip(), date.today().year-history_years-2, geo)
                fut = fetch_soybean_futures("max", "1d")
            if fut.empty:
                raise ValueError("無法取得 Yahoo Finance ZS=F 期貨行情。")
            fut = fut[["日期","Close"]].copy()
            fut["日期"] = pd.to_datetime(fut["日期"], utc=True).dt.tz_localize(None)
            fut["月份"] = fut["日期"].dt.to_period("M")
            fut = fut.groupby("月份", as_index=False)["Close"].mean().rename(columns={"Close":"期貨價格"})
            spot["月份"] = spot["日期"].dt.to_period("M")
            prices = spot.merge(fut, on="月份", how="inner").sort_values("月份")
            cutoff = pd.Period(pd.Timestamp.today(),freq="M") - history_years*12
            prices = prices[prices["月份"]>=cutoff].copy().reset_index(drop=True)
            prices["基差"] = prices["現貨價格"] - prices["期貨價格"]
            if len(prices) < horizon+3:
                raise ValueError("現貨與期貨可配對月份不足，請延長回測期間或縮短避險月數。")
            qty = quantity * BUSHELS_PER_METRIC_TON if unit=="公噸" else quantity
            price_by_month = prices.set_index("月份")
            rows=[]
            months = prices["月份"].tolist()[::1 if frequency=="每月一次" else 3]
            for month in months:
                entry_month = month-horizon
                if entry_month not in price_by_month.index:
                    continue
                e=price_by_month.loc[entry_month]; x=price_by_month.loc[month]
                for ratio in ratios:
                    contracts=int(np.floor(qty*ratio/CONTRACT_BUSHELS+0.5))
                    pnl=contracts*CONTRACT_BUSHELS*(x["期貨價格"]-e["期貨價格"])/100
                    cash=qty*x["現貨價格"]/100
                    effective=cash-pnl+contracts*transaction_cost
                    rows.append({"採購月份":str(month),"建倉月份":str(entry_month),
                                 "目標避險比例":ratio,"實際避險比例":contracts*CONTRACT_BUSHELS/qty,
                                 "期貨口數":contracts,"建倉現貨":e["現貨價格"],"採購現貨":x["現貨價格"],
                                 "建倉期貨":e["期貨價格"],"平倉期貨":x["期貨價格"],
                                 "建倉基差":e["基差"],"採購基差":x["基差"],
                                 "基差變動":x["基差"]-e["基差"],
                                 "不避險採購成本":cash,"期貨損益":pnl,
                                 "交易成本":contracts*transaction_cost,"避險後有效成本":effective})
            detail=pd.DataFrame(rows)
            if detail.empty:
                raise ValueError("沒有可配對的建倉與採購月份。")
            baseline=detail.drop_duplicates("採購月份")["不避險採購成本"].var(ddof=1)
            summary=[]
            for ratio,g in detail.groupby("目標避險比例"):
                var=g["避險後有效成本"].var(ddof=1)
                summary.append({"目標避險比例":ratio,"平均實際避險比例":g["實際避險比例"].mean(),
                                "平均期貨口數":g["期貨口數"].mean(),
                                "平均有效採購成本":g["避險後有效成本"].mean(),
                                "成本標準差":g["避險後有效成本"].std(ddof=1),
                                "平均基差變動":g["基差變動"].mean(),
                                "避險有效性":1-var/baseline if pd.notna(baseline) and baseline>0 else float("nan"),
                                "樣本數":len(g)})
            summary=pd.DataFrame(summary)
            st.success(f"成功取得 {len(spot)} 筆 USDA 原始月度價格；其中 {len(prices)} 個月可與期貨配對。")
            m1,m2,m3,m4=st.columns(4)
            m1.metric("有效採購月份", detail["採購月份"].nunique())
            m2.metric("最新配對基差",f'{prices.iloc[-1]["基差"]:+.2f} ¢/bu')
            m3.metric("平均基差",f'{prices["基差"].mean():+.2f} ¢/bu')
            best=summary.loc[summary["成本標準差"].idxmin()]
            m4.metric("成本最穩定策略",f'{best["目標避險比例"]:.0%}')
            st.subheader("USDA 現貨與 CBOT 期貨月均價格")
            chart=go.Figure()
            for col in ["現貨價格","期貨價格"]:
                chart.add_trace(go.Scatter(x=prices["日期"],y=prices[col],mode="lines+markers",name=col))
            chart.update_layout(yaxis_title="美分／蒲式耳",height=430,hovermode="x unified")
            st.plotly_chart(chart,width="stretch")
            st.subheader("歷史基差走勢（現貨－期貨）")
            basisfig=go.Figure(go.Scatter(x=prices["日期"],y=prices["基差"],mode="lines+markers",name="基差"))
            basisfig.add_hline(y=0,line_dash="dash")
            basisfig.update_layout(yaxis_title="美分／蒲式耳",height=420)
            st.plotly_chart(basisfig,width="stretch")
            st.subheader("避險策略回測摘要")
            st.dataframe(summary,width="stretch",hide_index=True)
            st.subheader("各策略有效採購成本")
            costfig=go.Figure()
            for ratio,g in detail.groupby("目標避險比例"):
                costfig.add_trace(go.Scatter(x=g["採購月份"],y=g["避險後有效成本"],
                                             mode="lines+markers",name=f"{ratio:.0%}"))
            costfig.update_layout(yaxis_title="美元",height=480,hovermode="x unified")
            st.plotly_chart(costfig,width="stretch")
            with st.expander("查看 USDA 與期貨配對資料"):
                st.dataframe(prices,width="stretch",hide_index=True)
            with st.expander("查看逐筆回測與基差變動"):
                st.dataframe(detail,width="stretch",hide_index=True)
            st.download_button("下載回測明細 CSV",detail.to_csv(index=False).encode("utf-8-sig"),
                               file_name="usda_basis_backtest.csv",mime="text/csv")
            st.session_state["soybean_backtest_result"]={"summary":summary,"detail":detail,"prices":prices}
        except (ValueError, requests.RequestException) as exc:
            st.error(f"無法完成 USDA 基差回測：{exc}")
        except Exception as exc:
            st.error(f"歷史回測失敗：{exc}")
    st.warning("研究限制：USDA NASS 為月均農民收取價格，非實際採購報價；ZS=F 為連續期貨代理，"
               "未追蹤指定到期合約與換月損益。此為真實公開資料的月度基差代理回測，"
               "並非逐筆可交易的精確避險績效。另須注意 USDA 數據公布延遲與修訂，不能用於即時交易決策。")
    st.caption("資料來源：USDA NASS Quick Stats（PRICE RECEIVED, MONTHLY, $/BU）；Yahoo Finance ZS=F。"
               "This product uses the NASS API but is not endorsed or certified by NASS.")
