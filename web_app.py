"""
web_app.py - Supply Chain Analytics Inventory Decision Agent (Interactive Web Dashboard)

Interactive Streamlit Web Application implementing the 7-stage architectural pipeline:
Customer Question -> Identify Model -> Extract Inputs -> Select Python Tool -> Calculate -> Verify -> Explain

Supports:
1. Economic Order Quantity (EOQ)
2. Quantity Discount Optimization
3. Safety Stock & Reorder Point (Continuous Review)
4. Fixed-Period Review Model (Order-Up-To)
5. Newsvendor Single-Period Optimization
"""

from __future__ import annotations

import streamlit as st
import numpy as np
import pandas as pd
import math

from inventory_agent import InventoryDecisionAgent, AgentResponse
import inventory_tools as tools


# ---------------------------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Supply Chain Inventory Decision Agent",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    /* Completely hide Streamlit sidebar and toggle control */
    [data-testid="stSidebar"], [data-testid="collapsedControl"] {
        display: none !important;
    }
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .pipeline-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        margin: 0.15rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        background-color: #E0E7FF;
        color: #3730A3;
    }
    .kpi-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .kpi-title {
        font-size: 0.85rem;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
    }
    .kpi-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0F172A;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_agent():
    return InventoryDecisionAgent()

agent = get_agent()


# ---------------------------------------------------------------------------
# Pre-loaded Business Scenarios
# ---------------------------------------------------------------------------
SCENARIOS = {
    "Automotive EOQ Analysis": {
        "question": (
            "We distribute automotive brake pads and sell 12,000 units annually. "
            "Placing an order costs $25 each time, and our inventory holding cost is $0.50 per unit per year. "
            "Our supplier has a 5-day lead time over 250 operating days. "
            "What is our optimal batch size and when should we reorder?"
        ),
        "context": None,
    },
    "Electronics Volume Quantity Discount": {
        "question": (
            "Our annual demand for HDMI splitters is 1,000 units, and placing a purchase order costs $40. "
            "Our inventory carrying cost rate is 20% of unit price. "
            "Vendor offers Tier 1 (0-499 units) at $10.00 each, and Tier 2 (500+ units) at $8.00 each. "
            "Should we take the discount tier?"
        ),
        "context": {
            "price_breaks": [
                {"min_qty": 0, "max_qty": 499, "unit_price": 10.0},
                {"min_qty": 500, "max_qty": None, "unit_price": 8.0},
            ]
        },
    },
    "Hospital Pharma Continuous Review (SS & ROP)": {
        "question": (
            "We manage critical surgical kits with average daily demand of 50 units and standard deviation of 10 units. "
            "Supplier replenishment lead time averages 4 days with standard deviation of 1.5 days. "
            "Hospital protocol mandates a 98% Cycle Service Level. "
            "What safety stock buffer and reorder point (ROP) should we establish?"
        ),
        "context": None,
    },
    "Grocery FMCG Periodic Review (Order-Up-To)": {
        "question": (
            "Our distribution center reviews bakery inventory every 7 days. "
            "Delivery lead time is 3 days. Mean daily demand is 20 units with std dev 5 units. "
            "We target a 95% service level. "
            "We currently have 40 units in warehouse stock and 20 units in transit pipeline. "
            "What is our target Order-Up-To level S, and how many units should we order today?"
        ),
        "context": None,
    },
    "Fashion Seasonal Goods (Newsvendor)": {
        "question": (
            "We are preparing our winter designer coat collection. Each coat costs $40 to procure and retails for $100. "
            "Unsold coats at the end of winter are liquidated at an off-price salvage markdown of $10 each. "
            "Seasonal demand averages 500 coats with standard deviation of 100 coats. "
            "How many units should we purchase before the season begins?"
        ),
        "context": None,
    },
}


# ---------------------------------------------------------------------------
# Main Application Content: Header & Supported Models
# ---------------------------------------------------------------------------
st.markdown('<div class="main-header">Supply Chain Analytics Decision Agent</div>', unsafe_allow_html=True)
st.markdown("""
**Supported Models:**
1. **Economic Order Quantity (EOQ)**
2. **Quantity Discount Breaks**
3. **Continuous Review $(s, Q)$ ROP**
4. **Periodic Review $(R, S)$ Order-Up-To**
5. **Newsvendor Critical Fractile**
""")
st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Question Input & Action Row
# ---------------------------------------------------------------------------
if "active_analysis" not in st.session_state:
    st.session_state["active_analysis"] = None
if "last_uploaded_file_sig" not in st.session_state:
    st.session_state["last_uploaded_file_sig"] = None
if "user_query_input" not in st.session_state:
    st.session_state["user_query_input"] = ""
if "scenario_context" not in st.session_state:
    st.session_state["scenario_context"] = None

def on_scenario_change():
    selected = st.session_state.get("scenario_selector")
    if selected and selected in SCENARIOS:
        st.session_state["user_query_input"] = SCENARIOS[selected]["question"]
        st.session_state["scenario_context"] = SCENARIOS[selected]["context"]
    elif selected == "(Select a benchmark case...)":
        st.session_state["user_query_input"] = ""
        st.session_state["scenario_context"] = None
        st.session_state["last_response"] = None
        if st.session_state.get("active_analysis") == "QUESTION":
            st.session_state["active_analysis"] = None

def on_csv_upload():
    if st.session_state.get("csv_uploader") is not None:
        st.session_state["active_analysis"] = "CSV"

user_query = st.text_area(
    "Enter your natural-language supply chain question:",
    key="user_query_input",
    height=95,
    placeholder="e.g. We sell 10,000 units per year. Placing an order costs $50, holding cost is $2 per unit/year. What is our optimal order quantity?"
)

col_btn, col_bench = st.columns([1, 2])
with col_btn:
    submit_pressed = st.button("🚀 Analyze & Decide", type="primary", use_container_width=True)
with col_bench:
    scenario_choice = st.selectbox(
        "Choose a benchmark case:",
        ["(Select a benchmark case...)", *SCENARIOS.keys()],
        key="scenario_selector",
        on_change=on_scenario_change,
        label_visibility="collapsed"
    )

# ---------------------------------------------------------------------------
# Data Upload Section
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("📁 Data Upload")
uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"],
    key="csv_uploader",
    on_change=on_csv_upload,
    help="Upload an inventory CSV for multicriteria classification (Item Number, Item Name, UoM, Sales ($), Hits) or inventory analysis (Product, Category, Units, Unit Cost)"
)

# Synchronize file upload signature to auto-activate CSV mode on upload
if uploaded_file is not None:
    current_sig = f"{uploaded_file.name}_{uploaded_file.size}"
    if current_sig != st.session_state.get("last_uploaded_file_sig"):
        st.session_state["last_uploaded_file_sig"] = current_sig
        st.session_state["active_analysis"] = "CSV"
else:
    st.session_state["last_uploaded_file_sig"] = None
    if st.session_state.get("active_analysis") == "CSV":
        st.session_state["active_analysis"] = "QUESTION" if st.session_state.get("last_response") else None

show_csv_analysis = (st.session_state.get("active_analysis") == "CSV") and (uploaded_file is not None)

if uploaded_file is not None and not show_csv_analysis:
    st.caption(f"📁 Dataset `{uploaded_file.name}` uploaded. Displaying last supply chain question analysis below.")
    if st.button("📊 Show Uploaded Dataset Analysis", key="btn_show_csv_mode"):
        st.session_state["active_analysis"] = "CSV"
        st.rerun()

if show_csv_analysis:
    st.markdown("---")
    st.markdown("### 📁 Uploaded Inventory Dataset & Analysis")

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading uploaded CSV file: {e}")
        df = None

    if df is not None:
        # Schema Detection
        col_lookup = {str(c).strip().lower(): c for c in df.columns}
        has_item_num = any(k in col_lookup for k in ["item number", "item_number", "item no", "item no.", "item #", "sku", "item"])
        has_item_name = any(k in col_lookup for k in ["item name", "item_name", "description", "product name", "product"])
        has_uom = any(k in col_lookup for k in ["uom", "unit of measure", "unit_of_measure"])
        has_sales = any(k in col_lookup for k in ["sales ($)", "sales", "sales($)", "annual sales", "sales amount", "total sales"])
        has_hits = any(k in col_lookup for k in ["hits", "hit", "order lines", "pick frequency", "picks", "frequency"])

        is_classification_schema = has_sales and has_hits
        is_units_schema = all(c in df.columns for c in ["Product", "Category", "Units", "Unit Cost"])

        if not is_classification_schema and not is_units_schema:
            st.error(
                "⚠️ **CSV Schema Validation Notice:** The uploaded CSV does not match either recognized schema:\n\n"
                "1. **Multicriteria Inventory Classification (A, B, C, D):**\n"
                "   - Required Columns: `Item Number`, `Item Name`, `UoM`, `Sales ($)`, `Hits`\n"
                "2. **Standard Inventory Valuation Analysis:**\n"
                "   - Required Columns: `Product`, `Category`, `Units`, `Unit Cost`\n\n"
                f"**Columns found in uploaded file:** {', '.join([f'`{c}`' for c in df.columns])}"
            )
        else:
            # 1. Dataset Metadata (Rows, Columns, Column Names)
            meta_c1, meta_c2, meta_c3 = st.columns([1, 1, 3])
            meta_c1.metric("Total Items / Rows", f"{len(df):,}")
            meta_c2.metric("Total Columns", f"{len(df.columns):,}")
            col_list_str = ", ".join([f"`{col}`" for col in df.columns])
            meta_c3.markdown(f"**Column Names:** {col_list_str}")

            # If both schemas are present, render tabs. If only one is present, render directly.
            if is_classification_schema and is_units_schema:
                tab_mcic, tab_units = st.tabs(["🎯 Multicriteria ABCD Classification", "📊 Units & Valuation Analysis"])
            elif is_classification_schema:
                tab_mcic, tab_units = st.container(), None
            else:
                tab_mcic, tab_units = None, st.container()

            # --- MULTICRITERIA INVENTORY CLASSIFICATION ---
            if is_classification_schema and tab_mcic is not None:
                with tab_mcic:
                    st.markdown("#### 🎯 Multicriteria Inventory Classification (Weighted Sales & Hits)")
                    st.caption(
                        "Classifies inventory items into **A, B, C, and D** categories using separate Pareto cumulative ranking for "
                        "**Sales** (0.00-60.99% A, 61.00-80.99% B, 81.00-90.99% C, 91.00-100% D) and "
                        "**Hits** (0.00-60.99% A, 61.00-80.99% B, 81.00-90.99% C, 91.00-100% D), "
                        "converted to scores (A=4, B=3, C=2, D=1), weighted as **0.65(Sales) + 0.35(Hits)**, and assigned to final categories "
                        "(A: 4.00-3.25, B: 3.24-2.50, C: 2.49-1.75, D: <1.75)."
                    )

                    try:
                        classified_df, mcic_metrics = tools.classify_inventory_multicriteria(df)

                        # Safe extraction of totals with graceful fallbacks
                        total_sales = mcic_metrics.get("total_sales")
                        if total_sales is None:
                            sales_c = mcic_metrics.get("sales_col", "Sales ($)")
                            total_sales = float(classified_df[sales_c].sum()) if sales_c in classified_df.columns else 0.0

                        total_h = mcic_metrics.get("total_hits")
                        if total_h is None:
                            hits_c = mcic_metrics.get("hits_col", "Hits")
                            total_h = float(classified_df[hits_c].sum()) if hits_c in classified_df.columns else 0.0

                        # Safe extraction of class sales & counts
                        class_sales_dict = mcic_metrics.get("class_sales")
                        if not class_sales_dict:
                            summary_tbl = mcic_metrics.get("summary_table")
                            if summary_tbl is not None and "Total Sales ($)" in summary_tbl.columns:
                                cat_col = "Final Category" if "Final Category" in summary_tbl.columns else ("Class" if "Class" in summary_tbl.columns else None)
                                if cat_col:
                                    class_sales_dict = dict(zip(summary_tbl[cat_col], summary_tbl["Total Sales ($)"]))
                            if not class_sales_dict:
                                sales_c = mcic_metrics.get("sales_col", "Sales ($)")
                                cat_c = "Final Category" if "Final Category" in classified_df.columns else ("Class" if "Class" in classified_df.columns else None)
                                if cat_c and sales_c in classified_df.columns:
                                    class_sales_dict = classified_df.groupby(cat_c)[sales_c].sum().to_dict()
                                else:
                                    class_sales_dict = {}

                        class_counts_dict = mcic_metrics.get("class_counts")
                        if not class_counts_dict:
                            cat_c = "Final Category" if "Final Category" in classified_df.columns else ("Class" if "Class" in classified_df.columns else None)
                            if cat_c and cat_c in classified_df.columns:
                                class_counts_dict = classified_df[cat_c].value_counts().to_dict()
                            else:
                                class_counts_dict = {}

                        # Classification KPI Cards
                        k1, k2, k3, k4 = st.columns(4)
                        k1.metric("Total Sales Across Inventory", f"${total_sales:,.2f}")
                        k2.metric("Total Hits (Picks)", f"{total_h:,.0f}" if float(total_h).is_integer() else f"{total_h:,.2f}")

                        a_count = class_counts_dict.get('A', 0)
                        a_sales = class_sales_dict.get('A', 0.0)
                        a_pct = (a_count / len(classified_df) * 100) if len(classified_df) > 0 else 0.0
                        k3.metric("Category A Items", f"{a_count} Items", delta=f"${a_sales:,.2f} sales ({a_pct:.1f}% items)")

                        d_count = class_counts_dict.get('D', 0)
                        d_sales = class_sales_dict.get('D', 0.0)
                        d_pct = (d_count / len(classified_df) * 100) if len(classified_df) > 0 else 0.0
                        k4.metric("Category D Items", f"{d_count} Items", delta=f"${d_sales:,.2f} sales ({d_pct:.1f}% items)", delta_color="inverse")

                        # Summary: Number of Items and Total Sales for each Final Category
                        st.markdown("##### 📋 Category Summary: Number of Items & Total Sales per Final Category")
                        summary_tbl = mcic_metrics.get("summary_table")
                        if summary_tbl is not None:
                            display_cols = [
                                "Final Category", "Number of Items", "Total Sales ($)",
                                "% of Total Items", "% of Total Sales", "Total Hits",
                                "% of Total Hits", "Weighted Score Range", "Strategic Policy"
                            ]
                            summary_display = summary_tbl[[c for c in display_cols if c in summary_tbl.columns]]

                            st.dataframe(
                                summary_display.style.format({
                                    "Number of Items": "{:,.0f}" if "Number of Items" in summary_display.columns else "{:,.0f}",
                                    "Total Sales ($)": "${:,.2f}" if "Total Sales ($)" in summary_display.columns else "${:,.2f}",
                                    "% of Total Items": "{:.2f}%" if "% of Total Items" in summary_display.columns else "{:.2f}%",
                                    "% of Total Sales": "{:.2f}%" if "% of Total Sales" in summary_display.columns else "{:.2f}%",
                                    "Total Hits": "{:,.0f}" if "Total Hits" in summary_display.columns else "{:,.0f}",
                                    "% of Total Hits": "{:.2f}%" if "% of Total Hits" in summary_display.columns else "{:.2f}%",
                                }),
                                use_container_width=True
                            )

                            # Distribution Visualizations
                            cat_index_col = "Final Category" if "Final Category" in summary_tbl.columns else ("Class" if "Class" in summary_tbl.columns else summary_tbl.columns[0])
                            sales_metric_col = "Total Sales ($)" if "Total Sales ($)" in summary_tbl.columns else summary_tbl.columns[1]
                            items_metric_col = "Number of Items" if "Number of Items" in summary_tbl.columns else ("SKU Count" if "SKU Count" in summary_tbl.columns else None)
                            hits_metric_col = "Total Hits" if "Total Hits" in summary_tbl.columns else None

                            chart_col1, chart_col2, chart_col3 = st.columns(3)
                            with chart_col1:
                                st.markdown("**Total Sales by Final Category ($):**")
                                st.bar_chart(summary_tbl.set_index(cat_index_col)[sales_metric_col], height=230)
                            with chart_col2:
                                if items_metric_col:
                                    st.markdown("**Number of Items by Final Category:**")
                                    st.bar_chart(summary_tbl.set_index(cat_index_col)[items_metric_col], height=230)
                            with chart_col3:
                                if hits_metric_col:
                                    st.markdown("**Total Hits by Final Category:**")
                                    st.bar_chart(summary_tbl.set_index(cat_index_col)[hits_metric_col], height=230)

                        # Filter and Full Classified Dataframe Display
                        st.markdown("##### 🔍 Item-Level Classification & Weighted Scoring Table")
                        selected_classes = st.multiselect(
                            "Filter by Final Category:",
                            options=["A", "B", "C", "D"],
                            default=["A", "B", "C", "D"],
                            key="mcic_class_filter"
                        )
                        category_filter_col = "Final Category" if "Final Category" in classified_df.columns else ("Class" if "Class" in classified_df.columns else None)
                        if category_filter_col:
                            filtered_df = classified_df[classified_df[category_filter_col].isin(selected_classes)]
                        else:
                            filtered_df = classified_df

                        format_dict = {}
                        sales_c = mcic_metrics["sales_col"]
                        hits_c = mcic_metrics["hits_col"]
                        if sales_c in filtered_df.columns:
                            format_dict[sales_c] = "${:,.2f}"
                        if "Sales Cum %" in filtered_df.columns:
                            format_dict["Sales Cum %"] = "{:.2f}%"
                        if "Hits Cum %" in filtered_df.columns:
                            format_dict["Hits Cum %"] = "{:.2f}%"
                        if hits_c in filtered_df.columns:
                            format_dict[hits_c] = "{:,.0f}"
                        if "Combined Weighted Score" in filtered_df.columns:
                            format_dict["Combined Weighted Score"] = "{:.2f}"

                        st.dataframe(
                            filtered_df.style.format(format_dict),
                            use_container_width=True
                        )

                        # Download classified CSV
                        csv_export = classified_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="📥 Download Classified Inventory CSV",
                            data=csv_export,
                            file_name="multicriteria_classified_inventory.csv",
                            mime="text/csv",
                            key="mcic_download_btn"
                        )
                    except Exception as e:
                        st.error(f"Error during multicriteria inventory classification: {e}")

            # --- STANDARD UNITS & INVENTORY VALUATION ---
            if is_units_schema and tab_units is not None:
                with tab_units:
                    st.markdown("#### 📊 Standard Inventory Valuation & Category Breakdown")
                    # Coerce numeric fields safely
                    df_u = df.copy()
                    if not pd.api.types.is_numeric_dtype(df_u["Units"]):
                        df_u["Units"] = df_u["Units"].astype(str).str.replace(",", "", regex=False).str.strip()
                    df_u["Units"] = pd.to_numeric(df_u["Units"], errors="coerce").fillna(0)

                    if not pd.api.types.is_numeric_dtype(df_u["Unit Cost"]):
                        df_u["Unit Cost"] = df_u["Unit Cost"].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False).str.strip()
                    df_u["Unit Cost"] = pd.to_numeric(df_u["Unit Cost"], errors="coerce").fillna(0.0)

                    total_units = df_u["Units"].sum()
                    avg_unit_cost = df_u["Unit Cost"].mean() if not df_u.empty else 0.0

                    if not df_u.empty and total_units > 0:
                        highest_idx = df_u["Units"].idxmax()
                        highest_product = str(df_u.loc[highest_idx, "Product"])
                        highest_units_val = df_u.loc[highest_idx, "Units"]
                    else:
                        highest_product = "N/A"
                        highest_units_val = 0

                    category_units_df = df_u.groupby("Category")["Units"].sum().reset_index()
                    category_units_df.columns = ["Category", "Total Units"]
                    category_units_df = category_units_df.sort_values(by="Total Units", ascending=False)

                    df_u["Inventory Value"] = df_u["Units"] * df_u["Unit Cost"]
                    total_inventory_value = df_u["Inventory Value"].sum()

                    uk1, uk2, uk3, uk4 = st.columns(4)
                    uk1.metric(
                        "Total Units",
                        f"{total_units:,.0f}" if float(total_units).is_integer() else f"{total_units:,.2f}"
                    )
                    uk2.metric("Average Unit Cost", f"${avg_unit_cost:,.2f}")
                    uk3.metric(
                        "Product with Highest Units",
                        f"{highest_product}",
                        delta=f"{highest_units_val:,.0f} units" if float(highest_units_val).is_integer() else f"{highest_units_val:,.2f} units"
                    )
                    uk4.metric("Total Inventory Value", f"${total_inventory_value:,.2f}")

                    subtab_df, subtab_cat = st.tabs(["📋 Complete Inventory Data (with Inventory Value)", "🏷️ Total Units by Category"])
                    with subtab_df:
                        st.markdown("**Uploaded Dataframe with Calculated `Inventory Value` Column:**")
                        st.dataframe(
                            df_u.style.format({
                                "Unit Cost": "${:,.2f}",
                                "Inventory Value": "${:,.2f}",
                                "Units": "{:,.0f}" if (df_u["Units"] % 1 == 0).all() else "{:,.2f}"
                            }),
                            use_container_width=True
                        )
                    with subtab_cat:
                        cc1, cc2 = st.columns([1.5, 2])
                        with cc1:
                            st.markdown("**Total Units by Category Table:**")
                            st.dataframe(
                                category_units_df.style.format({
                                    "Total Units": "{:,.0f}" if (category_units_df["Total Units"] % 1 == 0).all() else "{:,.2f}"
                                }),
                                use_container_width=True
                            )
                        with cc2:
                            st.markdown("**Category Volume Distribution:**")
                            st.bar_chart(category_units_df.set_index("Category")["Total Units"], height=260)

    st.markdown("---")

# ---------------------------------------------------------------------------
# Execution & Decision Results Rendering
# ---------------------------------------------------------------------------
if submit_pressed and user_query.strip():
    with st.spinner("Processing through Decision Agent Pipeline..."):
        response: AgentResponse = agent.process_question(
            user_query,
            context=st.session_state.get("scenario_context")
        )
        st.session_state["last_response"] = response
        st.session_state["active_analysis"] = "QUESTION"

show_question_analysis = (st.session_state.get("active_analysis") == "QUESTION") and (st.session_state.get("last_response") is not None)

if show_question_analysis:
    response: AgentResponse = st.session_state["last_response"]

    st.markdown("---")
    st.markdown("### 📋 Decision Analysis & Operational Recommendation")

    # 1. Pipeline Status & Identified Model Card
    c1, c2, c3, c4 = st.columns([2, 1.5, 1.5, 1.5])
    c1.metric("Identified Model", response.identified_model, help="Classified via semantic keyword & intent engine")
    c2.metric("Agent Confidence", f"{response.confidence * 100:.1f}%")
    c3.metric("Selected Tool", response.tool_selected or "None")
    c4.metric(
        "Verification Status",
        "Passed ✅" if response.verification_passed else "Action Required ⚠️",
        help="Operational boundary and sanity checks"
    )

    # 2. Executive Recommendation Alert
    if response.verification_passed and response.calculation_result:
        st.success(response.executive_summary)
    else:
        st.warning(response.executive_summary)

    # 3. Model-Specific KPI Metrics & Interactive Charts
    if response.calculation_result:
        res_data = response.calculation_result
        st.markdown("### 📊 Operational Decision Metrics")

        # --- MODEL 1: EOQ ---
        if response.identified_model == "EOQ":
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Optimal Order Quantity (Q*)", f"{res_data['optimal_order_quantity']:,.0f} units")
            k2.metric("Orders per Year", f"{res_data['orders_per_year']:.1f} orders")
            k3.metric("Order Cycle Interval", f"{res_data['order_cycle_days']:.1f} days")
            k4.metric("Total Annual Inventory Cost", f"${res_data['total_inventory_cost']:,.2f}")

            # Interactive EOQ Curve
            q_star = res_data["optimal_order_quantity"]
            q_range = np.linspace(max(10, q_star * 0.2), q_star * 2.2, 100)
            d = res_data["annual_demand"]
            s = res_data["ordering_cost"]
            h = res_data["holding_cost_per_unit"]

            holding_curve = (q_range / 2.0) * h
            ordering_curve = (d / q_range) * s
            total_curve = holding_curve + ordering_curve

            df_chart = pd.DataFrame({
                "Order Quantity (Q)": q_range,
                "Holding Cost ($)": holding_curve,
                "Ordering Cost ($)": ordering_curve,
                "Total Annual Cost ($)": total_curve,
            }).set_index("Order Quantity (Q)")

            st.line_chart(df_chart, height=320)
            st.caption("Figure: Total Annual Cost Curve showing Wilson cost equilibrium at optimal EOQ Q*.")

        # --- MODEL 2: QUANTITY DISCOUNT ---
        elif response.identified_model == "QUANTITY_DISCOUNT":
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Optimal Lot Size", f"{res_data['optimal_order_quantity']:,.0f} units")
            k2.metric("Optimal Tier Unit Price", f"${res_data['optimal_unit_price']:,.2f}")
            k3.metric("Total Landed Cost", f"${res_data['optimal_total_cost']:,.2f}")
            k4.metric("Annual Savings", f"${res_data.get('annual_savings', 0):,.2f}", delta=f"{res_data.get('percent_savings', 0):.1f}%")

            # Tier Comparison Table
            st.markdown("#### Price Break Comparison")
            tiers = res_data.get("tiers_evaluated", [])
            if tiers:
                df_tiers = pd.DataFrame(tiers)[
                    ["tier_index", "min_qty", "max_qty", "unit_price", "order_quantity", "annual_purchase_cost", "annual_holding_cost", "annual_ordering_cost", "total_annual_cost", "is_optimal"]
                ]
                df_tiers.columns = ["Tier", "Min Qty", "Max Qty", "Unit Price ($)", "Order Qty", "Purchase Cost ($)", "Holding Cost ($)", "Ordering Cost ($)", "Total Annual Cost ($)", "Optimal?"]
                st.dataframe(df_tiers, use_container_width=True)

        # --- MODEL 3: SAFETY STOCK & ROP ---
        elif response.identified_model == "SAFETY_STOCK_ROP":
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Reorder Point (ROP)", f"{res_data['reorder_point']:,.0f} units")
            k2.metric("Safety Stock Buffer", f"{res_data['safety_stock']:,.0f} units")
            k3.metric("Service Level (CSL)", f"{res_data['service_level'] * 100:.1f}%")
            k4.metric("Stockout Risk per Cycle", f"{res_data['stockout_risk_per_cycle'] * 100:.2f}%")

            # Normal Distribution Bell Curve for Lead Time Demand
            mu = res_data["lead_time_demand_mean"]
            sigma = res_data["lead_time_demand_std"]
            rop = res_data["reorder_point"]

            if sigma > 0:
                x_vals = np.linspace(max(0, mu - 4 * sigma), mu + 4 * sigma, 200)
                y_vals = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_vals - mu) / sigma) ** 2)
                df_dist = pd.DataFrame({"Lead Time Demand": x_vals, "Probability Density": y_vals}).set_index("Lead Time Demand")
                st.line_chart(df_dist, height=300)
                st.caption(f"Figure: Lead Time Demand Normal Distribution (Mean = {mu:.1f}, Std Dev = {sigma:.2f}). Reorder Point set at {rop:.1f} units.")

                # Service Level vs Safety Stock Exponential Trade-off Curve
                import scipy.stats as stats
                csl_levels = np.linspace(0.80, 0.999, 40)
                ss_levels = [float(stats.norm.ppf(l)) * sigma for l in csl_levels]
                df_tradeoff = pd.DataFrame({
                    "Target Service Level": [f"{l*100:.1f}%" for l in csl_levels],
                    "Safety Stock Units": ss_levels
                }).set_index("Target Service Level")
                st.line_chart(df_tradeoff, height=260)
                st.caption("Figure: Exponential Safety Stock growth as Cycle Service Level approaches 99.9%.")

        # --- MODEL 4: FIXED-PERIOD REVIEW ---
        elif response.identified_model == "FIXED_PERIOD_REVIEW":
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Target Level (S*)", f"{res_data['target_inventory_level']:,.0f} units")
            k2.metric("Immediate Order Quantity", f"{res_data['recommended_order_quantity']:,.0f} units")
            k3.metric("Current Inventory Position", f"{res_data['current_inventory_position']:,.0f} units")
            k4.metric("Protection Interval (R+L)", f"{res_data['protection_interval']:.1f} days")

        # --- MODEL 5: NEWSVENDOR ---
        elif response.identified_model == "NEWSVENDOR":
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Optimal Stocking (Q*)", f"{res_data['optimal_stocking_quantity']:,.0f} units")
            k2.metric("Critical Ratio", f"{res_data['critical_ratio'] * 100:.2f}%")
            k3.metric("Expected Net Profit", f"${res_data['expected_profit']:,.2f}")
            k4.metric("EVPI (Forecasting Value)", f"${res_data['evpi']:,.2f}")

            # Expected Profit vs Stocking Quantity Curve
            p = res_data["selling_price"]
            c = res_data["unit_cost"]
            s_val = res_data["salvage_value"]
            mu = res_data["demand_mean"]
            sigma = res_data["demand_std"]

            if sigma > 0:
                import scipy.stats as stats
                q_star = res_data["optimal_stocking_quantity"]
                q_vals = np.linspace(max(0, mu - 3 * sigma), mu + 3 * sigma, 80)
                profits = []
                for q_i in q_vals:
                    z_i = (q_i - mu) / sigma
                    phi_z = float(stats.norm.pdf(z_i))
                    Phi_z = float(stats.norm.cdf(z_i))
                    loss_z = phi_z - z_i * (1.0 - Phi_z)
                    els_i = sigma * loss_z
                    es_i = mu - els_i
                    elu_i = max(0.0, q_i - es_i)
                    rev_i = p * es_i + s_val * elu_i
                    prof_i = rev_i - (c * q_i)
                    profits.append(prof_i)

                df_nv = pd.DataFrame({"Order Quantity (Q)": q_vals, "Expected Profit ($)": profits}).set_index("Order Quantity (Q)")
                st.line_chart(df_nv, height=300)
                st.caption(f"Figure: Newsvendor Expected Profit Curve peaking at Q* = {q_star:.1f} units.")

        # --- Interactive What-If Sensitivity Sandbox ---
        with st.expander("🎛️ Interactive 'What-If' Parameter Sensitivity Sandbox", expanded=False):
            st.markdown("Adjust parameters dynamically to observe how decision thresholds and total costs respond in real time.")

            if response.identified_model == "EOQ":
                w_col1, w_col2, w_col3 = st.columns(3)
                base_d = float(res_data.get("annual_demand", 10000))
                base_s = float(res_data.get("ordering_cost", 50))
                base_h = float(res_data.get("holding_cost_per_unit", 2))

                sim_d = w_col1.slider("Simulated Demand (D)", min_value=max(100.0, base_d * 0.2), max_value=base_d * 2.5, value=base_d, step=100.0)
                sim_s = w_col2.slider("Simulated Order Cost (S)", min_value=max(1.0, base_s * 0.2), max_value=base_s * 3.0, value=base_s, step=5.0)
                sim_h = w_col3.slider("Simulated Holding Cost (H)", min_value=max(0.1, base_h * 0.2), max_value=base_h * 3.0, value=base_h, step=0.25)

                sim_res = tools.calculate_eoq(demand=sim_d, ordering_cost=sim_s, holding_cost=sim_h)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Simulated Q*", f"{sim_res.optimal_order_quantity:,.0f} units", delta=f"{sim_res.optimal_order_quantity - res_data['optimal_order_quantity']:+,.0f}")
                m2.metric("Orders / Year", f"{sim_res.orders_per_year:.1f}", delta=f"{sim_res.orders_per_year - res_data['orders_per_year']:+.1f}")
                m3.metric("Cycle Interval", f"{sim_res.order_cycle_days:.1f} days", delta=f"{sim_res.order_cycle_days - res_data['order_cycle_days']:+.1f}")
                m4.metric("Simulated Total Cost", f"${sim_res.total_inventory_cost:,.2f}", delta=f"${sim_res.total_inventory_cost - res_data['total_inventory_cost']:+,.2f}", delta_color="inverse")

            elif response.identified_model == "SAFETY_STOCK_ROP":
                w_col1, w_col2, w_col3 = st.columns(3)
                base_csl = float(res_data.get("service_level", 0.95))
                base_lt_std = float(res_data.get("lead_time_std", 0.0))
                base_d_std = float(res_data.get("demand_std", 0.0))
                base_d = float(res_data.get("demand_rate", 50))
                base_lt = float(res_data.get("lead_time", 4))

                sim_csl = w_col1.slider("Target Service Level (CSL)", min_value=0.80, max_value=0.999, value=min(0.999, max(0.80, base_csl)), step=0.005, format="%.3f")
                sim_d_std = w_col2.slider("Daily Demand Std Dev (sigma_d)", min_value=0.0, max_value=max(10.0, base_d_std * 3.0), value=base_d_std, step=1.0)
                sim_lt_std = w_col3.slider("Lead Time Std Dev (sigma_L)", min_value=0.0, max_value=max(5.0, base_lt_std * 3.0), value=base_lt_std, step=0.25)

                sim_res = tools.calculate_rop(demand_rate=base_d, lead_time=base_lt, service_level=sim_csl, demand_std=sim_d_std, lead_time_std=sim_lt_std)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Simulated ROP", f"{sim_res.reorder_point:,.1f} units", delta=f"{sim_res.reorder_point - res_data['reorder_point']:+,.1f}")
                m2.metric("Simulated Safety Stock", f"{sim_res.safety_stock:,.1f} units", delta=f"{sim_res.safety_stock - res_data['safety_stock']:+,.1f}")
                m3.metric("Normal z-score", f"{sim_res.z_score:.4f}", delta=f"{sim_res.z_score - res_data['z_score']:+.4f}")
                m4.metric("Cycle Stockout Risk", f"{sim_res.stockout_risk_per_cycle * 100:.2f}%", delta=f"{(sim_res.stockout_risk_per_cycle - res_data['stockout_risk_per_cycle']) * 100:+.2f}%", delta_color="inverse")

            elif response.identified_model == "NEWSVENDOR":
                w_col1, w_col2, w_col3 = st.columns(3)
                base_p = float(res_data.get("selling_price", 100))
                base_c = float(res_data.get("unit_cost", 40))
                base_s = float(res_data.get("salvage_value", 10))
                base_mu = float(res_data.get("demand_mean", 500))
                base_sig = float(res_data.get("demand_std", 100))

                sim_p = w_col1.slider("Selling Price (P)", min_value=base_c + 1.0, max_value=base_p * 2.0, value=base_p, step=5.0)
                sim_c = w_col2.slider("Unit Cost (C)", min_value=base_s + 1.0, max_value=sim_p - 1.0, value=min(base_c, sim_p - 1.0), step=5.0)
                sim_s = w_col3.slider("Salvage Value (S_val)", min_value=0.0, max_value=sim_c, value=min(base_s, sim_c), step=2.0)

                sim_res = tools.calculate_newsvendor(selling_price=sim_p, unit_cost=sim_c, demand_mean=base_mu, demand_std=base_sig, salvage_value=sim_s)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Simulated Q*", f"{sim_res.optimal_stocking_quantity:,.0f} units", delta=f"{sim_res.optimal_stocking_quantity - res_data['optimal_stocking_quantity']:+,.0f}")
                m2.metric("Critical Ratio", f"{sim_res.critical_ratio * 100:.1f}%", delta=f"{(sim_res.critical_ratio - res_data['critical_ratio']) * 100:+.1f}%")
                m3.metric("Expected Profit", f"${sim_res.expected_profit:,.2f}", delta=f"${sim_res.expected_profit - res_data['expected_profit']:+,.2f}")
                m4.metric("EVPI", f"${sim_res.evpi:,.2f}", delta=f"${sim_res.evpi - res_data['evpi']:+,.2f}", delta_color="inverse")

            elif response.identified_model == "FIXED_PERIOD_REVIEW":
                w_col1, w_col2, w_col3 = st.columns(3)
                base_r = float(res_data.get("review_period", 7))
                base_l = float(res_data.get("lead_time", 3))
                base_csl = float(res_data.get("service_level", 0.95))
                base_d = float(res_data.get("demand_rate", 20))
                base_d_std = float(res_data.get("demand_std", 5))
                base_oh = float(res_data.get("inputs", {}).get("current_on_hand", 40))
                base_pl = float(res_data.get("inputs", {}).get("pipeline_orders", 20))
                base_bo = float(res_data.get("inputs", {}).get("backorders", 0))

                sim_r = w_col1.slider("Review Period (R days)", min_value=1.0, max_value=30.0, value=base_r, step=1.0)
                sim_l = w_col2.slider("Lead Time (L days)", min_value=0.0, max_value=20.0, value=base_l, step=1.0)
                sim_csl = w_col3.slider("Service Level (CSL)", min_value=0.80, max_value=0.999, value=min(0.999, max(0.80, base_csl)), step=0.005, format="%.3f")

                sim_res = tools.calculate_periodic_review(review_period=sim_r, lead_time=sim_l, demand_rate=base_d, demand_std=base_d_std, service_level=sim_csl, current_on_hand=base_oh, pipeline_orders=base_pl, backorders=base_bo)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Order Quantity (q)", f"{sim_res.recommended_order_quantity:,.1f} units", delta=f"{sim_res.recommended_order_quantity - res_data['recommended_order_quantity']:+,.1f}")
                m2.metric("Target Level (S*)", f"{sim_res.target_inventory_level:,.1f} units", delta=f"{sim_res.target_inventory_level - res_data['target_inventory_level']:+,.1f}")
                m3.metric("Safety Stock", f"{sim_res.safety_stock:,.1f} units", delta=f"{sim_res.safety_stock - res_data['safety_stock']:+,.1f}")
                m4.metric("Protection Period", f"{sim_res.protection_interval:.1f} days", delta=f"{sim_res.protection_interval - res_data['protection_interval']:+.1f} days")

    # 4. Detailed Supply Chain Briefing & Verification
    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.markdown("### 📋 Executive Supply Chain Briefing")
        briefing_text = response.detailed_explanation
        if "### Analyst 8-Step Verification Audit" in briefing_text:
            briefing_text = briefing_text.split("### Analyst 8-Step Verification Audit")[0].rstrip("-\n ")
        st.markdown(briefing_text)

        if response.strategic_insights:
            st.markdown("#### 🧭 Managerial & Strategic Trade-Offs")
            for insight in response.strategic_insights:
                st.markdown(f"- {insight}")

    with col_right:
        st.markdown("### 🔍 Mathematical Verification & Audit")

        with st.expander("📊 Analyst 8-Step Verification", expanded=False):
            if response.variable_definitions:
                st.markdown("**1. Inputs Extracted:**")
                st.json(response.inputs_extracted if response.inputs_extracted else response.extracted_inputs)
                st.markdown("**2. Variable Definitions:**")
                for sym, defn in response.variable_definitions.items():
                    st.markdown(f"- **{sym}:** {defn}")
                st.markdown("**3. Formula:**")
                for fname, fval in response.formula.items():
                    st.code(fval, language="text")
                st.markdown("**4. Formula with Substituted Values:**")
                for sname, sval in response.formula_with_substituted_values.items():
                    st.code(sval, language="text")
                st.markdown("**5. Intermediate Calculations:**")
                st.json(response.intermediate_calculations)
                st.markdown("**6. Final Answer:**")
                st.json(response.final_answer)
                st.markdown("**7. Verification Checks:**")
                for chk in response.verification_checks:
                    st.markdown(f"- ✅ {chk}")
                st.markdown("**8. Assumptions:**")
                for asm in response.assumptions:
                    st.markdown(f"- 📌 {asm}")

        with st.expander("Operational Verification Checklist", expanded=False):
            for note in response.verification_notes:
                st.markdown(f"- {note}")

        if response.mathematical_trace:
            with st.expander("Step-by-Step Operations Research Trace", expanded=False):
                for idx, step in enumerate(response.mathematical_trace, 1):
                    st.markdown(f"**Step {idx}:** {step}")

        with st.expander("Extracted Slot Parameters", expanded=False):
            st.json(response.extracted_inputs)

    # 5. Export Decision Report & Data
    st.markdown("---")
    st.markdown("### 📥 Export Executive Decision Report")
    report_md = (
        f"# Supply Chain Decision Report: {response.identified_model}\n\n"
        f"**Question:** {response.question}\n\n"
        f"**Recommendation:** {response.executive_summary}\n\n"
        f"{response.detailed_explanation}\n\n"
        f"## Verification Notes\n" + "\n".join([f"- {n}" for n in response.verification_notes]) + "\n\n"
        f"## Mathematical Trace\n" + "\n".join([f"{idx}. {t}" for idx, t in enumerate(response.mathematical_trace, 1)]) + "\n\n"
        f"## Strategic Insights\n" + "\n".join([f"- {i}" for i in response.strategic_insights]) + "\n"
    )

    down_c1, down_c2 = st.columns([1, 4])
    down_c1.download_button(
        label="📄 Download Markdown Report",
        data=report_md,
        file_name=f"inventory_decision_{response.identified_model.lower()}.md",
        mime="text/markdown",
    )
    import json
    down_c2.download_button(
        label="📊 Download JSON Audit Data",
        data=json.dumps(response.model_dump(), indent=2),
        file_name=f"inventory_audit_{response.identified_model.lower()}.json",
        mime="application/json",
    )

elif st.session_state.get("last_response") is not None and not show_question_analysis:
    st.markdown("---")
    st.caption("💡 Last supply chain question analysis is available.")
    if st.button("📋 Show Last Question Analysis", key="btn_show_last_q_mode"):
        st.session_state["active_analysis"] = "QUESTION"
        st.rerun()

if submit_pressed and not user_query.strip():
    st.error("Please enter a question or select a benchmark case above.")

