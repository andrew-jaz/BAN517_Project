# Supply Chain Analytics Inventory Decision Agent

An autonomous Operations Research and Inventory Decision Agent that accepts natural-language supply chain questions, deterministically calculates optimal inventory policies using verified mathematical models, conducts boundary and operational sanity checks, and delivers executive-ready managerial briefings.

---

## 🏛️ Architecture Pipeline

The decision engine strictly implements the 7-stage architectural pipeline:

```
Customer Question
       │
       ▼
Identify Model (EOQ, Quantity Discount, Safety Stock/ROP, Fixed-Period, Newsvendor)
       │
       ▼
Extract Inputs (Normalized units, demand rates, cost parameters, service levels)
       │
       ▼
Select Python Tool (Dispatched to deterministic calculation function)
       │
       ▼
Calculate (Pure Python numerical execution via scipy & numpy)
       │
       ▼
Verify (Operational bounds, cost equilibrium, inventory feasibility)
       │
       ▼
Explain (Executive recommendation, formula traces, and managerial trade-offs)
```

---

## 📦 Supported Inventory Models & Operations Research Formulations

### 1. Economic Order Quantity (EOQ)
* **Objective:** Balance annual order setup costs against carrying/holding costs under constant demand.
* **Key Formulations:**
  $$Q^* = \sqrt{\frac{2 \cdot D \cdot S}{H}}$$
  $$\text{Orders per Year: } N = \frac{D}{Q^*}, \quad \text{Cycle Interval: } T = \frac{Q^*}{D} \times \text{working\_days}$$
  $$\text{Total Annual Cost: } TAC = \frac{D}{Q}S + \frac{Q}{2}H$$
* **Verification Checks:** Confirms Wilson cost balance ($\text{Annual Ordering Cost} \approx \text{Annual Holding Cost}$) and sensitivity around optimal lot size ($\pm 10\%, \pm 20\%$).

### 2. Quantity Discount Model (All-Units & Tiered Breaks)
* **Objective:** Determine if purchase discounts offset higher inventory carrying costs across multiple vendor price tiers.
* **Key Formulations:**
  For each price break tier $i$ with price $C_i$ and holding cost $H_i = h \cdot C_i$:
  1. Compute unconstrained EOQ: $Q_i = \sqrt{\frac{2DS}{H_i}}$
  2. Adjust to feasible boundary: if $Q_i < \min_i$, evaluate at $Q_{i,\text{eval}} = \min_i$
  3. Total Landed Cost: $TC_i = D \cdot C_i + \frac{D}{Q_{i,\text{eval}}}S + \frac{Q_{i,\text{eval}}}{2}H_i$
* **Verification Checks:** Identifies the global minimum landed cost and reports exact annual net dollar and percentage savings versus baseline EOQ.

### 3. Safety Stock & Reorder Point (Continuous Review $(s, Q)$)
* **Objective:** Maintain target Cycle Service Level ($\text{CSL}$) under demand and lead-time uncertainty.
* **Formulations under Joint Uncertainty:**
  $$\sigma_{DL} = \sqrt{L \cdot \sigma_d^2 + d^2 \cdot \sigma_L^2}$$
  $$SS = z \cdot \sigma_{DL}, \quad \text{where } z = \Phi^{-1}(\text{CSL})$$
  $$ROP = d \cdot L + SS$$
  $$\text{Expected Shortage per Cycle (ESC): } ESC = \sigma_{DL} \cdot \left[\phi(z) - z(1 - \Phi(z))\right]$$
  $$\text{Fill Rate (Type II Service Level): } \beta = 1 - \frac{ESC}{Q}$$
* **Verification Checks:** Confirms $ROP \ge SS$, verifies $0 < \text{CSL} < 1$, and alerts on high-risk lead time variances.

### 4. Fixed-Period Review Model (Periodic Review $(R, S)$ / Order-Up-To)
* **Objective:** Establish target inventory levels ($S^*$) and replenishment quantities for periodic review intervals ($R$) with supplier lead time ($L$).
* **Key Formulations:**
  $$\text{Protection Interval: } P = R + L$$
  $$\mu_P = d \cdot (R + L), \quad \sigma_P = \sigma_d \sqrt{R + L}$$
  $$\text{Target Order-Up-To Level: } S^* = \mu_P + z \cdot \sigma_P$$
  $$\text{Inventory Position (IP): } IP = I_{\text{on\_hand}} + I_{\text{pipeline}} - \text{Backorders}$$
  $$\text{Replenishment Order: } Q = \max(0, S^* - IP)$$
* **Verification Checks:** Quantifies the safety stock premium required to absorb demand during the review lag compared to continuous review.

### 5. Newsvendor Single-Period Optimization (Perishable & Seasonal Goods)
* **Objective:** Maximize expected profit for products with short lifecycles, seasonal demand, or perishability.
* **Key Formulations:**
  $$\text{Cost of Underage: } C_u = P - C + \text{Penalty}$$
  $$\text{Cost of Overage: } C_o = C - S_{\text{val}}$$
  $$\text{Critical Fractile: } CR = \frac{C_u}{C_u + C_o}$$
  $$Q^* = \mu + \Phi^{-1}(CR) \cdot \sigma$$
  $$\text{Expected Lost Sales: } ELS = \sigma \cdot L(z), \quad \text{Expected Leftover: } ELU = Q^* - \mu + ELS$$
  $$\text{Expected Value of Perfect Information: } EVPI = (P - C)\mu - \text{Expected Profit}(Q^*)$$
* **Verification Checks:** Asserts $0 < CR < 1$, verifies that high-margin goods stock above mean demand ($CR > 0.5 \implies Q^* > \mu$), and ensures $EVPI \ge 0$.

---

## 📊 Analyst 8-Step Verification Audit Architecture

Every calculation tool deterministically computes and returns an Analyst 8-Step Verification Audit structure so analysts, operations managers, and auditors can trace and verify the numerical answers:

1. **Inputs Extracted:** Exact parameter mapping without numerical invention or hallucination.
2. **Variable Definitions:** Mathematical notations, standard units, and business definitions.
3. **Formula:** Canonical Operations Research equations in symbolic notation.
4. **Formula with Substituted Values:** Formulas with exact extracted numerical parameters plugged in.
5. **Intermediate Calculations:** All intermediate numerators, denominators, z-scores, loss function values, and variance components.
6. **Final Answer:** Core decision metrics (optimal lot size, target inventory, reorder point, expected profit).
7. **Verification Checks:** Mathematical sanity checks (first-order derivative optimality, Wilson cost equilibrium, convexity, boundary conditions, inventory balance identities).
8. **Assumptions:** Explicit statistical and operational assumptions governing the validity of the model.

---

## 📂 File Structure

```
Inventory_Agent_V1/
├── app.py                     # Rich CLI terminal app & Streamlit launch bridge
├── web_app.py                 # Interactive Streamlit Web Dashboard with Analyst 8-Step Verification Audit & live charts
├── inventory_agent.py         # 7-stage Decision Agent orchestrator & NLP synonym parser
├── inventory_tools.py         # Deterministic numerical models with Analyst 8-Step Verification Audit schemas
├── requirements.txt           # Project dependencies
├── tests/
│   ├── test_inventory_tools.py # 17 Unit tests for mathematical accuracy, validation & Analyst 8-step audit
│   └── test_inventory_agent.py # 13 End-to-end NLP agent pipeline & synonym extraction tests
└── README.md                  # Project documentation
```

---

## 🚀 Getting Started

### Installation
```powershell
pip install -r requirements.txt
```

### 1. Launch Interactive Web Dashboard
Run the web application with live interactive charts, KPI cards, and export features:
```powershell
python app.py --web
# or
streamlit run web_app.py
```
Open **`http://localhost:8501`** in your browser.

### 2. Launch Terminal CLI Interface
Run the interactive terminal interface with Rich styling:
```powershell
python app.py
```

### 3. Run Pre-loaded Industry Scenarios (Batch Demo)
```powershell
python app.py --demo
```

### 4. Single-Query Execution via Command Line
```powershell
python app.py --query 'We sell 12,000 units annually. Order cost is $25, and carrying cost is $0.50 per unit per year. What is our optimal batch size?'
```

### 5. Run Test Suite
Run the full test suite (30 unit and pipeline tests):
```powershell
python -m pytest tests/ -v
```

---

## 💻 Python API Usage

You can also embed the decision agent directly into Python applications or ERP pipelines:

```python
from inventory_agent import InventoryDecisionAgent

agent = InventoryDecisionAgent()

# Submit any natural language inventory inquiry
query = (
    "Our daily demand is 50 units with standard deviation of 10. "
    "Lead time is 4 days with std dev 1.5 days. We target a 98% service level. "
    "What safety stock and ROP should we set?"
)

response = agent.process_question(query)

print("Identified Model:", response.identified_model)
print("Recommendation:", response.executive_summary)
print("Reorder Point:", response.calculation_result["reorder_point"])
print("Safety Stock:", response.calculation_result["safety_stock"])
```

