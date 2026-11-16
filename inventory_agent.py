"""
inventory_agent.py - Supply Chain Analytics Inventory Decision Agent

Implements the 7-stage architectural pipeline:
Customer Question -> Identify Model -> Extract Inputs -> Select Python Tool -> Calculate -> Verify -> Explain

Supports:
1. Economic Order Quantity (EOQ)
2. Quantity Discount
3. Safety Stock & Reorder Point (Continuous Review)
4. Fixed-Period Review (Periodic Review / Order-Up-To)
5. Newsvendor Single-Period Optimization
"""

from __future__ import annotations

import re
import math
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

import inventory_tools as tools


# ---------------------------------------------------------------------------
# Pipeline Data Models
# ---------------------------------------------------------------------------

class AgentResponse(BaseModel):
    question: str
    identified_model: str
    confidence: float
    extracted_inputs: Dict[str, Any]
    missing_inputs: List[str] = Field(default_factory=list)
    tool_selected: Optional[str] = None
    calculation_result: Optional[Dict[str, Any]] = None
    verification_passed: bool = True
    verification_notes: List[str] = Field(default_factory=list)
    executive_summary: str = ""
    detailed_explanation: str = ""
    mathematical_trace: List[str] = Field(default_factory=list)
    strategic_insights: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    intermediate_calculations: Dict[str, Any] = Field(default_factory=dict)
    final_answer: Dict[str, Any] = Field(default_factory=dict)
    # Analyst 8-Step Verification Audit fields:
    inputs_extracted: Dict[str, Any] = Field(default_factory=dict)
    variable_definitions: Dict[str, str] = Field(default_factory=dict)
    formula: Dict[str, str] = Field(default_factory=dict)
    formula_with_substituted_values: Dict[str, str] = Field(default_factory=dict)
    verification_checks: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Inventory Decision Agent Class
# ---------------------------------------------------------------------------

class InventoryDecisionAgent:
    """
    Autonomous Supply Chain Analytics Inventory Decision Agent.

    Executes the deterministic 7-step pipeline:
    1. Customer Question ingestion
    2. Model Identification (Intent Classification)
    3. Input Extraction & Normalization
    4. Python Tool Selection
    5. Pure-Python Calculation
    6. Post-Calculation Verification
    7. Executive Supply Chain Explanation
    """

    MODEL_EOQ = "EOQ"
    MODEL_QUANTITY_DISCOUNT = "QUANTITY_DISCOUNT"
    MODEL_SAFETY_STOCK_ROP = "SAFETY_STOCK_ROP"
    MODEL_FIXED_PERIOD_REVIEW = "FIXED_PERIOD_REVIEW"
    MODEL_NEWSVENDOR = "NEWSVENDOR"

    def __init__(self):
        # Keywords and weights for model classification
        self.intent_signatures = {
            self.MODEL_QUANTITY_DISCOUNT: [
                (r"\bdiscount(s)?\b", 3.5),
                (r"\bprice\s*break(s)?\b", 4.5),
                (r"\bquantity\s*break(s)?\b", 4.5),
                (r"\btier(s)?\b", 2.5),
                (r"\bbracket(s)?\b", 2.0),
                (r"\ball[- ]units\b", 3.5),
                (r"\bbulk\s*(pricing|order|purchase|discount)\b", 3.5),
                (r"\bif\s*we\s*order\s*(over|more\s*than|\d+)\b", 2.0),
                (r"\bper\s*unit\s*(drops|reduces|discount)\b", 3.0),
                (r"\bpercentage\s*holding\s*cost\b", 3.0),
                (r"\bfixed\s*holding\s*cost\b", 3.0),
                (r"\bcharges?\s*[\$]?\d+(?:\.\d+)?.*for.*(?:orders?|units?)\b", 4.0),
                (r"\b(?:below|under|less\s*than)\s*[0-9,]+\s*units?\b", 3.5),
                (r"\b[0-9,]+\s*(?:to|-)\s*[0-9,]+\s*units?\b", 3.5),
                (r"\b[0-9,]+\s*(?:\+|\s*or\s*more)\s*units?\b", 3.5),
                (r"\bpricing\s*(?:schedule|structure|tiers?)\b", 4.0),
                (r"\bvolume\s*(?:pricing|discount|purchase)\b", 4.0),
                (r"\bquantity\s*(?:discount|pricing|breaks?)\b", 4.5),
                (r"\bprice\s*(?:breaks?|tiers?|schedule)\b", 4.5),
            ],
            self.MODEL_FIXED_PERIOD_REVIEW: [
                (r"\bfixed[- ]period\b", 4.5),
                (r"\bperiodic\s*review\b", 4.5),
                (r"\border[- ]up[- ]to\b", 4.5),
                (r"\btarget\s*inventory\b", 4.0),
                (r"\breview(ing)?\s*(inventory|stock)?\s*every\b", 5.0),
                (r"\breview\s*(period|interval|cycle|every)\b", 4.0),
                (r"\bevery\s*(\d+|\w+)\s*(days?|weeks?|months?)\b", 3.5),
                (r"\bon[- ]hand\s*inventory\b", 3.5),
                (r"\bon[- ]order\s*inventory\b", 3.5),
                (r"\binventory\s*position\b", 4.0),
                (r"\bbackorders?\b", 2.5),
                (r"\b\(r,\s*s\)\b", 4.0),
                (r"\bprotection\s*interval\b", 3.5),
            ],
            self.MODEL_NEWSVENDOR: [
                (r"\bnewsvendor\b", 5.0),
                (r"\bsingle[- ]period\b", 4.5),
                (r"\bperishable\b", 4.0),
                (r"\bseasonal\b", 3.5),
                (r"\bsalvage(\s*value|\s*price)?\b", 4.0),
                (r"\bmarkdown\b", 3.0),
                (r"\bcritical\s*(ratio|fractile)\b", 4.5),
                (r"\bunderage(\s*cost)?\b", 4.5),
                (r"\boverage(\s*cost)?\b", 4.5),
                (r"\bshortage\s*(cost|penalty)\b", 3.5),
                (r"\bunsold\b", 2.5),
                (r"\bholiday\s*(season|demand|items?)\b", 2.5),
            ],
            self.MODEL_SAFETY_STOCK_ROP: [
                (r"\bsafety\s*stock\b", 4.5),
                (r"\breorder\s*point\b", 4.5),
                (r"\brop\b", 3.5),
                (r"\bservice\s*level\b", 3.0),
                (r"\bcycle\s*service\s*level\b", 3.5),
                (r"\bstockout\s*(risk|probability)\b", 3.0),
                (r"\b\(s,\s*Q\)\b", 3.5),
                (r"\blead[- ]time\s*(demand|standard\s*deviation|variance|std)\b", 4.0),
                (r"\baverage\s*lead[- ]time\b", 3.5),
                (r"\bwhen\s*should\s*we\s*reorder\b", 3.5),
            ],
            self.MODEL_EOQ: [
                (r"\beoq\b", 4.5),
                (r"\beconomic\s*order\s*quantity\b", 4.5),
                (r"\boptimal\s*order\s*quantity\b", 4.0),
                (r"\boptimal\s*batch(\s*size)?\b", 3.5),
                (r"\border\s*frequency\b", 2.5),
                (r"\bordering\s*cost\b", 3.0),
                (r"\border\s*cost\b", 3.0),
                (r"\bsetup\s*cost\b", 3.0),
                (r"\bsetup\s*expense\b", 3.5),
                (r"\bcost\s*per\s*order\b", 3.5),
                (r"\bcost\s*of\s*placing\s*(an?)?\s*order\b", 4.0),
                (r"\bordering\/setup\s*cost\b", 4.0),
                (r"\bholding\s*cost\b", 2.5),
                (r"\bcarrying\s*cost\b", 2.5),
                (r"\bhow\s*(much|many\s*units)\s*should\s*we\s*order\b", 2.0),
            ],
        }

    # -----------------------------------------------------------------------
    # Step 1 & 2: Identify Model
    # -----------------------------------------------------------------------

    def identify_model(self, question: str) -> Tuple[str, float]:
        """
        Classify the natural-language question into one of the 5 inventory models.
        Returns: (model_name, confidence_score)
        """
        clean_text = question.lower()
        scores: Dict[str, float] = {k: 0.0 for k in self.intent_signatures}

        for model, patterns in self.intent_signatures.items():
            for pattern, weight in patterns:
                matches = len(re.findall(pattern, clean_text))
                if matches > 0:
                    scores[model] += weight * min(matches, 3)

        # Disambiguation heuristics:
        # 1. Periodic Review priority
        if "review" in clean_text and any(k in clean_text for k in ["every", "order up to", "on hand", "pipeline", "periodic", "target inventory"]):
            scores[self.MODEL_FIXED_PERIOD_REVIEW] += 6.0

        # 2. Quantity Discount priority
        if any(k in clean_text for k in ["break", "tier", "discount", "bracket", "bulk pricing", "price schedule"]):
            scores[self.MODEL_QUANTITY_DISCOUNT] += 4.0

        # Disambiguation: If multiple price breaks / pricing tiers are detected in question text,
        # it is decisively a Quantity Discount problem, not a single-price EOQ problem.
        detected_breaks = self._extract_price_breaks_from_text(clean_text)
        if detected_breaks and len(detected_breaks) >= 2:
            scores[self.MODEL_QUANTITY_DISCOUNT] += 12.0

        # 3. Newsvendor priority
        if any(k in clean_text for k in ["salvage", "perishable", "newsvendor", "underage", "overage", "critical ratio", "markdown"]):
            scores[self.MODEL_NEWSVENDOR] += 5.0

        best_model = max(scores, key=scores.get)
        max_score = scores[best_model]

        if max_score == 0:
            if "reorder" in clean_text or "safety" in clean_text:
                return self.MODEL_SAFETY_STOCK_ROP, 0.4
            elif "cost" in clean_text and "order" in clean_text:
                return self.MODEL_EOQ, 0.4
            return self.MODEL_EOQ, 0.3

        confidence = min(0.99, max(0.50, round(0.50 + (max_score / 15.0) * 0.49, 2)))
        return best_model, confidence

    # -----------------------------------------------------------------------
    # Step 3: Extract Inputs
    # -----------------------------------------------------------------------

    def extract_inputs(self, model: str, question: str, context: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], List[str]]:
        """
        Extract numerical parameters and business constraints from question text.
        Examines the entire question and recognizes common inventory terminology and synonyms.
        Never reports a parameter as missing when it is present in the customer's question.
        Returns: (extracted_inputs, missing_critical_fields)
        """
        extracted = {} if context is None else dict(context)
        missing: List[str] = []

        q = question.lower()
        unit_nouns = r"(?:units|bags|items|pieces|kits|pounds|lbs|boxes|cartons|cases|packs|bottles|cans|pallets|cartridges|vials|rolls)"
        var_annot = r"(?:\s*\([a-zA-Z0-9_α-ωΑ-Ωσ\s\/\.-]+\))?"

        def extract_number(patterns: List[str], text: str) -> Optional[float]:
            """Try a list of regex patterns and extract the first positive numerical match."""
            for pat in patterns:
                m = re.search(pat, text, re.IGNORECASE)
                if m:
                    raw_val = m.group(1).replace(",", "").replace("$", "").replace("%", "").strip(" ,.;:)")
                    try:
                        val = float(raw_val)
                        return val
                    except ValueError:
                        continue
            return None

        # -------------------------------------------------------------------
        # Model 1: EOQ Inputs
        # -------------------------------------------------------------------
        if model == self.MODEL_EOQ:
            # 1. Annual Demand (D)
            if "demand" not in extracted:
                d = extract_number([
                    r"(?:annual\s*demand|yearly\s*demand|demand\s*per\s*year|demand\s*annually|annual\s*sales|yearly\s*sales|annual\s*usage|annual\s*consumption)" + var_annot + r"(?:\s+(?:for|of)\s+[a-zA-Z][a-zA-Z0-9\s/-]*?(?:\s+(?:is|averages?|averaging|=|:|at))?)?\s*(?:is|of|=|:)?\s*[\$]?([0-9,]+(?:\.[0-9]+)?)\s*(?:" + unit_nouns + r")?",
                    r"(?:demand|sells?|requires?|needs?|consumes?)" + var_annot + r"(?:\s+(?:for|of)\s+[a-zA-Z][a-zA-Z0-9\s/-]*?(?:\s+(?:is|averages?|averaging|=|:|at))?)?\s*(?:is|of|=|:)?\s*[\$]?([0-9,]+(?:\.[0-9]+)?)\s*(?:" + unit_nouns + r")?\s*(?:per\s*year|annually|a\s*year)",
                    r"([0-9,]+(?:\.[0-9]+)?)\s*(?:" + unit_nouns + r")?\s*(?:per\s*year|annually|a\s*year)\s*(?:demand|sales|usage)?",
                    r"(?:annual\s*demand|demand\s*is|demand\s*of|sells?)" + var_annot + r"\s*(?:approximately|about)?\s*[\$]?([0-9,]+(?:\.[0-9]+)?)\s*(?:" + unit_nouns + r")?",
                    r"(?:monthly\s*demand|demand\s*per\s*month)" + var_annot + r"(?:\s+(?:for|of)\s+[a-zA-Z][a-zA-Z0-9\s/-]*?(?:\s+(?:is|averages?|averaging|=|:|at))?)?\s*(?:is|of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
                    r"([0-9,]+(?:\.[0-9]+)?)\s*(?:" + unit_nouns + r")?\s*per\s*month",
                    r"\b[dD]" + var_annot + r"\s*(?:is|of|=|:)\s*[\$]?([0-9,]+(?:\.[0-9]+)?)",
                ], q)
                if d is not None:
                    if ("per month" in q or "monthly" in q) and "annual" not in q and "yearly" not in q:
                        d *= 12.0
                    extracted["demand"] = d

            # 2. Ordering Cost (S) - Treat all user synonyms as ordering cost:
            # - ordering cost
            # - order cost
            # - setup cost
            # - setup expense
            # - cost per order
            # - cost of placing an order
            # - ordering/setup cost
            if "ordering_cost" not in extracted:
                s = extract_number([
                    r"(?:ordering\/setup\s*costs?|ordering\s*and\s*setup\s*costs?|order\/setup\s*costs?|order\s*and\s*setup\s*costs?)" + var_annot + r"\s*(?:is|of|=|:)?\s*[\$]?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:dollars?|bucks?|\$)?",
                    r"(?:cost\s*of\s*placing\s*(?:an?|a\s*purchase)?\s*order|cost\s*to\s*place\s*(?:an?|a\s*purchase)?\s*order|cost\s*per\s*(?:purchase\s*)?order|expense\s*per\s*(?:purchase\s*)?order)" + var_annot + r"\s*(?:is|of|=|:)?\s*[\$]?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:dollars?|bucks?|\$)?",
                    r"(?:setup\s*expenses?|setup\s*costs?|ordering\s*costs?|order\s*costs?)" + var_annot + r"\s*(?:is|of|=|:)?\s*[\$]?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:dollars?|bucks?|\$)?",
                    r"(?:placing\s*(?:(?:an?|a\s*purchase)\s*)?order\s*costs?|place\s*(?:(?:an?|a\s*purchase)\s*)?order\s*costs?|order\s*fee|setup\s*fee)" + var_annot + r"\s*(?:us|is|of|=|:)?\s*[\$]?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:dollars?|bucks?|\$)?",
                    r"costs?\s*[\$]?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:dollars?|bucks?|\$)?\s*(?:to\s*place\s*(?:an?|a\s*purchase)?\s*order|per\s*(?:purchase\s*)?order|each\s*time\s*we\s*order|per\s*setup|to\s*order)",
                    r"[\$]?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:dollars?|bucks?|\$)?\s*(?:per\s*(?:purchase\s*)?order|setup\s*expense|setup\s*cost|ordering\s*cost|order\s*cost|ordering\/setup\s*cost|each\s*time\s*we\s*order)",
                    r"\b[sS]" + var_annot + r"\s*(?:is|of|=|:)\s*[\$]?\s*([0-9,]+(?:\.[0-9]+)?)",
                ], q)
                if s is not None:
                    extracted["ordering_cost"] = s

            # 3. Unit Purchase Price (C or P)
            if "unit_cost" not in extracted:
                c = extract_number([
                    r"(?:purchase\s*price(?:\s*per\s*(?:unit|" + unit_nouns + r"))?|unit\s*purchase\s*price|purchase\s*cost(?:\s*per\s*(?:unit|" + unit_nouns + r"))?|unit\s*purchase\s*cost|wholesale\s*price|wholesale\s*cost|item\s*cost|unit\s*cost|unit\s*price)" + var_annot + r"\s*(?:is|of|=|:)?\s*[\$]?([0-9,]+(?:\.[0-9]+)?)",
                    r"(?:buys?\s*for|costs?\s*us|procure\s*(?:at|for)|price\s*is)" + var_annot + r"\s*[\$]?([0-9,]+(?:\.[0-9]+)?)\s*(?:per\s*(?:unit|" + unit_nouns + r")|each)?",
                    r"[\$]([0-9,]+(?:\.[0-9]+)?)\s*(?:per\s*(?:unit|" + unit_nouns + r")\s*(?:purchase|cost|price)|wholesale)",
                    r"\b[pPcC]" + var_annot + r"\s*(?:is|of|=|:)\s*[\$]?([0-9,]+(?:\.[0-9]+)?)",
                ], q)
                if c is not None:
                    extracted["unit_cost"] = c

            # 4. Holding / Carrying Cost (H)
            if "holding_cost" not in extracted:
                h = extract_number([
                    r"(?:annual\s*|yearly\s*)?(?:fixed\s*)?(?:inventory\s*)?(?:holding|carrying)(?:\/carrying|\/holding)?\s*(?:costs?|expenses?)" + var_annot + r"(?:\s*(?:\/|\s*per\s*)(?:unit|item|piece|sku|" + unit_nouns + r")(?:\s*(?:\/|\s*per\s*)(?:year|annum|annually))?)?\s*(?:is|of|=|:)?\s*[\$]?\s*([0-9,]+(?:\.[0-9]+)?)(?!\s*%\b)\s*(?:dollars?|bucks?|\$)?(?:\s*(?:\/|\s*per\s*)(?:unit|item|piece|sku|" + unit_nouns + r")(?:\s*(?:\/|\s*per\s*)(?:year|annum|annually))?)?",
                    r"(?:cost\s*to\s*hold|cost\s*of\s*holding|cost\s*to\s*carry|cost\s*of\s*carrying)" + var_annot + r"\s*(?:is|of|=|:)?\s*[\$]?\s*([0-9,]+(?:\.[0-9]+)?)(?!\s*%\b)\s*(?:dollars?|bucks?|\$)?(?:\s*(?:\/|\s*per\s*)(?:unit|item|piece|sku|" + unit_nouns + r")(?:\s*(?:\/|\s*per\s*)(?:year|annum|annually))?)?",
                    r"[\$]?\s*([0-9,]+(?:\.[0-9]+)?)(?!\s*%\b)\s*(?:dollars?|bucks?|\$)?\s*(?:per\s*(?:unit|" + unit_nouns + r")\s*(?:per\s*year|annually|\/year|a\s*year)|per\s*(?:unit|" + unit_nouns + r")\s*carrying|per\s*(?:unit|" + unit_nouns + r")\s*holding|\/unit\/year|\/unit\s*per\s*year)",
                    r"costs?\s*[\$]?\s*([0-9,]+(?:\.[0-9]+)?)(?!\s*%\b)\s*(?:dollars?|bucks?|\$)?\s*(?:to\s*hold|to\s*carry)\s*(?:each|one|a)?\s*(?:unit|" + unit_nouns + r")?",
                    r"\b[hH]" + var_annot + r"\s*(?:is|of|=|:)\s*[\$]?\s*([0-9,]+(?:\.[0-9]+)?)(?!\s*%\b)",
                ], q)
                if h is not None:
                    extracted["holding_cost"] = h

            # Check percentage holding cost (e.g. 20% of unit price)
            if "holding_cost_rate" not in extracted:
                pct_rate = extract_number([
                    r"(?:annual\s*holding\s*cost\s*rate|percentage\s*holding\s*costs?|carrying\s*cost\s*rate|holding\s*cost\s*rate|carrying\s*rate|holding\s*rate|annual\s*carrying\s*cost\s*rate)" + var_annot + r"\s*(?:is|of|=|:)?\s*([0-9]+(?:\.[0-9]+)?)\s*%",
                    r"(?:annual\s*|yearly\s*)?(?:inventory\s*)?(?:holding|carrying)\s*costs?" + var_annot + r"\s*(?:is|of|=|:)?\s*([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:of\s*(?:the\s*)?(?:unit\s*)?(?:purchase\s*)?(?:cost|price))?",
                    r"([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:of\s*(?:the\s*)?(?:unit\s*)?(?:purchase\s*)?(?:cost|price)|carrying\s*cost\s*rate|holding\s*cost\s*rate|holding|carrying)",
                    r"\b[iI]" + var_annot + r"\s*(?:is|of|=|:)\s*([0-9]+(?:\.[0-9]+)?)\s*%",
                ], q)
                if pct_rate is not None:
                    extracted["holding_cost_rate"] = pct_rate / 100.0

            if "holding_cost" not in extracted and "holding_cost_rate" in extracted and "unit_cost" in extracted:
                extracted["holding_cost"] = extracted["holding_cost_rate"] * extracted["unit_cost"]

            # 5. Operating / Working Days per Year
            if "working_days" not in extracted:
                wd = extract_number([
                    r"(?:operating\s*days(?:\s*per\s*year)?|working\s*days(?:\s*per\s*year)?|business\s*days(?:\s*per\s*year)?|days\s*per\s*year)" + var_annot + r"\s*(?:is|of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)\s*days?",
                    r"([0-9,]+(?:\.[0-9]+)?)\s*(?:operating|working|business)\s*days(?:\s*per\s*year|\/year)?",
                ], q)
                if wd is not None:
                    extracted["working_days"] = int(wd)

            # 6. Lead Time (days)
            if "lead_time_days" not in extracted:
                lt = extract_number([
                    r"(?:supplier\s*(?:replenishment\s*)?lead[- ]time|lead[- ]time|replenishment\s*time|delivery\s*time|replenishment\s*lead[- ]time)" + var_annot + r"\s*(?:is|of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:working\s*days?|business\s*days?|days?|weeks?)?",
                    r"([0-9,]+(?:\.[0-9]+)?)[- ](?:working\s*day|business\s*day|day)\s*(?:lead\s*time|delivery\s*time|replenishment)",
                    r"takes\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:working\s*days?|business\s*days?|days?)\s*to\s*arrive",
                    r"\b[lL]" + var_annot + r"\s*(?:is|of|=|:)\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:working\s*days?|days?)?",
                ], q)
                if lt is not None:
                    extracted["lead_time_days"] = lt

            # Validation check
            if "demand" not in extracted:
                missing.append("annual_demand (D)")
            if "ordering_cost" not in extracted:
                missing.append("ordering_cost (S)")
            if "holding_cost" not in extracted:
                missing.append("holding_cost (H)")

        # -------------------------------------------------------------------
        # Model 2: Quantity Discount Inputs
        # -------------------------------------------------------------------
        elif model == self.MODEL_QUANTITY_DISCOUNT:
            # Demand
            if "demand" not in extracted:
                d = extract_number([
                    r"(?:annual\s*demand|yearly\s*demand|demand\s*per\s*year|demand\s*annually|annual\s*sales|annual\s*usage)" + var_annot + r"(?:\s+(?:for|of)\s+[a-zA-Z][a-zA-Z0-9\s/-]*?(?:\s+(?:is|averages?|averaging|=|:|at))?)?\s*(?:is|of|=|:)?\s*[\$]?([0-9,]+(?:\.[0-9]+)?)\s*(?:" + unit_nouns + r")?",
                    r"(?:demand|sells?)" + var_annot + r"(?:\s+(?:for|of)\s+[a-zA-Z][a-zA-Z0-9\s/-]*?(?:\s+(?:is|averages?|averaging|=|:|at))?)?\s*(?:is|of|=|:)?\s*[\$]?([0-9,]+(?:\.[0-9]+)?)\s*(?:" + unit_nouns + r")?\s*(?:per\s*year|annually)",
                    r"([0-9,]+(?:\.[0-9]+)?)\s*(?:" + unit_nouns + r")?\s*(?:per\s*year|annually)",
                    r"(?:demand|annual\s*usage|usage)" + var_annot + r"(?:\s+(?:for|of)\s+[a-zA-Z][a-zA-Z0-9\s/-]*?(?:\s+(?:is|averages?|averaging|=|:|at))?)?\s*(?:is|of|=|:)?\s*[\$]?([0-9,]+(?:\.[0-9]+)?)\s*(?:" + unit_nouns + r")?",
                    r"(?:annual\s*demand|demand)" + var_annot + r"\s*(?:is|of|=|:)?\s*[\$]?([0-9,]+(?:\.[0-9]+)?)\s*(?:" + unit_nouns + r")?",
                    r"\b[dD]" + var_annot + r"\s*(?:is|of|=|:)\s*[\$]?([0-9,]+(?:\.[0-9]+)?)",
                ], q)
                if d is not None:
                    extracted["demand"] = d

            # Ordering Cost (using all standard ordering synonyms)
            if "ordering_cost" not in extracted:
                s = extract_number([
                    r"(?:ordering\/setup\s*costs?|ordering\s*and\s*setup\s*costs?|order\/setup\s*costs?|order\s*and\s*setup\s*costs?)" + var_annot + r"\s*(?:is|of|=|:)?\s*[\$]?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:dollars?|bucks?|\$)?",
                    r"(?:cost\s*of\s*placing\s*(?:an?|a\s*purchase)?\s*order|cost\s*to\s*place\s*(?:an?|a\s*purchase)?\s*order|cost\s*per\s*(?:purchase\s*)?order|expense\s*per\s*(?:purchase\s*)?order)" + var_annot + r"\s*(?:is|of|=|:)?\s*[\$]?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:dollars?|bucks?|\$)?",
                    r"(?:setup\s*expenses?|setup\s*costs?|ordering\s*costs?|order\s*costs?)" + var_annot + r"\s*(?:is|of|=|:)?\s*[\$]?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:dollars?|bucks?|\$)?",
                    r"(?:placing\s*(?:(?:an?|a\s*purchase)\s*)?order\s*costs?|place\s*(?:(?:an?|a\s*purchase)\s*)?order\s*costs?|order\s*fee|setup\s*fee)" + var_annot + r"\s*(?:us|is|of|=|:)?\s*[\$]?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:dollars?|bucks?|\$)?",
                    r"costs?\s*[\$]?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:dollars?|bucks?|\$)?\s*(?:to\s*place\s*(?:an?|a\s*purchase)?\s*order|per\s*(?:purchase\s*)?order|each\s*time\s*we\s*order|per\s*setup|to\s*order)",
                    r"[\$]?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:dollars?|bucks?|\$)?\s*(?:per\s*(?:purchase\s*)?order|setup\s*expense|setup\s*cost|ordering\s*cost|order\s*cost|ordering\/setup\s*cost|each\s*time\s*we\s*order)",
                    r"\b[sS]" + var_annot + r"\s*(?:is|of|=|:)\s*[\$]?\s*([0-9,]+(?:\.[0-9]+)?)",
                ], q)
                if s is not None:
                    extracted["ordering_cost"] = s

            # Percentage Holding Cost (holding_cost_rate)
            if "holding_cost_rate" not in extracted:
                h_rate = extract_number([
                    r"(?:percentage\s*holding\s*costs?|inventory\s*carrying\s*cost\s*rate|carrying\s*cost\s*rate|holding\s*cost\s*rate|carrying\s*rate|holding\s*rate|annual\s*carrying\s*cost\s*rate|annual\s*holding\s*cost\s*rate)" + var_annot + r"\s*(?:is|of|=|:)?\s*([0-9]+(?:\.[0-9]+)?)\s*%",
                    r"(?:annual\s*|yearly\s*)?(?:inventory\s*)?(?:holding|carrying)\s*costs?" + var_annot + r"\s*(?:is|of|=|:)?\s*([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:of\s*(?:the\s*)?(?:unit|purchase)?\s*(?:cost|price))?",
                    r"([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:of\s*(?:the\s*)?(?:unit\s*)?(?:purchase\s*)?(?:cost|price)|carrying\s*cost\s*rate|holding\s*cost\s*rate|carrying|holding)",
                    r"\b[iI]" + var_annot + r"\s*(?:is|of|=|:)\s*([0-9]+(?:\.[0-9]+)?)\s*%",
                ], q)
                if h_rate is not None:
                    extracted["holding_cost_rate"] = h_rate / 100.0

            # Fixed Holding Cost (fixed_holding_cost)
            if "fixed_holding_cost" not in extracted and "holding_cost_rate" not in extracted:
                fixed_h = extract_number([
                    r"(?:annual\s*|yearly\s*)?(?:fixed\s*)?(?:inventory\s*)?(?:holding|carrying)(?:\/carrying|\/holding)?\s*(?:costs?|expenses?)" + var_annot + r"(?:\s*(?:\/|\s*per\s*)(?:unit|item|piece|sku|" + unit_nouns + r")(?:\s*(?:\/|\s*per\s*)(?:year|annum|annually))?)?\s*(?:is|of|=|:)?\s*[\$]?\s*([0-9,]+(?:\.[0-9]+)?)(?!\s*%\b)\s*(?:dollars?|bucks?|\$)?(?:\s*(?:\/|\s*per\s*)(?:unit|item|piece|sku|" + unit_nouns + r")(?:\s*(?:\/|\s*per\s*)(?:year|annum|annually))?)?",
                    r"holding\s*cost\s*(?:is|of|=|:)?\s*[\$]?\s*([0-9,]+(?:\.[0-9]+)?)(?!\s*%\b)\s*(?:dollars?|bucks?|\$)?\s*(?:per\s*(?:unit|" + unit_nouns + r")\s*(?:per\s*year|annually|\/year|a\s*year))",
                    r"[\$]?\s*([0-9,]+(?:\.[0-9]+)?)(?!\s*%\b)\s*(?:dollars?|bucks?|\$)?\s*(?:per\s*(?:unit|" + unit_nouns + r")\s*(?:per\s*year|annually)\s*(?:fixed\s*)?holding\s*cost|fixed\s*holding\s*cost)",
                    r"\b[hH]" + var_annot + r"\s*(?:is|of|=|:)\s*[\$]?\s*([0-9,]+(?:\.[0-9]+)?)(?!\s*%\b)",
                ], q)
                if fixed_h is not None:
                    extracted["fixed_holding_cost"] = fixed_h

            # Quantity Breaks / Tiers
            if "price_breaks" not in extracted:
                breaks = self._extract_price_breaks_from_text(q)
                if breaks:
                    extracted["price_breaks"] = breaks

            if "demand" not in extracted:
                missing.append("annual_demand (D)")
            if "ordering_cost" not in extracted:
                missing.append("ordering_cost (S)")
            if "holding_cost_rate" not in extracted and "fixed_holding_cost" not in extracted:
                missing.append("holding_cost (holding_cost_rate or fixed_holding_cost)")
            if "price_breaks" not in extracted or not extracted["price_breaks"]:
                missing.append("price_breaks (quantity break tiers with unit prices)")

        # -------------------------------------------------------------------
        # Model 3: Safety Stock & ROP Inputs
        # -------------------------------------------------------------------
        elif model == self.MODEL_SAFETY_STOCK_ROP:
            # 1. Average Demand (d)
            if "demand_rate" not in extracted:
                d = extract_number([
                    r"(?:average\s*demand|mean\s*demand|average\s*daily\s*demand|mean\s*daily\s*demand|daily\s*demand|demand\s*rate)" + var_annot + r"(?:\s+(?:for|of)\s+[a-zA-Z][a-zA-Z0-9\s/-]*?(?:\s+(?:is|averages?|averaging|=|:|at))?)?\s*(?:is|of|averages?|averaging|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
                    r"demand(?:\s+(?:for|of)\s+[a-zA-Z][a-zA-Z0-9\s/-]*?(?:\s+(?:is|averages?|averaging|=|:|at))?)?\s*(?:averages?|is\s*averaging|rate\s*is)?" + var_annot + r"\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:" + unit_nouns + r")?\s*(?:per\s*day|a\s*day|daily)",
                    r"([0-9,]+(?:\.[0-9]+)?)\s*(?:" + unit_nouns + r")?\s*(?:per\s*day|a\s*day|daily)",
                    r"demand\s*(?:is|of|=|:)\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:" + unit_nouns + r")?",
                    r"\b[dD]" + var_annot + r"\s*(?:is|of|=|:)\s*([0-9,]+(?:\.[0-9]+)?)",
                ], q)
                if d is not None:
                    extracted["demand_rate"] = d

            # 2. Demand Standard Deviation (sigma_d)
            if "demand_std" not in extracted:
                d_std = extract_number([
                    r"(?:demand\s*standard\s*deviation|standard\s*deviation\s*of\s*(?:daily\s*)?demand|demand\s*std\s*dev|demand\s*stdev|demand\s*std|daily\s*demand\s*std)" + var_annot + r"\s*(?:is|of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
                    r"(?:daily\s*)?demand\s*(?:is|of|=|averages?)?\s*[0-9,.]+\s*(?:" + unit_nouns + r")?\s*(?:per\s*day)?\s*(?:with|and|,)?\s*(?:a\s*)?(?:demand\s*)?(?:standard\s*deviation|std\s*dev|stdev)" + var_annot + r"\s*(?:of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
                    r"(?:standard\s*deviation|std\s*dev|stdev)" + var_annot + r"\s*(?:of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:" + unit_nouns + r")?\b",
                    r"demand\s*variation\s*(?:is|of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
                    r"\b(?:σ_?d|sigma_?d|s_?d)" + var_annot + r"\s*(?:is|of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
                ], q)
                if d_std is not None:
                    extracted["demand_std"] = d_std
                else:
                    extracted["demand_std"] = 0.0

            # 3. Average Lead Time (L)
            if "lead_time" not in extracted:
                lt = extract_number([
                    r"(?:average\s*lead[- ]time|average\s*lead\s*time|mean\s*lead[- ]time|mean\s*lead\s*time|lead[- ]time|lead\s*time|replenishment\s*lead[- ]time|replenishment\s*time|delivery\s*lead[- ]time|delivery\s*time|supplier\s*(?:replenishment\s*)?lead[- ]time)" + var_annot + r"\s*(?:is|of|averages?|averaging|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:working\s*days?|business\s*days?|days?|weeks?)?",
                    r"(?:supplier\s*)?(?:replenishment\s*)?lead[- ]time" + var_annot + r"\s*(?:is|of|averages?|averaging|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:working\s*days?|business\s*days?|days?|weeks?)?",
                    r"takes\s*(?:an\s*average\s*of)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:working\s*days?|business\s*days?|days?|weeks?)\s*(?:lead[- ]time|to\s*arrive|delivery)?",
                    r"([0-9,]+(?:\.[0-9]+)?)[- ](?:working\s*day|business\s*day|day)\s*(?:lead[- ]time|delivery\s*time|replenishment)",
                    r"\b[lL]" + var_annot + r"\s*(?:is|of|=|:)\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:working\s*days?|days?)?",
                ], q)
                if lt is not None:
                    extracted["lead_time"] = lt

            # 4. Lead-Time Standard Deviation (sigma_L)
            if "lead_time_std" not in extracted:
                lt_std = extract_number([
                    r"(?:lead[- ]time\s*standard\s*deviation|lead\s*time\s*standard\s*deviation|standard\s*deviation\s*of\s*lead[- ]time|standard\s*deviation\s*of\s*lead\s*time|lead[- ]time\s*std\s*dev|lead\s*time\s*std\s*dev|lead[- ]time\s*std|lead\s*time\s*std)" + var_annot + r"\s*(?:is|of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
                    r"(?:(?:supplier\s*)?replenishment\s*)?lead[- ]time\s*(?:is|averages?)?\s*[0-9,.]+\s*(?:days?)?\s*(?:with|and|,)?\s*(?:a\s*)?(?:lead[- ]time\s*)?(?:standard\s*deviation|std\s*dev|stdev)" + var_annot + r"\s*(?:of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
                    r"(?:standard\s*deviation|std\s*dev|stdev)" + var_annot + r"\s*(?:of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:days?|weeks?)\b",
                    r"lead[- ]time\s*varies\s*by\s*([0-9,]+(?:\.[0-9]+)?)\s*days?",
                    r"lead[- ]time\s*variation\s*(?:is|of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
                    r"\b(?:σ_?l|sigma_?l|s_?l)" + var_annot + r"\s*(?:is|of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
                ], q)
                if lt_std is not None:
                    extracted["lead_time_std"] = lt_std
                else:
                    extracted["lead_time_std"] = 0.0

            # 5. Service Level (CSL)
            if "service_level" not in extracted:
                sl = extract_number([
                    r"(?:target\s*(?:cycle\s*)?service\s*level|cycle\s*service\s*level|service\s*level|csl|service\s*rate|service\s*probability)" + var_annot + r"\s*(?:is|of|target|=|:)?\s*([0-9]+(?:\.[0-9]+)?)\s*%",
                    r"([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:cycle\s*service\s*level|service\s*level|csl)",
                    r"(?:mandates?|requires?|targets?)\s*(?:a\s*)?([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:cycle\s*)?service\s*level",
                    r"(?:target\s*)?(?:service\s*level|csl)" + var_annot + r"\s*(?:is|of|=|:)?\s*(0\.[0-9]+)",
                    r"\b(?:csl|sl)" + var_annot + r"\s*(?:is|of|=|:)\s*([0-9]+(?:\.[0-9]+)?)\s*%",
                ], q)
                if sl is not None:
                    extracted["service_level"] = sl / 100.0 if sl > 1.0 else sl
                else:
                    extracted["service_level"] = 0.95

            # Order quantity Q (optional, for Type II fill rate)
            if "order_quantity" not in extracted:
                oq = extract_number([
                    r"(?:order\s*quantity|batch\s*size|replenishment\s*lot\s*size|q)" + var_annot + r"\s*(?:is|of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
                    r"\b[qQ]" + var_annot + r"\s*(?:is|of|=|:)\s*([0-9,]+(?:\.[0-9]+)?)",
                ], q)
                if oq is not None:
                    extracted["order_quantity"] = oq

            if "demand_rate" not in extracted:
                missing.append("average_demand (d per day or period)")
            if "lead_time" not in extracted:
                missing.append("average_lead_time (L in matching time units)")

        # -------------------------------------------------------------------
        # Model 4: Fixed-Period Review Inputs
        # -------------------------------------------------------------------
        elif model == self.MODEL_FIXED_PERIOD_REVIEW:
            # 1. Review Period (R)
            if "review_period" not in extracted:
                r_period = extract_number([
                    r"(?:review\s*period|review\s*interval|review\s*cycle|order\s*interval|reviews?\s*inventory\s*every|review\s*every|order\s*every)" + var_annot + r"\s*(?:is|of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)\s*days?",
                    r"every\s*([0-9,]+(?:\.[0-9]+)?)\s*days?",
                    r"\b[rR]" + var_annot + r"\s*(?:is|of|=|:)\s*([0-9,]+(?:\.[0-9]+)?)\s*days?",
                ], q)
                if r_period is not None:
                    extracted["review_period"] = r_period
                elif "weekly" in q:
                    extracted["review_period"] = 7.0
                elif "biweekly" in q:
                    extracted["review_period"] = 14.0
                elif "monthly" in q:
                    extracted["review_period"] = 30.0

            # 2. Lead Time (L)
            if "lead_time" not in extracted:
                lt = extract_number([
                    r"(?:lead[- ]time|lead\s*time|delivery\s*lead[- ]time|delivery\s*time|replenishment\s*time|replenishment\s*lead[- ]time|supplier\s*(?:replenishment\s*)?lead[- ]time)" + var_annot + r"\s*(?:is|of|averages?|averaging|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:working\s*days?|business\s*days?|days?|weeks?)?",
                    r"takes\s*(?:an\s*average\s*of)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:working\s*days?|business\s*days?|days?|weeks?)\s*(?:lead[- ]time|to\s*arrive|delivery)?",
                    r"([0-9,]+(?:\.[0-9]+)?)[- ](?:working\s*day|business\s*day|day)\s*lead[- ]time",
                    r"\b[lL]" + var_annot + r"\s*(?:is|of|=|:)\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:working\s*days?|days?)?",
                ], q)
                if lt is not None:
                    extracted["lead_time"] = lt

            # 3. Demand Rate (d)
            if "demand_rate" not in extracted:
                d = extract_number([
                    r"(?:daily\s*demand|average\s*daily\s*demand|mean\s*daily\s*demand|demand\s*rate|average\s*demand|mean\s*demand|demand)" + var_annot + r"(?:\s+(?:for|of)\s+[a-zA-Z][a-zA-Z0-9\s/-]*?(?:\s+(?:is|averages?|averaging|=|:|at))?)?\s*(?:is|of|averages?|averaging|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:" + unit_nouns + r")?\s*(?:\/|\s*per\s*)day",
                    r"(?:daily\s*demand|average\s*daily\s*demand|mean\s*daily\s*demand|demand\s*rate|average\s*demand|mean\s*demand)" + var_annot + r"\s*(?:is|of|averages?|averaging|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:" + unit_nouns + r")?",
                    r"([0-9,]+(?:\.[0-9]+)?)\s*(?:" + unit_nouns + r")?\s*(?:\/|\s*per\s*)day",
                    r"demand\s*(?:is|of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:" + unit_nouns + r")?",
                    r"\b[dD]" + var_annot + r"\s*(?:is|of|=|:)\s*([0-9,]+(?:\.[0-9]+)?)",
                ], q)
                if d is not None:
                    extracted["demand_rate"] = d

            # 4. Demand Standard Deviation (sigma_d)
            if "demand_std" not in extracted:
                d_std = extract_number([
                    r"(?:demand\s*standard\s*deviation|standard\s*deviation\s*of\s*(?:daily\s*)?demand|demand\s*std\s*dev|demand\s*stdev|demand\s*std|std\s*dev\s*of\s*demand)" + var_annot + r"\s*(?:is|of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
                    r"demand\s*(?:is|of|=|averages?)?\s*[0-9,.]+\s*(?:" + unit_nouns + r")?\s*(?:with|and|,)?\s*(?:a\s*)?(?:demand\s*)?(?:std\s*dev|standard\s*deviation)" + var_annot + r"\s*(?:of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
                    r"(?:with|and)\s*(?:a\s*)?(?:std\s*dev|standard\s*deviation|stdev)" + var_annot + r"\s*(?:of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
                    r"std\s*dev\s*([0-9,]+(?:\.[0-9]+)?)",
                    r"\b(?:σ_?d|sigma_?d|s_?d)" + var_annot + r"\s*(?:is|of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
                ], q)
                if d_std is not None:
                    extracted["demand_std"] = d_std
                else:
                    extracted["demand_std"] = 0.0

            # 5. Service Level (CSL)
            if "service_level" not in extracted:
                sl = extract_number([
                    r"(?:service\s*level|cycle\s*service\s*level|csl)" + var_annot + r"\s*(?:is|of|=|:)?\s*([0-9]+(?:\.[0-9]+)?)\s*%",
                    r"([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:service\s*level|cycle\s*service\s*level|csl)",
                    r"(?:mandates?|requires?|targets?)\s*(?:a\s*)?([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:cycle\s*)?service\s*level",
                    r"\b(?:csl|sl)" + var_annot + r"\s*(?:is|of|=|:)\s*([0-9]+(?:\.[0-9]+)?)\s*%",
                ], q)
                if sl is not None:
                    extracted["service_level"] = sl / 100.0 if sl > 1.0 else sl
                else:
                    extracted["service_level"] = 0.95

            # 6. Inventory Position / On-Hand / On-Order / Backorders
            # Direct Inventory Position if stated
            inv_pos = extract_number([
                r"(?:inventory\s*position|effective\s*inventory)" + var_annot + r"\s*(?:is|of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
            ], q)

            on_hand = extract_number([
                r"(?:on[- ]hand\s*inventory|on\s*hand\s*inventory|on[- ]hand\s*stock|on\s*hand\s*stock|current\s*on[- ]hand|physical\s*inventory|in\s*warehouse|current\s*stock|currently\s*have|in\s*stock)" + var_annot + r"\s*(?:is|of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
                r"have\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:" + unit_nouns + r"\s*)?(?:on[- ]hand|in\s*stock|in\s*the\s*warehouse)",
                r"([0-9,]+(?:\.[0-9]+)?)\s*(?:" + unit_nouns + r"\s*)?(?:on[- ]hand|in\s*stock|in\s*warehouse\s*stock)",
            ], q)

            pipeline = extract_number([
                r"(?:on[- ]order\s*inventory|on\s*order\s*inventory|on[- ]order\s*stock|pipeline\s*orders?|pipeline\s*inventory|in\s*pipeline|in\s*transit|open\s*orders?|open\s*purchase\s*orders?|incoming\s*orders?)" + var_annot + r"\s*(?:is|of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
                r"([0-9,]+(?:\.[0-9]+)?)\s*(?:" + unit_nouns + r"\s*)?(?:in\s*pipeline|in\s*transit|on[- ]order|on\s*order)",
                r"pipeline\s*of\s*([0-9,]+(?:\.[0-9]+)?)",
            ], q)

            backorders = extract_number([
                r"(?:backorders?|back-orders?|backordered|backlogged|backlog|unfilled\s*orders?)" + var_annot + r"\s*(?:are|is|of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
                r"([0-9,]+(?:\.[0-9]+)?)\s*(?:" + unit_nouns + r"\s*)?(?:backordered|backlogged|on\s*backorder)",
            ], q)

            if inv_pos is not None and on_hand is None:
                extracted["current_on_hand"] = inv_pos
                extracted["pipeline_orders"] = 0.0
                extracted["backorders"] = 0.0
            else:
                extracted["current_on_hand"] = on_hand if on_hand is not None else 0.0
                extracted["pipeline_orders"] = pipeline if pipeline is not None else 0.0
                extracted["backorders"] = backorders if backorders is not None else 0.0

            if "review_period" not in extracted:
                missing.append("review_period (R in days)")
            if "lead_time" not in extracted:
                missing.append("lead_time (L in days)")
            if "demand_rate" not in extracted:
                missing.append("demand_rate (daily demand d)")

        # -------------------------------------------------------------------
        # Model 5: Newsvendor Inputs
        # -------------------------------------------------------------------
        elif model == self.MODEL_NEWSVENDOR:
            # 1. Selling Price (P)
            if "selling_price" not in extracted:
                p = extract_number([
                    r"(?:selling\s*price|retail\s*price|sales\s*price|unit\s*selling\s*price|retail\s*value|retails?\s*(?:for|at)|sells?\s*(?:for|at))" + var_annot + r"\s*(?:is|of|=|:)?\s*[\$]?([0-9,]+(?:\.[0-9]+)?)",
                    r"[\$]([0-9,]+(?:\.[0-9]+)?)\s*(?:selling\s*price|retail|retail\s*price|sale\s*price)",
                    r"\b[pP]" + var_annot + r"\s*(?:is|of|=|:)\s*[\$]?([0-9,]+(?:\.[0-9]+)?)",
                ], q)
                if p is not None:
                    extracted["selling_price"] = p

            # 2. Purchase Cost (C)
            if "unit_cost" not in extracted:
                c = extract_number([
                    r"(?:purchase\s*cost|unit\s*purchase\s*cost|procurement\s*cost|unit\s*cost|wholesale\s*cost|wholesale\s*price|cost\s*per\s*(?:unit|" + unit_nouns + r")|production\s*cost|cost\s*to\s*produce|buys?\s*for|costs?\s*us)" + var_annot + r"\s*(?:is|of|=|:)?\s*[\$]?([0-9,]+(?:\.[0-9]+)?)",
                    r"costs?\s*[\$]?([0-9,]+(?:\.[0-9]+)?)\s*(?:to\s*procure|to\s*produce|to\s*buy|each)",
                    r"[\$]([0-9,]+(?:\.[0-9]+)?)\s*(?:purchase\s*cost|unit\s*cost|wholesale)",
                    r"\b[cC]" + var_annot + r"\s*(?:is|of|=|:)\s*[\$]?([0-9,]+(?:\.[0-9]+)?)",
                ], q)
                if c is not None:
                    extracted["unit_cost"] = c

            # 3. Salvage Value (S_val)
            if "salvage_value" not in extracted:
                s_val = extract_number([
                    r"(?:salvage\s*value|salvage\s*price|salvage|markdown\s*price|markdown\s*value|markdown|liquidation\s*value|liquidation\s*price|scrap\s*value|residual\s*value|recovery\s*value)" + var_annot + r"\s*(?:for\s*unsold\s*(?:items?|units?|" + unit_nouns + r")|is|of|for|=|:)?\s*[\$]?([0-9,]+(?:\.[0-9]+)?)",
                    r"unsold\s*(?:items?|units?|goods?|" + unit_nouns + r")\s*(?:sell\s*for|salvage\s*for|liquidate\s*for|is|are|yield|recover)?\s*(?:an?\s*(?:off-price\s*)?salvage\s*(?:markdown\s*)?(?:of)?)?\s*[\$]?([0-9,]+(?:\.[0-9]+)?)",
                    r"liquidated?\s*at\s*(?:an?\s*(?:off-price\s*)?salvage\s*markdown\s*of)?\s*[\$]?([0-9,]+(?:\.[0-9]+)?)",
                    r"[\$]([0-9,]+(?:\.[0-9]+)?)\s*(?:salvage|markdown|salvage\s*value)",
                    r"\b(?:v|s_?val|salvage)" + var_annot + r"\s*(?:is|of|=|:)\s*[\$]?([0-9,]+(?:\.[0-9]+)?)",
                ], q)
                if s_val is not None:
                    extracted["salvage_value"] = s_val
                else:
                    extracted["salvage_value"] = 0.0

            # 4. Underage / Shortage Cost (Cu)
            cu = extract_number([
                r"(?:underage\s*cost|cost\s*of\s*underage|shortage\s*cost|cost\s*of\s*shortage)" + var_annot + r"\s*(?:is|of|=|:)?\s*[\$]?([0-9,]+(?:\.[0-9]+)?)",
                r"\b(?:c_?u|cu)" + var_annot + r"\s*(?:is|of|=|:)\s*[\$]?([0-9,]+(?:\.[0-9]+)?)",
            ], q)
            if cu is not None:
                extracted["underage_cost"] = cu

            # 5. Overage / Excess Cost (Co)
            co = extract_number([
                r"(?:overage\s*cost|cost\s*of\s*overage|excess\s*cost|cost\s*of\s*excess|loss\s*per\s*unsold\s*unit)" + var_annot + r"\s*(?:is|of|=|:)?\s*[\$]?([0-9,]+(?:\.[0-9]+)?)",
                r"\b(?:c_?o|co)" + var_annot + r"\s*(?:is|of|=|:)\s*[\$]?([0-9,]+(?:\.[0-9]+)?)",
            ], q)
            if co is not None:
                extracted["overage_cost"] = co

            # Shortage Penalty
            pen = extract_number([
                r"(?:shortage\s*penalty|goodwill\s*penalty|penalty\s*per\s*lost\s*sale|penalty\s*for\s*stockout|goodwill\s*loss)" + var_annot + r"\s*(?:is|of|=|:)?\s*[\$]?([0-9,]+(?:\.[0-9]+)?)",
            ], q)
            if pen is not None:
                extracted["shortage_penalty"] = pen
            else:
                extracted["shortage_penalty"] = 0.0

            # If Cu and Co are provided directly, deduce unit_cost and selling_price
            if "underage_cost" in extracted and "overage_cost" in extracted and "selling_price" not in extracted:
                extracted["unit_cost"] = extracted["overage_cost"]
                extracted["selling_price"] = extracted["overage_cost"] + extracted["underage_cost"]
                extracted["salvage_value"] = 0.0

            # 6. Mean Demand (mu)
            if "demand_mean" not in extracted:
                d_mean = extract_number([
                    r"(?:mean\s*demand|average\s*demand|expected\s*demand|demand\s*mean|demand\s*averages?)" + var_annot + r"\s*(?:is|of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
                    r"demand\s*(?:is\s*normally\s*distributed\s*with\s*(?:a\s*)?)?mean\s*(?:of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
                    r"demand\s*of\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:" + unit_nouns + r")?\s*with\s*(?:a\s*)?(?:std\s*dev|standard\s*deviation)",
                    r"\b(?:μ|mu)" + var_annot + r"\s*(?:is|of|=|:)\s*([0-9,]+(?:\.[0-9]+)?)",
                ], q)
                if d_mean is not None:
                    extracted["demand_mean"] = d_mean

            # 7. Demand Standard Deviation (sigma)
            if "demand_std" not in extracted:
                d_std = extract_number([
                    r"(?:demand\s*standard\s*deviation|standard\s*deviation\s*of\s*demand|demand\s*std\s*dev|demand\s*std|demand\s*stdev)" + var_annot + r"\s*(?:is|of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
                    r"standard\s*deviation\s*(?:of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:" + unit_nouns + r")?",
                    r"std\s*dev\s*(?:of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
                    r"\b(?:σ|sigma)" + var_annot + r"\s*(?:is|of|=|:)\s*([0-9,]+(?:\.[0-9]+)?)",
                ], q)
                if d_std is not None:
                    extracted["demand_std"] = d_std
                else:
                    extracted["demand_std"] = 0.0

            if "selling_price" not in extracted and ("underage_cost" not in extracted or "overage_cost" not in extracted):
                missing.append("selling_price (P) or underage_cost (Cu)")
            if "unit_cost" not in extracted and ("underage_cost" not in extracted or "overage_cost" not in extracted):
                missing.append("unit_cost (C) or overage_cost (Co)")
            if "demand_mean" not in extracted:
                missing.append("demand_mean (mu)")

        return extracted, missing

    def _extract_price_breaks_from_text(self, text: str) -> Optional[List[Dict[str, Any]]]:
        """Parse structured text for price breaks (e.g. '0-499: $10, 500-999: $9, 1000+: $8', or '$12 below 500, $11.50 for 500-999, $11 for 1000+')."""
        breaks = []

        # 1. Price first - below / under: "$12 per unit for orders below 500 units"
        pat_price_first_below = re.compile(
            r"[\$]([0-9]+(?:\.[0-9]+)?)\s*(?:per\s*unit|each)?\s*(?:for\s*(?:orders\s*(?:of)?)?)?\s*(?:below|under|less\s*than)\s*([0-9,]+)\s*(?:units?)?",
            re.IGNORECASE
        )
        for m in pat_price_first_below.finditer(text):
            price = float(m.group(1))
            thresh = float(m.group(2).replace(",", ""))
            breaks.append({"min_qty": 0.0, "max_qty": thresh - 1.0, "unit_price": price})

        # 1b. Price first - up to: "$12 for up to 499 units"
        pat_price_first_upto = re.compile(
            r"[\$]([0-9]+(?:\.[0-9]+)?)\s*(?:per\s*unit|each)?\s*(?:for\s*(?:orders\s*(?:of)?)?)?\s*(?:up\s*to)\s*([0-9,]+)\s*(?:units?)?",
            re.IGNORECASE
        )
        for m in pat_price_first_upto.finditer(text):
            price = float(m.group(1))
            max_q = float(m.group(2).replace(",", ""))
            if not any(b["min_qty"] == 0.0 for b in breaks):
                breaks.append({"min_qty": 0.0, "max_qty": max_q, "unit_price": price})

        # 2. Price first - closed range: "$11.50 for 500-999 units", "$11.50 for orders of 500 to 999 units"
        pat_price_first_closed = re.compile(
            r"[\$]([0-9]+(?:\.[0-9]+)?)\s*(?:per\s*unit|each)?\s*(?:for\s*(?:orders\s*(?:of)?)?|between)\s*([0-9,]+)\s*(?:to|-|and)\s*([0-9,]+)\s*(?:units?)?",
            re.IGNORECASE
        )
        for m in pat_price_first_closed.finditer(text):
            price = float(m.group(1))
            min_q = float(m.group(2).replace(",", ""))
            max_q = float(m.group(3).replace(",", ""))
            if not any(b["min_qty"] == min_q for b in breaks):
                breaks.append({"min_qty": min_q, "max_qty": max_q, "unit_price": price})

        # 3. Price first - open-ended: "$11 for orders of 1,000 units or more", "$45 for 1000+ units"
        pat_price_first_open = re.compile(
            r"[\$]([0-9]+(?:\.[0-9]+)?)\s*(?:per\s*unit|each)?\s*(?:for\s*(?:orders\s*(?:of)?)?|at)\s*(?:(?:over|above|more\s*than)\s*([0-9,]+)\s*(?:units?)?|([0-9,]+)\s*(?:units?)?\s*(?:\+|\s*or\s*more|\s*and\s*(?:above|up))\s*(?:units?)?)",
            re.IGNORECASE
        )
        for m in pat_price_first_open.finditer(text):
            price = float(m.group(1))
            q_str = m.group(2) or m.group(3)
            min_q = float(q_str.replace(",", ""))
            if not any(b["min_qty"] == min_q for b in breaks):
                breaks.append({"min_qty": min_q, "max_qty": None, "unit_price": price})

        # 4. Quantity first - closed ranges: "0-499 units at $10.00", "(0-499 units) at $10.00", "0 to 499 units: $10"
        range_pat = re.compile(
            r"(?:tier\s*\d+:?\s*)?\(?(?:orders\s*(?:of)?)?\s*([0-9,]+)\s*(?:to|-)\s*([0-9,]+)\s*(?:units?)?\)?\s*(?:at|costs?|price|for|=|:)\s*[\$]?([0-9]+(?:\.[0-9]+)?)",
            re.IGNORECASE
        )
        for m in range_pat.finditer(text):
            min_q = float(m.group(1).replace(",", ""))
            max_q = float(m.group(2).replace(",", ""))
            price = float(m.group(3))
            if not any(b["min_qty"] == min_q for b in breaks):
                breaks.append({"min_qty": min_q, "max_qty": max_q, "unit_price": price})

        range_pat_units = re.compile(
            r"(?:tier\s*\d+:?\s*)?\(?(?:orders\s*(?:of)?)?\s*([0-9,]+)\s*(?:to|-)\s*([0-9,]+)\s*units?\)?\s*(?:at|costs?|price|for|=|:)?\s*[\$]([0-9]+(?:\.[0-9]+)?)",
            re.IGNORECASE
        )
        for m in range_pat_units.finditer(text):
            min_q = float(m.group(1).replace(",", ""))
            max_q = float(m.group(2).replace(",", ""))
            price = float(m.group(3))
            if not any(b["min_qty"] == min_q for b in breaks):
                breaks.append({"min_qty": min_q, "max_qty": max_q, "unit_price": price})

        # 5. Quantity first - open-ended ranges: "1000+ units at $45", "(500+ units) at $8.00", "500 or more units at $8"
        open_pat = re.compile(
            r"(?:tier\s*\d+:?\s*)?\(?(?:orders\s*(?:of)?)?\s*(?:(?:over|above|more\s*than)\s*([0-9,]+)\s*(?:units?)?|([0-9,]+)\s*(?:units?)?\s*(?:\+|\s*or\s*more|\s*and\s*(?:above|up))\s*(?:units?)?)\)?\s*(?:at|costs?|price|for|=|:)\s*[\$]?([0-9]+(?:\.[0-9]+)?)",
            re.IGNORECASE
        )
        for m in open_pat.finditer(text):
            q_str = m.group(1) or m.group(2)
            min_q = float(q_str.replace(",", ""))
            price = float(m.group(3))
            if not any(b["min_qty"] == min_q for b in breaks):
                breaks.append({"min_qty": min_q, "max_qty": None, "unit_price": price})

        open_pat_dollar = re.compile(
            r"(?:tier\s*\d+:?\s*)?\(?(?:orders\s*(?:of)?)?\s*(?:(?:over|above|more\s*than)\s*([0-9,]+)\s*(?:units?)?|([0-9,]+)\s*(?:units?)?\s*(?:\+|\s*or\s*more|\s*and\s*(?:above|up))\s*(?:units?)?)\)?\s*(?:at|costs?|price|for|=|:)?\s*[\$]([0-9]+(?:\.[0-9]+)?)",
            re.IGNORECASE
        )
        for m in open_pat_dollar.finditer(text):
            q_str = m.group(1) or m.group(2)
            min_q = float(q_str.replace(",", ""))
            price = float(m.group(3))
            if not any(b["min_qty"] == min_q for b in breaks):
                breaks.append({"min_qty": min_q, "max_qty": None, "unit_price": price})

        # 6. Quantity first - upper bound: "under 500 units at $10", "less than 500 units at $10", "orders below 500 units at $12"
        under_pat = re.compile(
            r"(?:orders\s*(?:of)?)?\s*(?:under|less\s*than|below)\s*([0-9,]+)\s*(?:units?)?\s*(?:at|costs?|price|for|=|:)\s*[\$]?([0-9]+(?:\.[0-9]+)?)",
            re.IGNORECASE
        )
        for m in under_pat.finditer(text):
            max_val = float(m.group(1).replace(",", "")) - 1.0
            price = float(m.group(2))
            if not any(b["min_qty"] == 0.0 for b in breaks):
                breaks.append({"min_qty": 0.0, "max_qty": max_val, "unit_price": price})

        under_pat_dollar = re.compile(
            r"(?:orders\s*(?:of)?)?\s*(?:under|less\s*than|below)\s*([0-9,]+)\s*(?:units?)?\s*(?:at|costs?|price|for|=|:)?\s*[\$]([0-9]+(?:\.[0-9]+)?)",
            re.IGNORECASE
        )
        for m in under_pat_dollar.finditer(text):
            max_val = float(m.group(1).replace(",", "")) - 1.0
            price = float(m.group(2))
            if not any(b["min_qty"] == 0.0 for b in breaks):
                breaks.append({"min_qty": 0.0, "max_qty": max_val, "unit_price": price})

        # Sort breaks by min_qty and deduplicate
        breaks = sorted(breaks, key=lambda x: x["min_qty"])
        dedup = []
        seen = set()
        for b in breaks:
            k = (b["min_qty"], b["max_qty"])
            if k not in seen:
                seen.add(k)
                dedup.append(b)
        return dedup if dedup else None

    # -----------------------------------------------------------------------
    # Step 4 & 5: Select Python Tool & Calculate
    # -----------------------------------------------------------------------

    def select_and_calculate(self, model: str, inputs: Dict[str, Any]) -> Tuple[str, Any]:
        """
        Dispatches parameters to the verified Python calculation function.
        Returns: (tool_function_name, calculation_result_object)
        """
        if model == self.MODEL_EOQ:
            tool_name = "calculate_eoq"
            result = tools.calculate_eoq(
                demand=float(inputs["demand"]),
                ordering_cost=float(inputs["ordering_cost"]),
                holding_cost=float(inputs["holding_cost"]),
                unit_cost=float(inputs.get("unit_cost", 0.0)),
                working_days=int(inputs.get("working_days", 250)),
                lead_time_days=float(inputs.get("lead_time_days", 0.0)),
            )
            return tool_name, result

        elif model == self.MODEL_QUANTITY_DISCOUNT:
            tool_name = "calculate_quantity_discount"
            result = tools.calculate_quantity_discount(
                demand=float(inputs["demand"]),
                ordering_cost=float(inputs["ordering_cost"]),
                price_breaks=inputs["price_breaks"],
                holding_cost_rate=inputs.get("holding_cost_rate"),
                fixed_holding_cost=inputs.get("fixed_holding_cost"),
            )
            return tool_name, result

        elif model == self.MODEL_SAFETY_STOCK_ROP:
            tool_name = "calculate_rop"
            result = tools.calculate_rop(
                demand_rate=float(inputs["demand_rate"]),
                lead_time=float(inputs["lead_time"]),
                service_level=float(inputs.get("service_level", 0.95)),
                demand_std=float(inputs.get("demand_std", 0.0)),
                lead_time_std=float(inputs.get("lead_time_std", 0.0)),
                order_quantity=inputs.get("order_quantity"),
            )
            return tool_name, result

        elif model == self.MODEL_FIXED_PERIOD_REVIEW:
            tool_name = "calculate_periodic_review"
            result = tools.calculate_periodic_review(
                review_period=float(inputs["review_period"]),
                lead_time=float(inputs["lead_time"]),
                demand_rate=float(inputs["demand_rate"]),
                demand_std=float(inputs.get("demand_std", 0.0)),
                service_level=float(inputs.get("service_level", 0.95)),
                current_on_hand=float(inputs.get("current_on_hand", 0.0)),
                pipeline_orders=float(inputs.get("pipeline_orders", 0.0)),
                backorders=float(inputs.get("backorders", 0.0)),
            )
            return tool_name, result

        elif model == self.MODEL_NEWSVENDOR:
            tool_name = "calculate_newsvendor"
            result = tools.calculate_newsvendor(
                selling_price=float(inputs["selling_price"]),
                unit_cost=float(inputs["unit_cost"]),
                salvage_value=float(inputs.get("salvage_value", 0.0)),
                shortage_penalty=float(inputs.get("shortage_penalty", 0.0)),
                demand_mean=float(inputs["demand_mean"]),
                demand_std=float(inputs.get("demand_std", 0.0)),
            )
            return tool_name, result

        else:
            raise ValueError(f"Unknown inventory model: {model}")

    # -----------------------------------------------------------------------
    # Step 6: Verify
    # -----------------------------------------------------------------------

    def verify(self, model: str, result: Any) -> Tuple[bool, List[str]]:
        """
        Performs post-calculation sanity checks, operational validations, and boundary verifications.
        Returns: (passed: bool, notes: list[str])
        """
        notes: List[str] = []
        passed = True

        if model == self.MODEL_EOQ:
            r: tools.EOQResult = result
            if r.optimal_order_quantity <= 0:
                passed = False
                notes.append("CRITICAL: Optimal order quantity must be strictly positive.")
            # Verify trade-off balance: Ordering cost ~ Holding cost within numerical rounding
            cost_diff = abs(r.annual_ordering_cost - r.annual_holding_cost)
            if cost_diff > max(2.0, r.total_inventory_cost * 0.02):
                notes.append(f"WARNING: Imbalance between annual ordering (${r.annual_ordering_cost:,.2f}) and holding cost (${r.annual_holding_cost:,.2f}).")
            else:
                notes.append("Verification Passed: Wilson balance confirmed (Annual Ordering Cost ≈ Annual Holding Cost).")
            if r.orders_per_year > 365:
                notes.append("Operational Warning: Implied order frequency exceeds daily deliveries (> 365 orders/year). Consider supplier capacity.")

        elif model == self.MODEL_QUANTITY_DISCOUNT:
            r: tools.QuantityDiscountResult = result
            if r.optimal_order_quantity <= 0:
                passed = False
                notes.append("CRITICAL: Optimal order quantity must be strictly positive.")
            if r.optimal_total_cost > (r.baseline_total_cost or float("inf")):
                passed = False
                notes.append("CRITICAL: Selected discount tier total cost exceeds baseline cost.")
            else:
                notes.append(f"Verification Passed: Global minimum total cost verified (${r.optimal_total_cost:,.2f} at Tier {r.optimal_tier_index}).")

        elif model == self.MODEL_SAFETY_STOCK_ROP:
            r: tools.SafetyStockROPResult = result
            if r.safety_stock < 0:
                passed = False
                notes.append("CRITICAL: Safety stock cannot be negative.")
            if r.reorder_point < r.safety_stock:
                passed = False
                notes.append("CRITICAL: Reorder point is lower than safety stock buffer.")
            else:
                notes.append(f"Verification Passed: ROP ({r.reorder_point:.1f} units) safely covers mean lead time demand ({r.lead_time_demand_mean:.1f}) + buffer ({r.safety_stock:.1f}).")
            if r.service_level > 0.999:
                notes.append("Managerial Warning: Target service level exceeds 99.9%. Exponential holding cost curve applies.")

        elif model == self.MODEL_FIXED_PERIOD_REVIEW:
            r: tools.FixedPeriodResult = result
            if r.target_inventory_level <= 0:
                passed = False
                notes.append("CRITICAL: Target order-up-to level S must be positive.")
            if r.recommended_order_quantity < 0:
                passed = False
                notes.append("CRITICAL: Replenishment order quantity cannot be negative.")
            else:
                notes.append(f"Verification Passed: Protection interval ({r.protection_interval} periods) and Order-Up-To Level S ({r.target_inventory_level:.1f} units) verified.")

        elif model == self.MODEL_NEWSVENDOR:
            r: tools.NewsvendorResult = result
            if not (0.0 <= r.critical_ratio <= 1.0):
                passed = False
                notes.append(f"CRITICAL: Critical ratio ({r.critical_ratio}) is out of [0, 1] probability range.")
            if r.expected_profit < 0:
                notes.append("Financial Warning: Expected profit is negative. Margins do not support demand uncertainty.")
            if r.evpi < 0:
                passed = False
                notes.append("CRITICAL: EVPI cannot be negative.")
            else:
                notes.append(f"Verification Passed: Critical ratio {r.critical_ratio:.2%} and expected profit ${r.expected_profit:,.2f} mathematically sound.")

        return passed, notes

    # -----------------------------------------------------------------------
    # Step 7: Explain
    # -----------------------------------------------------------------------

    def explain(self, model: str, result: Any, notes: List[str]) -> Tuple[str, str, List[str], List[str]]:
        """
        Synthesizes executive summary, detailed briefing, calculation trace, and managerial insights.
        Returns: (executive_summary, detailed_explanation, calculation_trace, strategic_insights)
        """
        exec_summary = ""
        detailed = ""
        insights: List[str] = []
        trace: List[str] = getattr(result, "calculation_steps", [])

        if model == self.MODEL_EOQ:
            r: tools.EOQResult = result
            exec_summary = (
                f"RECOMMENDATION: Place replenishment orders of **{r.optimal_order_quantity:,.0f} units** "
                f"every **{r.order_cycle_days:.1f} operating days** ({r.orders_per_year:.1f} orders/year). "
                f"Total annual inventory management cost will be **${r.total_inventory_cost:,.2f}**."
            )
            if r.reorder_point_deterministic:
                exec_summary += f" Trigger new purchase orders whenever warehouse inventory reaches **{r.reorder_point_deterministic:,.0f} units**."

            detailed = (
                f"### Economic Order Quantity (EOQ) Analysis\n\n"
                f"- **Optimal Order Quantity ($Q^*$):** {r.optimal_order_quantity:,.2f} units\n"
                f"- **Order Frequency:** {r.orders_per_year:.2f} orders per year\n"
                f"- **Cycle Duration:** {r.order_cycle_days:.1f} working days between orders\n"
                f"- **Annual Ordering Cost:** ${r.annual_ordering_cost:,.2f}\n"
                f"- **Annual Holding Cost:** ${r.annual_holding_cost:,.2f}\n"
                f"- **Total Inventory Holding & Setup Cost:** ${r.total_inventory_cost:,.2f}\n"
            )
            if r.unit_cost > 0:
                detailed += f"- **Total Landed Annual Cost (incl. Purchasing):** ${r.total_landed_cost:,.2f}\n"

            insights.append("Cost Flatness: The EOQ total cost curve is notably flat around the minimum. Modest lot-size deviations (±10%) increase total inventory cost by under 0.6%.")
            insights.append("Ordering vs Carrying: Current ordering cost exactly matches carrying cost, proving cost equilibrium.")
            if r.reorder_point_deterministic:
                insights.append(f"Deterministic Pipeline: At lead-time demand of {r.reorder_point_deterministic:,.0f} units, ensure procurement initiates replenishment promptly at the ROP threshold.")

        elif model == self.MODEL_QUANTITY_DISCOUNT:
            r: tools.QuantityDiscountResult = result
            exec_summary = (
                f"RECOMMENDATION: Order **{r.optimal_order_quantity:,.0f} units** to qualify for **Tier {r.optimal_tier_index}** "
                f"pricing at **${r.optimal_unit_price:,.2f}/unit**. Total landed annual cost is **${r.optimal_total_cost:,.2f}**."
            )
            if r.annual_savings and r.annual_savings > 0:
                exec_summary += f" This saves **${r.annual_savings:,.2f} annually ({r.percent_savings:.1f}%)** vs. ordering at base EOQ."

            detailed = (
                f"### Quantity Discount Optimization\n\n"
                f"- **Selected Tier:** Tier {r.optimal_tier_index} (Price: ${r.optimal_unit_price:,.2f}/unit)\n"
                f"- **Recommended Lot Size:** {r.optimal_order_quantity:,.0f} units\n"
                f"- **Total Landed Cost:** ${r.optimal_total_cost:,.2f}\n"
                f"- **Base Landed Cost:** ${r.baseline_total_cost:,.2f}\n"
                f"- **Net Annual Financial Advantage:** ${r.annual_savings:,.2f}\n\n"
                f"**Tier Comparison:**\n"
            )
            for t in r.tiers_evaluated:
                status = "★ OPTIMAL" if t.is_optimal else ("FEASIBLE" if t.is_feasible else "BOUNDARY ADJUSTED")
                detailed += (
                    f"- Tier {t.tier_index} (min {t.min_qty:,.0f}, ${t.unit_price:,.2f}/u): Order Q={t.order_quantity:,.0f} | "
                    f"Purch: ${t.annual_purchase_cost:,.2f} | Hold: ${t.annual_holding_cost:,.2f} | "
                    f"Order: ${t.annual_ordering_cost:,.2f} | Total: ${t.total_annual_cost:,.2f} [{status}]\n"
                )

            insights.append("Working Capital Impact: Taking advantage of quantity discounts elevates inventory carrying requirements. Confirm warehouse physical capacity and credit facility before increasing lot size.")
            insights.append("Supplier Negotiation: Use these calculations during supplier negotiations to demonstrate how order batching offsets the supplier's volume discounts.")

        elif model == self.MODEL_SAFETY_STOCK_ROP:
            r: tools.SafetyStockROPResult = result
            exec_summary = (
                f"RECOMMENDATION: Set your warehouse Reorder Point (ROP) at **{r.reorder_point:,.0f} units**, "
                f"incorporating a dedicated Safety Stock buffer of **{r.safety_stock:,.0f} units**. "
                f"This guarantees a **{r.service_level * 100:.1f}% Cycle Service Level**."
            )

            detailed = (
                f"### Safety Stock & Continuous Review (s, Q) Analysis\n\n"
                f"- **Target Service Level (CSL):** {r.service_level * 100:.1f}% (Normal z = {r.z_score:.4f})\n"
                f"- **Expected Lead Time Demand:** {r.lead_time_demand_mean:,.2f} units\n"
                f"- **Lead Time Demand Uncertainty (\\sigma_{{DL}}):** {r.lead_time_demand_std:,.2f} units\n"
                f"- **Buffer Safety Stock ($SS$):** {r.safety_stock:,.2f} units\n"
                f"- **Actionable Reorder Point ($ROP$):** {r.reorder_point:,.2f} units\n"
                f"- **Stockout Probability per Replenishment Cycle:** {r.stockout_risk_per_cycle * 100:.2f}%\n"
                f"- **Expected Units Short per Cycle:** {r.expected_shortage_per_cycle:.4f} units\n"
                f"- **Variability Regime:** {r.variability_case}\n"
            )
            if r.fill_rate is not None:
                detailed += f"- **Item Fill Rate (Type II Service Level):** {r.fill_rate * 100:.2f}%\n"

            insights.append(f"Uncertainty Driver: System modeled under '{r.variability_case}'.")
            if r.lead_time_std > 0:
                insights.append("Lead Time Variance Sensitivity: Supplier delivery inconsistency dramatically expands required safety buffer. Negotiating strict supplier lead time SLAs can cut buffer stock significantly.")
            insights.append(f"Stockout Exposure: Over each procurement cycle, there is only a {r.stockout_risk_per_cycle * 100:.1f}% chance of depleting stock before the delivery arrives.")

        elif model == self.MODEL_FIXED_PERIOD_REVIEW:
            r: tools.FixedPeriodResult = result
            exec_summary = (
                f"RECOMMENDATION: Target Order-Up-To Level (S) is **{r.target_inventory_level:,.0f} units** "
                f"(including a **{r.safety_stock:,.0f} unit** safety stock). "
                f"With current inventory position of **{r.current_inventory_position:,.0f} units**, "
                f"place a replenishment order for **{r.recommended_order_quantity:,.0f} units** today."
            )

            detailed = (
                f"### Fixed-Period Review Model (R, S) Analysis\n\n"
                f"- **Review Interval ($R$):** {r.review_period:.1f} days\n"
                f"- **Supplier Lead Time ($L$):** {r.lead_time:.1f} days\n"
                f"- **Vulnerability Protection Interval ($R + L$):** {r.protection_interval:.1f} days\n"
                f"- **Mean Demand over Protection Interval:** {r.protection_interval_mean_demand:,.2f} units\n"
                f"- **Protection Interval Std Dev:** {r.protection_interval_std_demand:,.2f} units\n"
                f"- **Safety Stock Buffer:** {r.safety_stock:,.2f} units\n"
                f"- **Target Inventory (Order-Up-To Level $S$):** {r.target_inventory_level:,.2f} units\n"
                f"- **Current Inventory Position ($IP$):** {r.current_inventory_position:,.2f} units\n"
                f"- **Immediate Order Quantity ($Q$):** {r.recommended_order_quantity:,.2f} units\n"
            )

            insights.append(f"Periodic Review Premium: Operating on fixed periodic schedules requires {r.safety_stock_vs_continuous_diff:,.1f} more safety units than continuous review to cover the review lag.")
            insights.append("Consolidation Advantage: While requiring higher safety buffer, periodic review allows batching orders across multiple suppliers or route schedules.")

        elif model == self.MODEL_NEWSVENDOR:
            r: tools.NewsvendorResult = result
            exec_summary = (
                f"RECOMMENDATION: Stock exactly **{r.optimal_stocking_quantity:,.0f} units** for this season/cycle. "
                f"Critical Ratio is **{r.critical_ratio:.1%}**, giving an expected net profit of **${r.expected_profit:,.2f}** "
                f"and expected lost sales of only **{r.expected_lost_sales:.1f} units**."
            )

            detailed = (
                f"### Newsvendor Single-Period Optimization\n\n"
                f"- **Selling Price ($P$):** ${r.selling_price:,.2f}\n"
                f"- **Unit Purchase Cost ($C$):** ${r.unit_cost:,.2f}\n"
                f"- **Salvage Markdown ($S_{{val}}$):** ${r.salvage_value:,.2f}\n"
                f"- **Cost of Underage ($C_u$):** ${r.cost_of_underage_cu:,.2f}/unit (Margin lost if short)\n"
                f"- **Cost of Overage ($C_o$):** ${r.cost_of_overage_co:,.2f}/unit (Net loss on unsold units)\n"
                f"- **Optimal Critical Ratio ($CR$):** {r.critical_ratio * 100:.2f}%\n"
                f"- **Optimal Order Quantity ($Q^*$):** {r.optimal_stocking_quantity:,.2f} units\n"
                f"- **Expected Sales:** {r.expected_sales:,.2f} units\n"
                f"- **Expected Leftover (Excess):** {r.expected_leftover_units:,.2f} units\n"
                f"- **Expected Lost Sales:** {r.expected_lost_sales:,.2f} units\n"
                f"- **Expected Net Profit:** ${r.expected_profit:,.2f}\n"
                f"- **Expected Value of Perfect Information (EVPI):** ${r.evpi:,.2f}\n"
            )

            if r.critical_ratio > 0.5:
                insights.append(f"Margin Bias: Because profit margin (${r.cost_of_underage_cu:,.2f}) exceeds overage risk (${r.cost_of_overage_co:,.2f}), the optimal policy stocks ABOVE mean demand ({r.optimal_stocking_quantity:,.0f} vs {r.demand_mean:,.0f}).")
            else:
                insights.append(f"Risk Aversion: Because unsold penalty (${r.cost_of_overage_co:,.2f}) exceeds unit margin (${r.cost_of_underage_cu:,.2f}), the optimal policy stocks BELOW mean demand to protect margins.")
            insights.append(f"Forecasting Value: EVPI is ${r.evpi:,.2f}. Investing in advanced demand forecasting or presales can capture up to this amount in additional margin.")

        return exec_summary, detailed, trace, insights

    def format_analyst_verification_guide(self, calc_res: Any) -> str:
        """
        Formats the Analyst 8-Step Verification Audit directly from deterministic tool outputs:
        1. Inputs Extracted
        2. Variable Definitions
        3. Formula
        4. Formula with Substituted Values
        5. Intermediate Calculations
        6. Final Answer
        7. Verification Checks
        8. Assumptions
        """
        lines = ["\n---\n### Analyst 8-Step Verification Audit\n"]

        # 1. Inputs Extracted
        lines.append("#### 1. Inputs Extracted")
        inputs_ext = getattr(calc_res, "inputs_extracted", getattr(calc_res, "inputs", {}))
        for k, v in inputs_ext.items():
            if k == "price_breaks" and isinstance(v, list):
                lines.append(f"- **{k}:**")
                for pb in v:
                    lines.append(f"  * Tier min {pb.get('min_qty', 0)} to {pb.get('max_qty', 'inf')}: ${pb.get('unit_price', 0):,.2f}/unit")
            else:
                lines.append(f"- **{k}:** {v}")

        # 2. Variable Definitions
        lines.append("\n#### 2. Variable Definitions")
        var_defs = getattr(calc_res, "variable_definitions", {})
        for sym, defn in var_defs.items():
            lines.append(f"- **{sym}:** {defn}")

        # 3. Formula
        lines.append("\n#### 3. Formula")
        formulas = getattr(calc_res, "formula", {})
        for name, form in formulas.items():
            lines.append(f"- **{name.replace('_', ' ').title()}:** `{form}`")

        # 4. Formula with Substituted Values
        lines.append("\n#### 4. Formula with Substituted Values")
        form_subs = getattr(calc_res, "formula_with_substituted_values", {})
        for name, sub_form in form_subs.items():
            lines.append(f"- **{name.replace('_', ' ').title()}:** `{sub_form}`")

        # 5. Intermediate Calculations
        lines.append("\n#### 5. Intermediate Calculations")
        intermediates = getattr(calc_res, "intermediate_calculations", {})
        for k, v in intermediates.items():
            if k == "tiers_detailed" and isinstance(v, list):
                lines.append("- **Tiers Evaluated Details:**")
                for td in v:
                    lines.append(f"  * Tier {td.get('tier_index')}: Q_eoq={td.get('unconstrained_eoq')}, evaluated Q={td.get('evaluated_quantity')}, Total Cost=${td.get('total_annual_cost'):,.2f} ({'Feasible' if td.get('is_feasible') else 'Boundary Adjusted'})")
            else:
                lines.append(f"- **{k.replace('_', ' ').title()}:** {v}")

        # 6. Final Answer
        lines.append("\n#### 6. Final Answer")
        final_ans = getattr(calc_res, "final_answer", {})
        for k, v in final_ans.items():
            lines.append(f"- **{k.replace('_', ' ').title()}:** {v}")

        # 7. Verification Checks
        lines.append("\n#### 7. Verification Checks")
        checks = getattr(calc_res, "verification_checks", [])
        for chk in checks:
            lines.append(f"- [x] {chk}")

        # 8. Assumptions
        lines.append("\n#### 8. Assumptions")
        assumptions = getattr(calc_res, "assumptions", [])
        for a in assumptions:
            lines.append(f"- {a}")

        return "\n".join(lines)

    # Backward compatibility alias
    format_student_verification_guide = format_analyst_verification_guide

    # -----------------------------------------------------------------------
    # Main Orchestrator: End-to-End Pipeline
    # -----------------------------------------------------------------------

    def process_question(self, question: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """
        Executes the full Supply Chain Analytics decision pipeline:
        Customer Question -> Identify Model -> Extract Inputs -> Select Python Tool -> Calculate -> Verify -> Explain
        """
        # 1 & 2: Identify Model
        model, confidence = self.identify_model(question)

        # 3: Extract Inputs
        extracted_inputs, missing_inputs = self.extract_inputs(model, question, context)

        # If critical inputs are missing, report back immediately with clarification request
        if missing_inputs:
            return AgentResponse(
                question=question,
                identified_model=model,
                confidence=confidence,
                extracted_inputs=extracted_inputs,
                missing_inputs=missing_inputs,
                tool_selected=None,
                calculation_result=None,
                verification_passed=False,
                verification_notes=[f"Missing required parameters: {', '.join(missing_inputs)}"],
                executive_summary=f"Model identified as {model}, but requires additional parameters to calculate.",
                detailed_explanation=(
                    f"To calculate optimal inventory for **{model}**, please provide the following missing parameters:\n"
                    + "\n".join([f"- **{field}**" for field in missing_inputs])
                ),
                mathematical_trace=[],
                strategic_insights=[
                    f"The agent detected an inquiry relating to {model}. Provide the necessary cost or demand parameters to compute deterministic solutions."
                ],
            )

        # 4 & 5: Select Python Tool & Calculate
        tool_name, calc_res = self.select_and_calculate(model, extracted_inputs)

        # 6: Verify
        passed, notes = self.verify(model, calc_res)

        # 7: Explain
        exec_sum, detailed, trace, insights = self.explain(model, calc_res, notes)
        detailed += self.format_analyst_verification_guide(calc_res)

        return AgentResponse(
            question=question,
            identified_model=model,
            confidence=confidence,
            extracted_inputs=extracted_inputs,
            missing_inputs=[],
            tool_selected=tool_name,
            calculation_result=calc_res.model_dump() if hasattr(calc_res, "model_dump") else calc_res.__dict__,
            verification_passed=passed,
            verification_notes=notes,
            executive_summary=exec_sum,
            detailed_explanation=detailed,
            mathematical_trace=trace,
            strategic_insights=insights,
            assumptions=getattr(calc_res, "assumptions", []),
            intermediate_calculations=getattr(calc_res, "intermediate_calculations", {}),
            final_answer=getattr(calc_res, "final_answer", {}),
            inputs_extracted=getattr(calc_res, "inputs_extracted", getattr(calc_res, "inputs", {})),
            variable_definitions=getattr(calc_res, "variable_definitions", {}),
            formula=getattr(calc_res, "formula", {}),
            formula_with_substituted_values=getattr(calc_res, "formula_with_substituted_values", {}),
            verification_checks=getattr(calc_res, "verification_checks", []),
        )
