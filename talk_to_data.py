# Paste this into a Custom Code component's Code tab
# Talk to Data Pipeline  unified 5-stage NL-to-SQL pipeline
# Exposed as a Tool for Worker Node via tool_mode=True
#
# Stages:
#   1. Query Analyzer    (CODE  normalize, classify intent, resolve aliases)
#   2. Schema Linker     (LLM   resolve NL terms to DB columns)
#   3. Context Builder   (CODE  filter knowledge, select examples, assemble prompt)
#   4. SQL Generator     (LLM   template match or LLM generation)
#   5. SQL Processor     (CODE + DB  validate, fix, execute, format)

from agentcore.custom import Node
from decimal import Decimal
import json
import re
import sys
import time
import traceback

# SQL Server / pyodbc returns numeric columns (money, decimal(p,s)) as Python
# Decimal, not float. Every formatter and type-detector must accept Decimal,
# otherwise money columns silently fall through to str(val) and render as
# "30545446.861598" instead of "30.5M". Single source of truth to avoid drift.
_NUMERIC_TYPES = (int, float, Decimal)

# 
# STAGE 1 CONSTANTS  Query Analyzer
# 

ABBREVIATIONS = {
    "ytd": "year to date", "yoy": "year over year", "mom": "month over month",
    "qty": "quantity", "amt": "amount", "avg": "average",
    "mfg": "manufacturing", "mgmt": "management", "dept": "department",
    "org": "organization", "fy": "fiscal year",
    "q1": "quarter 1", "q2": "quarter 2", "q3": "quarter 3", "q4": "quarter 4",
}

FILLER_RE = re.compile(
    r"\b(please|can you|could you|show me|i want to see|i need|"
    r"i would like|tell me|give me|display|find me|help me)\b",
    re.IGNORECASE,
)

INTENT_PATTERNS = {
    "enumerate": [r"\blist\b", r"\bshow all\b", r"\bwhat are\b", r"\bwhich\b", r"\bdistinct\b"],
    "top_n": [r"\btop\s+\d+\b", r"\bhighest\s+\d+\b", r"\blargest\s+\d+\b", r"\bbiggest\s+\d+\b"],
    "bottom_n": [r"\bbottom\s+\d+\b", r"\blowest\s+\d+\b", r"\bsmallest\s+\d+\b"],
    "count": [r"\bhow many\b", r"\bnumber of\b", r"\bcount\b"],
    "average": [r"\baverage\b", r"\bavg\b", r"\bmean\b"],
    "time_series": [r"\bmonthly\b", r"\bquarterly\b", r"\bweekly\b", r"\bdaily\b", r"\bby month\b"],
    "trend": [r"\btrend\b", r"\bover time\b", r"\byear over year\b", r"\bgrowth\b"],
    "comparison": [r"\bcompare\b", r"\bvs\b", r"\bversus\b", r"\bdifference\b"],
    "aggregation_grouped_filtered": [r"\bspend\b.*\b(in|for|from)\s+\w+\b.*\bby\s+\w+\b", r"\bby\s+\w+\b.*\b(in|for|from)\s+\w+\b"],
    "aggregation_grouped": [r"\bby\s+\w+\b", r"\bper\s+\w+\b", r"\bfor each\b"],
    "aggregation": [r"\btotal\b", r"\bsum\b", r"\boverall\b", r"\bspend\b"],
    "filter": [r"\bexcluding\b", r"\bexcept\b", r"\bwithout\b", r"\babove\b", r"\bbelow\b"],
}

SPECIFICITY = {
    "top_n": 10, "bottom_n": 10, "comparison": 8, "trend": 8,
    "time_series": 7, "average": 6, "count": 6, "enumerate": 6,
    "aggregation_grouped_filtered": 6, "aggregation_grouped": 5, "aggregation": 4,
    "filter": 3,
}

# Stopwords that should NEVER be matched as entity aliases even if they
# appear in the alias dictionary.  These are common English prepositions,
# conjunctions, and articles that cause false positives (e.g. "in"  India).
ALIAS_STOPWORDS = {
    "a", "an", "the", "in", "on", "at", "to", "of", "by", "for", "or",
    "and", "is", "it", "as", "if", "so", "no", "do", "be", "we", "he",
    "me", "my", "up", "am", "us", "vs", "per", "all", "any", "but",
    "not", "our", "its", "has", "had", "was", "are", "can", "may",
    "get", "got", "let", "new", "old", "top", "low", "big", "net",
    "how", "who", "what", "when", "with", "from", "into", "over",
    "than", "that", "this", "each", "some", "show", "give", "find",
}

# Minimum alias length (characters) to consider for matching.
# Aliases shorter than this are skipped unless they are uppercase
# acronyms (e.g. "US", "UK", "EU") that the user explicitly defined.
MIN_ALIAS_LENGTH = 3

# 
# TABLE REGISTRY  extensible config for multi-table support
# 

TABLE_REGISTRY = {
    "direct": {
        "table_name": "DBO.VW_DIRECT_SPEND_ALL",
        "label": "Direct Spend",
        "amount_is_eur": True,
        "detection_keywords": [
            "direct spend", "direct material", "production material",
            "direct procurement", "production spend", "direct cost",
            "direct purchase",
            "oem", "commodity description", "kss", "sbu", "fahrzeugtyp",
            "customer name", "negotiated price", "org_negotiated_price",
            "project name",
        ],
        "price_columns": [
            "ORG_NEGOTIATED_PRICE",
        ],
        "mandatory_filter": "INVOICE_DATE >= '2024-04-01'",
        "column_semantics": (
            "  - 'commodity' / 'category' = COMMODITY or COMMODITY_DESCRIPTION\n"
            "  - 'customer' / 'OEM customer' = CUSTOMER_NAME (automaker names: BMW, Audi, Daimler, VW, Porsche)\n"
            "  - 'kss' / 'material group' = KSS (code like E011-04); MG_DESCRIPTION for description\n"
            "  - 'sbu' / 'business unit' = SBU\n"
            "  - 'project' = PROJECT_NAME\n"
            "  - 'vehicle type' / 'fahrzeugtyp' = FAHRZEUGTYP\n"
            "  - 'negotiated price' = ORG_NEGOTIATED_PRICE\n"
            "  - 'article' / 'material no' / 'part number' = ARTICLE_NO; MATERIAL_DESCRIPTION for description\n"
            "  - 'country' = COUNTRY (short codes: GER, USA, HUN, MEX, BRA, SLO, SRB, POL)\n"
            "\n"
            "  **Material Classification (OEM column — Direct view only):**\n"
            "  OEM column stores nomination flags, NOT manufacturer names.\n"
            "  - 'OEM nominated' / 'OEM directed' = OEM IN ('J','Y')  (J=German Ja, Y=English Yes)\n"
            "  - 'MPP nominated' / 'Motherson nominated' = (OEM = 'N' OR OEM IS NULL)\n"
            "  - 'Group nominated' = OEM = 'G'\n"
            "  - To show classification as a column: CASE WHEN OEM IN ('J','Y') THEN 'OEM Nominated' WHEN OEM = 'G' THEN 'Group' ELSE 'MPP Nominated' END\n"
            "  - 'OEM nominated spend' = SUM(AMOUNT) WHERE OEM IN ('J','Y')\n"
            "  - 'Motherson nominated spend' = SUM(AMOUNT) WHERE (OEM = 'N' OR OEM IS NULL)\n"
            "  - 'Group nominated spend' = SUM(AMOUNT) WHERE ABC_INDICATOR = 'G'\n"
            "\n"
            "  **Supplier Classification (ABC_INDICATOR):**\n"
            "  - 'Motherson Group' / 'group suppliers' = ABC_INDICATOR = 'G'\n"
            "  - 'External suppliers' = ABC_INDICATOR IN ('O','N','S') OR ABC_INDICATOR IS NULL\n"
            "  - 'Strategic suppliers' = ABC_INDICATOR = 'S'\n"
            "  - To show as column: CASE WHEN ABC_INDICATOR = 'G' THEN 'Motherson Group' ELSE 'External' END\n"
            "\n"
            "  **Incoterm Responsibility (INCO_TERM):**\n"
            "  - 'Buyer Responsible' = INCO_TERM IN ('EXW','FCA','FOB','CIF')\n"
            "  - 'Supplier Responsible' = INCO_TERM IN ('DAP','DDP','DDU')\n"
            "  - 'Shared Responsibility' = INCO_TERM IN ('CPT','CIP')\n"
            "  - 'Unknown incoterm' = INCO_TERM = '999'\n"
            "  - 'Unclassified incoterm' = INCO_TERM IS NULL\n"
            "  - To show as column: CASE WHEN INCO_TERM IN ('EXW','FCA','FOB','CIF') THEN 'Buyer' WHEN INCO_TERM IN ('DAP','DDP','DDU') THEN 'Supplier' WHEN INCO_TERM IN ('CPT','CIP') THEN 'Shared' WHEN INCO_TERM = '999' THEN 'Unknown' ELSE 'Unclassified' END\n"
            "\n"
            "  **LCC Analysis (ABC_INDICATOR + PROPOSED_CATEGORY):**\n"
            "  - 'Insourcing' = ABC_INDICATOR = 'G'\n"
            "  - 'HCC' / 'High Cost Country' = (ABC_INDICATOR != 'G' OR ABC_INDICATOR IS NULL) AND PROPOSED_CATEGORY = 'HCC'\n"
            "  - 'MCC/LCC' / 'Low Cost Country' = (ABC_INDICATOR != 'G' OR ABC_INDICATOR IS NULL) AND (PROPOSED_CATEGORY != 'HCC' OR PROPOSED_CATEGORY IS NULL)\n"
            "  - To show as column: CASE WHEN ABC_INDICATOR = 'G' THEN 'Insourcing' WHEN PROPOSED_CATEGORY = 'HCC' THEN 'HCC' ELSE 'MCC/LCC' END\n"
            "\n"
            "  **Currency Conversion (EXCH_RATE + EXCH_CURRENCY):**\n"
            "  - AMOUNT is in EUR by default. If user asks for spend 'in USD'/'in MXN'/'in INR' etc.:\n"
            "    Use SUM(AMOUNT * EXCH_RATE) and add WHERE EXCH_CURRENCY = '<currency_code>'\n"
            "  - Available currencies: EUR, USD, MXN, BRL, RSD, PLN\n"
            "\n"
            "  **Top X% / Tail Spend (window functions):**\n"
            "  - 'Top 80% suppliers' = suppliers whose cumulative spend reaches 80% of total:\n"
            "    Use SUM(spend) OVER (ORDER BY spend DESC) / SUM(spend) OVER () as cumulative_pct, then filter <= 0.8\n"
            "  - 'Tail spend' / 'bottom 5%/10%/20%' = suppliers in the bottom X% of total spend (inverse of top%):\n"
            "    Same window function, filter cumulative_pct > (1 - threshold)\n"
            "\n"
            "  **Multi-Region Material:**\n"
            "  - 'same material in multiple regions' = ARTICLE_NO appearing in more than one REGION:\n"
            "    Use HAVING COUNT(DISTINCT REGION) > 1 when grouping by ARTICLE_NO\n"
        ),
    },
    "indirect": {
        "table_name": "DBO.VW_INDIRECT_SPEND_ALL",
        "label": "Indirect Spend",
        "amount_is_eur": True,
        "net_price_requires_user_filter": True,
        "detection_keywords": [
            "indirect spend", "indirect material", "services spend",
            "mro", "mro spend", "non-production", "overhead",
            "overhead spend", "indirect procurement", "indirect cost",
            "indirect purchase", "services cost",
            "material group", "main account", "net price", "net_price",
            "industry", "source", "sales org", "sales_org",
            "voucher amount", "voucher_amount", "payment term",
        ],
        "price_columns": [
            "NET_PRICE",
        ],
        "mandatory_filter": "INVOICE_DATE >= '2024-04-01'",
        "column_semantics": (
            "  - 'material group' = MATERIAL_GROUP or MG_DESCRIPTION\n"
            "  - 'main account' = MAIN_ACCOUNT or MAIN_ACCOUNT_DESCRIPTION\n"
            "  - 'net price' = NET_PRICE\n"
            "  - 'industry' = INDUSTRY\n"
            "  - 'source' = SOURCE\n"
            "  - 'sales org' / 'sales organization' = SALES_ORG\n"
            "  - 'voucher' / 'voucher amount' = VOUCHER_AMOUNT\n"
            "  - 'payment term' / 'payment terms' = PAYMENT_TERM (code); use TERMS_OF_PAYMENT_DESCRIPTION only if user asks for description\n"
            "\n"
            "  **Supplier Classification (ABC_INDICATOR):**\n"
            "  - 'Motherson Group' / 'group suppliers' = ABC_INDICATOR = 'G'\n"
            "  - 'External suppliers' = ABC_INDICATOR IN ('O','N','S') OR ABC_INDICATOR IS NULL\n"
            "  - 'Strategic suppliers' = ABC_INDICATOR = 'S'\n"
            "  - To show as column: CASE WHEN ABC_INDICATOR = 'G' THEN 'Motherson Group' ELSE 'External' END\n"
            "\n"
            "  **Incoterm Responsibility (INCO_TERM):**\n"
            "  - 'Buyer Responsible' = INCO_TERM IN ('EXW','FCA','FOB','CIF')\n"
            "  - 'Supplier Responsible' = INCO_TERM IN ('DAP','DDP','DDU')\n"
            "  - 'Shared Responsibility' = INCO_TERM IN ('CPT','CIP')\n"
            "  - 'Unknown incoterm' = INCO_TERM = '999'\n"
            "  - 'Unclassified incoterm' = INCO_TERM IS NULL\n"
            "  - To show as column: CASE WHEN INCO_TERM IN ('EXW','FCA','FOB','CIF') THEN 'Buyer' WHEN INCO_TERM IN ('DAP','DDP','DDU') THEN 'Supplier' WHEN INCO_TERM IN ('CPT','CIP') THEN 'Shared' WHEN INCO_TERM = '999' THEN 'Unknown' ELSE 'Unclassified' END\n"
            "\n"
            "  **LCC Analysis (ABC_INDICATOR + PROPOSED_CATEGORY):**\n"
            "  - 'Insourcing' = ABC_INDICATOR = 'G'\n"
            "  - 'HCC' / 'High Cost Country' = (ABC_INDICATOR != 'G' OR ABC_INDICATOR IS NULL) AND PROPOSED_CATEGORY = 'HCC'\n"
            "  - 'MCC/LCC' / 'Low Cost Country' = (ABC_INDICATOR != 'G' OR ABC_INDICATOR IS NULL) AND (PROPOSED_CATEGORY != 'HCC' OR PROPOSED_CATEGORY IS NULL)\n"
            "  - To show as column: CASE WHEN ABC_INDICATOR = 'G' THEN 'Insourcing' WHEN PROPOSED_CATEGORY = 'HCC' THEN 'HCC' ELSE 'MCC/LCC' END\n"
            "\n"
            "  **Currency Conversion (EXCH_RATE + INVOICE_AMOUNT_CURRENCY):**\n"
            "  - AMOUNT is in EUR by default. If user asks for spend 'in USD'/'in MXN'/'in INR' etc.:\n"
            "    Use SUM(AMOUNT * EXCH_RATE) and add WHERE INVOICE_AMOUNT_CURRENCY = '<currency_code>'\n"
            "  - Available currencies: EUR, USD, MXN, BRL, RSD, PLN\n"
            "\n"
            "  **Top X% / Tail Spend (window functions):**\n"
            "  - 'Top 80% suppliers' = suppliers whose cumulative spend reaches 80% of total:\n"
            "    Use SUM(spend) OVER (ORDER BY spend DESC) / SUM(spend) OVER () as cumulative_pct, then filter <= 0.8\n"
            "  - 'Tail spend' / 'bottom 5%/10%/20%' = suppliers in the bottom X% of total spend (inverse of top%):\n"
            "    Same window function, filter cumulative_pct > (1 - threshold)\n"
            "\n"
            "  **Multi-Region Material:**\n"
            "  - 'same material in multiple regions' = MATERIAL_GROUP appearing in more than one REGION:\n"
            "    Use HAVING COUNT(DISTINCT REGION) > 1 when grouping by MATERIAL_GROUP\n"
        ),
    },
}

# All price-sensitive columns across all views (for access control).
# Only these exact columns are RBAC-gated:
#   - Direct view:   ORG_NEGOTIATED_PRICE
#   - Indirect view: NET_PRICE
# Other price-shaped columns (LAST_SAP_PRICE, PRICE_AFTER_DISCOUNT,
# NEGOTIATED_PRICE, PRICE_UNIT, AMOUNT, VOUCHER_AMOUNT, ...) are NOT gated.
PRICE_SENSITIVE_COLUMNS = {
    "ORG_NEGOTIATED_PRICE",
    "NET_PRICE",
}
PRICE_SENSITIVE_KEYWORDS = {
    "net price",
    "net_price",
    "org negotiated price",
    "org_negotiated_price",
}

# Access control tables  live in Oracle (queried via the rbac_db_connection
# input). Spend tables live in SQL Server. _check_price_access reads these
# Oracle tables directly to fetch user entitlements, then
# _apply_commodity_access_filter injects SQL Server WHERE clauses on the spend
# query  no cross-DB JOIN.
ACCESS_USER_TABLE = "PISLOAD.EIS_ACCESS_USER_LIST"
ACCESS_COMMODITY_TABLE = "PISLOAD.EIS_USER_COMMODITY_LIST"
ACCESS_SUPP_CLASS_TABLE = "PISVIEW.EIS_USER_SUPPLIER_CLASSIFICATION_LIST"
ACCESS_SUPERUSER_TABLE = "PISLOAD.EIS_SUPERUSER_LIST"
# Commodity entitlements are stored as 4-character prefixes and must be
# matched against SUBSTRING(table.COMMODITY, 1, 4) on SQL Server.
COMMODITY_PREFIX_LEN = 4


# 
# COMPONENT
# 

class CodeEditorNode(Node):
    display_name = "Talk to Data Pipeline"
    description = "Enterprise NL-to-SQL pipeline: analyzes query, links schema, builds context, generates SQL, executes and formats results. Use this tool when the user asks a data question."
    icon = "database"
    name = "TalkToDataPipeline"

    inputs = [
        MessageTextInput(
            name="input_value",
            display_name="User Question",
            info="The natural language question about your data.",
            tool_mode=True,
        ),
        MessageTextInput(
            name="spend_type",
            display_name="Spend Type",
            info="Which view to query: 'direct', 'indirect', or 'auto' (auto-detect from query). Worker Node passes this.",
            value="auto",
            tool_mode=True,
        ),
        MessageTextInput(
            name="user_email",
            display_name="User Email",
            info="User's email for access control on price-sensitive columns. Leave empty to skip access check.",
            value="",
            tool_mode=True,
        ),
        HandleInput(
            name="knowledge_context",
            display_name="Knowledge Context",
            input_types=["Data"],
            info="From Knowledge Processor (unified knowledge dict  may be tagged by view).",
            required=False,
        ),
        HandleInput(
            name="db_connection",
            display_name="Spend DB Connection (SQL Server)",
            input_types=["Data"],
            info="SQL Server connector for the spend tables (DBO.VW_DIRECT_SPEND_ALL, DBO.VW_INDIRECT_SPEND_ALL).",
            required=True,
        ),
        HandleInput(
            name="rbac_db_connection",
            display_name="RBAC DB Connection (Oracle)",
            input_types=["Data"],
            info="Oracle connector for the access-control tables (PISLOAD.EIS_*, PISVIEW.EIS_*). Required when User Email is set.",
            required=False,
        ),
        HandleInput(
            name="llm",
            display_name="Language Model",
            input_types=["LanguageModel"],
            info="LLM for Schema Linking and SQL Generation.",
            required=True,
        ),
        IntInput(
            name="max_rows",
            display_name="Max Result Rows",
            value=100,
            info="Hard ceiling on rows. Any TOP N larger than this is capped.",
        ),
        IntInput(
            name="default_row_limit",
            display_name="Default Row Limit",
            value=10,
            info=(
                "Default TOP N applied when the user does not specify a limit "
                "(e.g. 'top suppliers in Germany' -> TOP 10). The LLM is also "
                "asked to honor this. If the user asks for 'top 25', that wins."
            ),
        ),
        IntInput(
            name="query_timeout",
            display_name="Query Timeout (seconds)",
            value=30000,
        ),
        MultilineInput(
            name="mandatory_filter",
            display_name="Mandatory Date Filter",
            value="INVOICE_DATE >= '2024-04-01'",
            info="Auto-injected WHERE clause. Leave empty to disable.",
        ),
        BoolInput(
            name="enable_templates",
            display_name="Enable Template Matching",
            value=True,
            info="Try SQL templates before calling LLM for generation.",
        ),
        DropdownInput(
            name="sql_dialect",
            display_name="SQL Dialect",
            options=["auto", "sqlserver", "postgresql"],
            value="sqlserver",
            info="Auto-detected from DB connector if set to auto.",
        ),
        MultilineInput(
            name="extra_rules",
            display_name="Extra SQL Rules",
            value="",
            info="Additional rules appended to the LLM prompt.",
        ),
        MultilineInput(
            name="price_filter_rule",
            display_name="Price Filter Rule",
            value="",
            info="Optional mandatory filter applied when NET_PRICE is used (e.g. SOURCE = 'M').",
        ),
        IntInput(
            name="max_examples",
            display_name="Max Few-Shot Examples",
            value=20,
            info=(
                "Top-K few-shot examples passed to the Stage 4 SQL generator. "
                "More examples = better accuracy; 20 is the recommended default "
                "— do not lower for latency reasons (accuracy first)."
            ),
        ),
        BoolInput(
            name="enable_retry",
            display_name="Enable SQL Retry (Judge LLM)",
            value=True,
            info="If SQL fails or returns 0 rows, a judge LLM analyzes the error and corrects the query.",
        ),
        IntInput(
            name="max_retries",
            display_name="Max Retry Attempts",
            value=2,
            info="Maximum number of retry attempts (1-3).",
        ),
        IntInput(
            name="max_value_hints",
            display_name="Max Column Value Hints",
            value=200,
            info="Max values shown per column in the LLM prompt. Higher = more accurate filters but more tokens.",
        ),
    ]

    outputs = [
        Output(display_name="Results", name="output", method="build_output"),
    ]

    # 
    # USER EMAIL RESOLUTION  auto-detect from logged-in user
    # 

    _cached_user_email = None

    def _resolve_user_email(self):
        """Look up the logged-in user's email via AgentCore's DatabaseService.
    
        Uses the platform's own connection pool (get_db_service) so that
        managed-identity auth, connection pooling, and URL sanitization
        are all handled correctly — instead of re-parsing DATABASE_URL
        with psycopg2 directly.
        """
        if self._cached_user_email is not None:
            return self._cached_user_email
    
        uid = getattr(self, "user_id", None)
        if not uid:
            return ""
    
        try:
            from uuid import UUID
            from sqlalchemy.orm import Session
            from sqlalchemy import create_engine
            from agentcore.services.deps import get_db_service
            from agentcore.services.database.models.user.model import User
    
            db_service = get_db_service()
            db_url = db_service.database_url
            # Strip async driver suffix — we need a sync engine here
            if "+asyncpg" in db_url:
                db_url = db_url.replace("+asyncpg", "")
            if "+psycopg" in db_url and "postgresql+psycopg" not in db_url:
                pass  # already sync-compatible
    
            engine = create_engine(db_url, pool_pre_ping=True, pool_size=2)
            with Session(engine) as session:
                row = session.get(User, UUID(str(uid)))
                email = row.email if row and row.email else ""
    
            self._cached_user_email = email
            self.log(f"user email auto-resolve passed: {email}")
            return email
    
        except Exception as e:
            try:
                self.log(f"user email auto-resolve failed: {e}")
            except Exception:
                pass
            return ""

    # 
    # MAIN ENTRY POINT
    # 

    def build_output(self) -> Message:
        raw_query = self.input_value or ""
        if raw_query.startswith('"') and raw_query.endswith('"'):
            raw_query = raw_query[1:-1]
        if not raw_query.strip():
            return Message(text="No query provided.")

        #  Resolve spend_type: explicit or auto-detect from query 
        spend_type = (self.spend_type or "auto").strip().lower()
        if spend_type not in ("direct", "indirect", "both"):
            spend_type = self._detect_spend_type(raw_query)

        #  Resolve user email: explicit input > auto-detect from session 
        user_email = (self.user_email or "").strip()
        if not user_email:
            user_email = self._resolve_user_email()

        #  Extract knowledge context (may be tagged by view) 
        kc = self.knowledge_context
        knowledge_all = {}
        if kc and kc != "":
            knowledge_all = kc.data if isinstance(kc, Data) else (kc if isinstance(kc, dict) else {})

        # Extract Spend DB connection (SQL Server)
        db = self.db_connection
        db_data = db.data if isinstance(db, Data) else (db if isinstance(db, dict) else {})
        if not db_data:
            return Message(text="Error: No spend database connection provided.")

        provider = self.sql_dialect if self.sql_dialect != "auto" else db_data.get("provider", "postgresql")
        schema_ddl = db_data.get("schema_ddl", "")

        # Extract RBAC DB connection (Oracle)  optional, required only when user_email is set
        rbac_db = getattr(self, "rbac_db_connection", None)
        rbac_db_data = (
            rbac_db.data if isinstance(rbac_db, Data)
            else (rbac_db if isinstance(rbac_db, dict) else {})
        )

        #  Determine which views to query
        if spend_type == "both":
            tables_to_query = ["direct", "indirect"]
        else:
            tables_to_query = [spend_type]

        def _resolve_knowledge(table_key):
            """Pick the right knowledge dict for a given branch. Mirrors the
            prior inline logic so parallel and sequential paths stay in sync."""
            if "direct" in knowledge_all and isinstance(knowledge_all.get("direct"), dict):
                k = knowledge_all.get(table_key, knowledge_all.get("direct", {}))
                if table_key == "indirect" and not knowledge_all.get("indirect"):
                    self._log_safe(
                        "[knowledge warning] indirect branch is running with "
                        "DIRECT knowledge because the indirect Knowledge Base "
                        "slot is empty — wire both KBs to the Knowledge "
                        "Processor or expect wrong SQL on the indirect branch."
                    )
                return k
            if table_key == "indirect":
                self._log_safe(
                    "[knowledge warning] indirect branch is running with "
                    "FLAT (untagged) knowledge — the Knowledge Processor "
                    "is likely in single-view mode. Connect the indirect "
                    "Knowledge Base to get correct indirect SQL."
                )
            return knowledge_all

        def _run_branch(table_key):
            """Run stages 1-5 for a single view. Returns (table_key, label, Message).

            Extracted so auto mode can run direct+indirect concurrently in a
            ThreadPoolExecutor — the previous sequential loop blew ~60s of
            wall time because DB and LLM latencies stacked across branches.
            """
            table_config = TABLE_REGISTRY.get(table_key, TABLE_REGISTRY["direct"])
            label = table_config.get("label", table_key)
            try:
                knowledge = _resolve_knowledge(table_key)
                table_schema_ddl = self._filter_schema_ddl(schema_ddl, table_config["table_name"])

                self.status = f"{label} — analyzing query…"
                stage1 = self._stage1_query_analyzer(raw_query, knowledge)
                stage1["spend_type"] = table_key
                stage1["table_config"] = table_config

                self.status = f"{label} — linking schema…"
                stage2 = self._stage2_schema_linker(stage1, knowledge, db_data)

                self.status = f"{label} — building context…"
                stage3 = self._stage3_context_builder(stage2, knowledge, provider, table_schema_ddl)

                try:
                    _ex_ids = [e.get("id") for e in stage3.get("selected_examples", [])
                               if isinstance(e, dict)]
                    self._log_safe(
                        f"[stage-trace] table={table_key} "
                        f"intent={stage1.get('primary_intent')!r} "
                        f"entities={sorted(stage2.get('detected_entities', []) or [])} "
                        f"examples={_ex_ids} "
                        f"aliases={sorted((stage2.get('resolved_aliases') or {}).keys())}"
                    )
                except Exception:
                    pass

                self.status = f"{label} — generating SQL…"
                stage4 = self._stage4_sql_generator(stage3, knowledge)

                self.status = f"{label} — executing query…"
                result_msg = self._stage5_sql_processor(
                    stage4, knowledge, db_data, provider, table_schema_ddl,
                    user_email=user_email, table_config=table_config,
                    rbac_db_data=rbac_db_data,
                )
                return (table_key, label, result_msg, None)
            except Exception as e:
                tb = traceback.format_exc()
                return (table_key, label, None, (e, tb))

        #  Run pipeline for each view — parallel when more than one branch
        all_results = []
        if len(tables_to_query) > 1:
            # Auto mode: fire both branches concurrently. Each branch has its
            # own LLM calls and its own DB query, so they do not share any
            # mutable state. The DB driver used here (pyodbc via Data object)
            # creates a new cursor per query, so concurrent calls are safe.
            # If a deployment ever has a stricter driver, flip the following
            # worker count to 1 — behavior falls back to sequential.
            from concurrent.futures import ThreadPoolExecutor
            self.status = f"auto mode — running {len(tables_to_query)} branches in parallel…"
            with ThreadPoolExecutor(max_workers=len(tables_to_query)) as pool:
                futures = [pool.submit(_run_branch, tk) for tk in tables_to_query]
                branch_out = [f.result() for f in futures]
            # Preserve tables_to_query order (direct first, then indirect) so
            # the combined summary section reads in a consistent order.
            branch_by_key = {bo[0]: bo for bo in branch_out}
            ordered = [branch_by_key[tk] for tk in tables_to_query if tk in branch_by_key]
        else:
            ordered = [_run_branch(tables_to_query[0])]

        for table_key, label, result_msg, err in ordered:
            if err is None:
                all_results.append((table_key, label, result_msg))
                continue
            # --- failure path: same user-facing payload as the prior version ---
            e, tb = err
            self._log_safe(
                f"[pipeline error] table={table_key} label={label!r} "
                f"error={type(e).__name__}: {e}\n{tb}"
            )
            error_text = (
                f"**Pipeline error ({label}):** "
                f"{type(e).__name__}: {e}\n\n"
                f"_This branch failed — the other branch(es) and combined "
                f"summary reflect only the successful view(s)._"
            )
            all_results.append((table_key, label, Message(text=error_text)))

        #  Combine results if multiple views 
        if len(all_results) == 1:
            return all_results[0][2]

        # Multiple views: wrap each with a header, then add combined summary
        combined_parts = []
        extracted_data = []  # (label, columns, rows) for each view

        for table_key, label, msg in all_results:
            combined_parts.append(f"### {label}\n")
            msg_text = msg.text if hasattr(msg, "text") else str(msg)
            # Strip per-view wrapper markers (we'll wrap the combined output instead)
            msg_text = msg_text.replace("[TOOL OUTPUT — COPY EVERYTHING BETWEEN THE MARKERS INTO YOUR RESPONSE]\n", "")
            msg_text = msg_text.replace("===BEGIN===\n", "").replace("\n===END===", "")
            msg_text = re.sub(r'\[Paste everything between.*?omit the HTML table\.\]', '', msg_text)
            combined_parts.append(msg_text.strip())
            combined_parts.append("")  # blank line separator

            # Extract data_json from the hidden comment for summation
            dj_match = re.search(r'<!-- data_json:({.*?}) -->', msg_text)
            if dj_match:
                try:
                    dj = json.loads(dj_match.group(1))
                    # Also extract SQL from the details block for LLM-based merge
                    sql_match = re.search(r'<pre[^>]*><code>(.*?)</code></pre>', msg_text, re.DOTALL)
                    view_sql = ""
                    if sql_match:
                        import html as _html
                        view_sql = _html.unescape(sql_match.group(1).strip())
                    extracted_data.append((label, dj.get("columns", []), dj.get("rows", []), view_sql))
                except (json.JSONDecodeError, ValueError):
                    pass

        #  Build combined summary if we have numeric data from multiple views
        if len(extracted_data) >= 2:
            self.status = "merging views into combined summary…"
            summary = self._build_combined_summary(extracted_data, raw_query)
            if summary:
                combined_parts.append(f"### Combined Summary\n")
                combined_parts.append(summary)

        full_output = "\n".join(combined_parts)
        wrapped = (
            "[TOOL OUTPUT — COPY EVERYTHING BETWEEN THE MARKERS INTO YOUR RESPONSE]\n"
            "===BEGIN===\n"
            f"{full_output}\n"
            "===END===\n"
            "[Paste everything between ===BEGIN=== and ===END=== into your response VERBATIM, "
            "then add one insight sentence. Do NOT summarize. Do NOT omit the HTML tables.]"
        )
        return Message(text=wrapped)

    # 
    # COMBINED SUMMARY  merge numeric results from multiple views
    # 

    def _log_safe(self, msg):
        """Emit msg to the host logger when available, else stderr.

        Callers in `auto` mode need a place to surface Indirect-branch errors
        without crashing the whole run. Using self.log directly would raise on
        test harnesses or plain-Python smoke tests that don't wire up the node;
        stderr fallback keeps the signal visible in those contexts too.
        """
        try:
            self.log(msg)
        except Exception:
            try:
                print(f"[talk_to_data] {msg}", file=sys.stderr, flush=True)
            except Exception:
                pass

    def _llm_invoke_silent(self, prompt):
        """Invoke self.llm while suppressing agent-level callback events.

        Without this, every internal LLM call inside this tool (Stage 2 schema
        linker, Stage 4 SQL generator, combined-summary merge, retry judge)
        emits an `add_message` SSE event. The Playground filters those out, but
        the Orchestration / MyBuddy chat page forwards them all to the UI —
        causing raw Stage 2 JSON and raw Stage 4 SQL to appear after the
        formatted tables.

        Passing config={"callbacks": []} tells any LangChain-style runnable to
        skip its callback handlers for just this call, so the SSE event is
        never produced. Older LLM wrappers that don't accept a RunnableConfig
        fall back to the plain invoke path (behavior unchanged for them).
        """
        try:
            return self.llm.invoke(prompt, config={"callbacks": []})
        except TypeError:
            return self.llm.invoke(prompt)

    @staticmethod
    def _fmt_money_str(val):
        """Format a numeric value with K/M suffix. Single source of truth used by
        both the per-branch HTML tables and the combined-summary markdown tables,
        so the same number renders identically in every surface.

        Accepts int, float, and Decimal (pyodbc returns SQL money columns as
        Decimal — without this the formatter silently skips them and numbers
        render as raw str(Decimal) like '30545446.861598')."""
        if val is None:
            return ""
        if isinstance(val, Decimal):
            val = float(val)
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            return str(val)
        v = abs(val)
        sign = "-" if val < 0 else ""
        if v >= 1_000_000:
            return f"{sign}{v / 1_000_000:,.1f}M"
        if v >= 1_000:
            return f"{sign}{v / 1_000:,.1f}K"
        if isinstance(val, float):
            return f"{val:,.2f}"
        return f"{val:,}"

    # Column names that are numeric but NOT money — IDs, counts, rates, dates.
    # Matched as substrings against upper-cased column names.
    _NON_MONEY_NAME_PATTERNS = (
        "QUANTITY", "QTY", "COUNT", "YEAR", "MONTH", "QUARTER", "WEEK_NO",
        "DAY_OF", "INVOICE_NO", "ORDER_NO", "ARTICLE_NO", "PLANT_NO",
        "SUPPLIER_NO", "INVOICE_POS_NO", "ORDER_POS_NO", "PURCHASING_AGENT_NO",
        "SUPP_DUNS_NO", "PARENT_DUNS", "EXCH_RATE", "MIGRATION_BATCH",
        "MIGRATED_AT", "ROW_NUMBER", "RANK_IN_", "PERCENT", "PCT", "RATIO",
    )
    # Column names that strongly imply money even with empty samples.
    _MONEY_NAME_PATTERNS = (
        "AMOUNT", "SPEND", "COST", "EXPENDITURE", "EXPENSE", "PRICE", "PAID",
        "REVENUE", "BUDGET", "VALUE", "PAYMENT", "VOUCHER", "GROSS_", "NET_",
        "TOTAL", "SUM_", "AVG_", "MIN_", "MAX_",
    )

    @classmethod
    def _detect_money_cols(cls, columns, rows, sample_size=20):
        """Return the set of column indices that should get K/M formatting.

        Rules (checked in order):
        1. If the column name matches a non-money pattern (QUANTITY, *_NO, COUNT,
           YEAR, RATE, PCT, etc.), it is NOT money — skip even if the values look
           big. Prevents SUPPLIER_NO or PO_QUANTITY from getting a "45.3K" suffix.
        2. If the column name matches a money pattern (AMOUNT, SPEND, PRICE,
           VALUE, TOTAL, ...), it IS money — skip the sample check so empty
           result sets still render with K/M on future rows.
        3. Otherwise, sample up to `sample_size` rows: if every non-null cell is
           numeric AND the max absolute value is >= 1000, treat as money.
        This closes the gap where NET_PRICE or aliases like avg_unit_price used
        to render as raw floats in per-branch tables while the combined summary
        LLM formatted them with K/M.
        """
        money_cols: set[int] = set()
        for i, col in enumerate(columns):
            col_upper = str(col).upper().replace(" ", "_").replace(".", "_")

            if any(p in col_upper for p in cls._NON_MONEY_NAME_PATTERNS):
                continue
            if any(p in col_upper for p in cls._MONEY_NAME_PATTERNS):
                money_cols.add(i)
                continue

            max_abs = 0.0
            has_numeric = False
            for row in rows[:sample_size]:
                if i >= len(row):
                    continue
                v = row[i]
                if isinstance(v, bool):
                    continue
                if isinstance(v, _NUMERIC_TYPES):
                    has_numeric = True
                    av = float(abs(v))
                    if av > max_abs:
                        max_abs = av
            if has_numeric and max_abs >= 1000:
                money_cols.add(i)
        return money_cols

    def _build_combined_summary(self, extracted_data, user_query=""):
        """Build a combined summary using LLM to intelligently merge results.

        extracted_data: list of (label, columns, rows, sql)

        The LLM understands the SQL queries and column semantics, so it can
        correctly merge PART_COUNT + ITEM_COUNT, handle different column
        counts, etc.  Falls back to heuristic merge if LLM is unavailable.
        """
        if not extracted_data or len(extracted_data) < 2:
            return ""

        # Try LLM-based merge first
        if self.llm and self.llm != "":
            try:
                llm_result = self._llm_combined_summary(extracted_data, user_query)
                if llm_result:
                    return llm_result
            except Exception:
                pass

        # Fallback: heuristic merge
        return self._heuristic_combined_summary(extracted_data)

    def _llm_combined_summary(self, extracted_data, user_query=""):
        """Ask LLM to build a combined summary from multiple view results."""
        # Build compact view descriptions. Money columns are pre-formatted with
        # K/M here so the LLM sees the same representation that the per-branch
        # HTML tables use — its job becomes copy-through, not reformat.
        view_sections = []
        for item in extracted_data:
            label, columns, rows = item[0], item[1], item[2]
            view_sql = item[3] if len(item) > 3 else ""
            money_cols = self._detect_money_cols(columns, rows)

            def _cell(v, is_money):
                if v is None:
                    return ""
                if isinstance(v, bool):
                    return str(v)
                if is_money and isinstance(v, _NUMERIC_TYPES):
                    return self._fmt_money_str(v)
                if isinstance(v, Decimal):
                    return f"{float(v):,.2f}"
                if isinstance(v, float):
                    return f"{v:,.2f}"
                if isinstance(v, int) and abs(v) >= 1000:
                    return f"{v:,}"
                return str(v)

            table_lines = [" | ".join(str(c) for c in columns)]
            for row in rows[:15]:  # cap at 15 rows
                table_lines.append(" | ".join(
                    _cell(v, i in money_cols) for i, v in enumerate(row)
                ))

            section = f"**{label}:**\n"
            if view_sql:
                section += f"SQL: {view_sql}\n"
            section += "Data:\n" + "\n".join(table_lines)
            view_sections.append(section)

        prompt = (
            "You are merging results from multiple database views into one Combined Summary.\n\n"
            f"User question: {user_query}\n\n"
            + "\n\n".join(view_sections) + "\n\n"
            "TASK: Create a combined markdown table that intelligently merges the above results.\n"
            "RULES:\n"
            "1. Match columns by MEANING not name (e.g. PART_COUNT and ITEM_COUNT are both counts)\n"
            "2. Sum numeric columns that represent the same metric across views\n"
            "3. Include a Breakdown column showing per-view contributions (e.g. 'Direct: 6,721, Indirect: 11')\n"
            "4. Add a GRAND TOTAL row at the bottom\n"
            "5. Sort by the primary spend/amount column descending\n"
            "6. Use | pipe markdown table format with bold totals\n"
            "7. Return ONLY the markdown table, no explanation\n"
            "8. Money/spend/amount values in the input are ALREADY K/M-formatted (e.g. '45.3M', '892.1K'). Preserve that formatting — do NOT re-scale or expand them. To SUM values, parse the K/M suffix back to a number, add, then re-format the sum with K/M. Non-money columns (counts, IDs, etc.) use plain commas.\n"
        )

        response = self._llm_invoke_silent(prompt)
        text = response.content if hasattr(response, "content") else str(response)
        text = text.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            text = re.sub(r'^```\w*\n?', '', text)
        if text.endswith("```"):
            text = text[:-3].rstrip()

        # Validate we got a markdown table
        if "|" not in text or text.count("\n") < 2:
            return None

        # Also try to emit data_json for the visualizer from the LLM table
        data_json_str = self._extract_data_json_from_markdown(text, extracted_data)
        if data_json_str:
            text += f"\n{data_json_str}"

        return text

    @staticmethod
    def _parse_km_number(s):
        """Parse a number string that may have K/M suffix (e.g. '45.3M' → 45300000)."""
        s = s.replace(",", "").replace("*", "").strip()
        if not s:
            return None
        suffix = s[-1].upper()
        if suffix == 'M':
            try:
                return float(s[:-1]) * 1_000_000
            except ValueError:
                return None
        if suffix == 'K':
            try:
                return float(s[:-1]) * 1_000
            except ValueError:
                return None
        try:
            return float(s)
        except ValueError:
            return None

    def _extract_data_json_from_markdown(self, md_table, extracted_data):
        """Parse LLM markdown table to emit a data_json for the visualizer."""
        try:
            lines = [l.strip() for l in md_table.strip().split("\n") if l.strip().startswith("|")]
            if len(lines) < 3:
                return None
            # Parse header
            headers = [c.strip().strip("*") for c in lines[0].split("|") if c.strip()]
            # Parse data rows (skip separator)
            chart_rows = []
            for line in lines[2:]:
                cells = [c.strip().strip("*") for c in line.split("|") if c.strip()]
                if not cells or "GRAND TOTAL" in cells[0].upper():
                    continue
                if len(cells) >= 2:
                    # First cell = dimension, find first numeric cell for value
                    dim = cells[0]
                    val = 0
                    for cell in cells[1:]:
                        parsed = self._parse_km_number(cell)
                        if parsed is not None:
                            val = parsed
                            break
                    chart_rows.append([dim, val])
            if chart_rows:
                primary_col = headers[1] if len(headers) > 1 else "Combined Total"
                obj = {
                    "columns": [headers[0], primary_col],
                    "rows": chart_rows,
                    "source": "Combined",
                }
                return f"<!-- data_json:{json.dumps(obj)} -->"
        except Exception:
            pass
        return None

    def _heuristic_combined_summary(self, extracted_data):
        """Fallback: merge results using heuristic position-based matching."""
        if not extracted_data or len(extracted_data) < 2:
            return ""

        # Pattern 1: Single-row aggregates
        all_single_row = all(len(item[2]) == 1 for item in extracted_data)
        if all_single_row:
            summary_lines = []
            combined_totals = {}

            for item in extracted_data:
                label, columns, rows = item[0], item[1], item[2]
                row = rows[0]
                for i, col in enumerate(columns):
                    val = row[i] if i < len(row) else None
                    if val is not None and isinstance(val, _NUMERIC_TYPES) and not isinstance(val, bool):
                        if isinstance(val, Decimal):
                            val = float(val)
                        col_upper = col.upper()
                        base_name = col_upper
                        for prefix in ("DISTINCT_", "TOTAL_", "AVG_", "SUM_"):
                            if base_name.startswith(prefix):
                                base_name = base_name[len(prefix):]
                        for suffix in ("_EUR", "_COUNT", "_TOTAL", "_SUM", "_AVG"):
                            if base_name.endswith(suffix):
                                base_name = base_name[:-len(suffix)]
                        if base_name not in combined_totals:
                            combined_totals[base_name] = {"label": col, "values": [], "table_labels": []}
                        combined_totals[base_name]["values"].append(val)
                        combined_totals[base_name]["table_labels"].append(label)

            if combined_totals:
                table_rows = []
                for base_name, info in combined_totals.items():
                    total = sum(info["values"])
                    breakdown_parts = []
                    for v, vl in zip(info["values"], info["table_labels"]):
                        breakdown_parts.append(f"{vl}: {self._fmt_money_str(v)}")
                    breakdown = " + ".join(breakdown_parts)
                    total_str = f"**{self._fmt_money_str(total)}**"
                    table_rows.append(f"| {info['label']} | {breakdown} | {total_str} |")
                if table_rows:
                    summary_lines.append("| Metric | Breakdown | Combined Total |")
                    summary_lines.append("|--------|-----------|----------------|")
                    summary_lines.extend(table_rows)
                    return "\n".join(summary_lines)

        # Pattern 2: Multi-row — position-based merge
        first_cols = [item[1][0].upper() if item[1] else "" for item in extracted_data]
        if len(set(first_cols)) == 1 and first_cols[0]:
            dim_name = extracted_data[0][1][0]
            table_labels = []
            numeric_positions = {}
            for item in extracted_data:
                label, columns, rows = item[0], item[1], item[2]
                table_labels.append(label)
                for i, col in enumerate(columns[1:], 1):
                    has_numeric = any(isinstance(row[i], _NUMERIC_TYPES) and not isinstance(row[i], bool) for row in rows if i < len(row))
                    if has_numeric:
                        if i not in numeric_positions:
                            numeric_positions[i] = []
                        numeric_positions[i].append((label, col))

            if numeric_positions:
                pos_list = sorted(numeric_positions.keys())
                merged = {}
                for item in extracted_data:
                    label, columns, rows = item[0], item[1], item[2]
                    for row in rows:
                        dim_val = str(row[0]) if row else ""
                        if dim_val not in merged:
                            merged[dim_val] = {}
                        if label not in merged[dim_val]:
                            merged[dim_val][label] = {}
                        for pos in pos_list:
                            val = row[pos] if pos < len(row) else None
                            if isinstance(val, _NUMERIC_TYPES) and not isinstance(val, bool):
                                if isinstance(val, Decimal):
                                    val = float(val)
                                merged[dim_val][label][pos] = merged[dim_val][label].get(pos, 0) + val

                primary_pos = pos_list[-1]
                primary_col = numeric_positions[primary_pos][0][1]

                combined_rows = []
                for dim_val, view_data in merged.items():
                    total = 0
                    parts = []
                    for vl in table_labels:
                        v = view_data.get(vl, {}).get(primary_pos, 0)
                        total += v
                        if v:
                            parts.append(f"{vl}: {self._fmt_money_str(v)}")
                    combined_rows.append((dim_val, total, parts))
                combined_rows.sort(key=lambda x: x[1], reverse=True)

                summary_lines = []
                summary_lines.append(f"| {dim_name} | Combined {primary_col} | Breakdown |")
                summary_lines.append("|-----------|----------------------|-----------|")
                for dim_val, total, parts in combined_rows[:10]:
                    total_str = f"**{self._fmt_money_str(total)}**"
                    summary_lines.append(f"| {dim_val} | {total_str} | {', '.join(parts)} |")

                grand_total = sum(t for _, t, _ in combined_rows)
                gt_str = f"**{self._fmt_money_str(grand_total)}**"
                summary_lines.append(f"| **GRAND TOTAL** | {gt_str} | |")

                combined_chart_rows = [[dv, t] for dv, t, _ in combined_rows]
                combined_chart_obj = {
                    "columns": [dim_name, f"Combined {primary_col}"],
                    "rows": combined_chart_rows,
                    "source": "Combined",
                }
                summary_lines.append(f"\n<!-- data_json:{json.dumps(combined_chart_obj)} -->")
                return "\n".join(summary_lines)

        #  Pattern 3: Incompatible  just show grand totals per view
        total_lines = []
        grand_total = 0
        has_numeric = False
        for item in extracted_data:
            label, columns, rows = item[0], item[1], item[2]
            view_total = 0
            for row in rows:
                for i, col in enumerate(columns):
                    val = row[i] if i < len(row) else None
                    if isinstance(val, _NUMERIC_TYPES) and not isinstance(val, bool) and any(
                        kw in col.upper() for kw in ("SPEND", "AMOUNT", "TOTAL", "SUM", "PRICE", "COUNT")
                    ):
                        view_total += float(val) if isinstance(val, Decimal) else val
                        has_numeric = True
            if has_numeric:
                total_lines.append(f"- **{label}:** {self._fmt_money_str(view_total)}")
                grand_total += view_total

        if total_lines:
            total_lines.append(f"- **Combined Total:** **{self._fmt_money_str(grand_total)}**")
            return "\n".join(total_lines)

        return ""

    # 
    # VIEW DETECTION  auto-detect spend type from query keywords
    # 

    # Spend-type synonym phrases — compiled as word-boundary regex patterns.
    _DIRECT_PATTERNS = [re.compile(r'\b' + re.escape(p) + r'\b') for p in [
        "direct spend", "direct material", "production material",
        "direct procurement", "production spend", "direct cost",
        "direct purchase",
    ]]
    _INDIRECT_PATTERNS = [re.compile(r'\b' + re.escape(p) + r'\b') for p in [
        "indirect spend", "indirect material", "services spend",
        "mro spend", "mro", "non-production", "overhead spend",
        "overhead", "indirect procurement", "indirect cost",
        "indirect purchase", "services cost",
    ]]

    def _detect_spend_type(self, query):
        """Detect 'direct', 'indirect', or 'both' from keywords in the query.

        Checks expanded synonym lists first (word-boundary regex), then falls
        back to bare 'direct' / 'indirect' word matching.
        """
        q = query.lower()

        # Phrase-level matching with word boundaries
        has_direct = any(p.search(q) for p in self._DIRECT_PATTERNS)
        has_indirect = any(p.search(q) for p in self._INDIRECT_PATTERNS)

        # Also check bare word matching (covers "direct" / "indirect" alone)
        if not has_direct:
            has_direct = bool(re.search(r'\bdirect\b', q))
        if not has_indirect:
            has_indirect = bool(re.search(r'\bindirect\b', q))

        if has_indirect and not has_direct:
            return "indirect"
        if has_direct and not has_indirect:
            return "direct"
        # Both mentioned or neither → query both views
        return "both"

    def _filter_schema_ddl(self, full_ddl, target_table_name):
        """If DDL contains multiple TABLE/VIEW blocks, extract only the target view."""
        if not full_ddl or not target_table_name:
            return full_ddl
        # Extract just the view name without schema prefix for matching
        short_name = target_table_name.split(".")[-1].upper()
        # If DDL only has one table or already matches, return as-is
        if full_ddl.upper().count("TABLE ") <= 1:
            return full_ddl
        # Try to extract the matching TABLE block
        pattern = re.compile(
            r'(TABLE\s+(?:\w+\.)?' + re.escape(short_name) + r'\s*\(.*?\))',
            re.IGNORECASE | re.DOTALL,
        )
        match = pattern.search(full_ddl)
        if match:
            return match.group(1)
        return full_ddl  # fallback: return all

    # 
    # STAGE 1: Query Analyzer (CODE)
    # 

    def _stage1_query_analyzer(self, raw, knowledge):
        text = raw.strip()
        expansions = []
        for abbr, full in ABBREVIATIONS.items():
            pat = re.compile(r"\b" + re.escape(abbr) + r"\b", re.IGNORECASE)
            if pat.search(text):
                text = pat.sub(full, text)
                expansions.append(f"{abbr} -> {full}")

        alias_resolutions = []
        entity_aliases = knowledge.get("entity_aliases", {})
        if entity_aliases:
            text_lower = text.lower()
            for alias in sorted(entity_aliases.keys(), key=len, reverse=True):
                alias_lower = alias.lower().strip()
                # Skip common English stopwords that cause false positives
                if alias_lower in ALIAS_STOPWORDS:
                    continue
                # Skip very short aliases unless they are uppercase acronyms (US, UK, EU)
                if len(alias_lower) < MIN_ALIAS_LENGTH and alias != alias.upper():
                    continue
                # Word-boundary match to avoid false positives (e.g. "esp" matching "responsible")
                if re.search(r"\b" + re.escape(alias) + r"\b", text_lower):
                    info = entity_aliases[alias]
                    alias_resolutions.append({
                        "alias": alias,
                        "canonical_value": info.get("canonical_value", ""),
                        "sql_filter": info.get("sql_filter", ""),
                    })

        extracted_numbers = [int(m) for m in re.findall(r"\b(\d+)\b", text)]

        #  Fiscal year resolution  convert FY references to concrete date ranges
        # Supported formats:
        #   - FY25, FY'25, FY 25, FY2025                              (simple)
        #   - FY24/25, FY 2024/25, FY2024/2025                        (slash with FY prefix)
        #   - 2025-26, 2025/26, 2025-2026                             (bare YYYY-YY/YYYY)
        #   - FY 2025-26, fiscal year 2025-26                         (prefix + YYYY-YY)
        # Convention: the ending year names the FY. FY25 = Apr 2024-Mar 2025.
        # "2025-26" must have end = start+1 (e.g. 2025-26 or 2025-2026) to match
        # — this prevents false positives on random hyphenated numbers/dates.
        # Supports MULTIPLE FY references: "FY25 vs FY26" → range spans both
        fy_resolved = None
        _fy_all = []

        # Claimed character positions — later patterns skip any position
        # already owned by an earlier, more-specific match.
        _claimed_spans: set = set()

        # 1. FY-prefixed slash format: FY24/25, FY 2024/2025
        for _m in re.finditer(r"\bFY\s*'?(\d{2,4})\s*/\s*(\d{2,4})\b", text, re.IGNORECASE):
            _s = int(_m.group(1))
            _e = int(_m.group(2))
            if _s < 100: _s = 2000 + _s
            if _e < 100: _e = 2000 + _e
            _fy_all.append({"original": _m.group(0), "fy_year": _e,
                            "start_date": f"{_s}-04-01", "end_date": f"{_e}-03-31"})
            _claimed_spans.update(range(_m.start(), _m.end()))

        # 2. Bare range format: "2025-26", "2025/26", "2025-2026". Accepts with
        # or without a preceding "FY" / "fiscal year". Requires end = start + 1
        # so it doesn't false-match random hyphenated numbers or ISO dates.
        # This is the pattern users most commonly type ("last fiscal year i.e.
        # 2025-26") — without it the resolver falls to `None` and the SQL LLM
        # picks its own (often wrong) year range.
        for _m in re.finditer(r"\b(\d{4})\s*[-/]\s*(\d{2,4})\b", text):
            if any(p in _claimed_spans for p in range(_m.start(), _m.end())):
                continue
            _s = int(_m.group(1))
            _e_raw = int(_m.group(2))
            # Anchor to the modern fiscal-year window so 1999-00 or 2050-51
            # don't accidentally resolve.
            if _s < 2000 or _s > 2049:
                continue
            if _e_raw < 100:
                if _e_raw != (_s + 1) % 100:
                    continue
                _e = _s + 1
            else:
                if _e_raw != _s + 1:
                    continue
                _e = _e_raw
            _fy_all.append({"original": _m.group(0), "fy_year": _e,
                            "start_date": f"{_s}-04-01", "end_date": f"{_e}-03-31"})
            _claimed_spans.update(range(_m.start(), _m.end()))

        # 3. Simple FY: FY25, FY'25, FY 2025 — skip any overlap with #1 or #2
        # (e.g. in "FY 2025-26" the bare-range match already claimed "2025-26",
        # so the simple pattern doesn't double-book it as FY2025 = FY25).
        for _m in re.finditer(r"\bFY\s*'?(\d{2,4})\b", text, re.IGNORECASE):
            if any(p in _claimed_spans for p in range(_m.start(), _m.end())):
                continue
            _n = int(_m.group(1))
            if _n < 100: _n = 2000 + _n
            _fy_all.append({"original": _m.group(0), "fy_year": _n,
                            "start_date": f"{_n - 1}-04-01", "end_date": f"{_n}-03-31"})
            _claimed_spans.update(range(_m.start(), _m.end()))

        if _fy_all:
            # Use the widest range across all mentioned FYs
            _earliest = min(f["start_date"] for f in _fy_all)
            _latest = max(f["end_date"] for f in _fy_all)
            _originals = " & ".join(f["original"] for f in _fy_all)
            _fy_years = [f["fy_year"] for f in _fy_all]
            fy_resolved = {
                "original": _originals,
                "fy_year": _fy_years[0] if len(_fy_years) == 1 else _fy_years,
                "start_date": _earliest,
                "end_date": _latest,
                "sql_filter": f"INVOICE_DATE >= '{_earliest}' AND INVOICE_DATE <= '{_latest}'",
                "multi": len(_fy_all) > 1,
            }
        elif re.search(r'\b(current|this)\s+fiscal\s+year\b', text, re.IGNORECASE):
            from datetime import date as _d
            _t = _d.today()
            # Current FY: if month >= Apr, FY ends next Mar; else FY ends this Mar
            _start_yr = _t.year if _t.month >= 4 else _t.year - 1
            fy_start = f"{_start_yr}-04-01"
            fy_end = f"{_start_yr + 1}-03-31"
            fy_resolved = {
                "original": "current fiscal year",
                "fy_year": _start_yr + 1,  # FY label = ending year (FY27 = Apr 2026-Mar 2027)
                "start_date": fy_start,
                "end_date": fy_end,
                "sql_filter": f"INVOICE_DATE >= '{fy_start}' AND INVOICE_DATE <= '{fy_end}'",
            }
        elif re.search(r'\b(last|previous)\s+fiscal\s+year\b', text, re.IGNORECASE):
            from datetime import date as _d
            _t = _d.today()
            _start_yr = (_t.year if _t.month >= 4 else _t.year - 1) - 1
            fy_start = f"{_start_yr}-04-01"
            fy_end = f"{_start_yr + 1}-03-31"
            fy_resolved = {
                "original": "last fiscal year",
                "fy_year": _start_yr + 1,  # FY label = ending year
                "start_date": fy_start,
                "end_date": fy_end,
                "sql_filter": f"INVOICE_DATE >= '{fy_start}' AND INVOICE_DATE <= '{fy_end}'",
            }

        cleaned = FILLER_RE.sub("", text)
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
        if len(cleaned) < 3:
            cleaned = text

        # Intent classification
        query_lower = cleaned.lower()
        query_tokens = {w.strip(".,?!'\"") for w in query_lower.split() if len(w.strip(".,?!'\"")) > 1}

        scores = []
        matched_phrases = []

        intent_index = knowledge.get("intent_index", {})
        for intent_name, intent_data in intent_index.items():
            overlap = len(query_tokens & intent_data.get("tokens", set()))
            if overlap > 0:
                score = overlap / max(len(intent_data.get("tokens", set())), 1)
                scores.append([intent_name, score])

        for intent_name, patterns in INTENT_PATTERNS.items():
            for pat in patterns:
                m = re.search(pat, query_lower)
                if m:
                    matched_phrases.append(f"{intent_name}: {m.group()}")
                    found = False
                    for i, (name, score) in enumerate(scores):
                        if name == intent_name:
                            scores[i][1] = min(score + 0.3, 1.0)
                            found = True
                            break
                    if not found:
                        scores.append([intent_name, 0.4])
                    break

        for i, (name, score) in enumerate(scores):
            scores[i][1] = score + SPECIFICITY.get(name, 0) * 0.03
        scores.sort(key=lambda x: x[1], reverse=True)

        if scores:
            confidence = round(min(scores[0][1], 1.0), 3)
            intent = {
                "primary_intent": scores[0][0],
                "secondary_intents": [s[0] for s in scores[1:4]],
                "confidence": confidence,
                "confidence_level": "high" if confidence >= 0.6 else ("medium" if confidence >= 0.3 else "low"),
                "matched_phrases": matched_phrases,
            }
        else:
            intent = {"primary_intent": "unknown", "secondary_intents": [], "confidence": 0.0, "confidence_level": "low", "matched_phrases": []}

        result = {
            "raw_query": raw,
            "normalized_query": cleaned,
            "normalizer": {
                "expansions": expansions,
                "alias_resolutions": alias_resolutions,
                "extracted_numbers": extracted_numbers,
            },
            "intent": intent,
        }
        if fy_resolved:
            result["fy_resolved"] = fy_resolved
        return result

    # 
    # STAGE 2: Schema Linker (LLM)
    # 

    def _stage2_schema_linker(self, ctx, knowledge, db_data):
        normalized_query = ctx.get("normalized_query", ctx.get("raw_query", ""))
        alias_resolutions = ctx.get("normalizer", {}).get("alias_resolutions", [])

        synonym_map = knowledge.get("synonym_map", {})
        entities = knowledge.get("entities", {})
        col_hints = knowledge.get("column_value_hints", {})

        syn_lines = [f'  "{t}" -> {info.get("column", "?")}' for t, info in list(synonym_map.items())[:100]]
        ent_lines = [f"  {n}: PK={info.get('primary_key','?')}, Display={info.get('display_column','?')}" for n, info in entities.items()]
        hint_lines = [f"  {col}: {', '.join(str(v) for v in h.get('examples',[])[:8])}" for col, h in col_hints.items() if h.get("examples")]

        alias_section = ""
        if alias_resolutions:
            alias_section = "\nALIAS RESOLUTIONS:\n" + "\n".join(f"  {a['alias']} -> {a['sql_filter']}" for a in alias_resolutions)

        prompt = f"""You are a schema linking agent. Resolve natural language terms to database column names.

SYNONYM MAP:
{chr(10).join(syn_lines) if syn_lines else '  (none)'}

ENTITIES:
{chr(10).join(ent_lines) if ent_lines else '  (none)'}

COLUMN VALUE EXAMPLES:
{chr(10).join(hint_lines) if hint_lines else '  (none)'}
{alias_section}

User query: "{normalized_query}"

CRITICAL RULES for suggested_filters:
- ONLY suggest filters that the user EXPLICITLY mentioned in their query.
- Do NOT invent filters based on column names or categories (e.g. SOURCE, REGION, COUNTRY) unless the user specifically asked to filter by them.
- Treat "direct", "indirect", "direct procurement", and "indirect procurement" as view-routing hints, not SQL filter values.
- Do NOT add date range filters  date filtering is handled separately by the pipeline.
- Do NOT map fiscal year (FY24, FY25, etc.) to G_JAHR  G_JAHR is a calendar year column, NOT a fiscal year column. Leave fiscal year resolution to the pipeline.
- If the user says "top 10 suppliers by spend", avoid extra dimensional filters (SOURCE, REGION, COUNTRY). Data-quality filters from business rules are applied by pipeline code.
- When in doubt, return an EMPTY suggested_filters list. Fewer filters is better than phantom filters.

Respond with JSON:
{{"resolved_columns": {{}}, "detected_entities": [], "suggested_groupby": [], "suggested_filters": [], "suggested_orderby": null, "suggested_limit": null}}

Return ONLY JSON."""

        schema_linking = {}
        try:
            response = self._llm_invoke_silent(prompt)
            text = response.content if hasattr(response, "content") else str(response)
            text = text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            schema_linking = json.loads(text.strip())
        except Exception as e:
            schema_linking = {"error": str(e)}

        # Sanitize suggested_filters  convert dicts to SQL strings, drop garbage
        raw_filters = schema_linking.get("suggested_filters", [])
        clean_filters = []
        for f in raw_filters:
            if isinstance(f, str) and f.strip():
                clean_filters.append(f.strip())
            elif isinstance(f, dict):
                # Convert {"column": "X", "operator": "=", "value": "Y"} to SQL
                col = f.get("column", "")
                op = f.get("operator", "=")
                val = f.get("value", "")
                if col and val and not any(placeholder in str(val).upper() for placeholder in ("LAST_", "CURRENT_", "NEXT_", "{", "}")):
                    if isinstance(val, str) and not val.replace(".", "").replace("-", "").isdigit():
                        clean_filters.append(f"{col} {op} '{val}'")
                    else:
                        clean_filters.append(f"{col} {op} {val}")
                # Skip filters with placeholder values like LAST_2_YEARS  the temporal context handles these
        schema_linking["suggested_filters"] = clean_filters

        # Inject alias-resolved filters
        if alias_resolutions:
            for a in alias_resolutions:
                sf = a.get("sql_filter", "")
                if sf and sf not in clean_filters:
                    clean_filters.append(sf)
            schema_linking["suggested_filters"] = clean_filters

        return {**ctx, "schema_linking": schema_linking}

    # 
    # STAGE 3: Context Builder (CODE)
    # 

    def _stage3_context_builder(self, ctx, knowledge, provider, schema_ddl):
        schema_linking = ctx.get("schema_linking", {})
        intent_result = ctx.get("intent", {})
        normalized_query = ctx.get("normalized_query", ctx.get("raw_query", ""))
        query_lower = normalized_query.lower()
        mr = self.max_rows

        # Compute resolved columns
        resolved_cols = set()
        for col in schema_linking.get("resolved_columns", {}).values():
            # LLM may return col as dict ({"column": "X", "table": "Y"}) or string
            if isinstance(col, dict):
                col = col.get("column", col.get("name", ""))
            if col:
                resolved_cols.add(str(col).upper())
        entities = knowledge.get("entities", {})
        for ent_name in schema_linking.get("detected_entities", []):
            # LLM may return entities as dicts ({"name": "Supplier"}) instead of strings
            if isinstance(ent_name, dict):
                ent_name = ent_name.get("name", ent_name.get("entity", ""))
            if not isinstance(ent_name, str) or not ent_name:
                continue
            for col in entities.get(ent_name, {}).get("columns", []):
                resolved_cols.add(str(col).upper())

        # Smart context filtering
        fk = knowledge
        filtered = {}

        cm = fk.get("column_metadata", {})
        filtered["column_metadata"] = {k: v for k, v in cm.items() if k.upper() in resolved_cols} if resolved_cols else cm

        ch = fk.get("column_value_hints", {})
        filtered["column_value_hints"] = {k: v for k, v in ch.items() if k.upper() in resolved_cols} if resolved_cols else ch

        rules = fk.get("business_rules", {})
        fr = {"sqlserver_syntax": rules.get("sqlserver_syntax", {})}
        if any(t in query_lower for t in ("total", "sum", "average", "avg", "count", "spend", "cost", "price", "kpi")):
            fr["metrics"] = rules.get("metrics", {})
            fr["exclusion_rules"] = rules.get("exclusion_rules", {})
        if any(t in query_lower for t in ("year", "month", "quarter", "week", "day", "ytd", "trend", "fiscal", "date")):
            fr["time_filters"] = rules.get("time_filters", {})
        if any(t in query_lower for t in ("type", "category", "class", "material", "oem", "abc")):
            fr["classification_rules"] = rules.get("classification_rules", {})
        filtered["business_rules"] = fr

        hierarchies = fk.get("hierarchies", {})
        if resolved_cols:
            filtered["hierarchies"] = {
                n: info for n, info in hierarchies.items()
                if {l.get("column", "").upper() for l in info.get("levels", [])} & resolved_cols
            }
        else:
            filtered["hierarchies"] = hierarchies

        filtered["additional_domain_context"] = fk.get("additional_domain_context", "")
        filtered["additional_business_rules"] = fk.get("additional_business_rules", "")

        # Example selection
        all_examples = fk.get("examples", [])
        primary_intent = intent_result.get("primary_intent", "")
        detected_ents = schema_linking.get("detected_entities", [])
        # Normalize: LLM may return entities as dicts or strings
        entity_set = set()
        for e in detected_ents:
            if isinstance(e, dict):
                e = e.get("name", e.get("entity", ""))
            if isinstance(e, str) and e:
                entity_set.add(e.lower())

        _STOP = {"the","a","an","by","for","in","of","to","and","or","from","with","is","are","me","my","show","what","how","all","total","give"}
        query_tokens = {w.strip(".,?!'\"") for w in query_lower.split() if len(w.strip(".,?!'\"")) > 1} - _STOP
        scored = []
        for ex in all_examples:
            score = 0.0
            ql = (ex.get("question") or ex.get("input", "")).lower()
            sl = (ex.get("sql") or ex.get("output", "")).lower()
            tags = {t.lower() for t in ex.get("tags", [])}
            cat = (ex.get("category", "") or ex.get("intent", "") or "").lower()
            # Intent match (category or tag or substring)
            if primary_intent:
                pi = primary_intent.lower()
                if pi == cat:
                    score += 4
                elif pi in tags:
                    score += 3
                elif pi in ql:
                    score += 1
            # Entity overlap
            ex_entities = {e.lower() for e in ex.get("entities", []) if isinstance(e, str) and e}
            for entity in entity_set:
                if entity in ex_entities:
                    score += 3.5  # strongest: exact entity list match
                elif entity in ql:
                    score += 2.5  # entity substring in question
                elif entity in sl:
                    score += 1.5  # entity substring in SQL
            # Keyword overlap between query and example question
            ex_tokens = {w.strip(".,?!'\"") for w in ql.split() if len(w.strip(".,?!'\"")) > 1} - _STOP
            overlap = len(query_tokens & ex_tokens)
            score += min(overlap * 0.5, 3.0)
            # Complexity tiebreaker
            score += (5 - min(ex.get("complexity", 2), 5)) * 0.05
            scored.append((score, ex))
        scored.sort(key=lambda x: x[0], reverse=True)
        selected_examples = [ex for _, ex in scored[:self.max_examples]]

        # Assemble prompt
        table_cfg = ctx.get("table_config", TABLE_REGISTRY.get("direct", {}))
        table_name = table_cfg.get("table_name", "")
        spend_type = ctx.get("spend_type", "direct")

        sections = [
            f"You are an expert SQL analyst for {provider.upper()} databases. "
            "Generate a precise SQL query for the question below.",
            f"\n**Target Table:** {table_name} ({TABLE_REGISTRY.get(spend_type, {}).get('label', spend_type)} data)",
            f"\n**Database Schema:**\n```sql\n{schema_ddl}\n```",
        ]

        resolved = schema_linking.get("resolved_columns", {})
        if resolved:
            lines = [f'  "{t}" -> {c}' for t, c in resolved.items()]
            sections.append("\n**Resolved Columns:**\n" + "\n".join(lines))
        sug_f = schema_linking.get("suggested_filters", [])
        if sug_f:
            sections.append("Suggested filters: " + ", ".join(str(f) for f in sug_f))
        sug_g = schema_linking.get("suggested_groupby", [])
        if sug_g:
            sections.append("Suggested GROUP BY: " + ", ".join(str(g) for g in sug_g))

        if intent_result.get("primary_intent", "unknown") != "unknown":
            sections.append(f"\n**Intent:** {intent_result['primary_intent']} (confidence: {intent_result.get('confidence', 0)})")

        cd_lines = [f"  {c}: {info.get('description', '')}" for c, info in filtered["column_metadata"].items() if info.get("description")]
        if cd_lines:
            sections.append("\n**Column Descriptions:**\n" + "\n".join(cd_lines))

        cvh = filtered["column_value_hints"]
        if cvh:
            h_lines = []
            for c, h in cvh.items():
                examples = h.get("examples", [])
                if not examples:
                    continue
                card = h.get("cardinality", "?")
                # Show all values for low-cardinality columns, top N for high-cardinality (configurable)
                mvh = self.max_value_hints
                limit = len(examples) if (isinstance(card, int) and card <= mvh) else mvh
                vals = ", ".join(str(v) for v in examples[:limit])
                tag = " [COMPLETE]" if (isinstance(card, int) and card <= 200 and len(examples) >= card) else ""
                h_lines.append(f"  {c} ({card}{tag}): {vals}")
            if h_lines:
                sections.append("\n**Column Values:**\n" + "\n".join(h_lines))

        metrics = filtered["business_rules"].get("metrics", {})
        if metrics:
            metric_lines = []
            for n, e in list(metrics.items())[:15]:
                if isinstance(e, dict):
                    formula = self._sanitize_prompt_text(e.get("formula", ""))
                    mf = self._sanitize_prompt_text(e.get("filter", ""))
                    metric_lines.append(f"  {n}: {formula}" + (f" | filter: {mf}" if mf else ""))
                else:
                    metric_lines.append(f"  {n}: {self._sanitize_prompt_text(e)}")
            sections.append("\n**KPI Definitions:**\n" + "\n".join(metric_lines))

        excl = filtered["business_rules"].get("exclusion_rules", [])
        if excl:
            if isinstance(excl, dict):
                lines = [f"  - {k}: {v}" for k, v in list(excl.items())[:10]]
            else:
                lines = [f"  - {r}" for r in excl[:10]]
            sections.append("\n**Exclusion Rules:**\n" + "\n".join(lines))

        tfilters = filtered["business_rules"].get("time_filters", {})
        if tfilters:
            tf_lines = []
            for n, e in list(tfilters.items())[:10]:
                if isinstance(e, dict):
                    fexpr = e.get("filter") or e.get("filter_template") or ""
                    tf_lines.append(f"  {n}: {fexpr}")
                else:
                    tf_lines.append(f"  {n}: {e}")
            sections.append("\n**Time Filters:**\n" + "\n".join(tf_lines))

        osyn = filtered["business_rules"].get("sqlserver_syntax", {})
        if osyn:
            os_lines = []
            for n, e in list(osyn.items())[:15]:
                if isinstance(e, dict):
                    txt = e.get("rule") or e.get("description") or e.get("syntax") or ""
                    os_lines.append(f"  {n}: {self._sanitize_prompt_text(txt)}")
                else:
                    os_lines.append(f"  {n}: {self._sanitize_prompt_text(e)}")
            if os_lines:
                sections.append("\n**SQL Server Syntax Rules:**\n" + "\n".join(os_lines))

        crules = filtered["business_rules"].get("classification_rules", {})
        if crules:
            cr_lines = []
            for n, e in list(crules.items())[:15]:
                if isinstance(e, dict):
                    txt = e.get("rule") or e.get("description") or e.get("filter") or ""
                    cr_lines.append(f"  {n}: {self._sanitize_prompt_text(txt)}")
                else:
                    cr_lines.append(f"  {n}: {self._sanitize_prompt_text(e)}")
            if cr_lines:
                sections.append("\n**Classification Rules:**\n" + "\n".join(cr_lines))

        adc = filtered.get("additional_domain_context", "")
        if adc:
            sections.append(f"\n**Domain Context:**\n{adc}")

        abr = filtered.get("additional_business_rules", "")
        if abr:
            sections.append(f"\n**Additional Business Rules:**\n{self._sanitize_prompt_text(abr)}")

        if selected_examples:
            ex_lines = []
            for ex in selected_examples:
                q = ex.get("question") or ex.get("input", "")
                s = self._sanitize_prompt_text(ex.get("sql") or ex.get("output", ""))
                if q and s:
                    ex_lines.append(f"Q: {q}\nSQL: {s}")
            if ex_lines:
                sections.append("\n**Examples:**\n" + "\n\n".join(ex_lines))

        # Temporal context  give LLM today's date and fiscal year boundaries
        from datetime import date as _date
        _today = _date.today()
        if _today.month >= 4:
            _fy_start = _date(_today.year, 4, 1)
            _fy_end = _date(_today.year + 1, 3, 31)
            _prev_fy_start = _date(_today.year - 1, 4, 1)
            _prev_fy_end = _date(_today.year, 3, 31)
        else:
            _fy_start = _date(_today.year - 1, 4, 1)
            _fy_end = _date(_today.year, 3, 31)
            _prev_fy_start = _date(_today.year - 2, 4, 1)
            _prev_fy_end = _date(_today.year - 1, 3, 31)
        _q = (_today.month - 4) % 12 // 3 + 1  # fiscal quarter 1-4
        _fq_month = 4 + (_q - 1) * 3
        _fq_year = _today.year if _fq_month >= 4 else _today.year
        if _today.month < 4:
            _fq_year = _today.year - 1
            _fq_month = 4 + (_q - 1) * 3
        _fq_start = _date(_fq_year if _fq_month <= 12 else _fq_year + 1, ((_fq_month - 1) % 12) + 1, 1)
        _fq_end_month = _fq_start.month + 2
        _fq_end_year = _fq_start.year
        if _fq_end_month > 12:
            _fq_end_month -= 12
            _fq_end_year += 1
        import calendar
        _fq_end = _date(_fq_end_year, _fq_end_month, calendar.monthrange(_fq_end_year, _fq_end_month)[1])

        temporal_ctx = (
            f"\n**Temporal Context:**\n"
            f"  Today: {_today.isoformat()}\n"
            f"  Current year: {_today.year}\n"
            f"  Fiscal year: April 1 to March 31\n"
            f"  Current FY: '{_fy_start}' to '{_fy_end}'\n"
            f"  Previous FY: '{_prev_fy_start}' to '{_prev_fy_end}'\n"
            f"  Current fiscal quarter (Q{_q}): '{_fq_start}' to '{_fq_end}'\n"
            f"  Date column: INVOICE_DATE\n"
            f"  NOTE: G_JAHR is a CALENDAR year column (VARCHAR 'YYYY'), NOT a fiscal year column. Do NOT use G_JAHR for fiscal year filtering  always use INVOICE_DATE date ranges instead.\n"
            f"\n  CALENDAR YEAR vs FISCAL YEAR (CRITICAL):\n"
            f"    - 'in 2025' / 'for 2025' / 'year 2025' = CALENDAR year  YEAR(INVOICE_DATE) = 2025\n"
            f"    - 'current year' / 'this year' = CALENDAR year  YEAR(INVOICE_DATE) = {_today.year}\n"
            f"    - 'last year' / 'previous year' = CALENDAR year  YEAR(INVOICE_DATE) = {_today.year - 1}\n"
            f"    - 'current fiscal year' / 'FY{_fy_end.year % 100}' / 'FY {_fy_end.year}' = FISCAL year  INVOICE_DATE >= '{_fy_start}' AND INVOICE_DATE <= '{_fy_end}'\n"
            f"    - 'last fiscal year' / 'FY{_prev_fy_end.year % 100}' = FISCAL year  INVOICE_DATE >= '{_prev_fy_start}' AND INVOICE_DATE <= '{_prev_fy_end}'\n"
            f"    Convention: FY25 = April 1, 2024 to March 31, 2025 (the year the FY ENDS, not starts).\n"
            f"    Use FISCAL year ONLY when user explicitly says 'fiscal year' or 'FY'. Otherwise default to CALENDAR year.\n"
            f"\n  MONTH RULES:\n"
            f"    - 'in June' (month without year, today is {_today.isoformat()})  INVOICE_DATE >= '2025-06-01' AND INVOICE_DATE <= '2025-06-30'\n"
            f"      Use the MOST RECENT past occurrence. If the month has passed this year, use this year. If not yet, use last year.\n"
            f"    - 'June 2024' (month with year)  INVOICE_DATE >= '2024-06-01' AND INVOICE_DATE <= '2024-06-30'\n"
            f"    - Do NOT use MONTH(INVOICE_DATE) = N without also filtering the year.\n"
            f"\n  OTHER:\n"
            f"    - 'last quarter'  compute previous quarter from current Q{_q}\n"
            f"    - 'last 6 months'  INVOICE_DATE >= DATEADD(MONTH, -6, GETDATE())\n"
            f"    - 'last N months'  INVOICE_DATE >= DATEADD(MONTH, -N, GETDATE())\n"
        )
        sections.append(temporal_ctx)

        # If Stage 1 pre-resolved a fiscal year, inject it as a hard constraint
        fy_resolved = ctx.get("fy_resolved")
        if fy_resolved:
            if fy_resolved.get("multi"):
                # Multiple FYs (e.g. "FY25 vs FY26") — give the LLM the full date range
                # but tell it the user wants comparison across those FYs
                sections.append(
                    f"\n**PRE-RESOLVED FISCAL YEARS (MULTIPLE — USE EXACTLY THESE DATES):**\n"
                    f"  User referenced: \"{fy_resolved['original']}\"\n"
                    f"  Combined date range: INVOICE_DATE >= '{fy_resolved['start_date']}' AND INVOICE_DATE <= '{fy_resolved['end_date']}'\n"
                    f"  Your WHERE clause MUST use the combined range above to include data from ALL referenced fiscal years.\n"
                    f"  Use CASE WHEN on INVOICE_DATE to label each row's fiscal year (e.g. 'FY25', 'FY26').\n"
                    f"  Do NOT use G_JAHR. Do NOT compute different dates.\n"
                )
            else:
                sections.append(
                    f"\n**PRE-RESOLVED FISCAL YEAR (USE EXACTLY  DO NOT RECOMPUTE):**\n"
                    f"  User said: \"{fy_resolved['original']}\"\n"
                    f"  Resolved to: INVOICE_DATE >= '{fy_resolved['start_date']}' AND INVOICE_DATE <= '{fy_resolved['end_date']}'\n"
                    f"  You MUST use exactly these dates. Do NOT use G_JAHR. Do NOT compute different dates.\n"
                )

        default_limit = getattr(self, "default_row_limit", 10) or 10
        if provider == "sqlserver":
            sections.append(
                f"\n**SQL Server Rules:**\n"
                f"1. Use SELECT TOP N (NEVER LIMIT, NEVER FETCH FIRST). "
                f"If the user explicitly asks for 'top N', 'first N', or 'N suppliers/plants/etc.', "
                f"use that N. Otherwise default to TOP {default_limit}. Hard ceiling is {mr}.\n"
                f"2. Use ISNULL() where needed, UPPER() for case-insensitive text matching\n"
                f"3. MANDATORY: EVERY query must include date filter: INVOICE_DATE >= '2024-04-01'\n"
                f"4. Use GETDATE() for relative date calculations (DATEADD, last N months). For fiscal year boundaries, use the concrete dates from Temporal Context.\n"
                f"5. Use + for string concatenation (not ||)\n"
                f"6. CURRENCY RULES:\n"
                f"   - AMOUNT is already normalized to EUR in this dataset; use AMOUNT directly for spend metrics.\n"
                f"   - If user explicitly asks for spend in a non-EUR currency (e.g. 'in USD', 'in MXN', 'in local currency'),\n"
                f"     use SUM(AMOUNT * EXCH_RATE) and filter by the currency column (Direct: EXCH_CURRENCY, Indirect: INVOICE_AMOUNT_CURRENCY).\n"
                f"   - Do NOT apply EXCH_RATE conversion unless the user explicitly requests a non-EUR currency.\n"
                f"   - NET_PRICE is a separate price metric and should be used only when user asks for price."
            )
        else:
            sections.append(f"\n**SQL Rules:**\nUse LIMIT {mr} to cap results.")

        # View-specific column semantics from registry
        table_cfg = ctx.get("table_config", TABLE_REGISTRY.get("direct", {}))
        table_semantics = table_cfg.get("column_semantics", "")
        sections.append(
            f"\n**Column Semantics (use these to resolve ambiguous terms):**\n"
            f"  - 'part number' / 'item number' / 'article number' = ARTICLE_NO (use COUNT(DISTINCT ARTICLE_NO) to count parts)\n"
            f"  - 'part description' / 'item description' = ARTICLE_DESCRIPTION\n"
            f"  - 'quantity' / 'volume' / 'units' / 'total volume' = QUANTITY (use SUM(QUANTITY) for totals)\n"
            f"  - 'spend' / 'amount' / 'cost' / 'value' = AMOUNT (already EUR)\n"
            f"  - 'unit price' / 'price per unit' = ROUND(SUM(AMOUNT) / NULLIF(SUM(QUANTITY), 0), 2) (EUR per unit)\n"
            f"  - 'supplier' / 'vendor' = SUPPLIER_NAME\n"
            f"  - 'plant' / 'factory' / 'location' = PLANT_NAME\n"
            f"  - 'parent supplier' / 'group' = \"Parent Supplier\"\n"
            f"  - 'payment terms' = PAYMENT_TERM (code); use TERMS_OF_PAYMENT_DESCRIPTION only if user asks for description\n"
            + table_semantics
        )

        sections.append(f"\n**User Question:** {normalized_query}")
        sections.append(
            f"\n**Rules (MUST FOLLOW):**\n"
            f"1. SELECT only\n"
            f"2. Use exact column names from the schema\n"
            f"3. GROUP BY for aggregations\n"
            f"4. Return ONLY the SQL, no explanations\n"
            f"5. For name/text filters (SUPPLIER_NAME, PLANT_NAME, etc.), use UPPER(col) LIKE UPPER('%value%') for partial matching, NOT exact equality\n"
            f"6. **NO PHANTOM FILTERS**  this is the #1 cause of query failures:\n"
            f"   - ONLY add WHERE conditions that the user EXPLICITLY asked for\n"
            f"   - Data-quality rules (e.g. AMOUNT > 0, AMOUNT != 0, SUPPLIER_NO IS NOT NULL, PLANT_NO IS NOT NULL) are allowed when they come from business rules\n"
            f"   - NO extra dimension filters (SOURCE, REGION, COUNTRY, INDUSTRY, etc.) unless the user specifically mentions them\n"
            f"   - NO date range filters beyond the mandatory one  do NOT add arbitrary date ranges like 'INVOICE_DATE >= 2023-01-01 AND INVOICE_DATE < 2025-01-01' unless the user asked for a specific period\n"
            f"   - The ONLY mandatory filter is INVOICE_DATE >= '2024-04-01' (injected automatically by the pipeline  you do NOT need to include it)\n"
            f"   - If the user says 'top 10 suppliers by spend', avoid extra dimensional filters; data-quality filters may still be applied by business rules\n"
            f"7. **NO ARBITRARY DATE RANGES**  only add date filters when the user mentions a specific time period:\n"
            f"   - 'in 2025'  add year filter\n"
            f"   - 'last 6 months'  add date filter\n"
            f"   - 'top 10 suppliers by spend' (no time mention)  no extra date filter from you\n"
            f"8. 'in 2025' = calendar year (Jan-Dec). Only use fiscal year when user explicitly says 'fiscal year' or 'FY'."
        )

        if self.extra_rules and self.extra_rules.strip():
            sections.append(self.extra_rules.strip())

        sections.append("\n**SQL Query:**")

        prompt_text = "\n".join(sections)
        token_est = len(prompt_text) // 4

        return {
            **ctx,
            "prompt_text": prompt_text,
            "token_estimate": token_est,
            "selected_examples_count": len(selected_examples),
            "total_examples_count": len(all_examples),
            "provider": provider,
            "schema_ddl": schema_ddl,
        }

    # 
    # STAGE 4: SQL Generator (LLM or Template)
    # 

    def _stage4_sql_generator(self, ctx, knowledge):
        intent = ctx.get("intent", {})
        schema_linking = ctx.get("schema_linking", {})
        prompt_text = ctx.get("prompt_text", "")

        sql = ""
        method = "llm"

        # Try template matching
        query_lower = ctx.get("normalized_query", "").lower()
        has_time_terms = any(t in query_lower for t in (
            "fiscal", "year", "month", "quarter", "date", "ytd", "yoy", "period",
            "current", "last", "previous", "recent", "this year", "last year",
        ))
        # Templates that handle dates internally and don't need LLM date resolution
        _TIME_SAFE_TEMPLATES = {"time_series_monthly"}
        if self.enable_templates and knowledge and intent.get("confidence_level") == "high":
            templates = knowledge.get("sql_templates", {})
            if templates:
                tmap = {
                    "enumerate": "enumerate_distinct", "top_n": "top_n_by_spend",
                    "bottom_n": "top_n_by_spend", "aggregation_grouped": "spend_by_dimension",
                    "aggregation": "spend_by_dimension", "time_series": "time_series_monthly",
                    "count": "count_distinct",
                }
                tname = tmap.get(intent.get("primary_intent", ""))
                if tname and tname in templates and (not has_time_terms or tname in _TIME_SAFE_TEMPLATES):
                    tmpl = templates[tname].get("template", "")
                    if tmpl:
                        resolved = schema_linking.get("resolved_columns", {})
                        groupby = schema_linking.get("suggested_groupby", [])
                        filters = schema_linking.get("suggested_filters", [])
                        limit = schema_linking.get("suggested_limit")
                        dim = str(groupby[0]) if groupby else (next(iter(resolved.values()), "") if resolved else "")
                        if dim:
                            where = "WHERE " + " AND ".join(str(f) for f in filters) if filters else ""
                            try:
                                table_name = ctx.get("table_config", {}).get("table_name", "")
                                sql = tmpl.format(
                                    table_name=table_name,
                                    column=dim, dimension=dim, dimension1=dim,
                                    dimension2=groupby[1] if len(groupby) > 1 else dim,
                                    where_clause=where, n=limit or 10,
                                    filter_column="", filter_value="",
                                    count_column=dim, alias="COUNT",
                                    columns=f"{dim}, SUM(AMOUNT) AS TOTAL_SPEND",
                                    group_by=f"GROUP BY {dim}", threshold=0,
                                ).strip()
                                method = "template"
                            except (KeyError, IndexError):
                                sql = ""

        # Fall back to LLM
        if not sql:
            if not prompt_text:
                return {**ctx, "generated_sql": "", "generation_method": "none", "error": True, "message": "Empty prompt."}
            try:
                response = self._llm_invoke_silent(prompt_text)
                raw = response.content if hasattr(response, "content") else str(response)
                sql = raw.strip()
                if sql.startswith("```sql"):
                    sql = sql[6:]
                if sql.startswith("```"):
                    sql = sql[3:]
                if sql.endswith("```"):
                    sql = sql[:-3]
                sql = sql.strip()
            except Exception as e:
                return {**ctx, "generated_sql": "", "generation_method": "llm", "error": True, "message": f"LLM failed: {e}"}

        return {**ctx, "generated_sql": sql, "generation_method": method}

    #
    # SQL JUDGE  retry failed queries with LLM correction
    # 

    def _judge_and_fix_sql(self, failed_sql, error_msg, ctx, provider, zero_rows=False):
        """Ask the LLM to diagnose and fix a failed SQL query."""
        schema_ddl = ctx.get("schema_ddl", "")
        user_question = ctx.get("normalized_query", ctx.get("raw_query", ""))
        schema_linking = ctx.get("schema_linking", {})
        resolved = schema_linking.get("resolved_columns", {})
        filters = schema_linking.get("suggested_filters", [])

        if zero_rows:
            problem = (
                "The query executed successfully but returned **0 rows**. "
                "The filters are likely too restrictive. Common causes:\n"
                "  - String comparison is case-sensitive (use UPPER() on both sides)\n"
                "  - Exact match used instead of LIKE for partial text matching\n"
                "  - Filter value doesn't match actual data (check column_value_hints)\n"
                "  - Date range too narrow or wrong fiscal year\n"
                "  - Column value format mismatch (e.g. 'DE' vs 'Germany')\n"
                "Relax filters or fix value matching. Do NOT simply remove WHERE clause entirely."
            )
        else:
            problem = f"The query failed with this error:\n{error_msg}"

        col_hints = ctx.get("column_value_hints_snippet", "")
        if not col_hints:
            # Pull a compact snippet from knowledge for the judge
            knowledge_hints = {}
            for t, c in resolved.items():
                for col_name, hints in (ctx.get("_knowledge_col_hints") or {}).items():
                    if col_name.upper() == str(c).upper():
                        examples = hints.get("examples", [])[:5]
                        if examples:
                            knowledge_hints[col_name] = examples
            if knowledge_hints:
                col_hints = "\n".join(f"  {c}: {', '.join(str(v) for v in vals)}" for c, vals in knowledge_hints.items())

        judge_prompt = f"""You are a SQL judge/debugger for {provider.upper()} databases.

A SQL query was generated for the user's question but it FAILED.

**User Question:** {user_question}

**Failed SQL:**
```sql
{failed_sql}
```

**Problem:** {problem}

**Database Schema:**
```sql
{schema_ddl}
```

**Resolved Columns:** {json.dumps(resolved) if resolved else '(none)'}
**Suggested Filters:** {json.dumps(filters) if filters else '(none)'}
{f"**Column Value Examples:**{chr(10)}{col_hints}" if col_hints else ""}

**Your task:**
1. Diagnose WHY the query failed or returned 0 rows
2. Fix the SQL query
3. Return ONLY the corrected SQL query, nothing else

**Rules:**
- SELECT only (no DDL/DML)
- For {provider.upper()}: {"use SELECT TOP N (not LIMIT, not FETCH FIRST)" if provider == "sqlserver" else "use LIMIT"}
- Use UPPER(col) LIKE UPPER('%value%') for text matching
- AMOUNT is already EUR in this dataset; only use EXCH_RATE if user explicitly asks for non-EUR currency
- Keep the original intent  don't oversimplify the query
- Do NOT remove all WHERE conditions just to get results
- Do NOT add NEW filters that weren't in the original query (e.g. don't add G_JAHR, REGION, SOURCE filters)
- Focus on relaxing existing filters: broaden LIKE patterns, widen date ranges, fix case sensitivity
- If 0 rows: the most common fix is relaxing the text match (e.g. '%Bosch GmbH%'  '%Bosch%') or widening the date range

**Corrected SQL:**"""

        try:
            response = self._llm_invoke_silent(judge_prompt)
            raw = response.content if hasattr(response, "content") else str(response)
            sql = raw.strip()
            if sql.startswith("```sql"):
                sql = sql[6:]
            if sql.startswith("```"):
                sql = sql[3:]
            if sql.endswith("```"):
                sql = sql[:-3]
            return sql.strip()
        except Exception:
            return None

    # 
    # INTERACTIVE TABLE RENDERER
    # 

    def _build_interactive_table(self, columns, rows):
        """Build a styled HTML table with number formatting and alternating rows.

        Uses plain HTML (no iframe, no JS) so the LLM can pass it through reliably.
        The frontend's rehype-raw renders the HTML directly.
        """
        # Detect money columns using the shared dtype+magnitude+name heuristic.
        # Keeps per-branch HTML tables in sync with the combined-summary formatter.
        money_cols = self._detect_money_cols(columns, rows)

        def _fmt(val, is_money=False):
            if val is None:
                return '<span style="color:#94a3b8;font-style:italic"></span>'
            if isinstance(val, bool):
                return str(val)
            if is_money and isinstance(val, _NUMERIC_TYPES):
                return self._fmt_money_str(val)
            if isinstance(val, Decimal):
                # Non-money Decimal — format like a float with thousands separators.
                return f"{float(val):,.2f}"
            if isinstance(val, float):
                return f"{val:,.2f}"
            if isinstance(val, int) and abs(val) >= 1000:
                return f"{val:,}"
            return str(val)

        # Detect numeric columns for right-alignment (Decimal counts as numeric).
        numeric_cols = set()
        for i in range(len(columns)):
            for row in rows[:10]:
                if i < len(row) and row[i] is not None and isinstance(row[i], _NUMERIC_TYPES) and not isinstance(row[i], bool):
                    numeric_cols.add(i)
                    break

        # Build HTML table
        lines = []
        lines.append('<table style="width:100%;border-collapse:collapse;font-family:system-ui,sans-serif;font-size:13px">')

        # Header
        lines.append('<thead><tr>')
        for i, col in enumerate(columns):
            align = "right" if i in numeric_cols else "left"
            lines.append(f'<th style="background:#f8fafc;color:#1e293b;font-weight:600;padding:8px 12px;text-align:{align};border-bottom:2px solid #e2e8f0">{col}</th>')
        lines.append('</tr></thead>')

        # Body
        lines.append('<tbody>')
        for row_idx, row in enumerate(rows):
            bg = "#f8fafc" if row_idx % 2 == 1 else "#ffffff"
            lines.append(f'<tr style="background:{bg}">')
            for i, val in enumerate(row):
                align = "right" if i in numeric_cols else "left"
                font = "font-family:Consolas,monospace;font-size:12.5px;" if i in numeric_cols else ""
                lines.append(f'<td style="padding:7px 12px;border-bottom:1px solid #f1f5f9;text-align:{align};{font}">{_fmt(val, is_money=(i in money_cols))}</td>')
            lines.append('</tr>')
        lines.append('</tbody></table>')

        return "\n".join(lines)

    # 
    # STAGE 5: SQL Processor (CODE + DB)
    # 

    # 
    # ACCESS CONTROL  check user permissions for price-sensitive columns
    # 

    def _check_price_access(self, user_email, rbac_db_data, spend_type=""):
        """Check if user has access to price-sensitive columns.

        Dual-DB design: RBAC tables live in Oracle (PISLOAD.EIS_*, PISVIEW.EIS_*),
        spend tables live in SQL Server. We query Oracle directly for the user's
        entitlements (regions, 4-char commodity prefixes, supplier classes, etc.)
        and return them as plain Python values. _apply_commodity_access_filter()
        then injects SQL Server WHERE conditions from those values  no cross-DB
        JOINs are needed.

        Logic mirrors rbac_oracle.fetch_user_entitlements() (kept as the canonical
        standalone reference); inlined here because AgentCore loads custom
        components via exec(), which breaks sibling-file imports.

        Tables queried (Oracle):
          - PISLOAD.EIS_SUPERUSER_LIST                     superuser whitelist
          - PISLOAD.EIS_ACCESS_USER_LIST                   email -> APPROVAL_ID
          - PISLOAD.EIS_USER_COMMODITY_LIST                APPROVAL_ID -> region/commodity
          - PISVIEW.EIS_USER_SUPPLIER_CLASSIFICATION_LIST  APPROVAL_ID -> supp class
        """
        if not user_email:
            return {"has_access": False, "reason": "email_required"}

        st = (spend_type or "").strip().upper()
        if st not in ("DIRECT", "INDIRECT"):
            return {"has_access": False, "reason": "spend_type_required"}

        if not rbac_db_data:
            return {
                "has_access": False,
                "reason": "rbac_oracle_connection_missing",
                "user_email": user_email,
                "spend_type": st.lower(),
            }

        try:
            import oracledb
        except Exception as e:
            return {
                "has_access": False,
                "reason": f"oracledb_import_failed: {e!s}",
                "user_email": user_email,
                "spend_type": st.lower(),
            }

        try:
            host = rbac_db_data.get("host", "")
            port = rbac_db_data.get("port", 1521)
            service_name = rbac_db_data.get("database_name", "")
            username = rbac_db_data.get("username", "")
            password = rbac_db_data.get("password", "")
            if not host or not service_name or not username:
                return {
                    "has_access": False,
                    "reason": "rbac_oracle_connection_missing_fields",
                    "user_email": user_email,
                    "spend_type": st.lower(),
                }
            dsn = oracledb.makedsn(host, int(port), service_name=service_name)
            conn = oracledb.connect(user=username, password=password, dsn=dsn)
            conn.call_timeout = 10000
        except Exception as e:
            return {
                "has_access": False,
                "reason": f"rbac_oracle_connect_failed: {e!s}",
                "user_email": user_email,
                "spend_type": st.lower(),
            }

        try:
            cur = conn.cursor()

            # 1) Superuser short-circuit
            try:
                cur.execute(
                    f"SELECT 1 FROM {ACCESS_SUPERUSER_TABLE} "
                    f"WHERE UPPER(EMAIL) = UPPER(:email)",
                    {"email": user_email},
                )
                su_row = cur.fetchone()
            except Exception:
                su_row = None

            if su_row is not None:
                cur.close()
                return {
                    "has_access": True,
                    "is_superuser": True,
                    "access_types": [st],
                    "role_match": True,
                    "expected_role": st,
                    "regions": [],
                    "l0_levels": [],
                    "approval_ids": [],
                    "supp_classifications": [],
                    "commodities": [],
                    "user_email": user_email,
                    "spend_type": st.lower(),
                    "reason": "superuser",
                }

            # 2) Role-scoped entitlement join
            join_sql = f"""
                SELECT
                    m.APPROVAL_ID,
                    m.REGION,
                    m.L0_LEVEL,
                    SUBSTR(l.COMMODITY, 1, {COMMODITY_PREFIX_LEN}) AS COMMODITY,
                    sc.SUPP_CLASSIFICATION
                FROM {ACCESS_USER_TABLE} m
                LEFT JOIN {ACCESS_COMMODITY_TABLE} l
                    ON m.APPROVAL_ID = l.APPROVAL_ID
                   AND UPPER(l.ACCESSTYPE) = :role
                LEFT JOIN {ACCESS_SUPP_CLASS_TABLE} sc
                    ON m.APPROVAL_ID = sc.APPROVAL_ID
                   AND UPPER(sc.ACCESSTYPE) = :role
                WHERE UPPER(m.EMAIL) = UPPER(:email)
                  AND UPPER(m.ACCESSTYPE) = :role
            """
            try:
                cur.execute(join_sql, {"email": user_email, "role": st})
                rows = cur.fetchall() or []
            except Exception as e:
                cur.close()
                return {
                    "has_access": False,
                    "reason": f"rbac_query_failed: {e!s}",
                    "expected_role": st,
                    "user_email": user_email,
                    "spend_type": st.lower(),
                }
            cur.close()
        finally:
            try:
                conn.close()
            except Exception:
                pass

        if not rows:
            return {
                "has_access": False,
                "reason": "no_entitlements_for_role",
                "expected_role": st,
                "user_email": user_email,
                "spend_type": st.lower(),
            }

        def _u(v):
            return str(v).strip() if v is not None and str(v).strip() else ""

        approval_ids = sorted({_u(r[0]) for r in rows if _u(r[0])})
        regions = sorted({_u(r[1]).upper() for r in rows if _u(r[1])})
        l0_levels = sorted({_u(r[2]) for r in rows if _u(r[2])})
        commodities = sorted({_u(r[3]).upper() for r in rows if _u(r[3])})
        supp_classifications = sorted({_u(r[4]) for r in rows if _u(r[4])})

        return {
            "has_access": True,
            "is_superuser": False,
            "access_types": [st],
            "role_match": True,
            "expected_role": st,
            "regions": regions,
            "l0_levels": l0_levels,
            "approval_ids": approval_ids,
            "supp_classifications": supp_classifications,
            "commodities": commodities,
            "user_email": user_email,
            "spend_type": st.lower(),
            "reason": "entitlements_resolved",
        }

    def _is_price_sensitive_sql(self, sql, table_config):
        """Return True when SQL references price-sensitive columns."""
        if not sql:
            return False
        sql_upper = sql.upper()
        price_cols = set(PRICE_SENSITIVE_COLUMNS) | set(table_config.get("price_columns", []))
        for col in price_cols:
            if re.search(r"\b" + re.escape(str(col).upper()) + r"\b", sql_upper):
                return True
        return False

    def _is_price_sensitive_request(self, ctx):
        """Return True when user request indicates price-sensitive intent."""
        raw_q = str(ctx.get("raw_query", "") or "").lower()
        norm_q = str(ctx.get("normalized_query", "") or "").lower()
        q = f"{raw_q} {norm_q}"
        if any(k in q for k in PRICE_SENSITIVE_KEYWORDS):
            return True

        # Also detect if schema linker mapped to sensitive columns
        sl = ctx.get("schema_linking", {}) or {}
        for col in sl.get("resolved_columns", {}).values():
            if isinstance(col, dict):
                col = col.get("column", col.get("name", ""))
            if col and str(col).upper() in PRICE_SENSITIVE_COLUMNS:
                return True
        return False

    def _apply_commodity_access_filter(self, sql, access_result):
        """Restrict price queries using row-level access mapping tables.

        Enforces REGION + COMMODITY (4-char prefix) row filters for both
        DIRECT and INDIRECT. For INDIRECT, additionally applies a supplier-
        classification -> ABC_INDICATOR filter.

        Superusers bypass all row filters — _check_price_access returns an
        empty entitlement set with is_superuser=True and this method short-
        circuits to pass-through.
        """
        if not isinstance(access_result, dict):
            return sql, False

        # Superuser: no row filter
        if access_result.get("is_superuser"):
            return sql, False

        user_email = str(access_result.get("user_email", "") or "").strip()
        spend_type = str(access_result.get("spend_type", "") or "").strip().lower()
        if not user_email or spend_type not in ("direct", "indirect"):
            return sql, False

        def _esc(v):
            return str(v).strip().upper().replace("'", "''")

        commodities = [_esc(v) for v in access_result.get("commodities", []) if str(v).strip()]
        regions = [_esc(v) for v in access_result.get("regions", []) if str(v).strip()]

        scope_conditions = []
        if regions:
            region_list = ", ".join(f"'{v}'" for v in regions)
            scope_conditions.append(f"UPPER(LTRIM(RTRIM(REGION))) IN ({region_list})")
        if commodities:
            prefix_list = ", ".join(f"'{v}'" for v in commodities)
            scope_conditions.append(
                f"SUBSTRING(UPPER(ISNULL(COMMODITY, '')), 1, {COMMODITY_PREFIX_LEN}) IN ({prefix_list})"
            )

        if spend_type == "direct":
            if not scope_conditions:
                # User mapped to the role but no region/commodity entitlements
                # at all -> deny any rows.
                return self._inject_where_condition(sql, "1 = 0")
            return self._inject_where_condition(sql, " AND ".join(scope_conditions))

        # INDIRECT: add supplier-classification -> ABC_INDICATOR filter
        # G  => Motherson Group, O/N/S/blank => External Supplier.
        supp_classes = [str(v).upper() for v in access_result.get("supp_classifications", []) if str(v).strip()]
        allow_motherson = any("MOTHERSON" in s or s == "G" for s in supp_classes)
        allow_external = any("EXTERNAL" in s or s in ("O", "N", "S") for s in supp_classes)

        abc_allowed = []
        if allow_motherson:
            abc_allowed.append("'G'")
        if allow_external:
            abc_allowed.extend(["'O'", "'N'", "'S'", "''"])

        if not abc_allowed and not scope_conditions:
            # No entitlements at all for indirect -> deny.
            return self._inject_where_condition(sql, "1 = 0")

        if abc_allowed:
            scope_conditions.append(
                "UPPER(LTRIM(RTRIM(ISNULL(ABC_INDICATOR, '')))) IN (" + ", ".join(abc_allowed) + ")"
            )

        return self._inject_where_condition(sql, " AND ".join(scope_conditions))

    def _mask_price_columns_in_sql(self, sql, table_config):
        """Replace price-sensitive column references in SELECT with NULL AS <col>.

        Only masks columns in the SELECT clause  WHERE/GROUP BY refs are left alone
        since they don't expose the actual values.

        Handles nested expressions like ROUND(SUM(NET_PRICE), 2) AS TOTAL by
        splitting the SELECT list on commas and replacing any item that
        references a price column.
        """
        price_cols = set(table_config.get("price_columns", []))
        if not price_cols:
            return sql, []

        # Extract the SELECT ... FROM portion
        select_match = re.search(r'\bSELECT\b(.*?)\bFROM\b', sql, re.IGNORECASE | re.DOTALL)
        if not select_match:
            return sql, []

        select_part = select_match.group(1)

        # Split the SELECT list on top-level commas (not inside parentheses)
        items = []
        depth = 0
        current = []
        for ch in select_part:
            if ch == '(':
                depth += 1
                current.append(ch)
            elif ch == ')':
                depth -= 1
                current.append(ch)
            elif ch == ',' and depth == 0:
                items.append(''.join(current))
                current = []
            else:
                current.append(ch)
        if current:
            items.append(''.join(current))

        masked = []
        new_items = []
        for item in items:
            item_upper = item.upper()
            matched_col = None
            for col in price_cols:
                if re.search(r'\b' + re.escape(col) + r'\b', item_upper):
                    matched_col = col
                    break
            if matched_col:
                # Extract alias if present: ... AS ALIAS_NAME
                alias_match = re.search(r'\bAS\s+(\w+)\s*$', item.strip(), re.IGNORECASE)
                alias = alias_match.group(1) if alias_match else matched_col
                new_items.append(f" NULL AS {alias}")
                if matched_col not in masked:
                    masked.append(matched_col)
            else:
                new_items.append(item)

        if masked:
            new_select = ','.join(new_items)
            # Ensure there's whitespace before FROM  the regex consumed it
            if not new_select.endswith(('\n', ' ', '\t')):
                new_select += '\n'
            sql = sql[:select_match.start(1)] + new_select + sql[select_match.end(1):]

        return sql, masked

    def _strip_phantom_filters(self, sql, ctx):
        """Remove WHERE conditions that the user never asked for.

        Compares each AND-separated condition in the WHERE clause against
        the columns mentioned in the user's query (via schema_linking resolved_columns
        and suggested_filters). Conditions referencing columns the user didn't
        mention are stripped, EXCEPT for the mandatory date filter which is
        re-injected later.

        Returns (cleaned_sql, list_of_removed_conditions).
        """
        # Columns the user actually asked about (from schema linking)
        sl = ctx.get("schema_linking", {})
        resolved_cols = set()
        for col in sl.get("resolved_columns", {}).values():
            if isinstance(col, dict):
                col = col.get("column", col.get("name", ""))
            if col:
                resolved_cols.add(str(col).upper())
        for f in sl.get("suggested_filters", []):
            # Extract column name from filter string like "SUPPLIER_NAME = 'X'"
            fm = re.match(r'\s*(\w+)', str(f))
            if fm:
                resolved_cols.add(fm.group(1).upper())
        for g in sl.get("suggested_groupby", []):
            resolved_cols.add(str(g).upper())
        orderby = sl.get("suggested_orderby")
        if orderby:
            resolved_cols.add(str(orderby).upper())

        # Also include columns from alias resolutions
        for a in ctx.get("normalizer", {}).get("alias_resolutions", []):
            sf = a.get("sql_filter", "")
            fm = re.match(r'\s*(\w+)', sf)
            if fm:
                resolved_cols.add(fm.group(1).upper())

        if not resolved_cols:
            return sql, []

        # Known phantom filter patterns  conditions LLMs love to invent
        # These are removed if the column isn't in resolved_cols
        phantom_patterns = [
            # NOT NULL checks
            r"\b(\w+)\s+IS\s+NOT\s+NULL\b",
            # > 0 checks
            r"\b(\w+)\s*>\s*0\b",
            # LIKE '%...%' on columns user didn't mention
            r"UPPER\s*\(\s*(\w+)\s*\)\s+LIKE\s+UPPER\s*\(",
            r"\b(\w+)\s+LIKE\s+'[^']*'",
            # Equality/IN on columns user didn't mention
            r"\b(\w+)\s+IN\s*\(",
            r"\b(\w+)\s*=\s*'[^']*'",
        ]

        removed = []

        # Extract WHERE clause
        where_match = re.search(r'\bWHERE\b(.+?)(?:\bGROUP\s+BY\b|\bORDER\s+BY\b|\bFETCH\b|\bLIMIT\b|$)',
                                sql, re.IGNORECASE | re.DOTALL)
        if not where_match:
            return sql, []

        where_body = where_match.group(1)

        # Split on AND (top-level only, not inside parentheses)
        # Simple approach: split on \bAND\b, then check each condition
        conditions = re.split(r'\bAND\b', where_body, flags=re.IGNORECASE)
        kept = []
        for cond in conditions:
            cond_stripped = cond.strip()
            if not cond_stripped:
                continue

            # Always keep the mandatory date filter
            if "INVOICE_DATE" in cond_stripped.upper():
                kept.append(cond_stripped)
                continue

            # Check if this condition references a column the user asked about
            is_phantom = False
            for pat in phantom_patterns:
                m = re.search(pat, cond_stripped, re.IGNORECASE)
                if m:
                    col_in_cond = m.group(1).upper()
                    if col_in_cond not in resolved_cols:
                        is_phantom = True
                        break

            if is_phantom:
                removed.append(cond_stripped)
            else:
                kept.append(cond_stripped)

        if not removed:
            return sql, []

        # Rebuild the WHERE clause
        if kept:
            new_where = " WHERE " + " AND ".join(kept)
        else:
            new_where = " "

        # Replace the original WHERE...rest with new WHERE...rest
        before_where = sql[:where_match.start()]
        after_match = sql[where_match.end():]
        new_sql = before_where + new_where + after_match

        return new_sql, removed

    @staticmethod
    def _condition_implied(part_upper, sql_upper):
        """Check if a condition is already semantically covered by existing SQL.

        For example, AMOUNT > 0 implies both AMOUNT IS NOT NULL and AMOUNT != 0.
        """
        # Direct substring match
        if part_upper in sql_upper:
            return True
        # Semantic: "COL IS NOT NULL" is implied by "COL > 0" or "COL >= 1" etc.
        m = re.match(r'(\w+)\s+IS\s+NOT\s+NULL', part_upper)
        if m:
            col = m.group(1)
            if re.search(rf'\b{col}\b\s*>\s*0', sql_upper):
                return True
            if re.search(rf'\b{col}\b\s*>=\s*[1-9]', sql_upper):
                return True
        # Semantic: "COL != 0" is implied by "COL > 0"
        m = re.match(r'(\w+)\s*!=\s*0', part_upper)
        if m:
            col = m.group(1)
            if re.search(rf'\b{col}\b\s*>\s*0', sql_upper):
                return True
        return False

    def _inject_where_condition(self, sql, condition):
        """Inject a single condition into WHERE (or create WHERE) safely.

        For compound conditions (A AND B), only injects sub-parts that are
        not already present or semantically implied by existing conditions.
        """
        cond = (condition or "").strip()
        if not cond:
            return sql, False
        sql_upper = sql.upper()

        # Also skip if the full condition string is already present
        if cond.upper() in sql_upper:
            return sql, False

        # Split compound conditions and only keep missing parts
        sub_parts = [p.strip() for p in re.split(r'\bAND\b', cond, flags=re.IGNORECASE) if p.strip()]
        missing = [p for p in sub_parts if not self._condition_implied(p.upper(), sql_upper)]
        if not missing:
            return sql, False

        inject = " AND ".join(missing)

        su = sql.upper()
        if "WHERE" in su:
            wi = su.index("WHERE") + 5
            return sql[:wi] + f" {inject} AND" + sql[wi:], True
        if "GROUP BY" in su:
            gi = su.index("GROUP BY")
            return sql[:gi] + f"WHERE {inject}\n" + sql[gi:], True
        if "ORDER BY" in su:
            oi = su.index("ORDER BY")
            return sql[:oi] + f"WHERE {inject}\n" + sql[oi:], True
        return sql.rstrip() + f"\nWHERE {inject}", True

    @staticmethod
    def _format_sql_display(sql):
        """Break SQL onto multiple lines at major keywords for readability."""
        if not sql:
            return sql
        # Insert newline before major SQL keywords (only when not inside quotes)
        kw = r'\b(SELECT|FROM|WHERE|AND|OR|GROUP\s+BY|ORDER\s+BY|HAVING|FETCH|LIMIT|UNION|JOIN|LEFT\s+JOIN|RIGHT\s+JOIN|INNER\s+JOIN|ON)\b'
        formatted = re.sub(kw, r'\n\1', sql, flags=re.IGNORECASE)
        # Indent continuation keywords
        lines = []
        for line in formatted.split('\n'):
            stripped = line.strip()
            if not stripped:
                continue
            upper = stripped.upper()
            if upper.startswith(('AND ', 'OR ')):
                lines.append('  ' + stripped)
            elif upper.startswith(('SELECT', 'FROM', 'WHERE', 'GROUP', 'ORDER', 'HAVING', 'FETCH', 'LIMIT', 'UNION')):
                lines.append(stripped)
            else:
                lines.append('  ' + stripped)
        return '\n'.join(lines)

    def _sanitize_prompt_text(self, text):
        """Remove unresolved template placeholders from prompt snippets."""
        if not text:
            return text
        out = str(text)
        # Remove Jinja-style placeholders like {{SYSTEM_PRICE_FILTER}}
        out = re.sub(r"\{\{\s*[^{}]+\s*\}\}", "", out)
        # Remove single-brace placeholders that sometimes leak from templates
        out = re.sub(r"\{[A-Z_][A-Z0-9_]*\}", "", out)
        out = re.sub(r"\s{2,}", " ", out).strip()
        return out

    def _strip_unresolved_sql_placeholders(self, sql):
        """Remove conditions containing unresolved placeholders from SQL."""
        if not sql:
            return sql, []
        if "{" not in sql and "}" not in sql:
            return sql, []

        removed = []
        where_match = re.search(
            r'\bWHERE\b(.+?)(?:\bGROUP\s+BY\b|\bORDER\s+BY\b|\bFETCH\b|\bLIMIT\b|$)',
            sql, re.IGNORECASE | re.DOTALL,
        )
        if not where_match:
            # Last-resort cleanup if placeholders leaked outside WHERE
            cleaned = re.sub(r"\{\{\s*[^{}]+\s*\}\}", "", sql)
            cleaned = re.sub(r"\{[A-Z_][A-Z0-9_]*\}", "", cleaned)
            return cleaned, ["Removed unresolved template token(s)"]

        where_body = where_match.group(1)
        conditions = re.split(r'\bAND\b', where_body, flags=re.IGNORECASE)
        kept = []
        for cond in conditions:
            c = cond.strip()
            if not c:
                continue
            if "{" in c or "}" in c:
                removed.append(c)
                continue
            kept.append(c)

        new_where = (" WHERE " + " AND ".join(kept) + " ") if kept else " "
        before = sql[:where_match.start()]
        after = sql[where_match.start() + len("WHERE") + len(where_match.group(1)):]
        new_sql = before + new_where + after
        return new_sql, removed

    def _parse_rule_condition(self, rule):
        """Extract SQL condition from rule formats (dict/list/legacy string)."""
        if isinstance(rule, dict):
            for key in ("filter", "condition", "sql"):
                val = str(rule.get(key, "")).strip()
                if val:
                    return val
            return ""
        if isinstance(rule, str):
            txt = rule.strip()
            # Legacy format from Knowledge Processor: "exclude_credits: AMOUNT > 0"
            if ":" in txt:
                parts = txt.split(":", 1)
                rhs = parts[1].strip()
                if rhs:
                    return rhs
            return txt
        return ""

    def _apply_business_rule_filters(self, sql, ctx, knowledge):
        """Apply data-quality exclusion rules from knowledge in a controlled way."""
        rules = (knowledge or {}).get("business_rules", {})
        exclusions = rules.get("exclusion_rules", {})
        if not exclusions:
            return sql, []

        if isinstance(exclusions, list):
            ex_map = {}
            for item in exclusions:
                txt = str(item)
                if ":" in txt:
                    name, _ = txt.split(":", 1)
                    ex_map[name.strip()] = txt
                else:
                    ex_map[txt.strip()] = txt
            exclusions = ex_map

        raw_query = (ctx.get("raw_query", "") or "").lower()
        normalized_query = (ctx.get("normalized_query", "") or "").lower()
        q = raw_query + " " + normalized_query
        intent = (ctx.get("intent", {}) or {}).get("primary_intent", "")
        spend_like = (
            intent in {"aggregation", "aggregation_grouped", "top_n", "bottom_n", "trend", "time_series", "comparison", "average"}
            or any(t in q for t in ("spend", "cost", "amount", "expense", "price", "total", "ytd"))
        )
        include_credits = any(t in q for t in ("include credit", "including credit", "credit memo", "negative amount", "reversal"))
        supplier_like = any(t in q for t in ("supplier", "vendor"))
        plant_like = any(t in q for t in ("plant", "factory", "facility", "location"))

        resolved_cols = set()
        sl = ctx.get("schema_linking", {}) or {}
        for col in sl.get("resolved_columns", {}).values():
            if isinstance(col, dict):
                col = col.get("column", col.get("name", ""))
            if col:
                resolved_cols.add(str(col).upper())
        supplier_like = supplier_like or any(c in resolved_cols for c in ("SUPPLIER_NO", "SUPPLIER_NAME"))
        plant_like = plant_like or any(c in resolved_cols for c in ("PLANT_NO", "PLANT_NAME"))

        applied = []
        for name, rule in exclusions.items():
            lname = str(name).lower().strip()
            cond = self._parse_rule_condition(rule)
            if not cond:
                continue

            should_apply = False
            if "exclude_credits" in lname:
                should_apply = spend_like and not include_credits
            elif "valid_amount" in lname:
                should_apply = spend_like
            elif "valid_supplier" in lname:
                should_apply = supplier_like
            elif "valid_plant" in lname:
                should_apply = plant_like

            if not should_apply:
                continue

            sql_new, changed = self._inject_where_condition(sql, cond)
            if changed:
                sql = sql_new
                applied.append(f"{name}: {cond}")

        return sql, applied

    def _apply_price_filter_rule(self, sql):
        """Apply optional NET_PRICE guard filter supplied by system configuration."""
        clause = (self.price_filter_rule or "").strip()
        if not clause:
            return sql, False
        if "NET_PRICE" not in sql.upper():
            return sql, False
        return self._inject_where_condition(sql, clause)

    def _stage5_sql_processor(self, ctx, knowledge, db_data, provider, schema_ddl="",
                              user_email="", table_config=None, rbac_db_data=None):
        if table_config is None:
            table_config = ctx.get("table_config", TABLE_REGISTRY.get("direct", {}))

        if ctx.get("error"):
            return Message(text=f"Error from Stage {ctx.get('generation_method', '?')}: {ctx.get('message', 'Unknown')}")

        sql = ctx.get("generated_sql", "")
        if not sql:
            return Message(text="Error: No SQL generated.")

        mr = self.max_rows
        trace_events = []
        all_fixes = []

        #  Phantom filter stripping  remove LLM-invented conditions 
        sql, phantom_removed = self._strip_phantom_filters(sql, ctx)
        if phantom_removed:
            for pr in phantom_removed:
                all_fixes.append(f"Removed phantom filter: {pr}")
            trace_events.append(f"Stripped {len(phantom_removed)} phantom filter(s)")

        #  G_JAHR fiscal year fix  replace G_JAHR = 'YYYY' with INVOICE_DATE range 
        # G_JAHR is a calendar year column but LLMs mistakenly use it for fiscal years.
        # Detect patterns like G_JAHR = '2024' or G_JAHR IN ('2024', '2025') and replace
        # with the correct INVOICE_DATE range based on the user's query context.
        g_jahr_match = re.search(r"\bG_JAHR\s*=\s*'(\d{4})'", sql, re.IGNORECASE)
        if not g_jahr_match:
            g_jahr_match = re.search(r"\bG_JAHR\s+IN\s*\(\s*'?(\d{4})'?", sql, re.IGNORECASE)
        if g_jahr_match:
            raw_query_lower = ctx.get("raw_query", "").lower()
            # Only fix if the user mentioned fiscal year / FY
            if re.search(r"\b(fy\s*'?\d{2,4}(?:\s*/\s*\d{2,4})?|fiscal\s+year)\b", raw_query_lower):
                fy_year = int(g_jahr_match.group(1))
                # Ending-year convention: FY25 = Apr 2024-Mar 2025
                # G_JAHR value is calendar year; treat as FY ending year
                fy_start = f"'{fy_year - 1}-04-01'"
                fy_end = f"'{fy_year}-03-31'"
                # Remove the entire G_JAHR condition
                g_jahr_full = re.search(
                    r"\s*(?:AND\s+)?G_JAHR\s*(?:=\s*'?\d{4}'?|IN\s*\([^)]+\))\s*(?:AND\s+)?",
                    sql, re.IGNORECASE,
                )
                if g_jahr_full:
                    # Replace with INVOICE_DATE range
                    replacement = f" INVOICE_DATE >= {fy_start} AND INVOICE_DATE <= {fy_end} "
                    sql = sql[:g_jahr_full.start()] + replacement + sql[g_jahr_full.end():]
                    # Clean up any double AND
                    sql = re.sub(r'\bAND\s+AND\b', 'AND', sql, flags=re.IGNORECASE)
                    all_fixes.append(f"Replaced G_JAHR='{fy_year}' with INVOICE_DATE range ({fy_start} to {fy_end})")

        #  Deterministic FY date enforcement 
        # If Stage 1 pre-resolved a fiscal year, parse the WHERE clause,
        # remove ALL INVOICE_DATE / G_JAHR conditions, and inject the exact dates.
        fy_resolved = ctx.get("fy_resolved")
        if fy_resolved:
            correct_start = fy_resolved["start_date"]
            correct_end = fy_resolved["end_date"]
            correct_filter = f"INVOICE_DATE >= '{correct_start}' AND INVOICE_DATE <= '{correct_end}'"

            # Extract the WHERE clause body
            where_match = re.search(
                r'\bWHERE\b(.+?)(?:\bGROUP\s+BY\b|\bORDER\s+BY\b|\bFETCH\b|\bLIMIT\b|$)',
                sql, re.IGNORECASE | re.DOTALL,
            )
            if where_match:
                where_body = where_match.group(1)
                # Split on AND, keep only non-date conditions
                conditions = re.split(r'\bAND\b', where_body, flags=re.IGNORECASE)
                kept = []
                removed_dates = []
                for cond in conditions:
                    cond_stripped = cond.strip()
                    if not cond_stripped:
                        continue
                    cond_upper = cond_stripped.upper()
                    # Remove ANY condition that references INVOICE_DATE or G_JAHR
                    if "INVOICE_DATE" in cond_upper or "G_JAHR" in cond_upper:
                        removed_dates.append(cond_stripped)
                    else:
                        kept.append(cond_stripped)

                # Rebuild: correct FY filter + kept non-date conditions
                all_conditions = [correct_filter] + kept
                new_where = " WHERE " + " AND ".join(all_conditions) + " "

                # Replace original WHERE...rest
                before = sql[:where_match.start()]
                after = sql[where_match.start() + len("WHERE") + len(where_match.group(1)):]
                sql = before + new_where + after

                if removed_dates:
                    all_fixes.append(f"Enforced FY dates: {correct_filter} (replaced {len(removed_dates)} LLM date condition(s))")
                else:
                    all_fixes.append(f"Enforced FY dates: {correct_filter}")
            else:
                # No WHERE clause  add one with just the FY filter
                su = sql.upper()
                if "GROUP BY" in su:
                    gi = su.index("GROUP BY")
                    sql = sql[:gi] + f"WHERE {correct_filter}\n" + sql[gi:]
                elif "ORDER BY" in su:
                    oi = su.index("ORDER BY")
                    sql = sql[:oi] + f"WHERE {correct_filter}\n" + sql[oi:]
                else:
                    sql = sql.rstrip() + f"\nWHERE {correct_filter}"
                all_fixes.append(f"Enforced FY dates: {correct_filter}")

        #  Access control: enforce price visibility by user email + access tables 
        price_sensitive_sql = self._is_price_sensitive_sql(sql, table_config)
        price_sensitive_request = self._is_price_sensitive_request(ctx)
        requires_price_access = price_sensitive_sql or price_sensitive_request
        if requires_price_access and not user_email:
            return Message(
                text=(
                    "Access denied: price-sensitive query requires an authenticated user email.\n"
                    "Please ensure user email is passed from UI session."
                )
            )

        access_result = (
            self._check_price_access(
                user_email, rbac_db_data, spend_type=ctx.get("spend_type", "")
            )
            if requires_price_access
            else {"has_access": True}
        )
        if requires_price_access and not access_result.get("has_access", False):
            return Message(
                text=(
                    f"Access denied: user `{user_email}` is not authorized for price columns.\n"
                    f"Reason: {access_result.get('reason', 'access_check_failed')}"
                    + (
                        f"\nExpected role: {access_result.get('expected_role','')}, "
                        f"Assigned access types: {', '.join(access_result.get('access_types', []))}"
                        if access_result.get("expected_role")
                        else ""
                    )
                )
            )

        price_masked = []
        if requires_price_access and access_result.get("has_access", False):
            # Optional additional scope: if commodity restrictions exist, apply them.
            # This enforces row-level scope for price output using the commodity access table.
            sql_scoped, scoped = self._apply_commodity_access_filter(sql, access_result)
            if scoped:
                sql = sql_scoped
                all_fixes.append("Access control: applied commodity row-scope filter")
                trace_events.append("Commodity scope filter applied for price query")

        # Apply business-rule data-quality filters in a controlled way
        sql, rule_filters_applied = self._apply_business_rule_filters(sql, ctx, knowledge)
        if rule_filters_applied:
            all_fixes.append("Business rules applied: " + ", ".join(rule_filters_applied))

        # Apply optional NET_PRICE guard filter (for indirect price metrics)
        sql, price_filter_applied = self._apply_price_filter_rule(sql)
        net_price_rule_missing = False
        if price_filter_applied:
            all_fixes.append(f"Applied NET_PRICE filter rule: {self.price_filter_rule.strip()}")
        elif (
            "NET_PRICE" in sql.upper()
            and table_config.get("net_price_requires_user_filter")
            and not (self.price_filter_rule or "").strip()
        ):
            net_price_rule_missing = True
            trace_events.append("Warning: NET_PRICE used without configured Price Filter Rule")

        # Remove unresolved placeholders before anti-pattern/validation pass
        sql, removed_placeholders = self._strip_unresolved_sql_placeholders(sql)
        if removed_placeholders:
            all_fixes.append(f"Removed unresolved placeholder conditions: {len(removed_placeholders)}")
            trace_events.append("Sanitized unresolved template placeholders")

        # Anti-pattern fixes from knowledge
        anti_patterns = knowledge.get("anti_patterns", [])
        for ap in anti_patterns:
            compiled = ap.get("compiled")
            if not compiled or ap.get("required") or ap.get("forbidden"):
                continue
            if not compiled.search(sql):
                continue
            ap_name = ap.get("name", ap.get("id", "?"))
            # Convert LIMIT N -> TOP N (inject after SELECT, remove LIMIT)
            lm = re.search(r"\bLIMIT\s+(\d+)", sql, re.IGNORECASE)
            if lm and "LIMIT" in ap_name.upper():
                n = lm.group(1)
                sql = re.sub(r"\bLIMIT\s+\d+", "", sql, flags=re.IGNORECASE)
                sql = re.sub(r"\bSELECT\b", f"SELECT TOP {n}", sql, count=1, flags=re.IGNORECASE)
                all_fixes.append(f"LIMIT {n} -> TOP {n}")
                continue
            # Convert FETCH FIRST N ROWS ONLY -> TOP N
            ffm = re.search(r"FETCH\s+FIRST\s+(\d+)\s+ROWS?\s+ONLY", sql, re.IGNORECASE)
            if ffm and "FETCH" in ap_name.upper():
                n = ffm.group(1)
                sql = re.sub(r"\s*FETCH\s+FIRST\s+\d+\s+ROWS?\s+ONLY", "", sql, flags=re.IGNORECASE)
                sql = re.sub(r"\bSELECT\b", f"SELECT TOP {n}", sql, count=1, flags=re.IGNORECASE)
                all_fixes.append(f"FETCH FIRST {n} -> TOP {n}")
                continue
            if "semicolon" in ap_name.lower():
                sql = sql.rstrip().rstrip(";").rstrip()
                all_fixes.append("Removed semicolon")
                continue

        # Convert any remaining FETCH FIRST N ROWS ONLY -> TOP N
        ffm = re.search(r"FETCH\s+FIRST\s+(\d+)\s+ROWS?\s+ONLY", sql, re.IGNORECASE)
        if ffm:
            n = int(ffm.group(1))
            cap_n = min(n, mr)
            sql = re.sub(r"\s*FETCH\s+FIRST\s+\d+\s+ROWS?\s+ONLY", "", sql, flags=re.IGNORECASE)
            if not re.search(r"\bSELECT\s+TOP\s+\d+", sql, re.IGNORECASE):
                sql = re.sub(r"\bSELECT\b", f"SELECT TOP {cap_n}", sql, count=1, flags=re.IGNORECASE)
            all_fixes.append(f"FETCH FIRST {n} -> TOP {cap_n}")

        # Convert any remaining LIMIT N -> TOP N
        lm2 = re.search(r"\bLIMIT\s+(\d+)", sql, re.IGNORECASE)
        if lm2:
            n = int(lm2.group(1))
            cap_n = min(n, mr)
            sql = re.sub(r"\bLIMIT\s+\d+", "", sql, flags=re.IGNORECASE)
            if not re.search(r"\bSELECT\s+TOP\s+\d+", sql, re.IGNORECASE):
                sql = re.sub(r"\bSELECT\b", f"SELECT TOP {cap_n}", sql, count=1, flags=re.IGNORECASE)
            all_fixes.append(f"LIMIT {n} -> TOP {cap_n}")

        # Cap absurd TOP N
        tm = re.search(r"\bSELECT\s+TOP\s+(\d+)", sql, re.IGNORECASE)
        if tm and int(tm.group(1)) > mr:
            old_n = tm.group(1)
            sql = re.sub(r"\bSELECT\s+TOP\s+\d+", f"SELECT TOP {mr}", sql, flags=re.IGNORECASE)
            all_fixes.append(f"Capped TOP {old_n} -> {mr}")

        # Inject default TOP N when none is present. We ALWAYS inject — even
        # for "list all / export all" style questions — because the
        # Orchestration UI visibly chokes when a response carries hundreds of
        # rows of HTML table. If the user genuinely wants more, they can
        # follow up with "top 50" / "top 100" and the LLM will emit an
        # explicit TOP. Aggregate-only queries are unaffected (TOP N on a
        # 1-row result is a no-op). The "ctx['injected_default_top']" flag
        # below is consumed by stage 5's output builder to append a one-line
        # follow-up hint to the user.
        default_limit = getattr(self, "default_row_limit", 10) or 10
        default_limit = max(1, min(int(default_limit), mr))
        if not re.search(r"\bSELECT\s+(DISTINCT\s+)?TOP\s+\d+", sql, re.IGNORECASE):
            sql = re.sub(
                r"\bSELECT\b(\s+DISTINCT\b)?",
                lambda m: f"SELECT{m.group(1) or ''} TOP {default_limit}",
                sql,
                count=1,
                flags=re.IGNORECASE,
            )
            all_fixes.append(f"Injected default TOP {default_limit} (no limit in query)")
            ctx["injected_default_top"] = default_limit

        # Remove redundant fiscal year filters
        for pat, label in [
            (r"\s*AND\s+FORMAT\s*\(\s*INVOICE_DATE\s*,\s*'yyyy'\s*\)\s*=\s*FORMAT\s*\(\s*GETDATE\s*\(\s*\)\s*,\s*'yyyy'\s*\)", "FORMAT fiscal year"),
            (r"\s*AND\s+YEAR\s*\(\s*INVOICE_DATE\s*\)\s*=\s*YEAR\s*\(\s*GETDATE\s*\(\s*\)\s*\)", "YEAR(GETDATE()) fiscal year"),
        ]:
            if re.search(pat, sql, re.IGNORECASE):
                sql = re.sub(pat, "", sql, flags=re.IGNORECASE)
                all_fixes.append(f"Removed redundant {label} filter")

        # Strip trailing semicolons
        if sql.rstrip().endswith(";"):
            sql = sql.rstrip().rstrip(";").rstrip()
            all_fixes.append("Removed trailing semicolon")

        if all_fixes:
            trace_events.append(f"Fixes: {len(all_fixes)} applied")

        # Mandatory date filter injection
        mf = (self.mandatory_filter or "").strip()
        if mf:
            col_match = re.match(r"(\w+)", mf)
            col_name = col_match.group(1).upper() if col_match else ""
            if col_name and col_name not in sql.upper():
                su = sql.upper()
                if "WHERE" in su:
                    wi = su.index("WHERE") + 5
                    sql = sql[:wi] + f" {mf} AND" + sql[wi:]
                elif "GROUP BY" in su:
                    gi = su.index("GROUP BY")
                    sql = sql[:gi] + f"WHERE {mf}\n" + sql[gi:]
                elif "ORDER BY" in su:
                    oi = su.index("ORDER BY")
                    sql = sql[:oi] + f"WHERE {mf}\n" + sql[oi:]
                else:
                    sql = sql.rstrip() + f"\nWHERE {mf}"
                trace_events.append(f"Injected mandatory filter: {mf}")

        # Safety validation
        sql_stripped = sql.strip().rstrip(";")
        blocked = {"DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE", "TRUNCATE", "GRANT", "REVOKE", "EXEC", "MERGE"}
        tokens = sql_stripped.upper().split()
        for kw in blocked:
            if kw in tokens:
                return Message(text=f"Error: Blocked keyword '{kw}' in SQL.\n\n```sql\n{sql}\n```")
        first = tokens[0] if tokens else ""
        if first not in ("SELECT", "WITH"):
            return Message(text=f"Error: Query must start with SELECT. Got: {first}\n\n```sql\n{sql}\n```")

        trace_events.append("Validated OK")

        #  Execute SQL with retry/judge logic 
        max_attempts = 1 + (min(self.max_retries, 3) if self.enable_retry else 0)
        retry_history = []  # track each attempt for trace
        columns = []
        rows = []
        exec_ms = 0
        final_sql = sql

        # Store column hints in ctx for judge access
        ctx["_knowledge_col_hints"] = knowledge.get("column_value_hints", {})

        for attempt in range(1, max_attempts + 1):
            exec_error = None
            try:
                sql_exec = final_sql.strip().rstrip(";").strip()
                start = time.time()
                if provider == "sqlserver":
                    import pyodbc
                    conn_str = (
                        f"DRIVER={{ODBC Driver 18 for SQL Server}};"
                        f"SERVER={db_data['host']},{db_data['port']};"
                        f"DATABASE={db_data['database_name']};"
                        f"UID={db_data['username']};"
                        f"PWD={db_data['password']};"
                        f"TrustServerCertificate=yes;"
                    )
                    conn = pyodbc.connect(conn_str, timeout=15)
                else:
                    import psycopg2
                    conn = psycopg2.connect(
                        host=db_data["host"], port=db_data["port"],
                        dbname=db_data["database_name"],
                        user=db_data["username"], password=db_data["password"],
                        connect_timeout=15,
                        options=f"-c statement_timeout={self.query_timeout * 1000}",
                    )
                try:
                    cur = conn.cursor()
                    cur.execute(sql_exec)
                    columns = [d[0] for d in cur.description] if cur.description else []
                    rows = cur.fetchall() if columns else []
                    cur.close()
                finally:
                    conn.close()
                exec_ms = round((time.time() - start) * 1000, 2)
            except Exception as e:
                exec_error = str(e)
                exec_ms = round((time.time() - start) * 1000, 2)

            # Determine if retry is needed
            needs_retry = False
            if exec_error:
                retry_history.append({"attempt": attempt, "sql": final_sql, "error": exec_error, "rows": 0})
                trace_events.append(f"Attempt {attempt}: FAILED  {exec_error[:120]}")
                needs_retry = True
            elif not rows:
                retry_history.append({"attempt": attempt, "sql": final_sql, "error": None, "rows": 0, "zero_rows": True})
                trace_events.append(f"Attempt {attempt}: 0 rows returned")
                needs_retry = True
            else:
                retry_history.append({"attempt": attempt, "sql": final_sql, "error": None, "rows": len(rows)})
                trace_events.append(f"Attempt {attempt}: {len(rows)} rows in {exec_ms}ms")

            # If successful (has rows) or no more retries, break
            if not needs_retry or attempt >= max_attempts:
                break

            # Ask the judge LLM to fix the SQL
            trace_events.append(f"Retry {attempt}: invoking judge LLM...")
            fixed_sql = self._judge_and_fix_sql(
                failed_sql=final_sql,
                error_msg=exec_error or "0 rows returned",
                ctx={**ctx, "schema_ddl": schema_ddl},
                provider=provider,
                zero_rows=(exec_error is None),
            )
            if not fixed_sql or fixed_sql.strip().upper() == final_sql.strip().upper():
                trace_events.append(f"Judge returned same/empty SQL  stopping retry")
                break

            # Apply same safety fixes to the judge's SQL
            final_sql = fixed_sql.strip().rstrip(";").strip()
            all_fixes.append(f"Judge retry {attempt}: SQL corrected by LLM")

            # Re-validate the judge's SQL
            tokens_check = final_sql.upper().split()
            first_check = tokens_check[0] if tokens_check else ""
            if first_check not in ("SELECT", "WITH"):
                trace_events.append(f"Judge SQL invalid (starts with {first_check})  stopping retry")
                break
            blocked_found = False
            for kw in blocked:
                if kw in tokens_check:
                    trace_events.append(f"Judge SQL contains blocked keyword {kw}  stopping retry")
                    blocked_found = True
                    break
            if blocked_found:
                break

            # Re-apply FY date enforcement on judge SQL (judge will have wrong dates)
            if fy_resolved:
                correct_filter = f"INVOICE_DATE >= '{fy_resolved['start_date']}' AND INVOICE_DATE <= '{fy_resolved['end_date']}'"
                jw_match = re.search(
                    r'\bWHERE\b(.+?)(?:\bGROUP\s+BY\b|\bORDER\s+BY\b|\bFETCH\b|\bLIMIT\b|$)',
                    final_sql, re.IGNORECASE | re.DOTALL,
                )
                if jw_match:
                    jw_body = jw_match.group(1)
                    jw_conds = re.split(r'\bAND\b', jw_body, flags=re.IGNORECASE)
                    jw_kept = [c.strip() for c in jw_conds if c.strip() and "INVOICE_DATE" not in c.upper() and "G_JAHR" not in c.upper()]
                    jw_all = [correct_filter] + jw_kept
                    jw_new_where = " WHERE " + " AND ".join(jw_all) + " "
                    final_sql = final_sql[:jw_match.start()] + jw_new_where + final_sql[jw_match.start() + len("WHERE") + len(jw_match.group(1)):]

            # Re-apply price masking (judge SQL won't have it)
            if price_masked:
                final_sql, _ = self._mask_price_columns_in_sql(final_sql, table_config)

            # Re-apply mandatory date filter (judge SQL may not include it)
            mf = (self.mandatory_filter or "").strip()
            if mf:
                col_match = re.match(r"(\w+)", mf)
                col_name = col_match.group(1).upper() if col_match else ""
                if col_name and col_name not in final_sql.upper():
                    su = final_sql.upper()
                    if "WHERE" in su:
                        wi = su.index("WHERE") + 5
                        final_sql = final_sql[:wi] + f" {mf} AND" + final_sql[wi:]
                    elif "GROUP BY" in su:
                        gi = su.index("GROUP BY")
                        final_sql = final_sql[:gi] + f"WHERE {mf}\n" + final_sql[gi:]
                    else:
                        final_sql = final_sql.rstrip() + f"\nWHERE {mf}"

            # Re-apply business filters and NET_PRICE guard on judge SQL
            final_sql, _ = self._apply_business_rule_filters(final_sql, ctx, knowledge)
            final_sql, _ = self._apply_price_filter_rule(final_sql)

        # If all attempts failed with errors, return error message
        if exec_error and not rows:
            _safe_chars_err = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 \n")
            _err_sql = "".join(c if c in _safe_chars_err else f"&#{ord(c)};" for c in self._format_sql_display(final_sql))
            error_parts = [f'Error executing SQL after {len(retry_history)} attempt(s): {exec_error}\n\n<pre style="background:#1e293b;color:#e2e8f0;padding:12px;border-radius:6px;overflow-x:auto;font-size:12.5px"><code>{_err_sql}</code></pre>']
            if len(retry_history) > 1:
                error_parts.append("\n<details><summary>Retry History</summary>\n")
                for rh in retry_history:
                    status = f"Error: {rh['error'][:80]}" if rh.get("error") else f"{rh['rows']} rows"
                    _rh_sql = "".join(c if c in _safe_chars_err else f"&#{ord(c)};" for c in self._format_sql_display(rh['sql']))
                    error_parts.append(f'**Attempt {rh["attempt"]}:** {status}\n<pre style="background:#1e293b;color:#e2e8f0;padding:12px;border-radius:6px;overflow-x:auto;font-size:12.5px"><code>{_rh_sql}</code></pre>\n')
                error_parts.append("</details>")
            return Message(text="\n".join(error_parts))

        # Use final_sql for display
        sql = final_sql

        trace_events.append(f"Executed: {len(rows)} rows in {exec_ms}ms")

        # Post-result checks
        post_warnings = []
        if net_price_rule_missing:
            post_warnings.append("NET_PRICE query executed without configured 'Price Filter Rule'. Configure it to match business policy.")
        if not rows:
            post_warnings.append("Query returned 0 rows -- filters may be too restrictive.")
        if len(rows) == 1 and len(columns) == 1 and rows[0][0] is None:
            post_warnings.append("Aggregation returned NULL.")

        #  Build interactive HTML table (sortable, searchable, formatted) 
        if columns and rows:
            result_table = self._build_interactive_table(columns, rows)
        else:
            result_table = "_No data_"

        #  Pipeline trace (metadata) 
        raw_q = ctx.get("raw_query", "")
        norm_q = ctx.get("normalized_query", raw_q)
        normalizer = ctx.get("normalizer", {})
        intent = ctx.get("intent", {})
        sl = ctx.get("schema_linking", {})
        gen_method = ctx.get("generation_method", "?")
        tok_est = ctx.get("token_estimate", 0)
        n_ex = ctx.get("selected_examples_count", 0)
        tot_ex = ctx.get("total_examples_count", 0)

        #  Build RESULTS section (what the Worker LLM should show to user) 
        parts = []

        if rows:
            parts.append(result_table)
            parts.append(f"\n*{len(rows)} row{'s' if len(rows) != 1 else ''} returned in {exec_ms}ms*")
            # Follow-up hint: when we auto-capped the query AND the result is
            # at the cap, let the user know more rows may exist so the UI can
            # stay lean by default.
            injected_top = ctx.get("injected_default_top")
            if injected_top and len(rows) >= injected_top:
                parts.append(
                    f"\n> _Showing the default TOP {injected_top} rows to keep the "
                    f"response snappy. Reply with **'top 50'**, **'top 100'**, or "
                    f"**'show more'** to widen the result._"
                )
        else:
            parts.append("**No results found.** The filters may be too restrictive  try broadening your search.")

        if post_warnings:
            for w in post_warnings:
                if "0 rows" not in w:  # already handled above
                    parts.append(f"\n> **Note:** {w}")

        # Embed data_json for DataVisualizer follow-up ("plot this")
        # Wrapped in HTML comment so it's invisible to users but accessible to the LLM agent
        if columns and rows:
            serializable_rows = []
            for row in rows:
                serializable_row = []
                for v in row:
                    if v is None:
                        serializable_row.append(None)
                    elif isinstance(v, bool):
                        serializable_row.append(v)
                    elif isinstance(v, Decimal):
                        # json.dumps cannot serialize Decimal — coerce to float so the
                        # hidden data_json comment stays valid JSON and the visualizer
                        # can parse it downstream.
                        serializable_row.append(float(v))
                    elif isinstance(v, (int, float)):
                        serializable_row.append(v)
                    else:
                        serializable_row.append(str(v))
                serializable_rows.append(serializable_row)
            _table_label = ctx.get("table_config", {}).get("label", "")
            data_json_obj = {"columns": columns, "rows": serializable_rows, "source": _table_label}
            parts.append(f"\n<!-- data_json:{json.dumps(data_json_obj)} -->")

        results_section = "\n".join(parts)

        #  Build DETAILS section (collapsible) 
        details = []
        details.append(f"\n<details><summary>SQL Query & Pipeline Details</summary>\n")
        # Encode ALL non-alphanumeric characters as HTML numeric entities.
        # This prevents the Worker LLM from "reading" and mangling the SQL
        # (e.g. dropping SQL operators, single-quoted strings, etc.)
        _raw_sql = self._format_sql_display(sql)
        _safe_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 \n")
        _display_sql = "".join(c if c in _safe_chars else f"&#{ord(c)};" for c in _raw_sql)
        details.append(f'**SQL:**\n<pre style="background:#1e293b;color:#e2e8f0;padding:12px;border-radius:6px;overflow-x:auto;font-size:12.5px"><code>{_display_sql}</code></pre>\n')

        if all_fixes:
            details.append("**Auto-Fixes:**")
            for f in all_fixes:
                details.append(f"- {f}")
            details.append("")

        conf = intent.get("confidence", 0)
        bar_len = int(conf * 20)
        bar = "\u2588" * bar_len + "\u2591" * (20 - bar_len)

        details.append("**Pipeline Trace:**")
        _st = ctx.get("spend_type", "direct")
        _vc = ctx.get("table_config", {})
        details.append(f"0. **View**  `{_vc.get('table_name', '?')}` ({_vc.get('label', _st)})")
        if user_email:
            details.append(f"   - User: `{user_email}`" + (" (auto-resolved)" if not (self.user_email or "").strip() else ""))
        if price_masked:
            details.append(f"   - Access control: masked columns {', '.join(price_masked)}")
        details.append(f"1. **Query Analysis**  Intent: `{intent.get('primary_intent','?')}` ({conf:.0%}) {bar}")
        if raw_q != norm_q:
            details.append(f"   - Normalized: `{norm_q}`")
        exps = normalizer.get("expansions", [])
        if exps:
            details.append(f"   - Expansions: {', '.join(str(e) for e in exps)}")
        aliases = normalizer.get("alias_resolutions", [])
        for a in aliases:
            details.append(f"   - Alias: `{a.get('alias','')}`  `{a.get('sql_filter','')}`")

        if sl:
            mappings = [f"`{t}`  `{c}`" for t, c in sl.get("resolved_columns", {}).items()]
            details.append(f"2. **Schema Linking**  {', '.join(mappings)}")
            ents = sl.get("detected_entities", [])
            if ents:
                # Normalize: LLM may return entities as dicts or strings
                ent_names = []
                for e in ents:
                    if isinstance(e, dict):
                        ent_names.append(e.get("name", e.get("entity", str(e))))
                    else:
                        ent_names.append(str(e))
                details.append(f"   - Entities: {', '.join(ent_names)}")

        details.append(f"3. **Context**  ~{tok_est} tokens, {n_ex}/{tot_ex} examples")
        details.append(f"4. **SQL Generation**  {gen_method.upper()}")

        if len(retry_history) > 1:
            details.append(f"5. **Execution**  {len(rows)} rows, {exec_ms}ms (after {len(retry_history)} attempts)")
            for rh in retry_history:
                if rh.get("error"):
                    details.append(f"   - Attempt {rh['attempt']}: ERROR  {rh['error'][:100]}")
                elif rh.get("zero_rows"):
                    details.append(f"   - Attempt {rh['attempt']}: 0 rows  judge LLM corrected query")
                else:
                    details.append(f"   - Attempt {rh['attempt']}: {rh['rows']} rows (success)")
        else:
            details.append(f"5. **Execution**  {len(rows)} rows, {exec_ms}ms")

        for ev in trace_events:
            if "Executed" not in ev and "Validated" not in ev and "Attempt" not in ev and "Retry" not in ev and "Judge" not in ev:
                details.append(f"   - {ev}")

        details.append(f"\n</details>")

        details_section = "\n".join(details)

        retry_tag = f" | retries: {len(retry_history) - 1}" if len(retry_history) > 1 else ""
        self.status = f"{len(rows)} rows in {exec_ms}ms | {gen_method}{retry_tag}"

        # Wrap output with explicit copy instructions for the Worker LLM.
        # Without this, the LLM tends to summarize the output in one sentence
        # instead of passing through the HTML tables and details blocks.
        full_output = results_section + "\n" + details_section
        wrapped = (
            "[TOOL OUTPUT — COPY EVERYTHING BETWEEN THE MARKERS INTO YOUR RESPONSE]\n"
            "===BEGIN===\n"
            f"{full_output}\n"
            "===END===\n"
            "[Paste everything between ===BEGIN=== and ===END=== into your response VERBATIM, "
            "then add one insight sentence. Do NOT summarize. Do NOT omit the HTML table.]"
        )
        return Message(text=wrapped)
