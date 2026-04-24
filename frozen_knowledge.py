# Paste this into a Custom Code component's Code tab.
#
# Frozen Knowledge — single-component replacement for the
# [Knowledge Base x 2] + [Knowledge Processor] subgraph.
#
# Why this exists:
#   The stock Knowledge Base component downloads every selected YAML from
#   Azure blob storage on EVERY chat turn — ~8 seconds per KB = ~16 s of
#   avoidable wall time per question. This component fetches blobs ONCE
#   per pod lifetime, parses + indexes them once, and returns the cached
#   Data object on every subsequent call in microseconds.
#
# How it replaces 3 components with 1:
#   - Deletes the need for two "Knowledge Base" components (they downloaded
#     every call and emitted different temp paths each time).
#   - Deletes the need for the "Knowledge Processor" (it re-parsed and
#     rebuilt 14 indexes each call).
#   - Emits a Data object in the SAME tagged shape
#     {"direct": {...}, "indirect": {...}, "_tagged": True}
#     that Talk-to-Data already consumes — plug its output into the same
#     socket the old Knowledge Processor output fed.
#
# Update workflow:
#   1. Edit YAMLs and re-upload via the Knowledge page (same as today).
#   2. Either: `kubectl rollout restart` the deployment, OR toggle the
#      component's `refresh` input to True for one chat (wipes cache,
#      re-fetches, caches again), then toggle back to False.

from agentcore.custom import Node
import hashlib
import json
import os
import re
import time
from pathlib import Path


# ─── Module-level in-memory cache ─────────────────────────────────────────────
# Key = sha256 of (direct_kb_name, indirect_kb_name, additional_rules,
#                  additional_context). Value = (cached_at_epoch, Data).
# Survives across build_output() calls for the pod's lifetime. Cleared on
# pod restart or on explicit `refresh`.
_FROZEN_KNOWLEDGE_CACHE: dict = {}


# ─── File parsing (ported verbatim from knowledge_processor.py) ──────────────

def _parse_content(text, filename=""):
    """Parse file content — tries JSON first, then YAML, then flat YAML."""
    text = text.strip()
    if not text:
        return None

    if text.startswith(("{", "[")):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    try:
        import yaml
        result = yaml.safe_load(text)
        if isinstance(result, (dict, list)):
            if isinstance(result, dict) and len(result) > 3:
                none_count = sum(1 for v in result.values() if v is None)
                if none_count > len(result) * 0.5:
                    pass
                else:
                    return result
            else:
                return result
    except Exception:
        pass

    try:
        return json.loads(text)
    except Exception:
        pass

    result = _parse_flat_yaml(text)
    if result and isinstance(result, dict) and len(result) > 0:
        return result

    return None


# ─── Filename-Based Type Detection ───────────────────────────────────────────

_FILENAME_KEYWORDS = [
    (["knowledgegraph", "knowledge_graph", "kg"],          "knowledge_graph_file"),
    (["contextgraph", "context_graph"],                     "context_graph_file"),
    (["semanticlayer", "semantic_layer", "semantic"],        "semantic_layer_file"),
    (["ontology"],                                          "ontology_file"),
    (["synonym", "synonymn", "synonyms"],                   "synonyms_file"),
    (["businessrule", "business_rule", "business_rules"],   "business_rules_file"),
    (["example", "examples", "fewshot", "few_shot"],        "examples_file"),
    (["germanterm", "german_term", "german_column",
     "domainterm", "domain_term"],                          "domain_terms_file"),
    (["columnvalue", "column_value", "histogram",
     "columnvaluesprofiled"],                                "column_values_file"),
    (["datacontext", "data_context", "dataprofile",
     "data_profile"],                                        "data_context_file"),
    (["entityalias", "entities_alias", "alias"],            "entities_aliases_file"),
    (["entities", "entity"],                                 "knowledge_graph_file"),
    (["antipattern", "anti_pattern"],                       "anti_patterns_file"),
    (["sqltemplate", "sql_template", "template"],           "sql_templates_file"),
    (["columns", "schema_columns"],                         "schema_columns_file"),
]


def _normalize_filename(filename):
    base = os.path.basename(filename).rsplit(".", 1)[0]
    base = base.lower()
    base = re.sub(r'\(\d+\)', '', base)
    base = re.sub(r'[^a-z0-9]', '', base)
    return base


def _detect_type_by_filename(filename):
    if not filename:
        return None
    normalized = _normalize_filename(filename)
    if not normalized:
        return None
    for keywords, slot in _FILENAME_KEYWORDS:
        for kw in keywords:
            if kw in normalized:
                return slot
    return None


def _detect_type_by_content(parsed, filename=""):
    slot = _detect_type_by_filename(filename)
    if slot:
        return slot

    if not isinstance(parsed, dict):
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            if "question" in parsed[0] or "sql" in parsed[0]:
                return "examples_file"
        return None

    keys = set(parsed.keys())

    if "entities" in keys and ("relationships" in keys or "relations" in keys):
        return "knowledge_graph_file"
    if "hierarchies" in keys and ("valid_combinations" in keys or "constraints" in keys):
        return "ontology_file"
    if "columns" in keys and ("entity_mappings" in keys or "cardinality_summary" in keys):
        return "semantic_layer_file"
    if "question_types" in keys:
        return "context_graph_file"
    if "column_synonyms" in keys:
        return "synonyms_file"
    if "columns" in keys and ("aggregations" in keys or "patterns" in keys or "phrases" in keys):
        return "synonyms_file"
    if "metrics" in keys and ("exclusion_rules" in keys or "time_filters" in keys or "sqlserver_syntax" in keys):
        return "business_rules_file"
    if "examples" in keys:
        ex = parsed.get("examples")
        if isinstance(ex, list) and ex and isinstance(ex[0], dict):
            return "examples_file"
    if "column_mappings" in keys or "german_columns" in keys:
        return "domain_terms_file"
    if "table_name" in keys and "columns" in keys:
        return "schema_columns_file"
    if "templates" in keys:
        return "sql_templates_file"
    if "anti_patterns" in keys:
        return "anti_patterns_file"
    if any(k in keys for k in ("region_aliases", "oem_aliases", "business_concepts")):
        return "entities_aliases_file"

    return None


# ─── Flat YAML Parser ────────────────────────────────────────────────────────

_LIKELY_TOP_KEYS = {
    'metadata', 'metrics', 'time_filters', 'exclusion_rules', 'sqlserver_syntax',
    'query_templates', 'question_types', 'columns', 'column_synonyms',
    'column_mappings', 'german_columns', 'concepts', 'hierarchies',
    'valid_combinations', 'constraints', 'entity_mappings', 'cardinality_summary',
    'filter_hints', 'examples', 'entities', 'relationships', 'aggregations',
    'patterns', 'phrases', 'templates', 'anti_patterns', 'table_name',
    'extracted_at', 'total_columns', 'unique_columns', 'duplicate_count',
    'region_aliases', 'country_aliases', 'oem_aliases', 'commodity_aliases',
    'business_concepts', 'exclusions', 'properties', 'columns_by_tier',
}


def _yaml_value(val_str):
    if not val_str:
        return None
    if (val_str.startswith('"') and val_str.endswith('"')) or \
       (val_str.startswith("'") and val_str.endswith("'")):
        return val_str[1:-1]
    if val_str.startswith('[') and val_str.endswith(']'):
        inner = val_str[1:-1].strip()
        if not inner:
            return []
        return [item.strip().strip('"').strip("'") for item in re.split(r',\s*', inner) if item.strip()]
    if val_str.lower() in ('true', 'yes'):
        return True
    if val_str.lower() in ('false', 'no'):
        return False
    if val_str.lower() in ('null', 'none', '~'):
        return None
    try:
        return float(val_str) if '.' in val_str else int(val_str)
    except ValueError:
        return val_str


def _parse_flat_yaml(text):
    lines = text.replace('\r\n', '\n').split('\n')
    result = {}
    current_section = None
    current_item = None
    current_item_key = None
    items_in_section = {}
    section_list_items = []
    in_block_scalar = False
    block_lines = []
    block_key = None
    in_list = False
    list_key = None
    list_items = []
    section_list_data = {}

    def _flush_block():
        nonlocal in_block_scalar, block_lines, block_key
        if in_block_scalar and block_key and current_item is not None:
            current_item[block_key] = '\n'.join(block_lines)
        in_block_scalar = False
        block_lines = []
        block_key = None

    def _flush_list():
        nonlocal in_list, list_items, list_key
        if in_list and list_key and current_item is not None:
            current_item[list_key] = list_items
        in_list = False
        list_items = []
        list_key = None

    def _flush_item():
        nonlocal current_item, current_item_key
        _flush_block()
        _flush_list()
        if current_item_key and current_item is not None:
            if current_item_key == '__list_item__':
                section_list_items.append(current_item)
            else:
                items_in_section[current_item_key] = current_item
        current_item = None
        current_item_key = None

    def _flush_section():
        nonlocal current_section, items_in_section, section_list_items
        _flush_item()
        if current_section:
            if section_list_items and not items_in_section:
                result[current_section] = list(section_list_items)
            elif items_in_section:
                section_data = dict(items_in_section)
                if '__direct__' in section_data:
                    direct = section_data.pop('__direct__')
                    if isinstance(direct, dict):
                        section_data.update(direct)
                result[current_section] = section_data
            elif current_section in section_list_data:
                result[current_section] = section_list_data[current_section]
        items_in_section = {}
        section_list_items = []

    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped:
            _flush_block()
            _flush_list()
            if current_item_key and current_item is not None:
                _flush_item()
            continue
        if stripped.startswith('#'):
            continue
        if in_block_scalar:
            if re.match(r'^[A-Za-z_"\'][A-Za-z0-9_ .()"\'/]*:\s', stripped) or \
               (stripped.endswith(':') and not stripped.startswith(('SELECT', 'FROM', 'WHERE'))):
                _flush_block()
            else:
                block_lines.append(stripped)
                continue
        if stripped.startswith('- '):
            item_val = stripped[2:].strip()
            if not in_list and current_item is None and current_section:
                if current_section not in section_list_data:
                    section_list_data[current_section] = []
                if ': ' in item_val or item_val.endswith(':'):
                    current_item = {}
                    current_item_key = '__list_item__'
                    k, _, v = item_val.partition(':')
                    current_item[k.strip()] = v.strip() if v.strip() else None
                    continue
                else:
                    section_list_data[current_section].append(item_val.strip('"').strip("'"))
                    continue
            if in_list:
                list_items.append(item_val.strip('"').strip("'"))
            elif current_item is not None:
                in_list = True
                list_items = [item_val.strip('"').strip("'")]
            continue
        colon_idx = stripped.find(':')
        if colon_idx > 0:
            key = stripped[:colon_idx].strip().strip('"').strip("'")
            value = stripped[colon_idx + 1:].strip()
            _flush_list()
            if not value:
                if current_section is None or (key in _LIKELY_TOP_KEYS and (current_item is None or current_item_key == '__direct__')):
                    _flush_section()
                    current_section = key
                elif current_item is None:
                    current_item = {}
                    current_item_key = key
                else:
                    in_list = True
                    list_key = key
                    list_items = []
            elif value in ('|', '>'):
                if current_item is None:
                    current_item = {}
                    current_item_key = key
                    in_block_scalar = True
                    block_key = '__content__'
                    block_lines = []
                else:
                    in_block_scalar = True
                    block_key = key
                    block_lines = []
            else:
                if key in _LIKELY_TOP_KEYS and (current_item is None or current_item_key == '__direct__'):
                    _flush_section()
                    result[key] = _yaml_value(value)
                    continue
                if current_item is None:
                    if current_section:
                        current_item = {}
                        current_item_key = '__direct__'
                    else:
                        result[key] = _yaml_value(value)
                        continue
                current_item[key] = _yaml_value(value)
            continue
    _flush_section()
    return result if result else None


# ─── Index builders (ported verbatim) ────────────────────────────────────────

def _build_synonym_map(synonyms_data, semantic_layer, domain_terms):
    synonym_map = {}
    if synonyms_data:
        col_syns = synonyms_data.get("column_synonyms") or synonyms_data.get("columns") or synonyms_data
        if isinstance(col_syns, dict):
            for col_name, syn_info in col_syns.items():
                syns = syn_info if isinstance(syn_info, list) else (syn_info.get("synonyms", []) if isinstance(syn_info, dict) else [])
                for s in syns:
                    s_lower = str(s).lower().strip()
                    if s_lower:
                        synonym_map[s_lower] = {"column": col_name, "source": "synonyms_file"}
        for section_key in ("aggregations", "patterns"):
            section = synonyms_data.get(section_key, {})
            if isinstance(section, dict):
                for name, info in section.items():
                    if isinstance(info, dict):
                        for s in info.get("synonyms", []):
                            s_lower = str(s).lower().strip()
                            if s_lower and s_lower not in synonym_map:
                                synonym_map[s_lower] = {"column": name, "source": f"synonyms_{section_key}"}
        phrases = synonyms_data.get("phrases", {})
        if isinstance(phrases, dict):
            for phrase, info in phrases.items():
                p_lower = str(phrase).lower().strip()
                if p_lower and p_lower not in synonym_map:
                    sql = info.get("sql", "") if isinstance(info, dict) else str(info)
                    synonym_map[p_lower] = {"column": sql, "source": "synonyms_phrases"}
    if semantic_layer:
        for col_name, col_info in semantic_layer.get("columns", {}).items():
            if isinstance(col_info, dict):
                for s in col_info.get("synonyms", []):
                    s_lower = str(s).lower().strip()
                    if s_lower and s_lower not in synonym_map:
                        synonym_map[s_lower] = {"column": col_name, "entity": col_info.get("entity", ""), "source": "semantic_layer"}
    if domain_terms:
        terms = domain_terms.get("column_mappings") or domain_terms.get("german_columns") or domain_terms
        if isinstance(terms, dict):
            for col_name, info in terms.items():
                if isinstance(info, dict):
                    for s in info.get("translations", []):
                        s_lower = str(s).lower().strip()
                        if s_lower and s_lower not in synonym_map:
                            synonym_map[s_lower] = {"column": col_name, "source": "domain_terms"}
                    for k in ("german", "native", "term", "abbreviation"):
                        v = info.get(k, "")
                        if v:
                            synonym_map[v.lower().strip()] = {"column": col_name, "source": "domain_terms"}
    return synonym_map


def _build_entities(kg, sl):
    entities = {}
    if kg:
        for name, info in kg.get("entities", {}).items():
            if isinstance(info, dict):
                entities[name] = {
                    "type": info.get("type", "dimension"),
                    "primary_key": info.get("primary_key", ""),
                    "display_column": info.get("display_column", ""),
                    "columns": info.get("columns", []),
                    "measures": info.get("measures", []),
                    "description": info.get("description", ""),
                }
    if sl:
        for name, mapping in sl.get("entity_mappings", {}).items():
            if isinstance(mapping, dict) and name in entities:
                entities[name]["id_column"] = mapping.get("id_column", "")
    return entities


def _build_hierarchies(ontology):
    hierarchies = {}
    if not ontology:
        return hierarchies
    for name, h in ontology.get("hierarchies", {}).items():
        if isinstance(h, dict):
            levels = [{"level": l.get("level", 0), "name": l.get("name", ""), "column": l.get("column", "")}
                      for l in h.get("levels", []) if isinstance(l, dict)]
            hierarchies[name] = {"name": h.get("name", name), "levels": levels, "description": h.get("description", "")}
    return hierarchies


def _build_business_rules(rules_data):
    rules = {"metrics": {}, "exclusion_rules": [], "time_filters": {}, "sqlserver_syntax": {}}
    if not rules_data:
        return rules
    for name, info in rules_data.get("metrics", {}).items():
        if isinstance(info, dict):
            rules["metrics"][name] = {
                "formula": info.get("formula", info.get("sql", info.get("expression", ""))),
                "filter": info.get("filter", ""),
                "column": info.get("column", ""),
                "aggregation": info.get("aggregation", ""),
                "usage": info.get("usage", ""),
                "description": info.get("description", ""),
                "raw": info,
            }
        else:
            rules["metrics"][name] = {"formula": str(info), "raw": info}
    exclusions = rules_data.get("exclusion_rules", rules_data.get("exclusions", []))
    if isinstance(exclusions, dict):
        ex_struct = {}
        for name, info in exclusions.items():
            if isinstance(info, dict):
                ex_struct[name] = {
                    "filter": info.get("filter", info.get("condition", info.get("sql", ""))),
                    "description": info.get("description", ""),
                    "when_to_apply": info.get("when_to_apply", ""),
                    "raw": info,
                }
            else:
                ex_struct[name] = {"filter": str(info), "raw": info}
        rules["exclusion_rules"] = ex_struct
    elif isinstance(exclusions, list):
        rules["exclusion_rules"] = [str(e) for e in exclusions]
    tf = rules_data.get("time_filters", {})
    if isinstance(tf, dict):
        tf_struct = {}
        for name, info in tf.items():
            if isinstance(info, dict):
                tf_struct[name] = {
                    "filter": info.get("filter", ""),
                    "filter_template": info.get("filter_template", ""),
                    "triggers": info.get("triggers", []),
                    "description": info.get("description", ""),
                    "raw": info,
                }
            else:
                tf_struct[name] = str(info)
        rules["time_filters"] = tf_struct
    sqlserver = rules_data.get("sqlserver_syntax", rules_data.get("sqlserver_specific", {}))
    if isinstance(sqlserver, dict):
        rules["sqlserver_syntax"] = sqlserver
    return rules


def _build_column_metadata(schema_columns, semantic_layer):
    columns = {}
    if schema_columns:
        col_defs = schema_columns.get("columns", schema_columns)
        if isinstance(col_defs, dict):
            for name, info in col_defs.items():
                if isinstance(info, dict):
                    columns[name] = {"type": info.get("type", ""), "category": info.get("category", ""),
                                     "description": info.get("description", ""), "nullable": info.get("nullable", True)}
    if semantic_layer:
        for name, info in semantic_layer.get("columns", {}).items():
            if isinstance(info, dict):
                if name not in columns:
                    columns[name] = {}
                columns[name]["entity"] = info.get("entity", "")
                columns[name]["synonyms"] = info.get("synonyms", [])
                if not columns[name].get("description"):
                    columns[name]["description"] = info.get("description", "")
    return columns


def _index_examples(examples_data):
    examples = []
    if not examples_data:
        return examples
    ex_list = examples_data if isinstance(examples_data, list) else examples_data.get("examples", [])
    for ex in (ex_list if isinstance(ex_list, list) else []):
        if isinstance(ex, dict):
            examples.append({"question": ex.get("question") or ex.get("input", ""),
                             "sql": ex.get("sql") or ex.get("output", ""),
                             "category": ex.get("category", ""), "complexity": ex.get("complexity", 1),
                             "tags": ex.get("tags", []), "id": ex.get("id", "")})
    seen_ids = set()
    deduped = []
    for ex in examples:
        eid = ex.get("id", "")
        if eid and eid in seen_ids:
            continue
        if eid:
            seen_ids.add(eid)
        deduped.append(ex)
    return deduped


def _build_intent_index(context_graph):
    index = {}
    if not context_graph:
        return index
    for intent_name, intent_def in context_graph.get("question_types", {}).items():
        tokens = set()
        for p in intent_def.get("patterns", []):
            for word in re.split(r"\s+", p.lower()):
                word = word.strip(".,?!'\"")
                if len(word) > 1:
                    tokens.add(word)
        index[intent_name] = {"tokens": tokens, "definition": intent_def}
    return index


def _build_anti_patterns(data):
    if not data:
        return []
    if isinstance(data, list):
        return data
    ap = data.get("anti_patterns", data)
    if isinstance(ap, list):
        return ap
    if isinstance(ap, dict):
        result = []
        for name, info in ap.items():
            if isinstance(info, dict):
                result.append({"name": name, **info})
            else:
                result.append({"name": name, "description": str(info)})
        return result
    return []


def _build_sql_templates(data):
    if not data:
        return {}
    if isinstance(data, dict):
        return data.get("templates", data)
    return {}


def _build_column_values(data):
    if not data:
        return {}
    raw = data
    if isinstance(data, dict):
        raw = data.get("column_values", data.get("columns", data.get("histograms", data)))
    if not isinstance(raw, dict):
        return {}
    result = {}
    for col_name, col_info in raw.items():
        if isinstance(col_info, dict):
            examples = col_info.get("examples", col_info.get("values", []))
            if examples and isinstance(examples[0], dict):
                examples = [e.get("value", e) for e in examples if e.get("value") is not None]
            examples = [str(v).strip().strip('"').strip("'") for v in examples if v is not None and str(v).strip()]
            result[col_name] = {
                "cardinality": col_info.get("cardinality", len(examples)),
                "examples": examples,
                "complete": col_info.get("complete", False),
                "null_pct": col_info.get("null_pct", 0),
            }
        elif isinstance(col_info, list):
            examples = [str(v).strip().strip('"').strip("'") for v in col_info if v is not None]
            result[col_name] = {
                "cardinality": len(examples),
                "examples": examples,
                "complete": True,
            }
    return result


def _build_entity_aliases(aliases_data):
    aliases = {}
    if not aliases_data:
        return aliases
    for alias_type, col in [("region_aliases", "REGION"), ("country_aliases", "COUNTRY"),
                            ("oem_aliases", "CUSTOMER"), ("commodity_aliases", "COMMODITY_DESCRIPTION")]:
        for alias, canonical in aliases_data.get(alias_type, {}).items():
            aliases[str(alias).lower()] = {"type": alias_type.replace("_aliases", ""),
                                           "canonical_value": str(canonical), "sql_filter": f"{col} = '{canonical}'"}
    for alias, sql in aliases_data.get("business_concepts", {}).items():
        aliases[str(alias).lower()] = {"type": "business_concept", "sql_filter": str(sql)}
    return aliases


# ─── Azure blob fetch ────────────────────────────────────────────────────────

def _fetch_kb_files_from_blob(kb_name):
    """List every blob whose path contains '/<kb_name>/' and return a list of
    (filename, bytes) tuples. Uses the same env vars and auth flow as the
    stock Knowledge Base component, so no new config is needed — the pod
    already has what it takes to run this if the stock KB works there.

    Raises RuntimeError with a clear message on missing env vars or empty
    results so cold-start failures are obvious instead of silent.
    """
    account_url = os.environ.get("AZURE_STORAGE_ACCOUNT_URL", "").strip().strip("'\"")
    container_name = os.environ.get(
        "AZURE_STORAGE_CONTAINER_NAME", "agentcore-knowledge-container"
    ).strip().strip("'\"")

    if not account_url:
        raise RuntimeError(
            "AZURE_STORAGE_ACCOUNT_URL is not set. Frozen Knowledge cannot fetch "
            "blobs. This pod must have the same env vars as the stock Knowledge "
            "Base component."
        )

    from azure.storage.blob import BlobServiceClient
    from azure.identity import DefaultAzureCredential

    credential = DefaultAzureCredential(
        exclude_environment_credential=True,
        exclude_interactive_browser_credential=True,
    )
    sync_client = BlobServiceClient(account_url=account_url, credential=credential)

    # Guard against user pasting something like "/Knowledge_D_Spend/" or
    # trailing whitespace in the input — normalize before matching.
    needle = f"/{kb_name.strip().strip('/')}/"

    results = []
    try:
        container_client = sync_client.get_container_client(container_name)
        all_blobs = list(container_client.list_blobs())

        matching = [b for b in all_blobs if needle in f"/{b.name}"]
        if not matching:
            raise RuntimeError(
                f"No blobs found in container '{container_name}' matching KB "
                f"name '{kb_name}'. Check the KB name matches exactly what the "
                f"stock Knowledge Base component shows in the builder "
                f"(e.g. 'Knowledge_D_Spend'). Found {len(all_blobs)} blobs in "
                f"container but none with '/{kb_name}/' in the path."
            )

        for blob in matching:
            blob_path = blob.name
            ext = Path(blob_path).suffix.lower().lstrip(".")
            # Mirror the stock component's text-file filter. Binary files
            # (images, pptx, etc.) are intentionally skipped — they would
            # never parse as YAML/JSON knowledge anyway.
            if ext not in ("yaml", "yml", "json", "txt", "md"):
                continue
            try:
                blob_client = container_client.get_blob_client(blob_path)
                file_bytes = blob_client.download_blob().readall()
                filename = Path(blob_path).name
                results.append((filename, file_bytes))
            except Exception as e:
                # Per-file failure shouldn't kill the whole cold-start; record
                # a placeholder so the user can see which file was skipped.
                results.append((Path(blob_path).name, e))
    finally:
        sync_client.close()
        close_credential = getattr(credential, "close", None)
        if callable(close_credential):
            close_credential()

    return results


def _parse_kb_bytes(raw_files):
    """Convert (filename, bytes|Exception) tuples to (filename, parsed) tuples
    suitable for the knowledge_context builder."""
    files = []
    skipped = []
    for filename, payload in raw_files:
        if isinstance(payload, Exception):
            skipped.append((filename, f"download error: {payload}"))
            continue
        try:
            content = payload.decode("utf-8")
        except UnicodeDecodeError:
            skipped.append((filename, "non-UTF-8 content"))
            continue
        parsed = _parse_content(content, filename)
        if parsed is not None:
            files.append((filename, parsed))
            continue
        # Some files (data_context.txt variants) are free text, not YAML/JSON.
        norm = _normalize_filename(filename)
        matched_raw = False
        for keywords, slot in _FILENAME_KEYWORDS:
            if any(kw in norm for kw in keywords):
                if slot == "data_context_file":
                    files.append((filename, content))
                    matched_raw = True
                break
        if not matched_raw:
            skipped.append((filename, "parse failed, unknown slot"))
    return files, skipped


# ─── Component ────────────────────────────────────────────────────────────────

class CodeEditorNode(Node):
    display_name = "Frozen Knowledge"
    description = (
        "Downloads Direct + Indirect knowledge YAMLs from blob ONCE per pod, "
        "parses + indexes them, and caches the result in memory. Replaces the "
        "two Knowledge Base components + Knowledge Processor with a single "
        "custom component that saves ~16s on every chat after the pod's first."
    )
    icon = "snowflake"
    name = "FrozenKnowledge"

    inputs = [
        MessageTextInput(
            name="direct_kb_name",
            display_name="Direct KB Name",
            value="Knowledge_D_Spend",
            info=(
                "The name of the direct-spend Knowledge Base as shown in "
                "your Knowledge page / the stock Knowledge Base component "
                "dropdown. The component lists every blob whose path contains "
                "/<this-name>/ and treats them as Direct YAMLs."
            ),
            required=True,
        ),
        MessageTextInput(
            name="indirect_kb_name",
            display_name="Indirect KB Name",
            value="Knowledge_I_Spend",
            info=(
                "Same idea as Direct KB Name but for indirect-spend YAMLs. "
                "Leave blank if you only have a single KB."
            ),
            required=False,
        ),
        MultilineInput(
            name="additional_rules",
            display_name="Additional Business Rules",
            value="",
            info="Optional free-text rules appended to the knowledge context.",
        ),
        MultilineInput(
            name="additional_context",
            display_name="Additional Domain Context",
            value="",
            info="Optional free-text context appended to the knowledge context.",
        ),
        BoolInput(
            name="refresh",
            display_name="Refresh Cache",
            value=False,
            info=(
                "Toggle to True for ONE chat to force a re-fetch from blob "
                "(use after you upload updated YAMLs), then toggle back to "
                "False. A pod restart also refreshes automatically."
            ),
        ),
    ]

    outputs = [
        Output(display_name="Knowledge Context", name="output", method="build_output"),
    ]

    # ---- build helpers -------------------------------------------------------

    def _build_knowledge_context(self, all_files, additional_rules="", additional_context=""):
        """Same shape as knowledge_processor.py's _build_knowledge_context so
        Talk-to-Data receives an identical Data object."""
        _NAME_MAP = {
            "knowledge_graph_file": "Knowledge Graph", "ontology_file": "Ontology",
            "semantic_layer_file": "Semantic Layer", "context_graph_file": "Context Graph",
            "synonyms_file": "Synonyms", "business_rules_file": "Business Rules",
            "examples_file": "Few-Shot Examples", "domain_terms_file": "Domain Terms",
            "schema_columns_file": "Schema Columns", "sql_templates_file": "SQL Templates",
            "anti_patterns_file": "Anti-Patterns", "column_values_file": "Column Values",
            "entities_aliases_file": "Entity Aliases", "data_context_file": "Data Context",
        }
        slots = {}
        files_detected = []
        for filename, parsed in all_files:
            slot = _detect_type_by_content(parsed, filename)
            if slot:
                slots[slot] = parsed
                display = _NAME_MAP.get(slot, slot)
                count = len(parsed) if isinstance(parsed, (dict, list)) else 1
                files_detected.append({"name": display, "items": count, "filename": filename})

        loaded = len(files_detected)
        synonym_map = _build_synonym_map(slots.get("synonyms_file"), slots.get("semantic_layer_file"), slots.get("domain_terms_file"))
        entities = _build_entities(slots.get("knowledge_graph_file"), slots.get("semantic_layer_file"))
        hierarchies = _build_hierarchies(slots.get("ontology_file"))
        business_rules = _build_business_rules(slots.get("business_rules_file"))
        column_metadata = _build_column_metadata(slots.get("schema_columns_file"), slots.get("semantic_layer_file"))
        examples = _index_examples(slots.get("examples_file"))
        intent_index = _build_intent_index(slots.get("context_graph_file"))
        entity_aliases = _build_entity_aliases(slots.get("entities_aliases_file"))
        anti_patterns = _build_anti_patterns(slots.get("anti_patterns_file"))
        sql_templates = _build_sql_templates(slots.get("sql_templates_file"))
        column_values = _build_column_values(slots.get("column_values_file"))

        data_context_raw = slots.get("data_context_file")
        if data_context_raw:
            if isinstance(data_context_raw, dict):
                dc_text = json.dumps(data_context_raw, indent=2)
            elif isinstance(data_context_raw, str):
                dc_text = data_context_raw
            else:
                dc_text = str(data_context_raw)
            additional_context = (additional_context + "\n\n" + dc_text) if additional_context else dc_text

        return {
            "synonym_map": synonym_map,
            "entities": entities,
            "hierarchies": hierarchies,
            "business_rules": business_rules,
            "column_value_hints": column_values,
            "column_metadata": column_metadata,
            "examples": examples,
            "intent_index": intent_index,
            "sql_templates": sql_templates,
            "anti_patterns": anti_patterns,
            "entity_aliases": entity_aliases,
            "additional_business_rules": additional_rules,
            "additional_domain_context": additional_context,
            "knowledge_files_loaded": loaded,
            "total_knowledge_slots": 14,
            "synonym_count": len(synonym_map),
            "entity_count": len(entities),
            "hierarchy_count": len(hierarchies),
            "example_count": len(examples),
            "column_count": len(column_metadata),
            "entity_alias_count": len(entity_aliases),
            "anti_pattern_count": len(anti_patterns),
            "sql_template_count": len(sql_templates),
            "column_values_count": len(column_values),
            "files_detected": files_detected,
        }

    # ---- main entrypoint -----------------------------------------------------

    def build_output(self) -> Data:
        direct_kb = (getattr(self, "direct_kb_name", "") or "").strip()
        indirect_kb = (getattr(self, "indirect_kb_name", "") or "").strip()
        additional_rules = (getattr(self, "additional_rules", "") or "").strip()
        additional_context = (getattr(self, "additional_context", "") or "").strip()
        force_refresh = bool(getattr(self, "refresh", False))

        if not direct_kb and not indirect_kb:
            self.status = "ERROR: no KB name configured"
            return Data(data={"error": True, "message": (
                "Frozen Knowledge: set at least one of direct_kb_name / "
                "indirect_kb_name. Typical values: 'Knowledge_D_Spend', "
                "'Knowledge_I_Spend'."
            )})

        # Build a deterministic cache key over every input that affects output.
        h = hashlib.sha256()
        for part in (direct_kb, indirect_kb, additional_rules, additional_context):
            h.update(part.encode("utf-8", "ignore"))
            h.update(b"\x00")
        cache_key = h.hexdigest()

        if not force_refresh:
            cached = _FROZEN_KNOWLEDGE_CACHE.get(cache_key)
            if cached is not None:
                cached_at, cached_data = cached
                age_s = time.monotonic() - cached_at
                self.status = f"[frozen HIT] cache age {age_s:.1f}s (pod uptime)"
                return cached_data

        # Cache miss (or refresh forced) — do the one-time fetch + build.
        self.status = "[frozen MISS] fetching YAMLs from blob…"
        t0 = time.monotonic()

        direct_raw, direct_skipped = [], []
        indirect_raw, indirect_skipped = [], []

        if direct_kb:
            try:
                raw = _fetch_kb_files_from_blob(direct_kb)
                direct_raw, direct_skipped = _parse_kb_bytes(raw)
            except Exception as e:
                self.status = f"ERROR direct KB: {type(e).__name__}: {e}"
                return Data(data={"error": True, "message": f"Direct KB fetch failed: {e}"})

        if indirect_kb:
            try:
                raw = _fetch_kb_files_from_blob(indirect_kb)
                indirect_raw, indirect_skipped = _parse_kb_bytes(raw)
            except Exception as e:
                self.status = f"ERROR indirect KB: {type(e).__name__}: {e}"
                return Data(data={"error": True, "message": f"Indirect KB fetch failed: {e}"})

        direct_ctx = self._build_knowledge_context(direct_raw, additional_rules, additional_context)
        indirect_ctx = self._build_knowledge_context(indirect_raw, additional_rules, additional_context) \
                       if indirect_kb else {}

        context = {
            "direct": direct_ctx,
            "indirect": indirect_ctx,
            "_tagged": bool(indirect_kb),
        }

        # Summary for the `text_key` output — shown by the builder UI as the
        # component's text preview.
        direct_loaded = direct_ctx.get("knowledge_files_loaded", 0)
        indirect_loaded = indirect_ctx.get("knowledge_files_loaded", 0) if indirect_kb else 0
        summary_lines = [
            "**Frozen Knowledge** (in-memory cache, no re-download per chat)",
            f"**Direct KB ({direct_kb}):** {direct_loaded}/14 files "
            f"| {direct_ctx.get('synonym_count', 0)} synonyms "
            f"| {direct_ctx.get('example_count', 0)} examples",
        ]
        for fd in direct_ctx.get("files_detected", []):
            summary_lines.append(f"  - {fd['name']} ({fd['items']} items) [{fd['filename']}]")
        if direct_skipped:
            summary_lines.append(f"  skipped: {direct_skipped}")

        if indirect_kb:
            summary_lines.append(
                f"\n**Indirect KB ({indirect_kb}):** {indirect_loaded}/14 files "
                f"| {indirect_ctx.get('synonym_count', 0)} synonyms "
                f"| {indirect_ctx.get('example_count', 0)} examples"
            )
            for fd in indirect_ctx.get("files_detected", []):
                summary_lines.append(f"  - {fd['name']} ({fd['items']} items) [{fd['filename']}]")
            if indirect_skipped:
                summary_lines.append(f"  skipped: {indirect_skipped}")

        elapsed_ms = int((time.monotonic() - t0) * 1000)
        summary_lines.append(f"\nOne-time cold-start cost: {elapsed_ms} ms "
                             f"(subsequent chats reuse this cache).")
        context["summary"] = "\n".join(summary_lines)

        result = Data(data=context)
        result.text_key = "summary"

        # Store in process-level cache. Bounded at 16 distinct keys — typical
        # deployments only have 1-2 (one config per flow).
        _FROZEN_KNOWLEDGE_CACHE[cache_key] = (time.monotonic(), result)
        if len(_FROZEN_KNOWLEDGE_CACHE) > 16:
            oldest = min(_FROZEN_KNOWLEDGE_CACHE.items(), key=lambda kv: kv[1][0])[0]
            _FROZEN_KNOWLEDGE_CACHE.pop(oldest, None)

        self.status = (
            f"[frozen BUILT] Direct={direct_loaded}/14, Indirect={indirect_loaded}/14, "
            f"{elapsed_ms}ms — cached for pod lifetime"
        )
        return result
