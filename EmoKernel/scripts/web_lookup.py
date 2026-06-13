"""
Web Lookup - 网络新词/流行语实时查询模块
当静态词库遇到不认识的词时，通过 Tavily API 搜索网络了解含义并映射到情绪类别。
无 API Key 时使用字符特征猜测情绪。
"""

import json, os, time, hashlib, re, urllib.request
from typing import Dict, Optional, List

TAVILY_URL = "https://api.tavily.com/search"

BUILTIN_SLANG = {
  "emo": {
    "meaning": "emotional 的缩写，表示伤感、郁郁、情绪化",
    "emotion": {
      "悲伤": 0.6,
      "郁闷": 0.3
    }
  },
  "破防": {
    "meaning": "情绪失控、被击垮心理防线",
    "emotion": {
      "悲伤": 0.4,
      "委屈": 0.3,
      "惊讶": 0.2
    }
  },
  "我麻了": {
    "meaning": "表示无言以对、累了、不想说话",
    "emotion": {
      "无奈": 0.4,
      "郁闷": 0.3,
      "焦虑": 0.2
    }
  },
  "无语子": {
    "meaning": "无语的变体，无奈无言",
    "emotion": {
      "反感": 0.3,
      "烦躁": 0.3,
      "无奈": 0.2
    }
  },
  "CPU": {
    "meaning": "CPU给我干烧了，表示太复杂理解不了",
    "emotion": {
      "困惑": 0.5,
      "焦虑": 0.2
    }
  },
  "PUA": {
    "meaning": "网络用语中指精神控制、打击式教育",
    "emotion": {
      "愤怒": 0.4,
      "厌恶": 0.3
    }
  },
  "真的会谢": {
    "meaning": "反讽，表示无语、累感",
    "emotion": {
      "烦躁": 0.4,
      "反感": 0.3,
      "无奈": 0.2
    }
  },
  "绝了": {
    "meaning": "太过分、太离谱，多用于愤慨",
    "emotion": {
      "愤怒": 0.3,
      "惊讶": 0.3,
      "反感": 0.2
    }
  },
  "输了": {
    "meaning": "放弃、无奈，也作尿了",
    "emotion": {
      "失望": 0.4,
      "无奈": 0.3
    }
  },
  "算了算了": {
    "meaning": "不想纠缠了，放弃",
    "emotion": {
      "无奈": 0.3,
      "疲惫": 0.3
    }
  }
}

# 单个汉字的情绪倾向映射（用于无 API 兜底猜测）
STRONG_CHAR_MAP = {
    "悲": "悲伤", "惨": "悲伤", "痛": "悲伤", "哭": "悲伤", "泣": "悲伤",
    "伤": "悲伤", "哀": "悲伤", "苦": "悲伤",
    "怒": "愤怒", "气": "愤怒", "火": "愤怒", "愤": "愤怒", "恼": "愤怒",
    "烦": "烦躁", "躁": "烦躁",
    "怕": "恐惧", "恐": "恐惧", "惊": "恐惧", "慌": "恐惧", "惧": "恐惧",
    "爱": "亲密", "恋": "依恋", "念": "怀念", "亲": "亲密", "贴": "亲密",
    "恨": "怨恨", "厌": "厌恶", "嫌": "厌恶", "恶": "厌恶",
    "奇": "好奇", "疑": "困惑", "怪": "好奇",
    "愧": "愧疚", "疚": "愧疚",
    "想": "依恋", "盼": "期待", "望": "期待",
    "急": "焦虑", "虑": "焦虑", "焦": "焦虑",
    "尴": "尴尬", "尬": "尴尬",
    "羡": "嫉妒", "妒": "嫉妒",
    "鄙": "轻蔑", "蔑": "轻蔑", "藐": "轻蔑",
}


class WebLookup:
    def __init__(self, workspace_dir="", keyword_map=None):
        self.api_key = os.environ.get("TAVILY_API_KEY", "")
        self.cache_path = ""
        self._cache = {}
        self._keyword_map = keyword_map
        if self._keyword_map is None:
            try:
                from emotion_lexicon import RICH_KEYWORD_MAP
                self._keyword_map = {k: v["words"] for k, v in RICH_KEYWORD_MAP.items()}
            except ImportError:
                self._keyword_map = {}
        # 预计算所有词库关键词（展平+长词优先）
        self._all_known_words: List[str] = []
        for words in self._keyword_map.values():
            self._all_known_words.extend(words)
        self._all_known_words.sort(key=len, reverse=True)

        if workspace_dir:
            d = os.path.join(workspace_dir, "emotion_profile")
            os.makedirs(d, exist_ok=True)
            self.cache_path = os.path.join(d, "lookup_cache.json")
            self._load_cache()

    def _load_cache(self):
        if self.cache_path and os.path.exists(self.cache_path):
            with open(self.cache_path, "r", encoding="utf-8") as f:
                self._cache = json.load(f)

    def _save_cache(self):
        if self.cache_path:
            self._cache = dict(list(self._cache.items())[-200:])
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, ensure_ascii=False, indent=2)

    def is_available(self):
        return bool(self.api_key)

    def lookup(self, term):
        """Single-query lookup, kept for CLI slang backward compat."""
        return self._single_lookup(term)

    def multi_lookup(self, term):
        """Multi-angle search: multiple query formulations aggregated into one result.

        Each query uses a different angle, then all results are combined.
        This avoids the bias of a single query phrasing.
        """
        key = term.lower()
        if key in BUILTIN_SLANG:
            return dict(BUILTIN_SLANG[key])
        ck = hashlib.md5(key.encode()).hexdigest()
        if ck in self._cache:
            return self._cache[ck]
        if not self.is_available():
            return None

        queries = [
            f"{term} 网络用语 情绪 意思",
            f"{term} 表达什么情感 什么心情",
            f"{term} 是褒义还是贬义 什么情况下使用",
        ]

        all_emotions = {}
        all_meanings = []
        results_count = 0

        for q in queries:
            try:
                payload = json.dumps({
                    "api_key": self.api_key,
                    "query": q,
                    "search_depth": "basic",
                    "include_answer": True,
                    "max_results": 2,
                }).encode()
                req = urllib.request.Request(TAVILY_URL, data=payload,
                    headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read().decode())
                parsed = self._parse(data, term, q)
                if parsed:
                    results_count += 1
                    for emo, score in parsed.get("emotion_map", {}).items():
                        all_emotions[emo] = all_emotions.get(emo, 0) + score
                    if parsed.get("meaning"):
                        all_meanings.append(parsed["meaning"][:200])
            except Exception:
                continue

        if not all_emotions:
            return None

        total = sum(all_emotions.values())
        if total > 0:
            all_emotions = {k: round(v/total, 3) for k, v in all_emotions.items()}

        result = {
            "meaning": " | ".join(all_meanings)[:500],
            "emotion_map": all_emotions,
            "source": "tavily_multi",
            "queries_aggregated": results_count,
        }

        self._cache[ck] = result
        self._save_cache()
        return result

    def apply_feedback_to_cache(self, text, detected, corrected):
        """Update web_lookup cache when user corrects emotion detection."""
        if not self._cache:
            return
        wrong_emos = list(detected.keys()) if detected else []
        right_emos = list(corrected.keys()) if corrected else []
        if not wrong_emos or not right_emos:
            return
        terms = self._extract_unknown_terms(text)
        for term in terms:
            ck = hashlib.md5(term.lower().encode()).hexdigest()
            if ck in self._cache:
                entry = self._cache[ck]
                emap = entry.get('emotion_map', {})
                changed = False
                for wrong_emo in wrong_emos:
                    if wrong_emo in emap:
                        emap[wrong_emo] = max(0, emap[wrong_emo] - 0.2)
                        changed = True
                for right_emo in right_emos:
                    if right_emo not in emap:
                        emap[right_emo] = 0.0
                    emap[right_emo] += 0.25
                    changed = True
                if changed:
                    total = sum(emap.values())
                    if total > 0:
                        emap = {k: round(v/total, 3) for k, v in emap.items()}
                    entry['emotion_map'] = emap
                    entry['corrected_by_user'] = True
                    self._cache[ck] = entry
        self._save_cache()

    def _single_lookup(self, term):
        """Single-query Tavily search for CLI slang command."""
        key = term.lower()
        if key in BUILTIN_SLANG:
            return dict(BUILTIN_SLANG[key])
        ck = hashlib.md5(key.encode()).hexdigest()
        if ck in self._cache:
            return self._cache[ck]
        if not self.is_available():
            return None
        try:
            payload = json.dumps({
                "api_key": self.api_key,
                "query": f"{term} 网络用语 情绪 意思",
                "search_depth": "basic", "include_answer": True, "max_results": 3
            }).encode()
            req = urllib.request.Request(TAVILY_URL, data=payload,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
        except Exception:
            return None
        result = self._parse(data, term)
        if result:
            self._cache[ck] = result
            self._save_cache()
        return result

    def _parse(self, data, term):
        text = (data.get("answer") or "") + " " + " ".join([r.get("content","") for r in data.get("results",[])[:3]])
        emotion_keywords = {
            "悲伤": ["悲伤","伤心","难过","哭","忧郁","沮丧","sad","emotional","伤感"],
            "疲惫": ["累","疲惫","无力","糟糕","tired","趴","瘫"],
            "郁闷": ["郁闷","抑郁","压抑","depressed","emo","低落"],
            "焦虑": ["焦虑","紧张","不安","anxious","压力","担心","慌张"],
            "愤怒": ["愤怒","生气","火大","angry","不满","气愤","恼火"],
            "反感": ["反感","讨厌","无语","disgust","恶心","排斥"],
            "惊讶": ["惊讶","吃惊","surprise","shock","意外","震惊"],
            "困惑": ["困惑","不懂","复杂","confused","无语","麻了","迷茫"],
            "无奈": ["无奈","放弃","随便","whatever","算了","摆烂"],
            "烦躁": ["烦躁","烦人","焦躁","annoyed","不耐烦"],
        }
        scores = {}
        for emo, kws in emotion_keywords.items():
            cnt = sum(1 for kw in kws if kw in text)
            if cnt > 0:
                scores[emo] = cnt
        if not scores:
            return None
        total = sum(scores.values())
        emap = {k: round(v/total, 3) for k, v in scores.items()}
        return {"meaning": text[:200], "emotion_map": emap, "source": "tavily"}

    def analyze_text(self, text, base_scores, learning_manager=None):
        """分析文本，遇到不认识的说法就上网查。

        三级策略：
        1. 内置流行语词典
        2. 提取不认识的部分 → 有 API 就搜，没 API 用字符特征猜测
        3. 结果合并到 base_scores

        Args:
            text: 用户输入文本
            base_scores: 关键词/n-gram 分析的原始情绪向量
            learning_manager: 可选，传入后会把查到的新词晋升为用户关键词

        Returns:
            增强后的情绪向量
        """
        if not base_scores:
            return base_scores

        # --- 第一级：内置流行语 ---
        for term, info in BUILTIN_SLANG.items():
            if term in text:
                adj = dict(base_scores)
                for emo, boost in info.get("emotion", {}).items():
                    adj[emo] = adj.get(emo, 0) + boost
                total = sum(adj.values())
                return {k: round(v/total, 4) for k, v in adj.items()} if total > 0 else adj

        # --- 第二级：提取不认识的部分 ---
        unknown_terms = self._extract_unknown_terms(text)
        if not unknown_terms:
            return base_scores

        # 汇总查到的情绪
        web_emotions = {}
        for term in unknown_terms:
            if len(term) < 2:
                continue
            result = self.multi_lookup(term)
            if result and result.get("emotion_map"):
                for emo, score in result["emotion_map"].items():
                    web_emotions[emo] = web_emotions.get(emo, 0) + score * 0.5
                # 查到的词晋升到用户词表
                if learning_manager is not None:
                    top_emo = max(result["emotion_map"], key=result["emotion_map"].get)
                    learning_manager.promote_expression(term, top_emo, source="tavily")
            else:
                # 无 API / 查不到 → 字符特征猜测
                guessed = self._guess_emotion(term)
                for emo, score in guessed.items():
                    web_emotions[emo] = web_emotions.get(emo, 0) + score * 0.3

        if not web_emotions:
            return base_scores

        # --- 第三级：合并到原结果 ---
        adj = dict(base_scores)
        total_web = sum(web_emotions.values())
        if total_web > 0:
            web_emotions = {k: v / total_web for k, v in web_emotions.items()}
            for emo, score in web_emotions.items():
                # 用 0.35 权重合并，不覆盖原结果
                adj[emo] = adj.get(emo, 0) + score * 0.35

        total = sum(adj.values())
        if total > 0:
            adj = {k: round(v/total, 4) for k, v in adj.items()}

        return adj

    def _extract_unknown_terms(self, text: str) -> List[str]:
        """提取文本中不被词库覆盖的片段（连贯的未知中文字段）"""
        if not text:
            return []

        # 标记哪些位置被已知词覆盖
        covered = [False] * len(text)
        for kw in self._all_known_words:
            idx = text.find(kw)
            while idx >= 0:
                for i in range(idx, min(idx + len(kw), len(covered))):
                    covered[i] = True
                idx = text.find(kw, idx + 1)

        # 提取未覆盖的连贯中文片段
        unknown_terms = []
        current = ""
        for i, ch in enumerate(text):
            if not covered[i] and '\u4e00' <= ch <= '\u9fff':  # CJK 统一汉字
                current += ch
            else:
                if len(current) >= 2:
                    unknown_terms.append(current)
                current = ""
        if len(current) >= 2:
            unknown_terms.append(current)

        return unknown_terms

    def _guess_emotion(self, term: str) -> Dict[str, float]:
        """无 API 时用字符特征猜测情绪"""
        scores = {}
        for ch in term:
            if ch in STRONG_CHAR_MAP:
                cat = STRONG_CHAR_MAP[ch]
                scores[cat] = scores.get(cat, 0) + 0.3

        if not scores:
            return {}
        total = sum(scores.values())
        return {k: round(v/total, 3) for k, v in scores.items()}

    def search_coverage_ratio(self, text: str) -> float:
        """返回文本中被词库覆盖的比例（0~1），用于判断是否需要查网络"""
        if not text:
            return 1.0
        covered = [False] * len(text)
        for kw in self._all_known_words:
            idx = text.find(kw)
            while idx >= 0:
                for i in range(idx, min(idx + len(kw), len(covered))):
                    covered[i] = True
                idx = text.find(kw, idx + 1)
        return sum(covered) / len(text)


def get_default_lookup(workspace_dir="", keyword_map=None):
    return WebLookup(workspace_dir, keyword_map)
