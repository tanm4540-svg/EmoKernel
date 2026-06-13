"""
        return result
情感分析模块：对文本进行多标签情绪分类，输出情绪向量。
基于轻量 sentence-transformers 做语义嵌入 + 余弦相似度匹配。
"""

import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional

import time
import re

from emotion_lexicon import RICH_KEYWORD_MAP, COGNITIVE_DISTORTIONS, ATTACHMENT_PATTERNS
from model_cache import ensure_model, has_model, get_model, compute_embeddings

import numpy as np

# 惰性加载：不在导入时下载模型，避免网络超时阻塞

class EmotionAnalyzer:
    """文本情绪多标签分类器"""

    def __init__(
        self,
        tree_path: Optional[str] = None,
        top_k: int = 5,
        min_intensity: float = 0.15,
    ):
        self.top_k = top_k
        self.min_intensity = min_intensity
        # 始终加载词库（即便有模型也需要用于 web_lookup 的"不认识部分"判断）
        self._keyword_map = {k: v["words"] for k, v in RICH_KEYWORD_MAP.items()}
        self._psych_map = RICH_KEYWORD_MAP
        self._node_embeddings: Dict[str, np.ndarray] = {}
        self._tree_path = tree_path
        # 加载 n-gram 分类器作为辅助
        self._ngram_clf = None
        try:
            from ngram_classifier import get_default_classifier
            self._ngram_clf = get_default_classifier()
        except Exception:
            pass
        # 加载网络新词查询
        self._web_lookup = None
        try:
            from web_lookup import get_default_lookup
            self._web_lookup = get_default_lookup(keyword_map=self._keyword_map)
        except Exception:
            pass
        self._load_embeddings()

    def _load_embeddings(self):
        """预计算情感树所有节点的 embedding"""
        

        # 惰性加载模型
        if not has_model():
            ensure_model()

        if not has_model() or not self._tree_path:
            self._use_fallback()
            return
        tree_data = self._read_tree()
        if not tree_data:
            self._use_fallback()
            return

        labels = list(tree_data.keys())
        embeddings = get_model().encode(labels, convert_to_numpy=True)
        for label, emb in zip(labels, embeddings):
            self._node_embeddings[label] = emb

    def _read_tree(self) -> dict:
        if not self._tree_path or not os.path.exists(self._tree_path):
            return {}
        with open(self._tree_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _use_fallback(self):
        """退化模式：关键词词典匹配(使用增强词典)"""
        pass  # keyword_map and psych_map already set in __init__

    def _is_negated(self, text: str, idx: int) -> bool:
        """check if keyword at idx is negated"""
        if idx <= 0:
            return False
        if text[idx - 1] in "不没别无莫":
            return True
        if idx >= 2 and text[idx - 2:idx] in {"不是", "没有", "不太", "并非", "不会", "不要", "不用", "绝不", "从不"}:
            return True
        return False

    def _split_clauses(self, text: str) -> List[str]:
        parts = re.split(r'[。！？.!?\n]', text)
        clauses = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            sub_parts = re.split(r'[，,、；;]', part)
            for sp in sub_parts:
                sp = sp.strip()
                if sp:
                    clauses.append(sp)
        return clauses

    def _compute_clause_weights(self, clauses: List[str]) -> List[float]:
        n = len(clauses)
        weights = [1.0] * n
        if n == 0:
            return weights
        weights[-1] = 1.5
        ADVERSATIVES = {"但是", "可是", "不过", "然而", "却", "其实", "但", "可", "只是", "就是", "反倒", "反而"}
        for i, clause in enumerate(clauses):
            for adv in ADVERSATIVES:
                if adv in clause:
                    weights[i] = 2.0
                    if i + 1 < n:
                        weights[i + 1] = max(weights[i + 1], 1.8)
                    break
        return weights

    def _enhanced_keyword_analyze(self, text: str) -> Dict[str, float]:
        clauses = self._split_clauses(text)
        if not clauses:
            return {}
        weights = self._compute_clause_weights(clauses)
        scores = {}
        for label, keywords in self._keyword_map.items():
            wc = 0.0
            for clause, w in zip(clauses, weights):
                for kw in keywords:
                    idx = 0
                    while True:
                        idx = clause.find(kw, idx)
                        if idx == -1:
                            break
                        if self._is_negated(clause, idx):
                            wc -= 0.5 * w
                        else:
                            wc += 1.0 * w
                        idx += len(kw)
            if abs(wc) > 0:
                raw = wc / len(keywords) * 3
                scores[label] = max(min(raw, 1.0), -0.3)
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        result = dict(sorted_items[:self.top_k])
        result = {k: v for k, v in result.items() if v > 0}
        if result:
            total = sum(result.values())
            if total > 0:
                result = {k: round(v / total, 4) for k, v in result.items()}
        return result

    def psych_analyze(self, text: str) -> dict:
        """心理学分析：识别情绪背后的心理模式和应对策略"""
        if not text or not text.strip():
            return {"emotions": {}, "cognitive_distortions": [], "attachment": None, "coping_suggestions": [], "analysis": ""}
        emotions = self.analyze(text)
        distortions = []
        for d in COGNITIVE_DISTORTIONS:
            for kw in d["keywords"]:
                if kw in text:
                    distortions.append({"name": d["name"], "description": d["description"], "matched_keyword": kw})
                    break
        attachment_found = None
        for pattern_name, pattern_data in ATTACHMENT_PATTERNS.items():
            for kw in pattern_data["keywords"]:
                if kw in text:
                    attachment_found = {"pattern": pattern_name, "suggestion": pattern_data["suggestions"]}
                    break
            if attachment_found:
                break
        coping_suggestions = []
        analysis_parts = []
        psych_details = {}
        for emotion in emotions:
            if emotion in self._psych_map:
                p = self._psych_map[emotion]["psych"]
                psych_details[emotion] = {"valence": p["valence"], "arousal": p["arousal"], "triggers": p["triggers"]}
                if p["analysis"]:
                    analysis_parts.append(p["analysis"])
                for cs in p.get("coping", []):
                    if cs not in coping_suggestions:
                        coping_suggestions.append(cs)
        return {
            "emotions": emotions,
            "cognitive_distortions": distortions,
            "attachment": attachment_found,
            "psych_details": psych_details,
            "coping_suggestions": coping_suggestions,
            "analysis": " ".join(analysis_parts) if analysis_parts else "当前文本情绪特征不明显，未能进行深度心理学分析。",
        }

    def analyze(self, text: str) -> Dict[str, float]:
        """分析文本情绪，返回情绪向量（v1.0 同步投票）"""
        if not text or not text.strip():
            return {}

        # v1.0：始终计算增强关键词（含否定词+分句加权）
        kw_result = self._enhanced_keyword_analyze(text)

        # v1.0：embedding(70%) + 增强关键词(30%) 同步投票
        if has_model() and self._node_embeddings:
            emb_result = self._embedding_analyze(text)
            return self._synchronized_vote(emb_result, kw_result)

        # 退化模式：增强关键词 + n-gram 投票集成
            if kw_result or ng_result:
                result = self._vote_analyze(kw_result, ng_result)
                if self._should_web_lookup(text, kw_result, result):
                    enhanced = self._web_lookup.analyze_text(text, result)
                    if enhanced != result:
                        return enhanced
                return result

        # 回退：纯增强关键词（也走 web_lookup）
        if self._should_web_lookup(text, kw_result, kw_result):
            enhanced = self._web_lookup.analyze_text(text, kw_result)
            if enhanced != kw_result:
                return enhanced
        return kw_result

    def _should_web_lookup(self, text: str, kw_result: Dict, vote_result: Dict) -> bool:
        """判断是否需要触发网络查词：任何一条满足就触发"""
        if self._web_lookup is None:
            return False
        # 条件1：投票结果置信度低
        if vote_result and max(vote_result.values(), default=0) < 0.3:
            return True
        # 条件2：文本有长度但只匹配到1个以下关键词
        if len(text) > 4 and len(kw_result) <= 1:
            return True
        # 条件3：词库覆盖率低（超过60%的文本没被识别）
        try:
            ratio = self._web_lookup.search_coverage_ratio(text)
            if ratio < 0.4:
                return True
        except Exception:
            pass
        return False

    def _vote_analyze(self, kw: Dict[str, float], ng: Dict[str, float]) -> Dict[str, float]:
        """关键词(60%) + n-gram(40%) 投票集成"""
        combined = {}
        for k, v in kw.items():
            combined[k] = v * 0.6
        for k, v in ng.items():
            combined[k] = combined.get(k, 0) + v * 0.4
        total = sum(combined.values())
        if total > 0:
            combined = {k: round(v/total, 4) for k, v in combined.items()}
        return combined

    def _synchronized_vote(self, emb: Dict[str, float], kw: Dict[str, float]) -> Dict[str, float]:
        """v1.0 同步投票：embedding(70%) + 增强关键词(30%)"""
        if not emb:
            return kw
        if not kw:
            return emb
        combined = {}
        all_keys = set(emb.keys()) | set(kw.keys())
        for k in all_keys:
            ev = emb.get(k, 0.0)
            kv = kw.get(k, 0.0)
            combined[k] = ev * 0.7 + kv * 0.3
        total = sum(combined.values())
        if total > 0:
            combined = {k: round(v / total, 4) for k, v in combined.items()}
        return combined

    def _embedding_analyze(self, text: str) -> Dict[str, float]:
        """基于 embedding 的情感分析"""
        if get_model() is None:
            return self._enhanced_keyword_analyze(text)
        text_emb = get_model().encode([text], convert_to_numpy=True)[0]
        text_norm = text_emb / (np.linalg.norm(text_emb) + 1e-8)

        scores = {}
        for label, node_emb in self._node_embeddings.items():
            node_norm = node_emb / (np.linalg.norm(node_emb) + 1e-8)
            sim = float(np.dot(text_norm, node_norm))
            if sim >= self.min_intensity:
                scores[label] = round(sim, 4)

        # 取 top_k
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        result = dict(sorted_items[: self.top_k])

        # softmax 归一化
        if result:
            values = np.array(list(result.values()))
            exp_vals = np.exp((values - np.max(values)) * 3)
            norm_vals = exp_vals / exp_vals.sum()
            result = {k: round(float(v), 4) for k, v in zip(result.keys(), norm_vals)}

        return result
