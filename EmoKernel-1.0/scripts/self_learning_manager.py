"""
自学习管理器
为 emotion-companion 提供三点能力：
1. 用户纠错学习 (feedback_log)
2. 词汇晋升 (word_promotion)
3. 个性化情感模型
"""

import json
import os
import time
from collections import defaultdict
from typing import Dict, List, Optional


class LearningManager:
    PROMOTION_THRESHOLD = 3

    def __init__(self, workspace_dir: str):
        self.workspace_dir = workspace_dir
        self.profile_dir = os.path.join(workspace_dir, "emotion_profile")
        os.makedirs(self.profile_dir, exist_ok=True)

        self.feedback_path = os.path.join(self.profile_dir, "feedback.json")
        self.promotions_path = os.path.join(self.profile_dir, "promotions.json")
        self.learnings_path = os.path.join(self.profile_dir, "learnings.json")

        self._load()

    def _load(self):
        for key, path, default in [
            ("feedback", self.feedback_path, {"entries": [], "correction_stats": {}}),
            ("promotions", self.promotions_path, {"tracked_words": {}, "promoted_words": {}}),
            ("learnings", self.learnings_path, {"version": 1, "changelog": [], "correction_patterns": {}, "personalized_keywords": {}}),
        ]:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    setattr(self, key, json.load(f))
            else:
                setattr(self, key, dict(default))

    def _save(self):
        for key, path in [
            ("feedback", self.feedback_path),
            ("promotions", self.promotions_path),
            ("learnings", self.learnings_path),
        ]:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(getattr(self, key), f, ensure_ascii=False, indent=2)

    def record_feedback(self, text: str, detected: Dict[str, float], corrected: Dict[str, float]):
        """记录用户纠错"""
        entry = {"timestamp": time.time(), "text": text, "detected": detected, "corrected": corrected}
        self.feedback["entries"].append(entry)
        self.feedback["entries"] = self.feedback["entries"][-200:]

        for wrong_cat in detected:
            for right_cat in corrected:
                key = f"{wrong_cat}->{right_cat}"
                self.feedback["correction_stats"][key] = self.feedback["correction_stats"].get(key, 0) + 1

        self._track_words(text, detected, corrected)
        self._update_patterns()
        self._save()

    def _track_words(self, text: str, detected: Dict[str, float], corrected: Dict[str, float]):
        tracked = self.promotions["tracked_words"]
        promoted = self.promotions["promoted_words"]

        for right_cat in corrected:
            for i in range(len(text) - 1):
                seg = text[i:i+2]
                if len(seg) == 2 and "\u4e00" <= seg[0] <= "\u9fff" and "\u4e00" <= seg[1] <= "\u9fff":
                    if seg not in tracked:
                        tracked[seg] = {}
                    tracked[seg][right_cat] = tracked[seg].get(right_cat, 0) + 1

        for word, cat_counts in list(tracked.items()):
            for cat, count in list(cat_counts.items()):
                if count >= self.PROMOTION_THRESHOLD and not self._is_already_keyword(word, cat):
                    if cat not in promoted:
                        promoted[cat] = []
                    if word not in promoted[cat]:
                        promoted[cat].append(word)
                    cat_counts[cat] = 0

    def _update_patterns(self):
        stats = self.feedback.get("correction_stats", {})
        patterns = {}
        for key, count in stats.items():
            if count >= 2:
                wrong_cat, right_cat = key.split("->")
                delta = min(count * 0.05, 0.25)
                if right_cat not in patterns:
                    patterns[right_cat] = {}
                patterns[right_cat][wrong_cat] = -round(delta, 3)
        old_count = len(self.learnings.get("correction_patterns", {}))
        self.learnings["correction_patterns"] = patterns
        new_count = len(patterns)
        # 版本跟踪：新模式产生时自动升级
        if patterns and new_count > old_count:
            if new_count > old_count:
                self.learnings["version"] = self.learnings.get("version", 1) + 1
                for k, v in patterns.items():
                    if k not in self.learnings.get("correction_patterns", {}):
                        self.learnings.setdefault("changelog", []).append({
                            "version": self.learnings["version"],
                            "timestamp": time.time(),
                            "action": "pattern_add",
                            "category": k,
                            "details": v
                        })

    def _is_already_keyword(self, word: str, cat: str) -> bool:
        try:
            from emotion_lexicon import RICH_KEYWORD_MAP
            return cat in RICH_KEYWORD_MAP and word in RICH_KEYWORD_MAP[cat]["words"]
        except:
            return False

    def get_promoted_keywords(self, cat: str = None) -> Dict[str, List[str]]:
        pw = self.promotions.get("promoted_words", {})
        if cat:
            return {cat: pw.get(cat, [])}
        return dict(pw)

    def adjust_scores(self, text: str, scores: Dict[str, float]) -> Dict[str, float]:
        """根据历史学习调整情绪分析结果"""
        if not scores:
            return scores

        adj = dict(scores)

        # 应用纠错模式
        for right_cat, fixes in self.learnings.get("correction_patterns", {}).items():
            for wrong_cat, delta in fixes.items():
                if wrong_cat in adj:
                    adj[wrong_cat] = adj.get(wrong_cat, 0) + delta
                if right_cat in adj:
                    adj[right_cat] = adj.get(right_cat, 0) - delta

        # 加晋升词汇权重
        for cat, words in self.promotions.get("promoted_words", {}).items():
            for word in words:
                if word in text and cat in adj:
                    adj[cat] = adj.get(cat, 0) + 0.15

        total = sum(adj.values())
        if total > 0:
            adj = {k: round(v / total, 4) for k, v in adj.items()}
        return adj

    def promote_expression(self, expression, emotion, source="web"):
        """Promote an expression learned from web lookup to user keywords."""
        promoted = self.promotions.setdefault("promoted_words", {})
        if emotion not in promoted:
            promoted[emotion] = []
        if expression not in promoted[emotion]:
            promoted[emotion].append(expression)
            self.learnings.setdefault("changelog", []).append({
                "version": self.learnings.get("version", 1),
                "timestamp": __import__("time").time(),
                "action": "web_promote",
                "category": emotion,
                "expression": expression,
                "source": source,
            })
            self.learnings["version"] = self.learnings.get("version", 1) + 1
        self._save()

    def record_tree_event(self, event: dict):
        """Record tree growth event."""
        """记录情感树生长事件到学习记录"""
        self.learnings.setdefault("tree_events", []).append({
            "timestamp": event.get("timestamp", time.time()),
            "label": event.get("label", ""),
            "added": event.get("added", False),
            "parent": event.get("parent", ""),
        })
        self.learnings["tree_events"] = self.learnings["tree_events"][-50:]
        self._save()

    def export_knowledge(self) -> dict:
        """导出可 prtable 的学习知识"""
        return {
            "version": self.learnings.get("version", 1),
            "changelog": self.learnings.get("changelog", [])[-10:],
            "correction_patterns": dict(self.learnings.get("correction_patterns", {})),
            "personalized_keywords": dict(self.learnings.get("personalized_keywords", {})),
            "promoted_words": {k: v for k, v in self.promotions.get("promoted_words", {}).items() if v},
        }

    def get_report(self) -> dict:
        stats = self.feedback.get("correction_stats", {})
        promoted = self.promotions.get("promoted_words", {})
        return {
            "version": self.learnings.get("version", 1),
            "total_feedback": len(self.feedback.get("entries", [])),
            "top_corrections": dict(sorted(stats.items(), key=lambda x: -x[1])[:10]),
            "promoted_words_total": sum(len(v) for v in promoted.values()),
            "promoted_by_category": {k: len(v) for k, v in promoted.items()},
            "latest_changelog": self.learnings.get("changelog", [])[-3:],
}
