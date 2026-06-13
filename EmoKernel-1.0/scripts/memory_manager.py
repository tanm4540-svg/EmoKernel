"""
记忆管理模块：双层架构 — 会话级情绪队列 + 持久化用户情绪档案。
"""

import json
import os
import time
from collections import deque
from typing import Any, Dict, List, Optional


class MemoryManager:
    """双层情绪记忆管理"""

    SHORT_TERM_MAX = 20  # 会话级队列最大长度

    def __init__(self, profile_path: str, user_id: str = "default"):
        self.profile_path = profile_path
        self.user_id = user_id
        self._short_term: deque = deque(maxlen=self.SHORT_TERM_MAX)
        self._profile: dict = self._load_profile()

    def _load_profile(self) -> dict:
        """加载持久化情绪档案"""
        if os.path.exists(self.profile_path):
            with open(self.profile_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "user_id": self.user_id,
            "created_at": time.time(),
            "stats": {},
            "trends": {},
            "highlights": [],
            "tree_snapshots": [],
        }

    def _save_profile(self):
        """保存情绪档案"""
        os.makedirs(os.path.dirname(self.profile_path), exist_ok=True)
        self._profile["updated_at"] = time.time()
        with open(self.profile_path, "w", encoding="utf-8") as f:
            json.dump(self._profile, f, ensure_ascii=False, indent=2)

    def record(self, user_emotion: Dict[str, float], ai_emotion_state: Dict[str, float]):
        """记录一轮对话的情绪状态"""
        entry = {
            "round": len(self._short_term) + 1,
            "user_emotion": user_emotion,
            "ai_emotion_state": ai_emotion_state,
            "timestamp": time.time(),
        }
        self._short_term.append(entry)

        # 更新持久化统计
        self._update_stats(user_emotion)

        # 检测高光时刻
        self._detect_highlight(user_emotion)

    def _update_stats(self, emotion: Dict[str, float]):
        """增量更新长期统计"""
        stats = self._profile.setdefault("stats", {})

        for label, intensity in emotion.items():
            if label not in stats:
                stats[label] = {"mean": intensity, "m2": 0.0, "count": 1}
            else:
                s = stats[label]
                s["count"] += 1
                delta = intensity - s["mean"]
                s["mean"] += delta / s["count"]
                delta2 = intensity - s["mean"]
                s["m2"] += delta * delta2

        # 每日趋势
        today = time.strftime("%Y-%m-%d")
        trends = self._profile.setdefault("trends", {})
        if today not in trends:
            trends[today] = {"dominant_emotion": "", "avg_intensity": 0.0, "rounds": 0}
        day = trends[today]
        day["rounds"] += 1
        if emotion:
            dominant = max(emotion, key=emotion.get)
            avg = sum(emotion.values()) / len(emotion)
            day["dominant_emotion"] = dominant
            day["avg_intensity"] = avg

        self._save_profile()

    def _detect_highlight(self, emotion: Dict[str, float]):
        """检测情绪高光时刻（强度峰值）"""
        if not emotion:
            return
        max_intensity = max(emotion.values())
        highlights = self._profile.setdefault("highlights", [])

        # 强度超过 0.8 且不是最近 1 分钟内已记录过的才新增
        if max_intensity >= 0.8:
            now = time.time()
            recent = any(
                now - h.get("timestamp", 0) < 60 for h in highlights[-3:]
            )
            if not recent:
                top_label = max(emotion, key=emotion.get)
                highlights.append(
                    {
                        "label": top_label,
                        "intensity": max_intensity,
                        "timestamp": now,
                    }
                )
                # 只保留最近 50 条, 先 slice 再重新赋值
                if len(highlights) > 50:
                    self._profile["highlights"] = highlights[-50:]
                else:
                    self._profile["highlights"] = highlights
                self._save_profile()

    def get_current_ai_state(self) -> Dict[str, float]:
        """获取当前 AI 情绪状态（最近一轮）"""
        if not self._short_term:
            return {}
        return dict(self._short_term[-1]["ai_emotion_state"])

    def get_emotion_trend(self) -> Dict[str, Any]:
        """分析会话内情绪趋势：上升/下降/波动"""
        if len(self._short_term) < 3:
            return {"direction": "stable", "delta": 0.0, "message": "情绪数据不足"}

        recent = list(self._short_term)[-5:]
        intensities = [
            sum(e["user_emotion"].values()) / max(len(e["user_emotion"]), 1)
            for e in recent
        ]

        if len(intensities) < 2:
            return {"direction": "stable", "delta": 0.0}

        delta = intensities[-1] - intensities[0]
        if abs(delta) < 0.05:
            direction = "stable"
        elif delta > 0:
            direction = "rising"
        else:
            direction = "falling"

        return {"direction": direction, "delta": round(delta, 4)}

    def get_recent_history(self, n: int = 5) -> List[dict]:
        """获取最近 n 轮历史"""
        return list(self._short_term)[-n:]

    def get_profile_summary(self) -> Dict[str, Any]:
        """获取长期档案摘要"""
        stats = self._profile.get("stats", {})
        trends = self._profile.get("trends", {})
        highlights = self._profile.get("highlights", [])

        # 统计维度
        total_entries = sum(s["count"] for s in stats.values())
        top_emotions = sorted(
            stats.items(), key=lambda x: x[1]["count"], reverse=True
        )[:5]

        # 最近趋势
        recent_days = sorted(trends.keys())[-7:]
        recent_trends = {d: trends[d] for d in recent_days}

        return {
            "total_entries": total_entries,
            "top_emotions": [
                {"label": k, "count": v["count"], "mean_intensity": round(v["mean"], 4)}
                for k, v in top_emotions
            ],
            "recent_highlights": highlights[-5:],
            "recent_daily_trends": recent_trends,
        }


    def cleanup_old_entries(self, keep_days=30, max_highlights=20):
        cutoff = __import__("time").time() - keep_days * 86400
        old_count = 0
        new_highlights = []
        for h in self._profile.get("highlights", []):
            if isinstance(h, dict) and h.get("timestamp", 0) >= cutoff:
                new_highlights.append(h)
            elif isinstance(h, dict):
                old_count += 1
        self._profile["highlights"] = new_highlights[-max_highlights:]
        if "tree_snapshots" in self._profile:
            self._profile["tree_snapshots"] = self._profile["tree_snapshots"][-5:]
        if "trends" in self._profile:
            recent = sorted(self._profile["trends"].keys())[-keep_days:]
            self._profile["trends"] = {k: self._profile["trends"][k] for k in recent if k in self._profile["trends"]}
        self._save_profile()
        return {"cleaned": old_count}

    def archive_tree_snapshot(self, snapshot: dict):
        """归档情感树快照"""
        self._profile.setdefault("tree_snapshots", []).append(snapshot)
        # 只保留最近 10 个快照
        self._profile["tree_snapshots"] = self._profile["tree_snapshots"][-10:]
        self._save_profile()
