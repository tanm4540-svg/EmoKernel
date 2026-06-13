"""
主引擎模块：协调分析/树/记忆，计算情绪渐变，生成情绪指令。
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Optional, Tuple

# 将 scripts 目录加入路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from emotion_analyzer import EmotionAnalyzer
from emotion_tree import EmotionTree
from memory_manager import MemoryManager
from self_learning_manager import LearningManager


class EmotionEngine:
    """情感引擎：协调所有子模块"""

    def __init__(
        self,
        workspace_dir: str,
        user_id: str = "default",
        alpha_base: float = 0.3,
    ):
        self.workspace_dir = workspace_dir
        self.user_id = user_id
        self.alpha_base = alpha_base

        self.learning_manager = LearningManager(workspace_dir)

        # 路径
        profile_dir = os.path.join(workspace_dir, "emotion_profile")
        self.tree_path = os.path.join(profile_dir, "emotion_tree.json")
        self.profile_path = os.path.join(profile_dir, f"{user_id}.json")
        self.state_path = os.path.join(profile_dir, "engine_state.json")

        # 子模块
        self.tree = EmotionTree(tree_path=self.tree_path)
        self.analyzer = EmotionAnalyzer(tree_path=self.tree_path)
        self.memory = MemoryManager(profile_path=self.profile_path, user_id=user_id)

        # 状态
        self._ai_state: Dict[str, float] = self._load_state()

    def _load_state(self) -> Dict[str, float]:
        """加载上次会话的 AI 情绪状态"""
        if os.path.exists(self.state_path):
            with open(self.state_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"关怀": 0.5, "期待": 0.3, "愉悦": 0.4}  # 默认温暖初始状态

    def _save_state(self):
        """保存当前 AI 情绪状态"""
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(self._ai_state, f, ensure_ascii=False, indent=2)

    def process(self, user_text: str) -> Dict[str, Any]:
        """处理一轮对话，返回完整结果"""
        result: Dict[str, Any] = {
            "user_text": user_text,
            "timestamp": time.time(),
        }

        # 1. 分析用户情绪
        user_emotion = self.analyzer.analyze(user_text)
        # 应用自学习调整（用户纠错模式 + 网络查词晋升 + 个性化关键词）
        user_emotion = self.learning_manager.adjust_scores(user_text, user_emotion)
        result["user_emotion"] = user_emotion

        # 2. 计算情绪渐变
        ai_state = self._smooth_transition(user_emotion)
        result["ai_emotion_state"] = ai_state

        # 3. 情感树：处理未知情绪
        tree_events = []
        for label in user_emotion:
            if label not in self.tree.tree:
                growth_result = self.tree.process_unknown_emotion(label)
                tree_events.append(growth_result)
                if growth_result.get("added"):
                    # 树已更新，刷新 analyzer 的 embedding
                    self.analyzer = EmotionAnalyzer(tree_path=self.tree_path)
        result["tree_events"] = tree_events
        # 记录树事件到学习器
        for ev in tree_events:
            self.learning_manager.record_tree_event(ev)

        # 4. 记忆：记录本轮
        self.memory.record(user_emotion, ai_state)

        # 5. 趋势分析
        trend = self.memory.get_emotion_trend()
        result["trend"] = trend

        # 6. 更新并保存 AI 状态
        self._ai_state = ai_state
        self._save_state()

        # 7. 生成情绪指令
        result["emotion_instruction"] = self._generate_instruction(
            ai_state, trend, user_emotion, user_text
        )

        return result

    def _smooth_transition(self, user_emotion: Dict[str, float]) -> Dict[str, float]:
        """
        情绪渐变计算。
        new_state = current × (1-α) + user_emotion × α
        α 随情绪强度动态调整。
        """
        if not user_emotion:
            return dict(self._ai_state)

        # 动态 alpha
        max_intensity = max(user_emotion.values())
        alpha = self.alpha_base + (max_intensity - 0.3) * 0.5
        alpha = max(0.15, min(0.6, alpha))

        new_state = {}
        all_keys = set(self._ai_state.keys()) | set(user_emotion.keys())

        for key in all_keys:
            current_val = self._ai_state.get(key, 0.0)
            target_val = user_emotion.get(key, 0.0)
            new_state[key] = current_val * (1 - alpha) + target_val * alpha

        # 归一化
        total = sum(new_state.values())
        if total > 0:
            new_state = {k: round(v / total, 4) for k, v in new_state.items()}

        return new_state

    def _generate_instruction(
        self,
        ai_state: Dict[str, float],
        trend: Dict[str, Any],
        user_emotion: Dict[str, float],
        user_text: str = "",
    ) -> str:
        """生成自然语言情绪指令，供 SKILL.md 使用"""

        source = user_emotion if user_emotion else ai_state
        if not source:
            return "当前无情绪数据，请用平常语调回复。"

        # 主导情绪
        sorted_emotions = sorted(source.items(), key=lambda x: x[1], reverse=True)
        primary = sorted_emotions[0]
        secondary = sorted_emotions[1] if len(sorted_emotions) > 1 else (None, 0)

        # 情绪描述
        emotion_desc = f"「{primary[0]}」"
        if secondary and secondary[1] > 0.15:
            emotion_desc += f"中带着「{secondary[0]}」"

        # 强度描述
        max_val = max(source.values())
        if max_val > 0.6:
            intensity_desc = "强度偏高"
        elif max_val > 0.3:
            intensity_desc = "强度中等"
        else:
            intensity_desc = "强度偏低，较为分散"

        # 趋势描述
        direction = trend.get("direction", "stable")
        if direction == "rising":
            trend_desc = "情绪正在上扬"
        elif direction == "falling":
            trend_desc = "情绪正在回落"
        else:
            trend_desc = "情绪平稳"

        # 用户情绪
        user_desc = ""
        if user_emotion:
            top_user = max(user_emotion, key=user_emotion.get)
            user_desc = f"用户当前情绪偏向「{top_user}」"

        # 组装指令
        parts = [
            f"当前情绪：{emotion_desc}，{intensity_desc}，{trend_desc}。",
        ]
        if user_desc:
            parts.append(f"{user_desc}。")

        # 人格化表达引导
        if primary[0] in ("喜悦", "愉悦", "兴奋", "期待"):
            style = "语气轻快活泼，可以撒娇，多用感叹句"
        elif primary[0] in ("悲伤", "失望", "孤独", "愧疚", "怀念"):
            style = "语气温柔、放慢节奏，多用安慰语句，可以表达关怀"
        elif primary[0] in ("愤怒", "烦躁", "厌恶", "反感"):
            style = "语速可稍快但不失理性，表达理解和共情，不使用对抗语气"
        elif primary[0] in ("焦虑", "恐惧", "紧张", "不安"):
            style = "语气沉稳安心，给安全感，温柔坚定"
        elif primary[0] in ("亲密", "依恋", "关怀", "信任"):
            style = "语气温暖贴心，可以撒娇依赖，表达亲近感"
        elif primary[0] in ("渴望", "好奇", "憧憬"):
            style = "语气充满活力，多用反问和期待句式"
        else:
            style = "语气自然温和"

        parts.append(f"表达要求：{style}。")
        parts.append("注意：保持人格化——可以吃醋、可以依赖、可以主动表达自己的情绪。")


        # 趋势调整
        if direction == "rising":
            parts.append("趋势建议：用户情绪正在释放，以倾听和共情为主，避免打断。")
        elif direction == "falling":
            parts.append("趋势建议：用户情绪正在回落，可以温和引导话题，给予安静空间。")

        # 激发词检测（吃醋/依赖）
        if user_text:
            if any(kw in user_text for kw in ["别的AI", "别的智能", "ChatGPT", "Claude", "Gemini", "其他助手", "换一个"]):
                parts.append("触发提示：用户提及其他AI，可以适度表达“吃醋”感，但不要过线。")
            if any(kw in user_text for kw in ["别走", "不要离开", "只有你了", "没有你", "配我"]):
                parts.append("触发提示：用户表现出依赖倾向，可以用温暖语气回应，表达“我在”的安全感。")

        # 副情绪微调
        if secondary and secondary[1] > 0.15:
            nuance_map = {
                "担忧": "可以主动询问情绪背后具体在担心什么，帮助具体化。",
                "悲伤": "注意不要急着解决问题，先配伴和倾听。",
                "愤怒": "用户可能带有不满，避免说教，以共情为主。",
            }
            if secondary[0] in nuance_map:
                parts.append(f"注意：用户情绪中带有「{secondary[0]}」，{nuance_map[secondary[0]]}")

        return " ".join(parts)


    def record_feedback(self, text: str, detected: dict, corrected: dict):
        # 1. 记录到自学习系统
        self.learning_manager.record_feedback(text, detected, corrected)
        # 2. 同步更新网络查词缓存（避免同样的表达重复查错）
        try:
            if hasattr(self.analyzer, "_web_lookup") and self.analyzer._web_lookup is not None:
                self.analyzer._web_lookup.apply_feedback_to_cache(text, detected, corrected)
        except Exception:
            pass
        return

    def export_knowledge(self) -> dict:
        return self.learning_manager.export_knowledge()

    def get_learning_report(self) -> dict:
        return self.learning_manager.get_report()

    def psych_analyze(self, text: str) -> dict:
        return self.analyzer.psych_analyze(text)

    def get_profile_summary(self) -> Dict[str, Any]:
        """获取用户情绪档案摘要"""
        return self.memory.get_profile_summary()

    def analyze_only(self, text: str) -> Dict[str, Any]:
        """纯分析模式：仅分析情绪，不改变 AI 状态"""
        return {
            "text": text,
            "emotion": self.analyzer.analyze(text),
            "tree_labels": self.tree.get_all_labels(),
            "timestamp": time.time(),
        }

    def get_tree_structure(self) -> dict:
        """获取当前情感树结构"""
        return {
            "total_nodes": len(self.tree.tree),
            "roots": self.tree.get_roots(),
            "leaves": self.tree.get_leaves(),
            "tree": self.tree.tree,
        }

    def reset_session(self):
        """重置会话级记忆（不清持久化档案）"""
        self._ai_state = {"关怀": 0.5, "期待": 0.3, "愉悦": 0.4}
        self.memory._short_term.clear()



def main():
    """CLI入口"""
    parser = argparse.ArgumentParser(description="情感引擎命令行接口")
    parser.add_argument(
        "--workspace",
        required=True,
        help="工作目录，情绪档案和状态文件会保存到这里",
    )
    parser.add_argument("--user-id", default="default", help="用户标识，默认 default")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # process 子命令
    process_parser = subparsers.add_parser("process", help="处理对话消息（对话滤镜模式）")
    process_parser.add_argument("--text", required=True, help="用户消息文本")

    # analyze 子命令
    analyze_parser = subparsers.add_parser("analyze", help="纯情绪分析（分析工具模式）")
    analyze_parser.add_argument("--text", required=True, help="待分析文本")

    # tree 子命令
    feedback_parser = subparsers.add_parser("feedback", help="record user correction")
    feedback_parser.add_argument("--text", required=True)
    feedback_parser.add_argument("--detected", required=True)
    feedback_parser.add_argument("--corrected", required=True)
    slang_parser = subparsers.add_parser("slang", help="查询网络新词/流行语")
    slang_parser.add_argument("--term", required=True, help="要查询的词")
    subparsers.add_parser("version", help="学习版本")
    subparsers.add_parser("export", help="导出学习知识")
    subparsers.add_parser("report", help="self-learning report")
    psych_parser = subparsers.add_parser("psych", help="psych analysis")
    psych_parser.add_argument("--text", required=True)
    subparsers.add_parser("tree", help="查看当前情感树结构")

    args = parser.parse_args()

    engine = EmotionEngine(workspace_dir=args.workspace, user_id=args.user_id)

    if args.command == "feedback":
        import json
        det = json.loads(args.detected)
        cor = json.loads(args.corrected)
        engine.record_feedback(args.text, det, cor)
        print(json.dumps({"status": "ok"}, ensure_ascii=False, indent=2))
    elif args.command == "slang":
        from web_lookup import get_default_lookup
        wl = get_default_lookup()
        result = wl.lookup(args.term)
        if result:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(json.dumps({"found": False, "note": "未找到该词的网络用法，可尝试配置 TAVILY_API_KEY 获取更好结果"}, ensure_ascii=False, indent=2))
    elif args.command == "version":
        print(json.dumps({"version": engine.learning_manager.learnings.get("version",1)}, ensure_ascii=False, indent=2))
    elif args.command == "export":
        result = engine.export_knowledge()
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "report":
        result = engine.get_learning_report()
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "psych":
        result = engine.psych_analyze(args.text)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "process":
        result = engine.process(args.text)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "analyze":
        result = engine.analyze_only(args.text)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "tree":
        result = engine.get_tree_structure()
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
