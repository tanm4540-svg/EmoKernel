"""
情感树模块：树结构管理、自动生长判定、JSON 持久化。
"""

import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from model_cache import ensure_model, has_model, get_model

class EmotionTree:
    """情感分类树，支持自动生长"""

    DEFAULT_TREE = {
        "喜": {
            "parent": None,
            "children": ["愉悦", "兴奋", "满足", "自豪", "感激"],
            "meta": {"depth": 0, "embedding_ref": "", "created_reason": "root", "usage_count": 0, "last_used": 0.0},
        },
        "怒": {
            "parent": None,
            "children": ["烦躁", "愤怒", "暴怒", "嫉妒", "怨恨"],
            "meta": {"depth": 0, "embedding_ref": "", "created_reason": "root", "usage_count": 0, "last_used": 0.0},
        },
        "哀": {
            "parent": None,
            "children": ["悲伤", "失望", "孤独", "怀念", "愧疚"],
            "meta": {"depth": 0, "embedding_ref": "", "created_reason": "root", "usage_count": 0, "last_used": 0.0},
        },
        "惧": {
            "parent": None,
            "children": ["焦虑", "恐惧", "紧张", "不安", "担忧"],
            "meta": {"depth": 0, "embedding_ref": "", "created_reason": "root", "usage_count": 0, "last_used": 0.0},
        },
        "爱": {
            "parent": None,
            "children": ["亲密", "依恋", "关怀", "信任", "崇拜"],
            "meta": {"depth": 0, "embedding_ref": "", "created_reason": "root", "usage_count": 0, "last_used": 0.0},
        },
        "恶": {
            "parent": None,
            "children": ["厌恶", "轻蔑", "反感", "鄙夷", "排斥"],
            "meta": {"depth": 0, "embedding_ref": "", "created_reason": "root", "usage_count": 0, "last_used": 0.0},
        },
        "欲": {
            "parent": None,
            "children": ["渴望", "期待", "好奇", "憧憬", "贪恋"],
            "meta": {"depth": 0, "embedding_ref": "", "created_reason": "root", "usage_count": 0, "last_used": 0.0},
        },
    }

    def __init__(
        self,
        tree_path: str,
        similarity_threshold: float = 0.6,
        growth_trigger_count: int = 3,
    ):
        self.tree_path = tree_path
        self.similarity_threshold = similarity_threshold
        self.growth_trigger_count = growth_trigger_count
        self._uncertain_count: Dict[str, Dict[str, int]] = {}
        self.tree: dict = {}
        self._load_or_init()

    def _load_or_init(self):
        """加载已有树或初始化默认树"""
        if os.path.exists(self.tree_path):
            with open(self.tree_path, "r", encoding="utf-8") as f:
                self.tree = json.load(f)
        else:
            self.tree = self._build_default_tree()
            self._save()

    def _build_default_tree(self) -> dict:
        """构建完整的默认树（补充叶子节点定义）"""
        tree = dict(self.DEFAULT_TREE)
        for node_name, node_data in self.DEFAULT_TREE.items():
            for child in node_data["children"]:
                if child not in tree:
                    tree[child] = {
                        "parent": node_name,
                        "children": [],
                        "meta": {
                            "depth": 1,
                            "embedding_ref": "",
                            "created_reason": "default", "usage_count": 0, "last_used": 0.0,
                            "sample_count": 0,
                        },
                    }
        self._compute_embeddings(tree)
        return tree

    def _compute_embeddings(self, tree: dict):
        """为所有节点计算 embedding"""
        if not has_model():
            return
        labels = list(tree.keys())
        embeddings = get_model().encode(labels, convert_to_numpy=True)
        for label, emb in zip(labels, embeddings):
            tree[label]["meta"]["embedding_ref"] = ",".join(
                [f"{v:.6f}" for v in emb[:32]]
            )

    def _save(self):
        """持久化到 JSON"""
        os.makedirs(os.path.dirname(self.tree_path), exist_ok=True)
        with open(self.tree_path, "w", encoding="utf-8") as f:
            json.dump(self.tree, f, ensure_ascii=False, indent=2)

    def get_all_labels(self) -> List[str]:
        """获取所有情绪标签"""
        return list(self.tree.keys())

    def get_leaves(self) -> List[str]:
        """获取所有叶子节点"""
        return [k for k, v in self.tree.items() if not v["children"]]

    def get_roots(self) -> List[str]:
        """获取根节点"""
        return [k for k, v in self.tree.items() if v["parent"] is None]

    def get_parent(self, label: str) -> Optional[str]:
        """获取父节点"""
        node = self.tree.get(label)
        return node["parent"] if node else None

    def get_children(self, label: str) -> List[str]:
        """获取子节点"""
        node = self.tree.get(label)
        return node["children"] if node else []

    def _similarity(self, label_a: str, label_b: str) -> float:
        """计算两个节点的语义相似度"""
        node_a = self.tree.get(label_a)
        node_b = self.tree.get(label_b)
        if not node_a or not node_b:
            return 0.0

        emb_a_str = node_a["meta"].get("embedding_ref", "")
        emb_b_str = node_b["meta"].get("embedding_ref", "")
        if not emb_a_str or not emb_b_str:
            # 无 embedding 时回退到离散余弦
            return self._discrete_similarity(label_a, label_b)

        emb_a = np.array([float(v) for v in emb_a_str.split(",")])
        emb_b = np.array([float(v) for v in emb_b_str.split(",")])
        return float(
            np.dot(emb_a, emb_b)
            / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-8)
        )

    def _discrete_similarity(self, a: str, b: str) -> float:
        """无 embedding 时的字符级离散相似度"""
        chars_a = set(a)
        chars_b = set(b)
        if not chars_a or not chars_b:
            return 0.0
        intersection = chars_a & chars_b
        union = chars_a | chars_b
        return len(intersection) / len(union)

    def find_nearest_parent(self, new_label: str) -> Tuple[Optional[str], float]:
        """为新情绪标签找到最近的父节点"""
        best_parent = None
        best_sim = 0.0
        for node_name, node_data in self.tree.items():
            if node_name == new_label:
                continue
            sim = self._similarity(new_label, node_name)
            if sim > best_sim:
                best_sim = sim
                best_parent = node_name
        return best_parent, best_sim

    def check_growth(self, new_label: str) -> Tuple[bool, Optional[str], str]:
        """
        判定是否应该自动生长新分支。
        返回: (是否生长, 父节点名或None, 原因描述)
        """
        if new_label in self.tree:
            return False, None, f"'{new_label}' 已存在于情感树中"

        best_parent, best_sim = self.find_nearest_parent(new_label)

        if best_parent is None or best_sim < self.similarity_threshold * 0.5:
            return False, None, f"无法为 '{new_label}' 找到合适的父节点 (最高相似度: {best_sim:.2f})"

        # 检查是否与父节点的现有子节点区分度足够
        existing_children = self.tree[best_parent]["children"]
        for child in existing_children:
            child_sim = self._similarity(new_label, child)
            if child_sim >= self.similarity_threshold:
                return (
                    False,
                    None,
                    f"'{new_label}' 与已有节点 '{child}' 相似度过高 ({child_sim:.2f})，不创建新分支",
                )

        # 累积不确定计数
        if best_parent not in self._uncertain_count:
            self._uncertain_count[best_parent] = {}
        self._uncertain_count[best_parent][new_label] = (
            self._uncertain_count[best_parent].get(new_label, 0) + 1
        )

        if self._uncertain_count[best_parent][new_label] >= self.growth_trigger_count:
            return True, best_parent, f"触发自动生长：'{new_label}' 在 '{best_parent}' 下累积 {self.growth_trigger_count} 次归类不确定"

        remaining = (
            self.growth_trigger_count - self._uncertain_count[best_parent][new_label]
        )
        return (
            False,
            None,
            f"'{new_label}' 还需 {remaining} 次触发才能自动生长",
        )

    def add_node(self, label: str, parent: str, reason: str = "") -> bool:
        """添加新节点"""
        if label in self.tree:
            return False

        parent_node = self.tree.get(parent)
        if not parent_node:
            return False

        depth = parent_node["meta"]["depth"] + 1
        self.tree[label] = {
            "parent": parent,
            "children": [],
            "meta": {
                "depth": depth,
                "embedding_ref": "",
                "created_reason": reason,
                "sample_count": 0,
            },
        }
        parent_node["children"].append(label)

        # 计算新节点的 embedding
        if has_model():
            emb = get_model().encode([label], convert_to_numpy=True)[0]
            self.tree[label]["meta"]["embedding_ref"] = ",".join(
                [f"{v:.6f}" for v in emb[:32]]
            )

        self._save()
        return True

    def process_unknown_emotion(self, label: str) -> dict:
        """处理未知情绪的完整流程：检查 → 判定 → (可选)生长"""
        will_grow, parent, reason = self.check_growth(label)
        result = {
            "label": label,
            "will_grow": will_grow,
            "parent": parent,
            "reason": reason,
        }
        if will_grow and parent:
            success = self.add_node(label, parent, reason)
            result["added"] = success
        return result

    def increment_usage(self, label):
        if label in self.tree:
            self.tree[label]["meta"]["usage_count"] = self.tree[label]["meta"].get("usage_count", 0) + 1
            self.tree[label]["meta"]["last_used"] = __import__("time").time()

    def prune(self, min_usage=3, keep_min_nodes=35):
        if len(self.tree) <= keep_min_nodes:
            return {"pruned": 0, "reason": "already at minimum"}
        leaves = self.get_leaves()
        pruned = []
        for leaf in leaves:
            if len(self.tree) - 1 < keep_min_nodes:
                break
            usage = self.tree[leaf]["meta"].get("usage_count", 0)
            if usage < min_usage:
                parent_name = self.tree[leaf]["parent"]
                if parent_name and parent_name in self.tree:
                    parent = self.tree[parent_name]
                    if leaf in parent["children"]:
                        parent["children"].remove(leaf)
                del self.tree[leaf]
                pruned.append(leaf)
        self._save()
        return {"pruned": len(pruned), "removed": pruned}

    def get_tree_snapshot(self) -> dict:
        """获取树快照（用于长期记忆归档）"""
        return {
            "timestamp": time.time(),
            "total_nodes": len(self.tree),
            "roots": self.get_roots(),
            "leaves": self.get_leaves(),
            "tree": self.tree,
        }
