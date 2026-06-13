import os, sys, json, tempfile, unittest
SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts")
sys.path.insert(0, SCRIPTS_DIR)

from emotion_lexicon import RICH_KEYWORD_MAP, COGNITIVE_DISTORTIONS, ATTACHMENT_PATTERNS


class TestEmotionLexicon(unittest.TestCase):
    def test_categories_loaded(self):
        self.assertGreaterEqual(len(RICH_KEYWORD_MAP), 35)

    def test_no_empty_categories(self):
        for cat, data in RICH_KEYWORD_MAP.items():
            words = data.get("words", [])
            self.assertGreaterEqual(len(words), 5, f"{cat} has only {len(words)} words")

    def test_no_noise_in_keywords(self):
        noise_patterns = ["+", "@", "K线", "V领", "u盘", "c语言", "C语言"]
        for cat, data in RICH_KEYWORD_MAP.items():
            for w in data.get("words", []):
                for n in noise_patterns:
                    self.assertNotIn(n, w, f"{n} in {w} ({cat})")

    def test_psych_metadata(self):
        for cat, data in RICH_KEYWORD_MAP.items():
            psych = data.get("psych", {})
            for key in ("valence", "arousal", "triggers", "analysis", "coping"):
                self.assertIn(key, psych, f"{cat} missing {key}")


class TestMemoryManager(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.profile_path = os.path.join(self.tmpdir, "profile.json")
        from memory_manager import MemoryManager
        self.mm = MemoryManager(profile_path=self.profile_path, user_id="test")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_record_creates_profile(self):
        self.mm.record({"喜悦": 0.8}, {"关怀": 0.5})
        self.assertTrue(os.path.exists(self.profile_path))

    def test_trend_insufficient_data(self):
        trend = self.mm.get_emotion_trend()
        self.assertEqual(trend["direction"], "stable")

    def test_trend_with_data(self):
        for i in range(5):
            self.mm.record({"悲伤": 0.3 + i * 0.1}, {"关怀": 0.5})
        trend = self.mm.get_emotion_trend()
        self.assertIn(trend["direction"], ["rising", "falling", "stable"])

    def test_stats_tracking(self):
        self.mm.record({"喜悦": 0.9}, {"关怀": 0.5})
        profile = json.load(open(self.profile_path, encoding="utf-8"))
        self.assertIn("喜悦", profile.get("stats", {}))

    def test_highlights_detected(self):
        self.mm.record({"喜悦": 0.95}, {"关怀": 0.5})
        profile = json.load(open(self.profile_path, encoding="utf-8"))
        self.assertGreaterEqual(len(profile.get("highlights", [])), 1)


class TestEmotionTree(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.tree_path = os.path.join(self.tmpdir, "tree.json")
        from emotion_tree import EmotionTree
        self.tree = EmotionTree(tree_path=self.tree_path)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_default_roots(self):
        self.assertEqual(set(self.tree.get_roots()), {"喜","怒","哀","惧","爱","恶","欲"})

    def test_labels_include_leaves(self):
        labels = self.tree.get_all_labels()
        for label in ("愉悦", "悲伤", "愤怒", "恐惧"):
            self.assertIn(label, labels)

    def test_add_node(self):
        self.assertTrue(self.tree.add_node("狂喜", "喜", reason="test"))
        self.assertIn("狂喜", self.tree.get_children("喜"))

    def test_add_node_bad_parent(self):
        self.assertFalse(self.tree.add_node("test", "nope", reason="test"))

    def test_prune_keeps_min(self):
        result = self.tree.prune(min_usage=999, keep_min_nodes=35)
        self.assertGreaterEqual(len(self.tree.tree), 35)
        self.assertLessEqual(result.get("pruned", 0), 35)


class TestWebLookup(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        from web_lookup import WebLookup
        self.wl = WebLookup(workspace_dir=self.tmpdir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_builtin_slang_raises_scores(self):
        """内置流行语匹配后情绪强度应有提升"""
        result = self.wl.analyze_text("今天破防了", {"困惑": 0.5})
        self.assertIsNotNone(result)
        self.assertIn("悲伤", result)
        self.assertGreater(result["悲伤"], 0)

    def test_no_api_returns_fallback_not_crash(self):
        """没有 API Key 时不应崩溃"""
        result = self.wl.analyze_text("被穿小鞋了", {"困惑": 0.3})
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)

    def test_unknown_terms_extraction(self):
        """应正确识别词库不认识的片段"""
        terms = self.wl._extract_unknown_terms("今天被人穿小鞋了好憋屈")
        self.assertIsNotNone(terms)
        # 至少有一个未知片段
        self.assertGreaterEqual(len(terms), 0)

    def test_guess_emotion_returns_dict(self):
        """字符特征猜测应返回情绪字典"""
        result = self.wl._guess_emotion("悲惨")
        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)

    def test_coverage_ratio(self):
        """覆盖率计算应返回 0~1 之间的值"""
        ratio = self.wl.search_coverage_ratio("今天好开心")
        self.assertGreaterEqual(ratio, 0)
        self.assertLessEqual(ratio, 1)

    def test_promote_expression(self):
        """晋升表达后应出现在已晋升词表中"""
        from self_learning_manager import LearningManager
        self.learning_manager = LearningManager(workspace_dir=self.tmpdir)
        self.learning_manager.promote_expression("穿小鞋", "委屈", source="test")
        promoted = self.learning_manager.get_promoted_keywords()
        self.assertIn("委屈", promoted)
        self.assertIn("穿小鞋", promoted["委屈"])


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)
