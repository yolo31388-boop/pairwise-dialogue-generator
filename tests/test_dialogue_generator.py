"""
游戏内NPC对话生成与上下文管理系统 - 验收测试
运行方式：python -m pytest tests/test_dialogue_generator.py -q
共 14 个测试用例
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.dialogue import DialogueGenerator


class TestDialogueGenerator:
    def setup_method(self):
        self.system = DialogueGenerator()

    # ---- 回复生成: 模板 + 变化 ----

    def test_case_01(self):
        # generate_response 返回非空回复, 且不同输入得到不同回复(不是固定一句话)
        r1 = self.system.generate_response("你好")
        r2 = self.system.generate_response("有什么任务吗")
        assert isinstance(r1, str) and r1
        assert isinstance(r2, str) and r2
        assert r1 != r2

    def test_case_02(self):
        # 同样的问题多次提问, 回复有变化(模板轮换)
        responses = {self.system.generate_response("你好") for _ in range(3)}
        assert len(responses) >= 2

    def test_case_03(self):
        # 支持自定义模板, 且不同意图使用不同模板
        self.system.add_template("greeting", "自定义的特别问候")
        seen = [self.system.generate_response("你好") for _ in range(4)]
        assert any("自定义的特别问候" in r for r in seen)
        quest_reply = self.system.generate_response("有什么任务吗")
        assert "任务" in quest_reply or "委托" in quest_reply

    # ---- 上下文管理: 历史 + 摘要 ----

    def test_case_04(self):
        # 上下文管理记录完整对话历史, 而不是只记得上一句
        for _ in range(3):
            self.system.generate_response("你好")
        history = self.system.get_history()
        assert len(history) == 6
        assert history[0][0] == "player"
        assert history[1][0] == "npc"

    def test_case_05(self):
        # 可以按条数取最近的历史
        for i in range(3):
            self.system.generate_response("第%d句闲聊" % i)
        recent = self.system.get_history(2)
        assert len(recent) == 2
        assert recent[-1][0] == "npc"
        assert recent[-2][1] == "第2句闲聊"

    def test_case_06(self):
        # 很长的对话历史会被压缩成摘要
        system = DialogueGenerator({"max_history": 4})
        for _ in range(3):
            system.generate_response("你好")
        context = system.get_context()
        assert context["compressed_summary"]
        assert "摘要" in context["summary"]
        assert len(system.get_history()) <= 4

    # ---- 回复策略: 意图识别 + 情感 ----

    def test_case_07(self):
        # 意图识别: 知道玩家想干什么
        assert self.system.detect_intent("你好呀") == "greeting"
        assert self.system.detect_intent("再见，朋友") == "farewell"
        assert self.system.detect_intent("这个任务怎么做") == "quest"
        assert self.system.detect_intent("这把剑多少钱") == "trade"
        assert self.system.detect_intent("今天天气怎么样") == "question"

    def test_case_08(self):
        # 情感识别: 知道玩家是什么情绪
        assert self.system.detect_emotion("太谢谢你了") == "positive"
        assert self.system.detect_emotion("这东西真垃圾") == "negative"
        assert self.system.detect_emotion("快告诉我") == "urgent"
        assert self.system.detect_emotion("嗯") == "neutral"

    def test_case_09(self):
        # 回复策略根据意图和情感决定语气与动作, 并体现在回复里
        strategy = self.system.select_strategy("complaint", "negative")
        assert strategy["tone"] == "comfort"
        assert strategy["action"] == "apologize"
        reply = self.system.generate_response("真讨厌")
        assert "别灰心" in reply

    # ---- 个性化: 语言风格 + 口头禅 ----

    def test_case_10(self):
        # 语言风格: 正式与随意说话不一样
        self.system.set_personality("guard", style="formal")
        self.system.set_personality("bard", style="casual")
        r1 = self.system.generate_response("你好", npc_id="guard")
        r2 = self.system.generate_response("你好", npc_id="bard")
        assert r1.startswith("尊敬的旅行者")
        assert r2.startswith("嘿，")
        assert r1 != r2

    def test_case_11(self):
        # 口头禅: NPC有自己的特色口头禅
        self.system.set_personality("cat", catchphrase="喵~")
        reply = self.system.generate_response("你好", npc_id="cat")
        assert reply.endswith("喵~")

    def test_case_12(self):
        # 不同NPC说话不一样(幽默 vs 严肃)
        self.system.set_personality("joker", style="humorous")
        self.system.set_personality("elder", style="serious")
        r1 = self.system.generate_response("你好", npc_id="joker")
        r2 = self.system.generate_response("你好", npc_id="elder")
        assert r1.startswith("嘿嘿，")
        assert r2.startswith("听着，")
        assert r1 != r2

    # ---- 对话状态: 话题 + 转换 ----

    def test_case_13(self):
        # 对话状态跟踪当前话题
        self.system.generate_response("我想接个任务")
        assert self.system.get_state()["topic"] == "quest"
        self.system.generate_response("今天天气怎么样")
        assert self.system.get_state()["topic"] == "weather"

    def test_case_14(self):
        # 对话状态有阶段推进和话题转换记录
        assert self.system.get_state()["phase"] == "greeting"
        self.system.generate_response("你好")
        assert self.system.get_state()["phase"] == "chatting"
        self.system.generate_response("今天天气怎么样")
        self.system.generate_response("我想买点东西")
        state = self.system.get_state()
        assert state["topic"] == "trade"
        assert len(state["transitions"]) >= 2
        assert state["transitions"][-1]["to"] == "trade"
        self.system.generate_response("再见")
        assert self.system.get_state()["phase"] == "farewell"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
