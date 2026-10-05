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

    # ---- 回复生成：模板 + 变化 ----
    def test_case_01(self):
        reply = self.system.generate_response("你好")
        assert isinstance(reply, str)
        assert len(reply) > 0
        assert reply != "你好"

    def test_case_02(self):
        replies = {self.system.generate_response("你好") for _ in range(3)}
        assert len(replies) >= 2, "同样的问题应该产生不同的回复"

    def test_case_03(self):
        quest_reply = DialogueGenerator().generate_response("有什么任务吗？")
        trade_reply = DialogueGenerator().generate_response("我要买东西")
        assert quest_reply != trade_reply
        assert "任务" in quest_reply

    # ---- 上下文管理：历史 + 摘要 ----
    def test_case_04(self):
        self.system.generate_response("你好")
        self.system.generate_response("有什么任务吗？")
        self.system.generate_response("我要买东西")
        history = self.system.context_manager()["history"]
        assert len(history) == 6, "上下文应该记住多轮对话，而不是只记上一句"
        assert history[0]["text"] == "你好"

    def test_case_05(self):
        system = DialogueGenerator({"max_history": 4})
        for _ in range(6):
            system.generate_response("你好")
        context = system.context_manager()
        assert len(context["history"]) <= 4
        assert context["summary"], "很长的对话历史应该被总结成摘要"
        assert "话题" in context["summary"]

    def test_case_06(self):
        self.system.context_manager(speaker="player", text="你好")
        context = self.system.context_manager()
        assert set(context.keys()) >= {"history", "summary", "size"}
        assert context["size"] == 1
        assert context["history"][0]["speaker"] == "player"

    # ---- 回复策略：意图识别 + 情感 ----
    def test_case_07(self):
        assert self.system.reply_strategy("你好啊")["intent"] == "greeting"
        assert self.system.reply_strategy("再见")["intent"] == "farewell"
        assert self.system.reply_strategy("这是什么东西？")["intent"] == "question"
        assert self.system.reply_strategy("有什么任务吗")["intent"] == "quest"
        assert self.system.reply_strategy("这个多少钱")["intent"] == "trade"

    def test_case_08(self):
        assert self.system.reply_strategy("我真是太喜欢这里了")["emotion"] == "positive"
        assert self.system.reply_strategy("这东西真垃圾，气死我了")["emotion"] == "negative"
        assert self.system.reply_strategy("今天天气不错")["emotion"] == "neutral"

    def test_case_09(self):
        happy = DialogueGenerator().generate_response("谢谢你的帮助！")
        angry = DialogueGenerator().generate_response("你的东西太差劲了")
        assert happy.startswith("太好了")
        assert "抱歉" in angry

    # ---- 个性化：语言风格 + 口头禅 ----
    def test_case_10(self):
        formal = DialogueGenerator()
        formal.personality(style="formal")
        humorous = DialogueGenerator()
        humorous.personality(style="humorous")
        serious = DialogueGenerator()
        serious.personality(style="serious")
        assert "您" in formal.generate_response("你好")
        assert "哈哈" in humorous.generate_response("你好")
        assert "！" not in serious.generate_response("你好")

    def test_case_11(self):
        system = DialogueGenerator({"npc_id": "guard"})
        reply = system.generate_response("你好")
        assert "为了王国的荣耀" in reply, "NPC回复应该带上自己的口头禅"

    def test_case_12(self):
        guard_reply = DialogueGenerator().generate_response("你好", npc_id="guard")
        merchant_reply = DialogueGenerator().generate_response("你好", npc_id="merchant")
        assert guard_reply != merchant_reply, "不同NPC说话不应该一样"
        assert "童叟无欺" in merchant_reply

    # ---- 对话状态：话题 + 转换 ----
    def test_case_13(self):
        state = self.system.dialogue_state()
        assert "topic" in state and "stage" in state
        self.system.generate_response("有什么任务吗？")
        assert self.system.dialogue_state()["topic"] == "任务"

    def test_case_14(self):
        assert self.system.dialogue_state()["stage"] == "start"
        self.system.generate_response("你好")
        assert self.system.dialogue_state()["stage"] == "middle"
        self.system.generate_response("我要买东西")
        state = self.system.dialogue_state()
        assert state["topic"] == "交易"
        assert state["transitions"] >= 1, "换话题应该被记录为一次状态转换"
        self.system.generate_response("再见")
        assert self.system.dialogue_state()["stage"] == "end"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
