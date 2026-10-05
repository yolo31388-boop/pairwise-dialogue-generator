"""
游戏内NPC对话生成与上下文管理系统 - 核心模块

包含：
- generate_response(): 模板驱动的回复生成，同一意图多模板轮用，保证回复有变化
- context_manager(): 多轮对话历史管理，超长历史自动摘要
- reply_strategy(): 意图识别 + 情感识别，决定回复语气
- personality(): 语言风格（正式/随意/幽默/严肃）与口头禅个性化
- dialogue_state(): 对话状态机，跟踪当前话题与话题转换
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import time


@dataclass
class Config:
    npc_id: str = "default"
    player_name: str = "旅行者"
    max_history: int = 10


# 支持的语言风格
STYLES: Tuple[str, ...] = ("formal", "casual", "humorous", "serious")

# 回复模板：按意图组织，每个意图多条模板轮用，保证回复有变化
RESPONSE_TEMPLATES: Dict[str, List[str]] = {
    "greeting": [
        "你好呀，{player}！我是{npc}，今天想聊点什么？",
        "哟，{player}，又见面了！有什么我能帮你的吗？",
        "欢迎欢迎！{npc}这儿随时欢迎你来坐坐。",
    ],
    "farewell": [
        "再见，{player}，一路顺风！",
        "下次再来找我{npc}聊天哦。",
        "告辞了，愿好运常伴你左右。",
    ],
    "thanks": [
        "不客气，{player}，这是我应该做的。",
        "能帮上忙就好，别放在心上。",
        "客气啥，随时欢迎再来找我。",
    ],
    "question": [
        "这个问题嘛……让{npc}想想，你可以去问问村里的长者。",
        "关于这个，我知道的也不多，不过传闻里倒是提过。",
        "好问题！可惜我{npc}也不是无所不知。",
    ],
    "quest": [
        "正好有个任务想拜托你，{player}！",
        "任务嘛，我这儿倒有一个，感兴趣吗？",
        "你愿意帮忙真是太好了，咱们坐下来细说。",
    ],
    "trade": [
        "来看看货吧，{player}，价格公道！",
        "买东西还是卖东西？我这儿都可以。",
        "我这儿的商品物美价廉，随便挑！",
    ],
    "complaint": [
        "让你不高兴了，真是抱歉。",
        "别生气嘛，我会努力改进的。",
        "抱歉抱歉，是我考虑不周。",
    ],
    "compliment": [
        "谢谢夸奖，{player}你过奖了！",
        "被你夸得我都不好意思了。",
        "你也很棒啊，{player}！",
    ],
    "unknown": [
        "嗯……这个话题我还不太了解。",
        "有意思，再多跟我说说？",
        "我{npc}见识有限，没太听懂。",
    ],
}

# 意图 -> 对话话题
INTENT_TOPICS: Dict[str, str] = {
    "greeting": "寒暄",
    "farewell": "告别",
    "thanks": "道谢",
    "question": "咨询",
    "quest": "任务",
    "trade": "交易",
    "complaint": "投诉",
    "compliment": "闲聊",
    "unknown": "闲聊",
}

# 意图关键词（顺序即优先级）
_INTENT_KEYWORDS: List[Tuple[str, Tuple[str, ...]]] = [
    ("farewell", ("再见", "拜拜", "告辞", "走了", "bye")),
    ("greeting", ("你好", "您好", "早上好", "晚上好", "哈喽", "嗨", "hello", "hi")),
    ("thanks", ("谢谢", "感谢", "多谢", "thanks")),
    ("complaint", ("讨厌", "垃圾", "差劲", "无聊", "气死", "烦死")),
    ("compliment", ("厉害", "真棒", "好样的", "喜欢", "佩服")),
    ("quest", ("任务", "委托", "帮忙", "帮我", "请求")),
    ("trade", ("买", "卖", "交易", "商店", "多少钱", "价格", "货")),
    ("question", ("什么", "怎么", "为什么", "哪里", "哪个", "谁", "吗", "？", "?")),
]

_NEGATIVE_WORDS = ("讨厌", "垃圾", "差劲", "滚", "生气", "气死", "烦死", "无聊", "失望")
_URGENT_WORDS = ("救命", "紧急", "赶快", "赶紧", "马上", "快")
_POSITIVE_WORDS = ("谢谢", "感谢", "喜欢", "棒", "厉害", "哈哈", "太好了", "爱")

# 情感 -> 回复语气
EMOTION_TONES: Dict[str, str] = {
    "positive": "热情",
    "negative": "安抚",
    "urgent": "急促",
    "neutral": "平静",
}

# 情感语气前缀（让语气在回复文本上可感知）
_TONE_PREFIX: Dict[str, str] = {
    "positive": "太好了，",
    "negative": "抱歉，",
    "urgent": "马上！",
    "neutral": "",
}

# 默认NPC个性化档案：语言风格 + 口头禅
DEFAULT_PERSONALITIES: Dict[str, Dict[str, str]] = {
    "default": {"style": "casual", "catchphrase": ""},
    "guard": {"style": "formal", "catchphrase": "为了王国的荣耀！"},
    "merchant": {"style": "humorous", "catchphrase": "童叟无欺～"},
    "elder": {"style": "serious", "catchphrase": "年轻人，记住了。"},
}


class DialogueGenerator:
    def __init__(self, config: Optional[Dict] = None):
        self.config = dict(config or {})
        self._state: Dict = {}
        self._history: List[Dict] = []
        self._summary: str = ""

        self._current_npc: str = self.config.get("npc_id", "default")
        self._player_name: str = self.config.get("player_name", "旅行者")
        self._max_history: int = int(self.config.get("max_history", 10))

        self._personalities: Dict[str, Dict[str, str]] = {
            npc_id: dict(profile) for npc_id, profile in DEFAULT_PERSONALITIES.items()
        }
        self._personalities.update(self.config.get("personalities", {}))

        self._template_cursor: Dict[str, int] = {}

        # 对话状态
        self._stage: str = "start"
        self._topic: Optional[str] = None
        self._topic_history: List[str] = []
        self._transitions: int = 0
        self._elapsed: float = 0.0

    # ------------------------------------------------------------------
    # 基础生命周期
    # ------------------------------------------------------------------
    def update(self, dt: float):
        """推进对话系统时间。"""
        self._elapsed += max(0.0, float(dt))
        return self._elapsed

    def reset(self):
        """重置全部对话状态、历史与模板游标。"""
        config = self.config
        self.__init__(config)
        return None

    # ------------------------------------------------------------------
    # 回复策略：意图识别 + 情感识别
    # ------------------------------------------------------------------
    def _recognize_intent(self, text: str) -> str:
        text = (text or "").lower()
        for intent, keywords in _INTENT_KEYWORDS:
            if any(word in text for word in keywords):
                return intent
        return "unknown"

    def _recognize_emotion(self, text: str) -> str:
        text = text or ""
        if any(word in text for word in _NEGATIVE_WORDS):
            return "negative"
        if any(word in text for word in _URGENT_WORDS) or text.count("！") + text.count("!") >= 2:
            return "urgent"
        if any(word in text for word in _POSITIVE_WORDS):
            return "positive"
        return "neutral"

    def reply_strategy(self, player_input: str) -> Dict[str, str]:
        """识别玩家意图与情感，返回回复策略。"""
        intent = self._recognize_intent(player_input)
        emotion = self._recognize_emotion(player_input)
        return {
            "intent": intent,
            "emotion": emotion,
            "tone": EMOTION_TONES[emotion],
        }

    # ------------------------------------------------------------------
    # 个性化：语言风格 + 口头禅
    # ------------------------------------------------------------------
    def personality(
        self,
        npc_id: Optional[str] = None,
        style: Optional[str] = None,
        catchphrase: Optional[str] = None,
    ) -> Dict[str, str]:
        """读取或设置某个NPC的语言风格与口头禅。"""
        target = npc_id or self._current_npc
        profile = self._personalities.setdefault(
            target, {"style": "casual", "catchphrase": ""}
        )
        if style is not None:
            if style not in STYLES:
                raise ValueError(f"未知语言风格: {style}，可选: {STYLES}")
            profile["style"] = style
        if catchphrase is not None:
            profile["catchphrase"] = catchphrase
        if npc_id is not None:
            self._current_npc = npc_id
        return dict(profile)

    def _apply_style(self, text: str, style: str) -> str:
        if style == "formal":
            text = text.replace("你", "您")
        elif style == "humorous":
            text = text.rstrip("。！？") + "，哈哈。"
        elif style == "serious":
            text = text.replace("！", "。").replace("!", ".")
        return text

    def _apply_tone(self, text: str, emotion: str) -> str:
        prefix = _TONE_PREFIX.get(emotion, "")
        return prefix + text if prefix else text

    def _attach_catchphrase(self, text: str, npc_id: str) -> str:
        catchphrase = self._personalities.get(npc_id, {}).get("catchphrase", "")
        return f"{text} {catchphrase}".strip() if catchphrase else text

    # ------------------------------------------------------------------
    # 对话状态：话题与转换
    # ------------------------------------------------------------------
    def _update_dialogue_state(self, intent: str):
        topic = INTENT_TOPICS.get(intent, "闲聊")
        if self._topic is not None and topic != self._topic:
            self._transitions += 1
        if topic not in self._topic_history:
            self._topic_history.append(topic)
        self._topic = topic

        if intent == "farewell":
            self._stage = "end"
        elif self._stage == "end":
            self._stage = "middle"
        elif self._stage == "start":
            self._stage = "middle"

    def dialogue_state(self) -> Dict:
        """返回当前对话状态：阶段、话题、话题历史、转换次数。"""
        return {
            "stage": self._stage,
            "topic": self._topic,
            "topics": list(self._topic_history),
            "transitions": self._transitions,
        }

    # ------------------------------------------------------------------
    # 上下文管理：多轮历史 + 超长摘要
    # ------------------------------------------------------------------
    def context_manager(
        self,
        speaker: Optional[str] = None,
        text: Optional[str] = None,
        intent: Optional[str] = None,
    ) -> Dict:
        """记录一条发言并返回当前上下文（历史、摘要、长度）。"""
        if speaker is not None and text is not None:
            self._history.append(
                {
                    "speaker": speaker,
                    "text": text,
                    "intent": intent or self._recognize_intent(text),
                    "ts": time.time(),
                }
            )
            self._maybe_summarize()
        return {
            "history": list(self._history),
            "summary": self._summary,
            "size": len(self._history),
        }

    def _maybe_summarize(self):
        while len(self._history) > self._max_history:
            keep = max(1, self._max_history // 2)
            old = self._history[:-keep]
            self._history = self._history[-keep:]

            topics: List[str] = []
            for msg in old:
                topic = INTENT_TOPICS.get(msg["intent"], "闲聊")
                if topic not in topics:
                    topics.append(topic)
            topic_text = "、".join(topics) if topics else "日常闲聊"
            segment = f"此前对话{len(old)}条，涉及话题：{topic_text}"
            self._summary = f"{self._summary}；{segment}" if self._summary else segment

    # ------------------------------------------------------------------
    # 回复生成：策略 + 状态 + 模板 + 个性化 的完整流水线
    # ------------------------------------------------------------------
    def _pick_template(self, intent: str) -> str:
        templates = RESPONSE_TEMPLATES.get(intent, RESPONSE_TEMPLATES["unknown"])
        index = self._template_cursor.get(intent, 0)
        self._template_cursor[intent] = index + 1
        return templates[index % len(templates)]

    def generate_response(self, player_input: str, npc_id: Optional[str] = None) -> str:
        """根据玩家输入生成个性化、有变化的NPC回复。"""
        npc_id = npc_id or self._current_npc
        self._current_npc = npc_id

        strategy = self.reply_strategy(player_input)
        intent = strategy["intent"]
        emotion = strategy["emotion"]

        self._update_dialogue_state(intent)
        self.context_manager(speaker="player", text=player_input, intent=intent)

        template = self._pick_template(intent)
        body = template.format(
            npc=npc_id,
            player=self._player_name,
            topic=self._topic or "闲聊",
        )
        body = self._apply_tone(body, emotion)
        body = self._apply_style(body, self.personality(npc_id)["style"])
        reply = self._attach_catchphrase(body, npc_id)

        self.context_manager(speaker=npc_id, text=reply, intent=intent)
        return reply
