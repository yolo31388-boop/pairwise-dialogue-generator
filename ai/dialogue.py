"""
游戏内NPC对话生成与上下文管理系统 - 核心模块

功能:
- generate_response(): 模板化回复生成, 同一意图有多个模板并按轮次轮换,
  相同输入在不同轮次会得到不同回复(变化), 并支持自定义模板。
- 上下文管理: 完整历史记录、按条数截取、超长历史压缩成摘要。
- 回复策略: 基于关键词的意图识别 + 情感识别, 据此选择语气和应对动作。
- 个性化: 每个NPC独立的语言风格(formal/casual/humorous/serious)
  和口头禅, 不同NPC说话有差异。
- 对话状态: 记录所处阶段(greeting/chatting/farewell/ended)、当前话题,
  以及话题之间的转换记录。
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import time


@dataclass
class Config:
    max_history: int = 20
    default_style: str = "casual"
    default_place: str = "村子"
    idle_timeout: float = 300.0


# (意图, 关键词) —— 顺序即优先级, 先匹配先返回
INTENT_KEYWORDS: List[Tuple[str, List[str]]] = [
    ("greeting", ["你好", "您好", "嗨", "哈喽", "早上好", "晚上好", "hi", "hello"]),
    ("farewell", ["再见", "拜拜", "回头见", "告辞", "慢走", "bye"]),
    ("quest", ["任务", "委托", "使命", "quest"]),
    ("trade", ["买", "卖", "交易", "商店", "多少钱", "价格", "货"]),
    ("compliment", ["厉害", "佩服", "好样的", "点赞", "棒"]),
    ("complaint", ["讨厌", "垃圾", "差劲", "无聊", "废物", "不满"]),
    ("request", ["请", "帮我", "能不能", "麻烦", "拜托"]),
    ("question", ["什么", "怎么", "为什么", "哪里", "哪个", "吗", "？", "?"]),
]

# (情感, 关键词)
EMOTION_KEYWORDS: List[Tuple[str, List[str]]] = [
    ("positive", ["谢谢", "感谢", "太好了", "喜欢", "开心", "哈哈", "棒", "不错"]),
    ("negative", ["讨厌", "垃圾", "生气", "差劲", "无聊", "烦", "失望", "难过"]),
    ("urgent", ["紧急", "马上", "立刻", "赶紧", "快点", "快"]),
]

# (话题, 关键词)
TOPIC_KEYWORDS: List[Tuple[str, List[str]]] = [
    ("quest", ["任务", "委托", "使命"]),
    ("trade", ["买", "卖", "交易", "商店", "货"]),
    ("weather", ["天气", "下雨", "晴天", "下雪"]),
    ("rumor", ["传闻", "听说", "消息", "八卦"]),
]

TOPIC_LABELS = {
    "quest": "任务",
    "trade": "买卖",
    "weather": "天气",
    "rumor": "传闻",
    None: "这件事",
}

# 每个意图对应多个模板, 生成时按轮次轮换以产生变化
DEFAULT_TEMPLATES: Dict[str, List[str]] = {
    "greeting": [
        "你好，旅行者！欢迎来到{place}。",
        "又见面了，朋友。今天过得怎么样？",
        "嗨！见到你真高兴。",
    ],
    "farewell": [
        "再见，一路顺风！",
        "回头见，朋友。",
        "慢走，{place}的大门随时为你敞开。",
    ],
    "quest": [
        "我正好有个任务要交给你：去{place}附近看看吧。",
        "说到任务，最近确实有一件棘手的事需要帮手。",
        "任务嘛……让我想想，有了，这个委托非你莫属。",
    ],
    "trade": [
        "想看看货吗？我这儿应有尽有。",
        "买卖不成仁义在，随便看看。",
        "价格好商量，你出个数？",
    ],
    "question": [
        "关于「{topic}」，我所知有限，不过可以告诉你一些。",
        "问得好。让我想想该怎么回答你。",
        "这个问题嘛，答案可没那么简单。",
    ],
    "request": [
        "没问题，包在我身上。",
        "既然你开口了，我就帮你这个忙。",
        "可以是可以，不过你得答应我一个条件。",
    ],
    "compliment": [
        "过奖过奖，你也很有眼光。",
        "哈哈，这话我爱听。",
        "谢谢夸奖，愧不敢当。",
    ],
    "complaint": [
        "抱歉让你不满意了，我会改进的。",
        "别生气，有话好好说。",
        "你的意见我记下了。",
    ],
    "chat": [
        "今天天气不错，适合出门冒险。",
        "最近{place}里可发生了不少事。",
        "想聊聊{topic}吗？我正好知道一些。",
    ],
}

TONE_BY_EMOTION = {
    "positive": "warm",
    "negative": "comfort",
    "urgent": "calm",
    "neutral": "neutral",
}

ACTION_BY_INTENT = {
    "greeting": "greet",
    "farewell": "say_goodbye",
    "quest": "offer_quest",
    "trade": "open_shop",
    "question": "answer",
    "request": "assist",
    "compliment": "thank",
    "complaint": "apologize",
    "chat": "chat",
}

# 情感对措辞的影响
EMOTION_PREFIX = {"negative": "别灰心，", "urgent": "别急，", "positive": "", "neutral": ""}
EMOTION_SUFFIX = {"positive": "！", "negative": "", "urgent": "", "neutral": ""}

# 语言风格 -> 说话前缀
STYLE_PREFIX = {
    "formal": "尊敬的旅行者，",
    "casual": "嘿，",
    "humorous": "嘿嘿，",
    "serious": "听着，",
}


class _SafeDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


class DialogueGenerator:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.max_history = int(self.config.get("max_history", Config.max_history))
        self.default_style = self.config.get("default_style", Config.default_style)
        self.default_place = self.config.get("place", Config.default_place)
        self.idle_timeout = float(self.config.get("idle_timeout", Config.idle_timeout))
        self._templates = {intent: list(items) for intent, items in DEFAULT_TEMPLATES.items()}
        self._personalities: Dict[str, Dict] = {}
        self._init_conversation()

    def _init_conversation(self):
        self._state: Dict = {"phase": "greeting", "topic": None, "turn": 0, "transitions": []}
        self._history: List[Tuple[str, str]] = []
        self._summary = ""
        self._topic_history: List[str] = []
        self._template_counters: Dict[str, int] = {}
        self._idle_time = 0.0
        self._last_update = time.monotonic()

    def update(self, dt: float):
        """推进游戏时间; 对话中长时间无交互则结束当前对话。"""
        dt = max(0.0, float(dt))
        self._idle_time += dt
        if self._state["phase"] == "chatting" and self._idle_time >= self.idle_timeout:
            self._state["phase"] = "ended"

    def reset(self):
        """重置对话(保留已注册的NPC个性化配置与自定义模板)。"""
        self._init_conversation()

    # ---------- 回复策略: 意图识别 / 情感识别 ----------

    def detect_intent(self, text: str) -> str:
        lowered = text.lower()
        for intent, keywords in INTENT_KEYWORDS:
            if any(word in lowered for word in keywords):
                return intent
        return "chat"

    def detect_emotion(self, text: str) -> str:
        lowered = text.lower()
        for emotion, keywords in EMOTION_KEYWORDS:
            if any(word in lowered for word in keywords):
                return emotion
        return "neutral"

    def detect_topic(self, text: str) -> Optional[str]:
        lowered = text.lower()
        for topic, keywords in TOPIC_KEYWORDS:
            if any(word in lowered for word in keywords):
                return topic
        return None

    def select_strategy(self, intent: str, emotion: str = "neutral") -> Dict[str, str]:
        """根据意图与情感决定如何回复(语气 + 应对动作)。"""
        return {
            "intent": intent,
            "emotion": emotion,
            "tone": TONE_BY_EMOTION.get(emotion, "neutral"),
            "action": ACTION_BY_INTENT.get(intent, "chat"),
        }

    # 与缺陷报告中的命名保持一致的入口
    def reply_strategy(self, text_or_intent: str, emotion: str = "neutral") -> Dict[str, str]:
        intent = text_or_intent if text_or_intent in ACTION_BY_INTENT else self.detect_intent(text_or_intent)
        if text_or_intent not in ACTION_BY_INTENT:
            emotion = self.detect_emotion(text_or_intent)
        return self.select_strategy(intent, emotion)

    # ---------- 回复生成: 模板 + 变化 ----------

    def add_template(self, intent: str, template: str):
        self._templates.setdefault(intent, []).append(template)

    def _select_template(self, intent: str) -> str:
        templates = self._templates.get(intent) or self._templates["chat"]
        idx = self._template_counters.get(intent, 0)
        self._template_counters[intent] = idx + 1
        return templates[idx % len(templates)]

    def _render(self, template: str, npc_id: str) -> str:
        topic = self._state.get("topic")
        return template.format_map(_SafeDict(
            npc=npc_id,
            player="旅行者",
            place=self.default_place,
            topic=TOPIC_LABELS.get(topic, "这件事"),
        ))

    @staticmethod
    def _apply_emotion(text: str, emotion: str) -> str:
        return EMOTION_PREFIX.get(emotion, "") + text + EMOTION_SUFFIX.get(emotion, "")

    def generate_response(self, player_input: str, npc_id: str = "default") -> str:
        intent = self.detect_intent(player_input)
        emotion = self.detect_emotion(player_input)
        topic = self.detect_topic(player_input)
        self._advance_state(intent, topic)

        response = self._render(self._select_template(intent), npc_id)
        response = self._apply_emotion(response, emotion)
        response = self._apply_personality(response, npc_id)

        self._record("player", player_input)
        self._record("npc", response)
        self._idle_time = 0.0
        return response

    # ---------- 个性化: 语言风格 + 口头禅 ----------

    def set_personality(self, npc_id: str, style: Optional[str] = None,
                        catchphrase: Optional[str] = None,
                        catchphrases: Optional[List[str]] = None):
        profile = self._personalities.setdefault(npc_id, {})
        if style is not None:
            profile["style"] = style
        if catchphrase is not None:
            profile["catchphrases"] = [catchphrase]
        if catchphrases is not None:
            profile["catchphrases"] = list(catchphrases)

    def get_personality(self, npc_id: str) -> Dict:
        profile = self._personalities.get(npc_id, {})
        return {"style": profile.get("style", self.default_style),
                "catchphrases": list(profile.get("catchphrases", []))}

    def _apply_personality(self, text: str, npc_id: str) -> str:
        profile = self._personalities.get(npc_id, {})
        style = profile.get("style", self.default_style)
        prefix = STYLE_PREFIX.get(style, "")
        if prefix:
            text = prefix + text
        catchphrases = profile.get("catchphrases", [])
        if catchphrases:
            idx = (self._state["turn"] - 1) % len(catchphrases)
            text = text + catchphrases[idx]
        return text

    # 与缺陷报告中的命名保持一致的入口
    def personality(self, npc_id: str, style: Optional[str] = None,
                    catchphrase: Optional[str] = None,
                    catchphrases: Optional[List[str]] = None):
        self.set_personality(npc_id, style=style, catchphrase=catchphrase,
                             catchphrases=catchphrases)

    # ---------- 上下文管理: 历史 + 摘要 ----------

    def _record(self, speaker: str, text: str):
        self._history.append((speaker, text))
        self._compress_history()

    def _compress_history(self):
        while len(self._history) > self.max_history:
            cut = max(1, self.max_history // 2)
            old, self._history = self._history[:cut], self._history[cut:]
            digest = "；".join(
                ("玩家" if speaker == "player" else "NPC") + "提到“" + text[:10] + "”"
                for speaker, text in old[:3]
            )
            entry = "此前对话摘要：" + digest + "。"
            self._summary = (self._summary + " " + entry).strip() if self._summary else entry

    def get_history(self, limit: Optional[int] = None) -> List[Tuple[str, str]]:
        if limit is None or limit >= len(self._history):
            return list(self._history)
        return list(self._history[-limit:])

    def summarize(self) -> str:
        topics = "、".join(TOPIC_LABELS.get(t, "其他") for t in self._topic_history) or "暂无"
        overview = "共进行了" + str(self._state["turn"]) + "轮对话，聊过的话题：" + topics + "。"
        if self._summary:
            return self._summary + overview
        return overview

    def get_context(self) -> Dict:
        return {
            "summary": self.summarize(),
            "compressed_summary": self._summary,
            "recent": self.get_history(),
            "state": self.get_state(),
        }

    # 与缺陷报告中的命名保持一致的入口
    def context_manager(self) -> Dict:
        return self.get_context()

    # ---------- 对话状态: 话题 + 转换 ----------

    def _advance_state(self, intent: str, topic: Optional[str]):
        self._state["turn"] += 1
        if intent == "farewell":
            self._state["phase"] = "farewell"
        elif self._state["phase"] in ("greeting", "farewell"):
            self._state["phase"] = "chatting"
        if topic:
            self.transition_to(topic)

    def transition_to(self, topic: str):
        old = self._state.get("topic")
        if old == topic:
            return
        self._state["topic"] = topic
        self._topic_history.append(topic)
        self._state["transitions"].append(
            {"from": old, "to": topic, "turn": self._state["turn"]}
        )

    def get_state(self) -> Dict:
        return {
            "phase": self._state["phase"],
            "topic": self._state["topic"],
            "turn": self._state["turn"],
            "transitions": list(self._state["transitions"]),
            "topic_history": list(self._topic_history),
        }

    # 与缺陷报告中的命名保持一致的入口
    def dialogue_state(self) -> Dict:
        return self.get_state()
