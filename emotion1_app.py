import streamlit as st
import torch
from transformers import pipeline
from diffusers import StableDiffusionPipeline
from PIL import Image
import time

# PAGE CONFIG
st.set_page_config(
    page_title="Emotion AI Story Generator",
    page_icon="🎭",
    layout="centered"
)

# 🎨 FULL CSS (MATCHES YOUR HTML DESIGN)
st.markdown("""
<style>
/* 🎭 HEADING STYLE */
.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    margin-top: 40px;   /* ✅ pushes down */
    margin-bottom: 10px;

    background: linear-gradient(90deg, #ff758c, #ff7eb3, #ffd86f);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;

    text-shadow: 0 0 15px rgba(255,255,255,0.3);
}
/* 🌈 FULL BACKGROUND FIX */
.stApp {
    background: linear-gradient(270deg, #ff6ec4, #7873f5, #42e695, #f9ca24);
    background-size: 800% 800%;
    animation: gradientMove 10s ease infinite;
}

@keyframes gradientMove {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* REMOVE WHITE BACKGROUND */
.css-18e3th9, .css-1d391kg {
    background: transparent !important;
}

/* 🎭 EMOJIS */
.emoji-left {
    position: fixed;
    left: 15px;
    top: 80px;
}

.emoji-right {
    position: fixed;
    right: 15px;
    top: 80px;
}

.emoji {
    font-size: 50px;
    margin: 30px 0;
    animation: float 4s infinite ease-in-out;
}

@keyframes float {
    0% {transform: translateY(0);}
    50% {transform: translateY(-25px);}
    100% {transform: translateY(0);}
}

/* CENTER CARD */
.block-container {
    background: rgba(0,0,0,0.65);
    padding: 35px;
    border-radius: 18px;
}

/* TEXTAREA */
.stTextArea textarea {
    background: #ffffff !important;
    color: #333 !important;
    border-radius: 12px !important;
    border: 2px solid #ddd !important;
}

/* BUTTON */
.stButton > button {
    background: linear-gradient(135deg, #ff758c, #ff7eb3);
    color: white !important;
    border-radius: 25px !important;
    font-size: 16px;
}

/* EMOTION BADGE */
.emotion-badge {
    padding: 10px 25px;
    border-radius: 30px;
    display: inline-block;
    margin-top: 15px;
    font-weight: bold;
}

/* QUOTE */
.quote-box {
    background: #ffffff;
    color: #333;
    border-radius: 12px;
    padding: 12px;
    margin: 10px 0;
    text-align: center;
    font-style: italic;
}

/* STORY */
.story-box {
    background: #ffffff;
    color: #333;
    border-radius: 12px;
    padding: 20px;
    margin-top: 20px;
    font-size: 17px;
    line-height: 1.8;
}
/* HIDE MENU */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

</style>
""", unsafe_allow_html=True)

# 🎭 SIDE EMOJIS
st.markdown("""
<div class="emoji-left">
    <div class="emoji">😄</div>
    <div class="emoji">😢</div>
    <div class="emoji">😡</div>
    <div class="emoji">😍</div>
    <div class="emoji">🤯</div>
</div>

<div class="emoji-right">
    <div class="emoji">😎</div>
    <div class="emoji">😭</div>
    <div class="emoji">😴</div>
    <div class="emoji">🤔</div>
    <div class="emoji">😱</div>
</div>
""", unsafe_allow_html=True)

# EMOTION STYLE
EMOTION_STYLES = {
    "sadness":  {"bg": "#1e2a3a", "color": "#7ab3e0", "emoji": "💙"},
    "joy":      {"bg": "#2a2a1a", "color": "#f0d060", "emoji": "✨"},
    "fear":     {"bg": "#1f1a2e", "color": "#b388e8", "emoji": "🌑"},
    "anger":    {"bg": "#2e1a1a", "color": "#e87070", "emoji": "🔥"},
    "disgust":  {"bg": "#1a2e1a", "color": "#80c980", "emoji": "🌿"},
    "neutral":  {"bg": "#1e1e1e", "color": "#aaaaaa", "emoji": "🌫️"},
    "surprise": {"bg": "#2a1e2e", "color": "#f0a0e0", "emoji": "⚡"},
}

# 💬 QUOTES
EMOTION_QUOTES = {
    "sadness": "💙 It's okay to feel lost sometimes — even the darkest nights end in sunrise.",
    "anger": "🔥 Take a deep breath. Calm mind leads to better decisions.",
    "disgust": "🌿 Sometimes stepping away is the healthiest choice you can make.",
    "joy": "🎉 Celebrate even the smallest victories.",
    "fear": "🔥 Face your fears, and they will fade.",
    "neutral": "🌫️ Balance brings clarity to life.",
    "surprise": "⚡ Life is full of unexpected beauty."
}

# STORY
STORY_TEMPLATES = {
    "sadness": (
        "She felt a deep heaviness in her heart as memories slowly filled her mind. "
        "The silence around her made everything more painful, like a fog that refused to lift. "
        "Taking a long, slow breath, she closed her eyes and tried to accept the moment — "
        "not to fight it, but to let it pass through her like rain. "
        "In that quiet surrender, she found a small sense of peace beginning to grow. "
        "And somewhere deep within, hope whispered that she would heal again."
    ),

    "joy": (
        "A warm smile spread across her face as happiness filled every corner of the moment. "
        "Everything around her felt brighter, lighter, as if the world had been turned up just a little. "
        "She held onto that feeling with both hands, knowing it was rare and worth savoring fully. "
        "Laughter bubbled inside her, effortless and pure like a melody. "
        "For that moment, everything felt exactly as it should be."
    ),

    "fear": (
        "Her heart started beating faster as a cold sense of fear crept slowly up her spine. "
        "She looked around, unsure of what lay ahead, every shadow feeling like a threat. "
        "Then, gathering every last piece of courage she had, she took one small step forward. "
        "That single step felt heavier than anything she had ever done before. "
        "Yet deep down, she realized courage was not the absence of fear, but moving despite it."
    ),

    "anger": (
        "Her emotions burned with a fierce intensity as frustration took over her thoughts entirely. "
        "She struggled to stay calm, trying hard not to let the fire inside her spill over. "
        "Slowly, steadily, she pulled herself back — choosing clarity over the heat of the moment. "
        "With each breath, the storm inside her began to settle just a little. "
        "And in that stillness, she found strength she didn’t know she had."
    ),

    "disgust": (
        "She felt deeply uncomfortable, unsettled by what she had just witnessed or heard. "
        "It was difficult even to process — her mind recoiling from the weight of the moment. "
        "She stepped away, quietly, quickly, letting clean air slowly replace what had been there. "
        "Distance gave her space to think and regain her sense of balance. "
        "Sometimes, walking away was the most powerful decision she could make."
    ),

    "surprise": (
        "She stopped completely — the moment catching her entirely off guard and breathless. "
        "Her mind raced to catch up with what her eyes had just seen unfold before her. "
        "After a long pause, a slow smile of disbelief crept across her face. She hadn't expected this. "
        "The unexpected moment lingered, filling her with a strange sense of wonder. "
        "Life, she realized, had its own beautiful way of surprising her."
    ),

    "neutral": (
        "She paused for a moment, quietly observing everything that surrounded her in the space. "
        "The situation felt ordinary on the surface, yet held a quiet meaning she couldn't quite name. "
        "She continued forward with a calm and settled mind, neither rushed nor reluctant. "
        "Each step felt steady, grounded in a sense of quiet awareness. "
        "Sometimes, peace was simply found in moving forward without noise."
    ),
}

# MODELS
@st.cache_resource
def load_emotion_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=1,
        device=0 if torch.cuda.is_available() else -1
    )

@st.cache_resource
def load_sd_pipeline():
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    pipe.safety_checker = None
    return pipe

# FUNCTIONS
def detect_emotion(clf, text):
    return clf(text)[0][0]["label"].lower()

def generate_story(prompt, emotion):
    return STORY_TEMPLATES.get(emotion).format(prompt=prompt)

# UI
st.markdown("<div class='main-title'>🎭 Emotion AI Story Generator ✨</div>", unsafe_allow_html=True)
user_prompt = st.text_area("", placeholder="Type your feeling or story idea...")

if st.button("✨ Generate"):

    if not user_prompt.strip():
        st.warning("Please enter a prompt")
        st.stop()

    clf = load_emotion_model()
    emotion = detect_emotion(clf, user_prompt)

    style = EMOTION_STYLES.get(emotion)

    # 🎭 Emotion Badge
    st.markdown(
        f"<div class='emotion-badge' style='background:{style['bg']}; color:{style['color']}'>"
        f"{style['emoji']} {emotion.upper()}</div>",
        unsafe_allow_html=True
    )

    # 💬 Quote
    quote = EMOTION_QUOTES.get(emotion)
    st.markdown(
        f"<div class='quote-box'>{quote}</div>",
        unsafe_allow_html=True
    )

    # 📖 Story
    story = generate_story(user_prompt, emotion)
    st.markdown(
        f"<div class='story-box'>{story}</div>",
        unsafe_allow_html=True
    )

    # 🖼️ Image
    pipe = load_sd_pipeline()
    with st.spinner("Generating image..."):
        image = pipe(user_prompt).images[0]

    st.image(image, use_column_width=True)