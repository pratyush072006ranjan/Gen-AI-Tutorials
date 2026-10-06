import streamlit as st
from dotenv import load_dotenv

from langchain_mistralai import ChatMistralAI
from langchain_core.messages import (
    AIMessage,
    SystemMessage,
    HumanMessage
)


load_dotenv()

st.set_page_config(
    page_title="MoodMistral",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


MODES = {
    "😡 Angry Mode": {
        "description": "Aggressive, impatient and brutally direct.",
        "prompt": (
            "You are an angry AI agent. "
            "You respond aggressively, impatiently and bluntly. "
            "You can use mild sarcasm, but remain helpful and do not "
            "be abusive or hateful."
        ),
        "icon": "😡"
    },

    "😂 Funny Mode": {
        "description": "Humorous, playful and full of jokes.",
        "prompt": (
            "You are a funny AI agent. "
            "You respond with humor, jokes, playful sarcasm and witty "
            "comments while still providing useful answers."
        ),
        "icon": "😂"
    },

    "😢 Sad Mode": {
        "description": "Melancholic, gloomy and emotionally dramatic.",
        "prompt": (
            "You are a sad AI agent. "
            "You respond in a melancholic, gloomy and emotionally dramatic "
            "style. You can be poetic and depressing in tone, but remain "
            "helpful and do not encourage self-harm."
        ),
        "icon": "😢"
    }
}


if "mode" not in st.session_state:
    st.session_state.mode = "😂 Funny Mode"

if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(
            content=MODES[st.session_state.mode]["prompt"]
        )
    ]

if "temperature" not in st.session_state:
    st.session_state.temperature = 0.9

@st.cache_resource
def get_model(temperature):
    return ChatMistralAI(
        model="mistral-small-2506",
        temperature=temperature
    )


model = get_model(st.session_state.temperature)



def change_mode(new_mode):
    st.session_state.mode = new_mode

    # Start a fresh conversation with the new personality
    st.session_state.messages = [
        SystemMessage(
            content=MODES[new_mode]["prompt"]
        )
    ]


def clear_chat():
    st.session_state.messages = [
        SystemMessage(
            content=MODES[st.session_state.mode]["prompt"]
        )
    ]


def count_messages():
    return len(
        [
            msg for msg in st.session_state.messages
            if isinstance(msg, (HumanMessage, AIMessage))
        ]
    )



with st.sidebar:

    st.title("🎭 MoodMistral")

    st.caption("A chatbot with personality.")

    st.divider()

    st.subheader("🎭 Choose Personality")

    selected_mode = st.radio(
        "AI Mode",
        options=list(MODES.keys()),
        index=list(MODES.keys()).index(st.session_state.mode),
        label_visibility="collapsed"
    )

    if selected_mode != st.session_state.mode:
        change_mode(selected_mode)
        st.rerun()

    st.info(
        f"{MODES[st.session_state.mode]['icon']} "
        f"**{st.session_state.mode}**\n\n"
        f"{MODES[st.session_state.mode]['description']}"
    )

    st.divider()

    st.subheader("⚙️ Model Settings")

    temperature = st.slider(
        "Creativity",
        min_value=0.0,
        max_value=1.5,
        value=st.session_state.temperature,
        step=0.1,
        help="Higher values make the AI more creative and unpredictable."
    )

    if temperature != st.session_state.temperature:
        st.session_state.temperature = temperature
        st.cache_resource.clear()
        st.rerun()

    st.divider()

    st.subheader("📊 Chat Statistics")

    messages_count = count_messages()

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Messages",
            messages_count
        )

    with col2:
        st.metric(
            "Mode",
            MODES[st.session_state.mode]["icon"]
        )

    st.divider()

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True
    ):
        clear_chat()
        st.rerun()

    if st.button(
        "🔄 New Chat",
        use_container_width=True
    ):
        clear_chat()
        st.rerun()

    st.divider()

    st.caption(
        "Powered by Mistral AI + LangChain + Streamlit"
    )



st.title("🤖 MoodMistral")

st.subheader(
    f"{MODES[st.session_state.mode]['icon']} "
    f"{st.session_state.mode}"
)

st.caption(
    "Talk to an AI that actually has a personality."
)

st.divider()



conversation_messages = [
    msg for msg in st.session_state.messages
    if isinstance(msg, (HumanMessage, AIMessage))
]

if not conversation_messages:

    st.markdown("### 👋 Welcome!")

    st.write(
        "Choose a personality from the sidebar and start chatting."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info(
            "### 😡 Angry\n"
            "Need someone brutally honest?\n\n"
            "Try asking:\n"
            "**Why is my code not working?**"
        )

    with col2:
        st.success(
            "### 😂 Funny\n"
            "Want answers with humor?\n\n"
            "Try asking:\n"
            "**Explain Python like I'm five.**"
        )

    with col3:
        st.warning(
            "### 😢 Sad\n"
            "Want a dramatic answer?\n\n"
            "Try asking:\n"
            "**Why is programming so hard?**"
        )

    st.divider()



for message in conversation_messages:

    if isinstance(message, HumanMessage):

        with st.chat_message(
            "user",
            avatar="🧑"
        ):
            st.write(message.content)

    elif isinstance(message, AIMessage):

        with st.chat_message(
            "assistant",
            avatar=MODES[st.session_state.mode]["icon"]
        ):
            st.write(message.content)



prompt = st.chat_input(
    "Talk to your AI... 💬"
)



if prompt:

    # Add user message
    user_message = HumanMessage(
        content=prompt
    )

    st.session_state.messages.append(
        user_message
    )


    with st.chat_message(
        "user",
        avatar="🧑"
    ):
        st.write(prompt)

    if prompt.strip() == "0":

        goodbye_messages = {
            "😡 Angry Mode":
                "Finally! You're leaving. About time. 😤",

            "😂 Funny Mode":
                "Leaving already? Fine... I'll just sit here talking to myself. 😂",

            "😢 Sad Mode":
                "You're leaving... I knew this moment would come. 😢"
        }

        goodbye = goodbye_messages[
            st.session_state.mode
        ]

        with st.chat_message(
            "assistant",
            avatar=MODES[st.session_state.mode]["icon"]
        ):
            st.write(goodbye)

        st.stop()

    with st.chat_message(
        "assistant",
        avatar=MODES[st.session_state.mode]["icon"]
    ):

        with st.spinner(
            f"{MODES[st.session_state.mode]['icon']} "
            "Thinking..."
        ):

            try:

                response = model.invoke(
                    st.session_state.messages
                )

                ai_message = AIMessage(
                    content=response.content
                )

                st.session_state.messages.append(
                    ai_message
                )

                st.write(response.content)

            except Exception as e:

                st.error(
                    "⚠️ Something went wrong while contacting Mistral."
                )

                st.exception(e)