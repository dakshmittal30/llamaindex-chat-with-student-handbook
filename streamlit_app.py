import streamlit as st
from copilot import Copilot
import os
### set openai key, first check if it is in environment variable, if not, check if it is in streamlit secrets, if not, raise error


st.title("Chat with Columbia Business School Grid Copilot")
st.write(
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
)

openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key: ### get openai key from user input
    openai_api_key = st.text_input("Please enter your OpenAI API Key", type="password")

if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:
    if "messages" not in st.session_state.keys():  # Initialize the chat messages history
        st.session_state.messages = [
            {"role": "assistant", "content": "I am Columbia Copilot, your personal assistant. You can ask me about Columbia University."}
        ]

    @st.cache_resource
    def load_copilot():
        return Copilot()



    if "chat_copilot" not in st.session_state.keys():  # Initialize the chat engine
        st.session_state.chat_copilot = load_copilot()

    if prompt := st.chat_input(
        "Ask a question"
    ):  # Prompt for user input and save to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

    for message in st.session_state.messages:  # Write message history to UI
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # If last message is not from assistant, generate a new response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):

            retrived_info, answer = st.session_state.chat_copilot.ask(prompt, messages=st.session_state.messages[:-1], openai_key=openai_api_key)
            ### answer can be a generator or a string

            #print(retrived_info)
            if isinstance(answer, str):
                st.write(answer)
            else:
                ### write stream answer to UI
                def generate():
                    for chunk in answer:
                        content = chunk.choices[0].delta.content
                        if content:
                            yield content
                answer = st.write_stream(generate())

            st.session_state.messages.append({"role": "assistant", "content": answer})
