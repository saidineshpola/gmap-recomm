import streamlit as st
import requests
import uuid

st.title("Personalized Location-based Recommendation System")

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

user_id = "111423711419019424734"  # st.text_input("Enter your User ID (optional)")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("How can I help you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        # Make a request to the FastAPI backend
        response = requests.post(
            "http://localhost:8000/query",
            json={
                "query": prompt,
                "user_id": user_id,
                "conversation_id": st.session_state.conversation_id,
            },
        )
        if response.status_code == 200:
            data = response.json()
            full_response = data["response"]
            results = data["results"]
            conversation_history = data["conversation_history"]

            message_placeholder.markdown(full_response)

            for result in results:
                if "photos" in result["data"]:
                    st.subheader(f"Images for {result['data'].get('name', 'Business')}")
                    for pic_url in result["data"]["photos"]:
                        st.image(pic_url, width=200)

            # Update the session state with the new conversation history
            # st.session_state.messages = conversation_history
        else:
            message_placeholder.markdown(
                "Sorry, there was an error processing your request."
            )

    st.session_state.messages.append({"role": "assistant", "content": full_response})
