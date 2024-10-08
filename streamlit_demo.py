import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(layout="wide")

st.title("Local Business Assistant")

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = 1

if "messages" not in st.session_state:
    st.session_state.messages = []

if "previous_results" not in st.session_state:
    st.session_state.previous_results = None

user_id = "101039921652255289747"

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything about businesses!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        try:
            # Determine which API endpoint to use based on conversation state
            if st.session_state.conversation_id == 1:
                api_endpoint = "http://localhost:8000/query_business"
                params = {
                    "input": prompt,
                    "user_id": user_id,
                    "conversation_id": str(st.session_state.conversation_id),
                }
                response = requests.post(api_endpoint, params=params)
            else:
                api_endpoint = "http://localhost:8000/follow_up_query"
                payload = {
                    "query": prompt,
                    "previous_results": st.session_state.previous_results,
                    "conversation_history": [
                        {"role": msg["role"], "content": msg["content"]}
                        for msg in st.session_state.messages
                    ],
                }
                response = requests.post(api_endpoint, json=payload)

            response.raise_for_status()

            data = response.json()
            full_response = data["response"]
            results = data.get("results", [])

            # Display the AI's response
            st.markdown("### AI Response")
            st.info(full_response)

            if st.session_state.conversation_id == 1:
                if results:
                    result = results[0]
                    business_data = result.get("data", {})

                    st.markdown("---")
                    st.header("Additional Business Information")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Basic Details")
                        st.markdown(f"**Name:** {business_data.get('name', 'N/A')}")
                        st.markdown(
                            f"**Category:** {', '.join(business_data.get('category', ['N/A']))}"
                        )
                        st.markdown(
                            f"**Address:** {business_data.get('address', 'N/A')}"
                        )

                        avg_rating = business_data.get("avg_rating")
                        if avg_rating:
                            st.markdown(
                                f"**Rating:** {'⭐' * int(avg_rating)} ({avg_rating}/5)"
                            )

                        st.markdown(
                            f"**Number of Reviews:** {business_data.get('num_of_reviews', 'N/A')}"
                        )
                        st.markdown(
                            f"**Price Range:** {business_data.get('price', 'N/A')}"
                        )
                        st.markdown(
                            f"**Current Status:** {business_data.get('state', 'N/A')}"
                        )

                        if "url" in business_data:
                            st.markdown(
                                f"[View on Google Maps]({business_data['url']})"
                            )

                    with col2:
                        pass

                    if result.get("top_images"):
                        st.subheader("Related Images")
                        image_col1, image_col2, image_col3 = st.columns(3)
                        for idx, img_url in enumerate(result["top_images"]):
                            try:
                                img_response = requests.get(img_url)
                                img_response.raise_for_status()
                                img = Image.open(io.BytesIO(img_response.content))

                                fixed_height = 200
                                aspect_ratio = img.width / img.height
                                new_width = int(fixed_height * aspect_ratio)
                                img_resized = img.resize((new_width, fixed_height))

                                with [image_col1, image_col2, image_col3][idx % 3]:
                                    st.image(img_resized, use_column_width=True)
                            except Exception as e:
                                st.write(f"Error loading image: {str(e)}")

            st.session_state.conversation_id += 1
            st.session_state.previous_results = data

        except requests.RequestException as e:
            error_message = f"An error occurred: {str(e)}"
            if hasattr(e, "response") and e.response is not None:
                error_message += f"\nStatus code: {e.response.status_code}"
                try:
                    error_details = e.response.json()
                    error_message += f"\nError details: {error_details}"
                except ValueError:
                    error_message += f"\nError content: {e.response.text}"

            st.error(error_message)
            full_response = error_message

    st.session_state.messages.append({"role": "assistant", "content": full_response})
