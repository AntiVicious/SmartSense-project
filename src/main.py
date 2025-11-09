import streamlit as st
import requests



# The Streamlit UI code (unchanged)
st.set_page_config(page_title="Real Estate Search", layout="wide")
st.title("SmartSense Real Estate Search 🏠")

# --- 1. Ingestion UI ---
st.header("Data Ingestion")
# We now connect to localhost:8000, because FastAPI is in the same container
BACKEND_URL = "http://localhost:8000" 

uploaded_file = st.file_uploader("Upload Property Excel File", type=["xlsx", "csv"])
if st.button("Start Ingestion"):
    if uploaded_file is not None:
        files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        try:
            with st.spinner("Ingesting data... This may take a moment."):
                response = requests.post(f"{BACKEND_URL}/ingest", files=files)
            if response.status_code == 200:
                st.success(response.json()['message'])
            else:
                st.error(f"Error: {response.json()['detail']}")
        except requests.exceptions.ConnectionError:
            st.error("Backend connection failed. Is it running?")
    else:
        st.error("Please upload a file first.")

# --- 2. Phase 1 Debug UI (UPDATED) ---
st.divider()
st.header("Floorplan Image Parser")
uploaded_image = st.file_uploader("Upload a single floorplan image", type=["jpg", "png", "jpeg"])

# --- THIS IS THE FIX ---
# Show the image as soon as it's uploaded
if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded Floorplan", use_container_width=False)
# -----------------------

if st.button("Parse Floorplan"):
    if uploaded_image is not None:
        files = {'file': (uploaded_image.name, uploaded_image.getvalue(), uploaded_image.type)}
        try:
            with st.spinner("Parsing image..."):
                response = requests.post(f"{BACKEND_URL}/parse-floorplan-debug", files=files)
            
            if response.status_code == 200:
                st.success("Image parsed successfully!")
                st.json(response.json())
            else:
                st.error(f"Error from API: {response.json()['detail']}")
                
        except requests.exceptions.ConnectionError:
            st.error("Backend connection failed. Is it running?")
    else:
        st.error("Please upload an image first.")
st.divider()
# --- END OF SECTION ---

# --- 2. Chatbot UI ---
st.header("Ask me Property related questions: Multi-Agent Chatbot")
if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me about properties..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    api_history = []
    for i in range(0, len(st.session_state.messages) - 1, 2):
        if st.session_state.messages[i]["role"] == "user" and st.session_state.messages[i+1]["role"] == "assistant":
            api_history.append((st.session_state.messages[i]["content"], st.session_state.messages[i+1]["content"]))

    try:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = requests.post(
                    f"{BACKEND_URL}/chat",
                    json={"query": prompt, "history": api_history}
                )
            if response.status_code == 200:
                ai_response = response.json()['response']
                st.markdown(ai_response)
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
            else:
                st.error(f"Error from API: {response.json()['detail']}")
    except requests.exceptions.ConnectionError:
        st.error("Backend is unreachable. Please check if it's running.")

# ... (all your other streamlit code is fine) ...

# --- 3. System Status (THE FIX) ---
st.sidebar.header("System Status")
try:
    # We check the root URL of the backend
    response = requests.get(f"{BACKEND_URL}/") 
    
    # --- THIS IS THE FIX ---
    # We check the status code AND the content
    if response.status_code == 200 and response.json().get('status') == 'ok':
        st.sidebar.success(f"Backend Connected: {response.json()['message']}")
    else:
        st.sidebar.error(f"Backend Connected, but got status: {response.status_code}")
    # -----------------------
        
except requests.exceptions.ConnectionError:
    st.sidebar.error("Backend Connection Failed.")

st.sidebar.divider()
st.sidebar.header("Model Configuration")
model_choice = st.sidebar.selectbox(
    "Choose Floorplan Model",
    ("best_300.pt", "best_1000.pt"),
    help="Select the model to use for floorplan parsing. The 1000-epoch model may be more accurate but slower."
)
st.sidebar.caption(f"Using: {model_choice}")