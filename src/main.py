import time

import streamlit as st
import requests


# The Streamlit UI code
st.set_page_config(page_title="Real Estate Search", layout="wide")
st.title("SmartSense Real Estate Search 🏠")

# We now connect to localhost:8000, because FastAPI is in the same container
BACKEND_URL = "http://localhost:8000"

# How long to wait for the backend to come up before declaring it dead.
# The FastAPI lifespan loads YOLO + EasyOCR + SentenceTransformer + builds
# the LangChain SQL agent before HTTP starts answering, which is ~20–40s
# on a cold start.
BACKEND_BOOT_TIMEOUT_SEC = 60
BACKEND_POLL_INTERVAL_SEC = 1.5


def _probe_backend(timeout: float = 2.0):
    """One quick attempt. Returns (ok: bool, message: str)."""
    try:
        r = requests.get(f"{BACKEND_URL}/", timeout=timeout)
        if r.status_code == 200 and r.json().get("status") == "ok":
            return True, r.json().get("message", "OK")
        return False, f"HTTP {r.status_code}"
    except requests.exceptions.RequestException as e:
        return False, str(e.__class__.__name__)


st.sidebar.header("System Status")
status_placeholder = st.sidebar.empty()

# Allow a manual re-check after a previous "Disconnected" verdict.
if "backend_ready" not in st.session_state:
    st.session_state.backend_ready = False

# Fast path: if a previous run already confirmed the backend is up, just do
# one cheap probe to make sure it's still up. Avoids the 20s wait on every
# user interaction.
if st.session_state.backend_ready:
    ok, msg = _probe_backend()
    if ok:
        status_placeholder.success(f"Backend Connected: {msg}")
    else:
        # It was up, now it's not. Flip back to unknown and let the slow path
        # below decide whether it's a blip or a real outage.
        st.session_state.backend_ready = False

# Slow path: backend hasn't been confirmed up yet (cold start, or it just
# went down). Try once instantly; if that fails, poll with a live status so
# the user sees "starting up" instead of a misleading "disconnected".
if not st.session_state.backend_ready:
    ok, msg = _probe_backend()
    if ok:
        status_placeholder.success(f"Backend Connected: {msg}")
        st.session_state.backend_ready = True
    else:
        deadline = time.monotonic() + BACKEND_BOOT_TIMEOUT_SEC
        attempt = 1
        while time.monotonic() < deadline:
            elapsed = int(BACKEND_BOOT_TIMEOUT_SEC - (deadline - time.monotonic()))
            status_placeholder.info(
                f"⏳ Backend is starting up… ({elapsed}s elapsed, attempt {attempt})"
            )
            time.sleep(BACKEND_POLL_INTERVAL_SEC)
            ok, msg = _probe_backend()
            if ok:
                status_placeholder.success(f"Backend Connected: {msg}")
                st.session_state.backend_ready = True
                break
            attempt += 1
        else:
            # Loop ran to completion without break — give up gracefully.
            status_placeholder.error(
                f"❌ Backend disconnected (no response after "
                f"{BACKEND_BOOT_TIMEOUT_SEC}s). Last error: {msg}"
            )
            if st.sidebar.button("🔄 Retry connection"):
                st.rerun()

# --- 1. Ingestion UI ---
st.header("Data Ingestion")

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

# --- 2. Phase 1 Debug UI ---
st.divider()
st.header("Floorplan Image Parser")
uploaded_image = st.file_uploader("Upload a single floorplan image", type=["jpg", "png", "jpeg"])

# Show the image as soon as it's uploaded
if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded Floorplan", use_container_width=False)

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

# --- 3. Chatbot UI ---
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

    # Build (user, assistant) history from completed pairs only. This skips any
    # dangling user message left over from a failed previous turn, and naturally
    # excludes the just-added new user message at the tail.
    api_history = []
    msgs = st.session_state.messages
    i = 0
    while i + 1 < len(msgs):
        if msgs[i]["role"] == "user" and msgs[i + 1]["role"] == "assistant":
            api_history.append((msgs[i]["content"], msgs[i + 1]["content"]))
            i += 2
        else:
            i += 1

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