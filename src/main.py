import os
import time

import streamlit as st
import requests

st.set_page_config(page_title="Real Estate Search", layout="wide")
st.title("SmartSense Real Estate Search 🏠")

# Defaults to the Docker Compose service name; set to the API's URL elsewhere.
BACKEND_URL = os.environ.get("API_BASE_URL", "http://api:8000")

# Sent as X-API-Key to POST /ingest and POST /parse-floorplan-debug. Streamlit
# runs server-side, so the key never reaches the browser.
API_KEY = os.environ.get("API_KEY", "")

# The API loads its models and agent before it responds (~20-40s cold start).
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

# If the backend was already up, a single probe is enough.
if st.session_state.backend_ready:
    ok, msg = _probe_backend()
    if ok:
        status_placeholder.success(f"Backend Connected: {msg}")
    else:
        # Fall through to the polling path below.
        st.session_state.backend_ready = False

# Otherwise poll until the backend responds or the timeout passes, showing
# a "starting up" status meanwhile.
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
            status_placeholder.info(f"⏳ Backend is starting up… ({elapsed}s elapsed, attempt {attempt})")
            time.sleep(BACKEND_POLL_INTERVAL_SEC)
            ok, msg = _probe_backend()
            if ok:
                status_placeholder.success(f"Backend Connected: {msg}")
                st.session_state.backend_ready = True
                break
            attempt += 1
        else:
            # Timed out
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
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        try:
            with st.spinner("Uploading..."):
                response = requests.post(f"{BACKEND_URL}/ingest", files=files, headers={"X-API-Key": API_KEY})
            # /ingest queues an Airflow job and returns 202 with a job_id.
            if response.status_code == 202:
                body = response.json()
                st.success(f"Queued as job {body['job_id']}. Poll {body['status_url']} for status.")
            else:
                st.error(f"Error: {response.json()['detail']}")
        except requests.exceptions.ConnectionError:
            st.error("Backend connection failed. Is it running?")
    else:
        st.error("Please upload a file first.")

# --- 2. Floorplan parser ---
st.divider()
st.header("Floorplan Image Parser")
uploaded_image = st.file_uploader("Upload a single floorplan image", type=["jpg", "png", "jpeg"])

# Show the image as soon as it's uploaded
if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded Floorplan", use_container_width=False)

if st.button("Parse Floorplan"):
    if uploaded_image is not None:
        files = {"file": (uploaded_image.name, uploaded_image.getvalue(), uploaded_image.type)}
        try:
            with st.spinner("Parsing image..."):
                response = requests.post(
                    f"{BACKEND_URL}/parse-floorplan-debug", files=files, headers={"X-API-Key": API_KEY}
                )

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

    # Build (user, assistant) history from completed pairs only, skipping
    # unanswered messages, including the one just added.
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
                    f"{BACKEND_URL}/chat", json={"query": prompt, "history": api_history}
                )
            if response.status_code == 200:
                ai_response = response.json()["response"]
                st.markdown(ai_response)
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
            else:
                st.error(f"Error from API: {response.json()['detail']}")
    except requests.exceptions.ConnectionError:
        st.error("Backend is unreachable. Please check if it's running.")
