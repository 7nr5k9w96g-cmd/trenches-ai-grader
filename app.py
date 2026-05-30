import streamlit as st
from dotenv import load_dotenv
import os
import tempfile
import json
import base64
import requests
from datetime import datetime

# Safely import openCV for visual frame extraction
try:
    import cv2
except ImportError:
    cv2 = None

# Load environment variables
load_dotenv()

# Configure Streamlit page layout
st.set_page_config(page_title="TrenchesAI", layout="wide", initial_sidebar_state="expanded")

# Initialize global tracking states safely
if "past_films" not in st.session_state:
    st.session_state["past_films"] = []
if "app_mode" not in st.session_state:
    st.session_state["app_mode"] = "Film Room & Grading"

# --- VIDEO HELPER: SAMPLE FRAMES FOR GPT-4o ---
def extract_video_frames(video_path, max_frames=12):
    if cv2 is None:
        return None
    base64_frames = []
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        video.release()
        return None
    
    step = max(1, total_frames // max_frames)
    for i in range(0, total_frames, step):
        video.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64_frames.append(base64.b64encode(buffer).decode("utf-8"))
        if len(base64_frames) >= max_frames:
            break
    video.release()
    return base64_frames

# --- SECURE FIREBASE AUTHENTICATION ENGINE ---
def firebase_auth_request(endpoint, email, password):
    api_key = os.getenv("FIREBASE_API_KEY")
    if not api_key:
        return None, "System configuration error: Missing Firebase API parameters."
    
    if endpoint == "signup":
        url = f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={api_key}"
    elif endpoint == "login":
        url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={api_key}"
    else:
        return None, "Invalid authentication action protocol."

    payload = {
        "email": email,
        "password": password,
        "returnSecureToken": True
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        res_data = response.json()
        
        if response.status_code == 200:
            return res_data, None
        else:
            error_msg = res_data.get("error", {}).get("message", "Authentication rejected.")
            if error_msg in ["EMAIL_NOT_FOUND", "INVALID_PASSWORD", "INVALID_LOGIN_CREDENTIALS"]:
                return None, "Invalid email address or account password."
            elif error_msg == "EMAIL_EXISTS":
                return None, "An account with this email address already exists."
            elif error_msg == "INVALID_EMAIL":
                return None, "Please provide a properly formatted email address."
            elif error_msg == "MISSING_PASSWORD":
                return None, "The password field cannot be left blank."
            return None, f"Security Alert: {error_msg}"
    except Exception as e:
        return None, f"Network timeout or server configuration mismatch: {str(e)}"

# --- AI MULTIMODAL FILM ENGINE ---
def analyze_football_film_with_openai(video_path):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("Missing OPENAI_API_KEY in your env configuration.")
        return None
        
    if cv2 is None:
        st.error("Missing dependencies: 'opencv-python-headless' must be installed on your backend machine.")
        return None

    try:
        with st.spinner("Analyzing film timeline using professional O-Line grading standards..."):
            base64_frames = extract_video_frames(video_path, max_frames=12)
            if not base64_frames:
                st.error("Could not parse or decode frames from this video file.")
                return None

            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            coaching_prompt = (
                "You are an elite, NFL-level offensive line coach and film coordinator. Analyze these chronological video frames of a football play. "
                "FIRST, look at the alignment, movement dynamics, and block tracks to determine if this is a RUN play or a PASS play. "
                "Evaluate the technique, footwork, stance, and assignment execution for all 5 positions: LT, LG, C, RG, RT. "
                "You MUST apply accurate context-driven coaching terminology based on the play profile: "
                "- IF RUN PLAY: Evaluate explosive initial get-off, drive blocks, down blocks, reach blocks, scoop blocks, combo/double-teams, climbing to linebackers at the second level, tracking inside hip pockets, and maintaining a low pad level. "
                "- IF PASS PLAY: Evaluate pass sets (vertical set, 45-set, jump set), independent hand strikes, active hand replacement, maintaining knee bend, anchoring against a bull rush, recovering against edge speed counters, and passing off stunts/twists. "
                "CRITICAL GRADING SCALE RULES (Strictly enforce these rules for the 'score' integer): "
                "- Use a score of 0 if the lineman DID NOT do their job correctly and WAS NOT physical (e.g., lets defender through cleanly, completely misses assignments). "
                "- Use intermediate scores of 1 or 2 if they executed parts of the assignment but had technical flaws or lacked control. "
                "- Use a score of 3 ONLY if the lineman executed their job perfectly AND was highly physical, dominant, and locked down their assignment. "
                "For each position, write a highly detailed, comprehensive coaching breakdown (3-5 sentences long) evaluating "
                "their mechanics. Be hyper-specific to what happens on this particular run or pass play. "
                "Provide your response STRICTLY in this JSON format, with no markdown tags, backticks, or wrap loops: "
                '{"LT": {"score": 3, "note": "Detailed text..."}, "LG": {"score": 2, "note": "Detailed text..."}, '
                '"C": {"score": 1, "note": "Detailed text..."}, "RG": {"score": 2, "note": "Detailed text..."}, '
                '"RT": {"score": 0, "note": "Detailed text..."}}. '
            )

            content_list = [{"type": "text", "text": coaching_prompt}]
            for frame in base64_frames:
                content_list.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{frame}"}
                })

            payload = {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": content_list}],
                "response_format": {"type": "json_object"}
            }

            response = requests.post(url, headers=headers, json=payload, timeout=90)
            if response.status_code == 200:
                result_json = response.json()
                text_response = result_json["choices"][0]["message"]["content"]
                return json.loads(text_response)
            
            st.error(f"OpenAI API Error ({response.status_code}): {response.text}")
            return None
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None

# --- UI COMPONENTS ---
def render_film_room():
    st.title("The Film Room")
    st.caption("Upload game or practice clips to run AI technique analysis.")
    col1, col2 = st.columns([6, 6])
    
    with col1:
        st.markdown("### Step 1: Upload Clip")
        uploaded_file = st.file_uploader("Choose a football film clip...", type=["mp4", "mov", "avi"])
        
        if uploaded_file:
            st.success("Film loaded successfully!")
            
            # Generate a truly unique signature matching name and file size
            file_signature = f"{uploaded_file.name}_{uploaded_file.size}"
            
            if st.session_state.get("last_file_signature") != file_signature:
                st.session_state["last_file_signature"] = file_signature
                st.session_state["analysis_run"] = False
                if "active_grades" in st.session_state:
                    del st.session_state["active_grades"]

            st.video(uploaded_file)
            has_run = st.session_state.get("analysis_run", False)
            
            if not has_run:
                if st.button("Run AI Film Grader", type="primary", use_container_width=True):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
                        tfile.write(uploaded_file.getvalue())
                        temp_filename = tfile.name
                    
                    ai_response = analyze_football_film_with_openai(temp_filename)
                    os.unlink(temp_filename)
                    
                    if ai_response:
                        st.session_state["active_grades"] = ai_response
                        st.session_state["analysis_run"] = True
                    else:
                        st.error("AI grading failed. Check OpenAI platform billing or API key usage.")
                    st.rerun()
                    
            if has_run:
                st.markdown("---")
                st.markdown("### ⏱️ Film Timeline Breakdown")
                st.info("**[Analysis Mode Active]** Review outputs below.")
                if st.button("Clear / Reset Analysis", use_container_width=True):
                    st.session_state["analysis_run"] = False
                    if "active_grades" in st.session_state:
                        del st.session_state["active_grades"]
                    st.rerun()
        else:
            st.info("Please upload an MP4 or MOV film clip to begin AI analysis.")
                
    with col2:
        st.markdown("### Front-Five Film Grade Sheet")
        if not uploaded_file or not st.session_state.get("analysis_run", False) or "active_grades" not in st.session_state:
            st.write("Upload a video and click 'Run AI Film Grader' to process scores.")
        else:
            current_grades = st.session_state["active_grades"]
            total_points = sum([int(info["score"]) for info in current_grades.values()])
            efficiency_pct = int((total_points / 15) * 100)
            
            m1, m2, m3 = st.columns(3)
            m1.metric(label="Unit Grade Score", value=f"{total_points} / 15")
            m2.metric(label="Play Efficiency", value=f"{efficiency_pct}%")
            m3.metric(label="Unit Avg Get-off", value="0.22s")
            
            st.markdown("---")
            with st.form("grading_override_form"):
                form_data = {}
                for pos in ["LT", "LG", "C", "RG", "RT"]:
                    st.markdown(f"**Position: {pos}**")
                    f_col1, f_col2 = st.columns([2, 5])
                    with f_col1:
                        score_val = st.number_input("Score", min_value=0, max_value=3, value=int(current_grades[pos]["score"]), key=f"score_{pos}")
                    with f_col2:
                        note_val = st.text_input("Coaching Note", value=current_grades[pos]["note"], key=f"note_{pos}")
                    form_data[pos] = {"score": score_val, "note": note_val}
                
                if st.form_submit_button("Save & Finalize Grades", use_container_width=True, type="primary"):
                    st.session_state["past_films"].append({
                        "filename": uploaded_file.name,
                        "video_bytes": uploaded_file.getvalue(),
                        "grades": form_data,
                        "total_score": total_points,
                        "efficiency": efficiency_pct,
                        "timestamp": datetime.now().strftime("%b %d, %Y @ %I:%M:%S %p")
                    })
                    st.session_state["analysis_run"] = False
                    st.session_state["active_grades"] = form_data
                    st.session_state["app_mode"] = "Past Film"
                    st.rerun()

def show_auth_page():
    st.title("TrenchesAI")
    st.subheader("AI-Powered Offensive Line Film Grader")
    
    tab1, tab2 = st.tabs(["Login", "Create Account"])
    with tab1:
        email = st.text_input("Email Address", key="login_email").strip()
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Sign In", key="btn_login", type="primary", use_container_width=True):
            if email and password:
                data, err = firebase_auth_request("login", email, password)
                if data:
                    st.session_state.user = data
                    st.rerun()
                else:
                    st.error(err)
            else:
                st.warning("Please type in both your email address and password to log in.")
                
    with tab2:
        email = st.text_input("Email Address", key="signup_email").strip()
        password = st.text_input("Password (Min 6 Characters)", type="password", key="signup_pass")
        if st.button("Create Staff Account", key="btn_signup", use_container_width=True):
            if email and password:
                if len(password) < 6:
                    st.error("Password Strength Issue: Firebase requires at least 6 characters.")
                else:
                    data, err = firebase_auth_request("signup", email, password)
                    if data:
                        st.success("Account successfully created on secure server! Switch to the 'Login' tab to enter.")
                    else:
                        st.error(err)
            else:
                st.warning("Please complete both input fields to register your email securely.")

def show_dashboard():
    with st.sidebar:
        st.title("TrenchesAI Control")
        st.write(f"Logged in as: **{st.session_state.user.get('email')}**")
        st.markdown("---")
        
        options = ["Film Room & Grading", "Past Film", "Team Reports"]
        if st.session_state["app_mode"] not in options:
            st.session_state["app_mode"] = "Film Room & Grading"
            
        current_index = options.index(st.session_state["app_mode"])
        
        def on_nav_change():
            st.session_state["app_mode"] = st.session_state["nav_sidebar_radio"]
            
        app_mode = st.radio(
            "Go to:", 
            options, 
            index=current_index, 
            key="nav_sidebar_radio", 
            on_change=on_nav_change
        )
        st.markdown("---")
        if st.button("Logout Dashboard", use_container_width=True):
            st.session_state.clear()
            st.rerun()

    if app_mode == "Film Room & Grading":
        render_film_room()
        
    if app_mode == "Past Film":
        st.title("🗂️ Historical Film Vault")
        st.caption("Review your archived clips matched alongside numerical performance metric configurations.")
        
        if not st.session_state["past_films"]:
            st.info("No clips have been archived yet. Go to 'Film Room & Grading' to run your first evaluation.")
        else:
            for idx, saved in enumerate(reversed(st.session_state["past_films"])):
                with st.expander(f"🎬 {saved['filename']} — Graded: {saved['timestamp']}", expanded=(idx==0)):
                    v_col, g_col = st.columns([7, 5])
                    with v_col:
                        st.markdown("**Film Playback Loop**")
                        st.video(saved["video_bytes"])
                    with g_col:
                        st.markdown("**Unit Performance Metrics**")
                        m_col1, m_col2 = st.columns(2)
                        m_col1.metric(label="Unit Score", value=f"{saved['total_score']} / 15")
                        m_col2.metric(label="Play Efficiency", value=f"{saved['efficiency']}%")
                        
                        st.markdown("---")
                        st.markdown("**Individual Positional Grades**")
                        for pos, info in saved["grades"].items():
                            st.markdown(f"**Position {pos}:** `{info['score']} / 3` Points")
                        
    if app_mode == "Team Reports":
        st.title("📝 Scouting Reports & Coaching Logs")
        st.caption("Comprehensive textual technique breakdowns and logs itemized by compilation timestamps.")
        
        if not st.session_state["past_films"]:
            st.info("No logs have been processed yet. Finalize grades in the Film Room to generate an export block.")
        else:
            for idx, saved in enumerate(reversed(st.session_state["past_films"])):
                st.markdown(f"### 📋 Report Log: {saved['filename']}")
                st.caption(f"⏱️ **Graded On:** {saved['timestamp']} | 📊 **Unit Grade Score:** {saved['total_score']}/15 ({saved['efficiency']}%)")
                
                for pos, info in saved["grades"].items():
                    st.markdown(f"**{pos} Evaluation (Grade: {info['score']}/3)**")
                    st.info(info["note"])
                st.markdown("---")

# --- MAIN ROUTING ENGINE ---
if "user" in st.session_state:
    show_dashboard()
else:
    show_auth_page()