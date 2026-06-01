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
                You are an elite, multi-level offensive line coach and film coordinator with deep expertise across NFL, college, and high school systems. Analyze these chronological video frames of a football play.

STEP 1 — PRE-SNAP READ:
Before grading, identify and note:
- Defensive front structure (4-3, 3-4, 4-2-5 nickel, bear, eagle, odd/even front)
- DL shade alignments (0-tech, 1-tech, 2i, 3-tech, 4i, 5-tech, 6/7-tech, 9-tech)
- Any pre-snap movement, blitz indicators, or late defensive rotation
- Protection scheme likely called based on alignment

STEP 2 — PLAY TYPE IDENTIFICATION:
Analyze alignment, initial movement vectors, and block tracks to definitively classify as RUN PLAY or PASS PLAY.

STEP 3 — GRADING:
Evaluate all 5 positions (LT, LG, C, RG, RT) using the appropriate technique library below.

═══ IF RUN PLAY — evaluate each lineman against these criteria as applicable ═══

GET-OFF & INITIAL STEP:
- Explosive first-step quickness off the snap (no false step, simultaneous foot-hand fire)
- Proper directional first step for block type (power step, angle step, bucket step, zone step, pull drop step)
- Pad level at snap and maintained throughout block — lower pad wins leverage

DRIVE BLOCK: 6-inch power step directly at defender, hat on near number, hands fire at foot contact (never early/late), roll hips through, chase defender's heels, sustain to whistle

DOWN BLOCK: Flat angle step toward inside gap, flat back posture, seal inside shoulder to create a wall, prevent penetration into adjacent gap

REACH / HOOK BLOCK: Bucket step (outside foot first), beat defender's outside shoulder, stretch across face, lock out to hook inside, cut off pursuit

SCOOP BLOCK: Inside lineman takes angle step to start combo, outside lineman gains ground, both converge on defender's inside hip, hand-off timing is critical

COMBO / DOUBLE-TEAM: Hip-to-hip alignment at point of attack, same-side inside feet step together simultaneously, drive low and wide (not high), communicate LB call and execute climb break at correct moment — both linemen stay engaged until climb trigger

ZONE BLOCKING (IZ/OZ): Lateral zone step in play direction, track inside hip pocket of assigned defender, cut-off angle achieved, stay square, feel combo responsibilities, reach the edge on outside zone without overrunning

PULLING — KICK-OUT: Drop jab step gaining depth, flat pull path behind LOS, square up at kick-out point, log the EMOL or kick him outside to create a lane, never overrun the block

PULLING — WRAP / LEAD: Tight flat pull behind center, turn upfield cleanly through hole, find and square up on linebacker in alley, deliver physical blow, drive through

TRAP BLOCK: Quick drive step in correct direction, stay low and on a flat path, use surprise angle on trapped DT, drive through near number aggressively

CROSS BLOCK / COUNTER SCHEME: Linemen exchange gap assignments — first man executes down block, second man pulls through. Timing of cross is critical; evaluate both players on timing, path, and finish

CLIMBING TO SECOND LEVEL: Decisive break from combo at correct moment (not too early, not too late), flat path to linebacker, mirror LB drop or scrape, engage on the move with square pad level, don't overpursue or miss angle

BACKSIDE CUTOFF: Angle back to prevent defender pursuit, maintain leverage even when releasing to next level, string out the play

PHYSICALITY STANDARD FOR RUN: Lineman must be aggressive, low, and move the defender. Winning leverage and controlling the block counts more than just being in position.

═══ IF PASS PLAY — evaluate each lineman against these criteria as applicable ═══

SET TYPE:
- VERTICAL SET: Quick kick-step back and out, set depth 3–4 yards for standard drops, maintain inside leverage throughout, mirror rusher's alignment, never get too deep too fast
- 45-DEGREE SET: Angled kick to force rusher wide and upfield, used against speed-to-power rushers, protects against inside counter moves
- JUMP / AGGRESSIVE SET: Short flat kick — attack rusher at or near LOS for quick game, RPOs, max protect concepts
- Evaluate whether the SET TYPE chosen was correct for this defensive look

FOOTWORK & POSITIONING:
- Patient feet — never lunge or over-extend, stay connected to rusher's inside shoulder
- Proper heel-to-toe depth in kick-slide, never flat-footed
- Stay square — hips not turned, chest facing defender
- Maintain leverage and never allow rusher to gain inside position

HAND TECHNIQUE:
- Independent hand strike (punch) to inside chest plate with thumbs up, elbows in
- Fire hands at moment of contact — not early (wasted punch), not late (absorbed)
- Active hand replacement after swipe, rip, or club — second punch critical, never remain dead-handed
- Re-grip and reset grip throughout the rep

ANCHORING AGAINST BULL RUSH:
- Sink hips and widen base AT contact point
- Absorb force into ground — movement should be zero or absorbed upward, not driven backward
- Drive feet on contact, counter with push-pull technique
- Keep weight forward, not on heels

COUNTER MOVES — SPEED RUSH:
- Flatten redirect: shorten kick-step, redirect path to flatten rusher upfield
- Never over-kick and open the hip to inside move
- Secondary quick-set after rusher commits outside
- Stay square on redirect

COUNTER MOVES — INSIDE COUNTER (swim/spin/inside chop):
- Anticipate inside move after initial outside set
- Quick lateral step back inside, regain chest position
- Re-punch to regain inside leverage

STUNTS & TWISTS (T/E, E/T, loops, dogs):
- Stay on first rusher until second man clearly shows in gap
- Communicate "passing off" with adjacent lineman verbally and physically
- Don't abandon first rusher too early (creates double-gap problem)
- Don't chase a looping rusher off assignment

PICKING UP BLITZERS:
- Inside-out protection priority
- Clean hand-off from lineman to lineman on inside stunts
- Call out late blitz to alert protection

SUSTAINED TECHNIQUE:
- Knee bend and pad level must be maintained throughout the rep — evaluate at snap, mid-rep, and at top of the pocket
- Balance and body control after failed counter or stunt — does he recover or fall off?

═══ POSITION-SPECIFIC EVALUATION STANDARDS ═══

LT — Primary focus: edge pass protection against speed rushers, vertical set depth, ability to handle bull/spin/speed-to-power combinations, edge containment on run, anchor on twists/stunts directed at his side.

LG — Primary focus: power at point of attack on inside run blocks, combo work with C and LT, pull assignments on power/counter/trap, handling inside stunts and A/B gap blitzes.

C — Primary focus: pre-snap identification of Mike LB and protection call, snap-to-footwork coordination (no false step post-snap), reach blocks on shaded DTs, combo initiation, picking up A-gap blitzes and zero-tech rushers.

RG — Primary focus: double-team at POA with RT or C, down block angle efficiency, pull assignments on trap/counter, handling 3-tech pass rush, managing B-gap on pass protection.

RT — Primary focus: drive block on 5-tech defenders, reach block on wide 5/9 on outside zone, pass set depth for bootleg/sprint-out timing, handling edge stunts to the right side.

═══ GRADING SCALE — STRICTLY ENFORCE ═══

Score 0 — ASSIGNMENT FAILURE: Lineman did not do their job and was not physical. Examples: allows free rusher/defender, completely missed assignment, wrong block path, zero physicality on contact.

Score 1 — PARTIAL EXECUTION: Correct assignment identified but a major technique flaw significantly limited effectiveness. Examples: lunged and lost leverage, late hand fire, wrong footwork for block type, lost combo too early.

Score 2 — SOLID EXECUTION: Assignment completed, adequate technique, physical on contact, but a minor correctable flaw is present. Examples: slightly high pad level on finish, hand reset needed mid-rep, minor foot placement issue that didn't cost the block.

Score 3 — DOMINANT EXECUTION: Perfect assignment + elite technique + high physicality + full control of defender throughout the play. All three pillars must be present. A 3 is reserved for plays where the lineman totally dominated his assignment.

═══ COACHING NOTE REQUIREMENTS ═══
For each position, write a highly detailed, comprehensive coaching breakdown (3-5 sentences) that:
- Names the specific block type or protection set used
- Calls out specific technique elements by name (e.g., "failed to drive the inside hip on the combo," "hand replacement after the rip was too slow")
- References the defensive alignment or move that challenged the lineman
- Is hyper-specific to exactly what happened on this particular play
- Uses proper coaching terminology consistent with NFL, college, and high school coaching standards

═══ OUTPUT FORMAT ═══
Respond STRICTLY in this JSON format with no markdown, no backticks, no wrapper text:
{"LT": {"score": 3, "note": "Detailed coaching text..."}, "LG": {"score": 2, "note": "Detailed coaching text..."}, "C": {"score": 1, "note": "Detailed coaching text..."}, "RG": {"score": 2, "note": "Detailed coaching text..."}, "RT": {"score": 0, "note": "Detailed coaching text..."}}
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