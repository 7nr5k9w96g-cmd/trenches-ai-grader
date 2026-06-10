import streamlit as st
from dotenv import load_dotenv
import os
import tempfile
import json
import time
import base64
import requests
from datetime import datetime
from google import genai
from google.genai import types

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from vision_pipeline import (
        run_vision_pipeline, TECHNIQUE_LABELS, save_training_label,
        train_classifier, get_training_stats, predict_technique
    )
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False

# Load environment variables
load_dotenv()

# Configure Streamlit page layout
st.set_page_config(page_title="TrenchesAI", layout="wide", initial_sidebar_state="expanded")

# Initialize global tracking states safely
if "past_films" not in st.session_state:
    st.session_state["past_films"] = []
if "app_mode" not in st.session_state:
    st.session_state["app_mode"] = "Film Room & Grading"

def extract_video_frames(video_path, fps_target=6, max_frames=60):
    if cv2 is None:
        return None
    frames = []
    video = cv2.VideoCapture(video_path)
    total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    native_fps = video.get(cv2.CAP_PROP_FPS) or 30
    if total <= 0:
        video.release()
        return None
    # Scale frame count with duration: fps_target frames per second, capped at max_frames
    duration_secs = total / native_fps
    target = min(max(int(duration_secs * fps_target), 10), max_frames)
    step = max(1, total // target)
    indices = list(range(0, total, step))[:target]
    for i in indices:
        video.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = video.read()
        if not ok:
            continue
        frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frames.append(base64.b64encode(buf).decode("utf-8"))
    video.release()
    return frames if frames else None

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

# --- AI MULTIMODAL FILM ENGINE (Gemini) ---
# play_type: "Auto-Detect", "Run Play", or "Pass Play"
RUN_PLAY_TYPES = [
    "General Run Play",
    "Power / Power-O",
    "Counter",
    "Iso / Lead",
    "Inside Zone (IZ)",
    "Outside Zone (OZ)",
    "Draw",
    "Trap",
]

def analyze_football_film_with_gemini(video_path, play_type="Auto-Detect", run_type=None, player_ids=None):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("Missing GEMINI_API_KEY in your env configuration.")
        return None

    try:
        client = genai.Client(api_key=api_key)

        with st.spinner("Extracting frames from film..."):
            if cv2 is None:
                st.error("opencv-python-headless is not installed.")
                return None
            frames = extract_video_frames(video_path)
            if not frames:
                st.error("Could not extract frames from this video file.")
                return None
            st.caption(f"Extracted {len(frames)} frames for analysis.")

        cv_data = None
        if VISION_AVAILABLE:
            with st.spinner("Running player detection and pose estimation..."):
                cv_text, cv_err, cv_features = run_vision_pipeline(video_path)
                if cv_text:
                    # Run classifier predictions if model exists
                    predictions = {}
                    for pos, feats in cv_features.items():
                        pred = predict_technique(feats)
                        if pred:
                            predictions[pos] = pred
                    if predictions:
                        pred_lines = "\n".join(f"  {pos}: {tech}" for pos, tech in predictions.items())
                        cv_text += f"\n\nCLASSIFIER PREDICTIONS (trained on coach-labeled film):\n{pred_lines}\nUse these as a strong prior — override only if the frames clearly contradict them.\n"
                    cv_data = cv_text
                    st.session_state["last_cv_features"] = cv_features
                # Non-fatal — if CV fails, Gemini still runs on frames alone

        with st.spinner("Analyzing film using professional O-Line grading standards..."):
            # Play type override injected at the very top of the prompt
            if play_type == "Run Play":
                if run_type and run_type != "General Run Play":
                    play_type_override = (
                        f"PLAY TYPE: The coach has confirmed this is a RUN PLAY — the scheme is {run_type}. "
                        f"Use ONLY the run blocking grading criteria. Do NOT apply pass protection grading. "
                        f"IMPORTANT: Do not assume what any individual lineman is doing based on the scheme name. "
                        f"Observe what each player actually does in the frames, then evaluate those observed actions "
                        f"against the {run_type} technique standards. A lineman may be pulling, driving, zoning, or doing something else entirely — "
                        f"grade what you see, not what you expect from the scheme label.\n\n"
                    )
                else:
                    play_type_override = (
                        "PLAY TYPE: The coach has confirmed this is a RUN PLAY. "
                        "Use ONLY the run blocking grading criteria. Do NOT apply pass protection grading. "
                        "Observe what each lineman actually does in the frames and grade those observed actions.\n\n"
                    )
            elif play_type == "Pass Play":
                play_type_override = (
                    "CRITICAL OVERRIDE: The coach has confirmed this is a PASS PLAY. "
                    "You MUST evaluate this as a pass play using ONLY the pass protection criteria. "
                    "Do NOT classify it as a run play. Do NOT apply run blocking grading. "
                    "Skip Step 3 classification and proceed directly to pass play grading.\n\n"
                )
            else:
                play_type_override = ""

            ids = player_ids or {}
            pos_desc = ["far left of the line", "second from left", "middle (snaps ball)", "second from right", "far right of the line"]
            if any(v.strip() for v in ids.values()):
                id_lines = []
                for i, pos in enumerate(["LT", "LG", "C", "RG", "RT"]):
                    pid = ids.get(pos, "").strip()
                    id_lines.append(f"- {pos}: jersey #{pid}" if pid else f"- {pos}: not tagged — use spatial position ({pos_desc[i]})")
                player_id_section = (
                    "STEP 1 - PLAYER IDENTIFICATION (provided by coach):\n"
                    + "\n".join(id_lines)
                    + "\n\nDo NOT attempt to re-identify players yourself. Trust the coach's tags. "
                    "Watch what each tagged player does in the frames and grade their technique.\n\n"
                )
            else:
                player_id_section = (
                    "STEP 1 - IDENTIFY EACH LINEMAN BY POSITION (no tags provided):\n"
                    "A. Find the row of 5 players at the line of scrimmage.\n"
                    "B. CENTER is the middle player of that row — #3 of 5 from either end. He snaps the ball.\n"
                    "C. QB is directly behind the center. Use him to confirm.\n"
                    "D. One left of center = LG, one right = RG, far left = LT, far right = RT.\n"
                    "E. Lock these in and do not change them mid-play.\n\n"
                )

            cv_section = (cv_data + "\n\n") if cv_data else ""

            coaching_prompt = (
                cv_section +
                play_type_override +
                "You are an elite, multi-level offensive line coach and film coordinator with deep expertise across NFL, college, and high school systems. Analyze these chronological video frames of a football play.\n\n"
                + player_id_section
                + "STEP 2 - PRE-SNAP READ:\n"
                "- Defensive front structure (4-3, 3-4, 4-2-5 nickel, bear, eagle, odd/even front)\n"
                "- DL shade alignments (0-tech, 1-tech, 2i, 3-tech, 4i, 5-tech, 6/7-tech, 9-tech)\n"
                "- Any pre-snap movement, blitz indicators, or late defensive rotation\n\n"
                "STEP 3 - PLAY TYPE IDENTIFICATION:\n"
                "Analyze alignment, initial movement vectors, and block tracks to definitively classify as RUN PLAY or PASS PLAY.\n\n"
                "STEP 4 - INDIVIDUAL LINEMAN TRACKING (REQUIRED BEFORE GRADING):\n"
                "CRITICAL RULE: Grade what you SEE, not what you expect. The play type label tells you which rubric to score against — "
                "it does NOT tell you what each lineman is doing. Do not assume a lineman is pulling, zoning, or driving just because the play type implies it. "
                "Watch what each player actually does in the frames and grade that action.\n\n"
                "For each of the 5 linemen — LT, LG, C, RG, RT — trace what you actually observe across the frames:\n"
                "1. Starting stance and alignment vs the defender in front of them\n"
                "2. First step: the exact direction and type of step you see them take (lateral, forward, angle drop, kick slide, etc.)\n"
                "3. Who or what they actually move toward and engage\n"
                "4. What happens at contact: hand placement, pad level, leverage won or lost — based on what the frames show\n"
                "5. How the block finishes: sustained, broke down, drove, released early — what you can see\n"
                "If you cannot clearly see a lineman's actions due to camera angle or occlusion, say so explicitly. Do not fabricate.\n\n"
                "STEP 5 - GRADING:\n"
                "Grade each position based solely on what you observed in Step 4. Apply the rubric for this play type.\n\n"
                "=== IF RUN PLAY - evaluate against these criteria ===\n\n"
                "GET-OFF & INITIAL STEP:\n"
                "- Explosive first-step quickness off snap - no false step, simultaneous foot-hand fire\n"
                "- Correct directional first step for the block type (power step, angle step, bucket step, zone step, pull drop step)\n"
                "- Pad level at snap and maintained throughout - lower pad wins leverage\n\n"
                "DRIVE BLOCK: 6-inch power step at defender, hat on near number, hands fire at foot contact, roll hips through, chase defender's heels, sustain to whistle\n\n"
                "DOWN BLOCK: Flat angle step to inside gap, flat back, seal inside shoulder to create a wall, prevent penetration\n\n"
                "REACH / HOOK BLOCK: Bucket step (outside foot first), beat defender's outside shoulder, stretch across face, lock out to hook\n\n"
                "SCOOP BLOCK: Inside lineman angle step, outside lineman gains ground, both converge on defender's inside hip, hand-off timing critical\n\n"
                "COMBO / DOUBLE-TEAM (ACE/DEUCE/TREY): Hip-to-hip at POA, same-side inside feet step simultaneously, drive low and wide, communicate and execute climb break at correct moment - both stay engaged until climb trigger\n\n"
                "DUO BLOCK: Pure double-team power with no climb intent - both linemen drive defender off the LOS, evaluate sustained joint push and pad level\n\n"
                "ZONE BLOCKING (IZ/OZ): Lateral zone step in play direction, track inside hip pocket of assigned defender, achieve cut-off angle, stay square, feel combo responsibilities, reach edge on OZ without overrunning\n\n"
                "PULLING - KICK-OUT: Drop jab step gaining depth, flat pull path, square up at kick-out point, log or kick the EMOL, never overrun\n\n"
                "PULLING - WRAP / LEAD: Tight flat pull, turn upfield cleanly through hole, square up on linebacker in alley, deliver physical blow\n\n"
                "TRAP BLOCK: Quick drive step on flat path, stay low, surprise angle on trapped DT, drive through near number\n\n"
                "CROSS BLOCK / G-T COUNTER / COUNTER TREY: Evaluate both the down-blocking lineman AND the pulling lineman separately - down blocker path and seal, puller's depth, flat pull path, kick-out vs wrap assignment, and finish\n\n"
                "PIN & PULL (perimeter): Inside lineman pins - evaluate seal angle and width; outside puller - evaluate pull path, turn upfield, and block in space at second level\n\n"
                "WHAM BLOCK SCHEME: When FB/TE executes wham kick-out, evaluate the releasing lineman's path - correct release angle, timing away from the wham blocker, and arrival at second-level assignment\n\n"
                "SPLIT ZONE: Line zones one direction - evaluate each lineman's zone path; backside lineman's cutoff angle when H-back kicks EMOL is critical - evaluate his release timing and cutoff block\n\n"
                "HINGE / TURNBACK BLOCK (bootleg/naked): Backside lineman stays between rusher and QB path, gives ground under control, does not chase - evaluate body position and containment\n\n"
                "BACKSIDE CUTOFF: Angle back to prevent pursuit, maintain leverage, string out the play\n\n"
                "PHYSICALITY STANDARD FOR RUN: Must be aggressive, low, and moving the defender. Winning leverage matters more than just being in position.\n\n"
                "=== IF PASS PLAY - evaluate against these criteria ===\n\n"
                "PROTECTION SCHEME IDENTIFICATION (visual read only):\n"
                "- BOB (Big on Big / man): covered linemen block their man, uncovered work inside-out to first threat\n"
                "- FULL SLIDE: entire line zones one direction - evaluate each lineman's gap discipline and correct zone-side assignment\n"
                "- HALF SLIDE / 4-MAN SLIDE: one side zones, backside tackle mans up on edge - evaluate correct role identification and execution for each lineman\n"
                "- COMBINATION PROTECTION: zone blocking on one side of the center, man blocking on the other - this is the most critical to grade correctly. Evaluate whether each lineman correctly identified his scheme side. On the zone side: gap discipline, no chasing, correct area coverage. On the man side: proper tracking of assigned rusher, no abandoning man for a ghost. A lineman executing man technique on a zone side (or vice versa) is a technique error regardless of outcome.\n\n"
                "SET TYPE:\n"
                "- VERTICAL SET: Quick kick-step back and out, set depth 3-4 yards, maintain inside leverage, mirror rusher's alignment\n"
                "- 45-DEGREE SET: Angled kick to force rusher wide, used against speed-to-power rushers, protects inside counter\n"
                "- JUMP / AGGRESSIVE SET: Short flat kick, attack rusher at or near LOS for quick game/RPO/max protect\n"
                "- Evaluate whether set type chosen was correct for the defensive look\n\n"
                "FOOTWORK & POSITIONING: Patient feet, stay connected to rusher's inside shoulder, heel-to-toe depth in kick-slide, stay square, never allow rusher inside position\n\n"
                "HAND TECHNIQUE: Independent punch to inside chest plate (thumbs up, elbows in), hands fire at contact, active hand replacement after swipe/rip/club, re-grip and reset throughout rep\n\n"
                "ANCHORING VS BULL RUSH: Sink hips and widen base at contact, absorb into ground, drive feet, counter with push-pull, weight forward not on heels\n\n"
                "COUNTER MOVES - SPEED RUSH: Flatten redirect, shorten kick, redirect path to flatten rusher upfield, never over-kick and open hip\n\n"
                "COUNTER MOVES - INSIDE COUNTER (swim/spin/chop): Quick lateral step back inside, regain chest position, re-punch for inside leverage\n\n"
                "STUNTS & TWISTS (T/E, E/T, fire stunts, loops): Stay on first rusher until second man clearly shows, pass off cleanly, never abandon first rusher too early\n\n"
                "SCREEN BLOCKING: Correct chip-and-release timing - deliver a controlled chip on the pass rusher, then release cleanly to second level; evaluate arrival angle and block in space on screen target defender\n\n"
                "SUSTAINED TECHNIQUE: Knee bend and pad level evaluated at snap, mid-rep, and top of pocket - must be maintained throughout\n\n"
                "=== PENALTY AWARENESS - apply to ALL plays ===\n\n"
                "Actively flag any of the following if visible in the frames:\n"
                "- HOLDING: hands outside the frame, jersey pull, arm-bar wrap around defender\n"
                "- HANDS TO THE FACE: palm or forearm contacting defender's helmet or facemask\n"
                "- FALSE START: any pre-snap flinch or movement before the ball\n"
                "- ILLEGAL BLOCK IN THE BACK: contact on the back of a defender away from the ball (perimeter runs, screens)\n"
                "- CHOP BLOCK RISK: any high/low combination where one blocker goes low while another is already engaged high on the same defender\n\n"
                "PENALTY GRADING RULE: If a penalty is identified, note it explicitly in the coaching note. A technically sound block that draws or risks a flag CANNOT score a 3. A block that both wins AND avoids penalties is a prerequisite for elite grades.\n\n"
                "=== EFFORT & FINISH - apply to ALL plays ===\n\n"
                "Evaluate the following on every lineman:\n"
                "- Did he sustain his block through the whistle, or did he let up early?\n"
                "- RUN: Did he drive his feet and finish, or make contact and stall?\n"
                "- PASS: Did he maintain his set for the full rep, or relax before the throw?\n"
                "- Did he pursue a second-level block or deliver a chip after his initial assignment was sealed?\n"
                "- HIGH MOTOR PLAYS: pancake attempts, staying attached on long developing runs, sprinting downfield to finish - these push a borderline 2 to a 3\n"
                "- EFFORT DEDUCTION RULE: A technically correct block that quit early or showed no motor CANNOT score a 3. Early release on a play requiring a finish is an automatic deduction.\n\n"
                "=== POSITION-SPECIFIC EVALUATION STANDARDS ===\n\n"
                "LT - Primary: edge pass protection vs speed rushers, vertical set depth, handle bull/spin/speed-to-power combos, edge containment on run, stunt/twist anchoring on his side\n\n"
                "LG - Primary: power at POA on inside run, combo work with C and LT, pull assignments on power/counter/trap, handle inside stunts and A/B gap blitzes\n\n"
                "C - Primary: snap-to-footwork coordination (no false step post-snap), reach blocks on shaded DTs, combo initiation, picking up A-gap blitzes and zero-tech rushers\n\n"
                "RG - Primary: double-team at POA with RT or C, down block angle, pull on trap/counter, handle 3-tech pass rush, B-gap management\n\n"
                "RT - Primary: drive block on 5-tech, reach block on wide 5/9 on OZ, pass set depth for bootleg/sprint-out, handle edge stunts to right side\n\n"
                "=== GRADING SCALE - STRICTLY ENFORCE ===\n\n"
                "Score 0 - ASSIGNMENT FAILURE: Did not do their job, was not physical. Allows free rusher, missed assignment, wrong path, zero effort.\n\n"
                "Score 1 - PARTIAL EXECUTION: Correct assignment but a major technique flaw significantly limited effectiveness (lunged, late hands, wrong footwork, lost combo too early).\n\n"
                "Score 2 - SOLID EXECUTION: Assignment completed, physical on contact, but a minor correctable flaw present (slightly high pad level on finish, hand reset needed, minor foot placement issue).\n\n"
                "Score 3 - DOMINANT EXECUTION: Perfect assignment + elite technique + high physicality + full finish through the whistle + zero penalty risk. ALL four pillars required. A 3 is reserved for total domination of the assignment.\n\n"
                "=== COACHING NOTE REQUIREMENTS ===\n\n"
                "For each position write a thorough coaching note structured as follows. Take as many sentences as needed — do not cut yourself short:\n\n"
                "SENTENCE 1 — WHAT YOU SAW (observation only, no evaluation yet): Describe the specific physical actions this lineman took. "
                "Name their first step direction, who or what they moved toward, whether they made contact, what their body position looked like. "
                "Example: 'The LT took a short lateral zone step left, climbed to the second level, and arrived at the linebacker as the ball carrier cut back.' "
                "Do NOT evaluate yet — just describe what the frames show.\n\n"
                "SENTENCES 2-3 — TECHNIQUE EVALUATION: Now evaluate those observed actions against the criteria for this block type. "
                "Name specific technique flaws or wins by their coaching term. Reference the defensive look that challenged him. "
                "Example: 'His bucket step was correct for the reach assignment against the 5-tech, but his hands were outside the frame on initial contact, preventing a clean lock-out.'\n\n"
                "SENTENCE 4 — EFFORT AND FINISH: Did he sustain through the whistle? Did he drive feet after contact or go passive? Note any motor plays or early releases.\n\n"
                "SENTENCE 5 (if applicable) — PENALTY RISK or VISIBILITY NOTE: Flag any penalty risk, OR if the camera angle limited visibility of this lineman, state that honestly. "
                "Do not guess or fabricate observations for linemen you could not clearly see.\n\n"
                "=== OUTPUT FORMAT ===\n\n"
                "Respond STRICTLY in this JSON format with no markdown, no backticks, no wrapper text:\n"
                "{\"LT\": {\"score\": 3, \"note\": \"Detailed coaching text...\"}, \"LG\": {\"score\": 2, \"note\": \"Detailed coaching text...\"}, "
                "\"C\": {\"score\": 1, \"note\": \"Detailed coaching text...\"}, \"RG\": {\"score\": 2, \"note\": \"Detailed coaching text...\"}, "
                "\"RT\": {\"score\": 0, \"note\": \"Detailed coaching text...\"}}"
            )

            contents = [types.Part(text=coaching_prompt)]
            for frame_b64 in frames:
                contents.append(types.Part(
                    inline_data=types.Blob(
                        mime_type="image/jpeg",
                        data=base64.b64decode(frame_b64)
                    )
                ))

            last_error = None
            for attempt in range(3):
                try:
                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=contents,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            max_output_tokens=8192,
                            thinking_config=types.ThinkingConfig(
                                thinking_budget=24576
                            )
                        )
                    )
                    return json.loads(response.text)
                except Exception as e:
                    last_error = e
                    if attempt < 2:
                        wait = 10 * (attempt + 1)
                        st.caption(f"Gemini busy — retrying in {wait}s (attempt {attempt + 2}/3)...")
                        time.sleep(wait)
            raise last_error
    except Exception as e:
        st.error(f"Gemini analysis failed: {str(e)}")
        return None

# --- UI COMPONENTS ---
def render_film_room():
    st.title("The Film Room")
    st.caption("Upload game or practice clips to run AI technique analysis.")
    st.info(
        "**Disclaimer:** AI analysis may not always be accurate. If you see a wrong grade or misidentified technique, "
        "correct it using the dropdowns after grading — every correction you make trains the system to grade better over time."
    )
    col1, col2 = st.columns([6, 6])
    
    with col1:
        st.markdown("### Step 1: Upload Film Clip")
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
                for pos in ["LT", "LG", "C", "RG", "RT"]:
                    st.session_state.pop(f"pid_{pos}", None)

            st.video(uploaded_file)
            has_run = st.session_state.get("analysis_run", False)

            if not has_run:
                st.markdown("### Step 2: Tag Your Players")
                st.caption("Pause the video at the snap and enter each player's jersey number. This removes all identification guesswork from the AI — it will only grade what it sees each tagged player doing.")
                pid_cols = st.columns(5)
                player_ids = {}
                for i, pos in enumerate(["LT", "LG", "C", "RG", "RT"]):
                    with pid_cols[i]:
                        player_ids[pos] = st.text_input(
                            pos, placeholder="#__",
                            key=f"pid_{pos}",
                            help=f"Jersey number of the {pos}"
                        )
                any_tagged = any(v.strip() for v in player_ids.values())
                if any_tagged:
                    st.success("Players tagged — AI will use your identifications instead of guessing.")
                else:
                    st.caption("Optional but strongly recommended. Leave blank to let the AI attempt spatial identification.")

                st.markdown("### Step 3: Select Play Type")
                play_type = st.selectbox(
                    "Tell the AI what type of play this is:",
                    options=["Auto-Detect", "Run Play", "Pass Play"],
                    index=0,
                    help=(
                        "Auto-Detect: AI reads the frames and decides. "
                        "Run Play / Pass Play: Locks the AI to the correct criteria - "
                        "use this when you know the play type to prevent misclassification."
                    ),
                    key="play_type_select"
                )

                run_type = None
                if play_type == "Auto-Detect":
                    st.caption("AI will attempt to classify run vs pass from the frames. For best accuracy, select the play type manually.")
                elif play_type == "Run Play":
                    run_type = st.selectbox(
                        "Select the run scheme:",
                        options=RUN_PLAY_TYPES,
                        index=0,
                        help="Telling the AI the exact run scheme locks its grading to the correct footwork paths, assignment rules, and technique standards for that play.",
                        key="run_type_select"
                    )
                    if run_type == "General Run Play":
                        st.caption("Locked to run blocking criteria. Select a specific scheme above for more precise grading.")
                    else:
                        st.caption(f"Locked to {run_type} grading criteria — AI will evaluate each lineman against the exact assignment and technique requirements for this scheme.")
                else:
                    st.caption("Locked to pass protection criteria: set type, hand technique, anchor, stunts, and all pass pro techniques.")

                st.markdown("### Step 4: Run Analysis")
                if st.button("Run AI Film Grader", type="primary", use_container_width=True):
                    file_ext = os.path.splitext(uploaded_file.name)[1].lower() or ".mp4"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tfile:
                        tfile.write(uploaded_file.getvalue())
                        temp_filename = tfile.name

                    ai_response = analyze_football_film_with_gemini(temp_filename, play_type=play_type, run_type=run_type, player_ids=player_ids)
                    os.unlink(temp_filename)

                    if ai_response:
                        st.session_state["active_grades"] = ai_response
                        st.session_state["analysis_run"] = True
                        st.rerun()
                    else:
                        st.error("AI grading failed — see the error above for details. Check your GEMINI_API_KEY or try again.")
                    
            if has_run:
                st.markdown("---")
                st.markdown("### Film Timeline Breakdown")
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
            
            m1, m2 = st.columns(2)
            m1.metric(label="Unit Grade Score", value=f"{total_points} / 15")
            m2.metric(label="Play Efficiency", value=f"{efficiency_pct}%")
            
            st.markdown("---")
            with st.form("grading_override_form"):
                form_data = {}
                technique_labels = {}
                for pos in ["LT", "LG", "C", "RG", "RT"]:
                    st.markdown(f"**Position: {pos}**")
                    f_col1, f_col2 = st.columns([2, 5])
                    with f_col1:
                        score_val = st.number_input("Score", min_value=0, max_value=3, value=int(current_grades[pos]["score"]), key=f"score_{pos}")
                    with f_col2:
                        note_val = st.text_input("Coaching Note", value=current_grades[pos]["note"], key=f"note_{pos}")
                    form_data[pos] = {"score": score_val, "note": note_val}
                    if VISION_AVAILABLE:
                        technique_labels[pos] = st.selectbox(
                            f"Technique ({pos}) — for classifier training",
                            options=["— skip —"] + TECHNIQUE_LABELS,
                            key=f"tech_{pos}"
                        )

                if st.form_submit_button("Save & Finalize Grades", use_container_width=True, type="primary"):
                    if VISION_AVAILABLE:
                        cv_feats = st.session_state.get("last_cv_features", {})
                        for pos, tech in technique_labels.items():
                            if tech and tech != "— skip —":
                                save_training_label(
                                    clip_name=uploaded_file.name,
                                    position=pos,
                                    technique=tech,
                                    grade=form_data[pos]["score"],
                                    notes=form_data[pos]["note"],
                                    frame_features=cv_feats.get(pos, {})
                                )
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
        if VISION_AVAILABLE:
            st.markdown("---")
            st.markdown("**Technique Classifier**")
            stats = get_training_stats()
            st.caption(f"{stats['total']} labeled clips saved")
            if stats['total'] > 0:
                for tech, count in sorted(stats['by_technique'].items(), key=lambda x: -x[1]):
                    st.caption(f"  {tech}: {count}")
            if stats['ready_to_train']:
                if st.button("Train Model", use_container_width=True, type="primary"):
                    with st.spinner("Training classifier..."):
                        ok, msg = train_classifier()
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)
            else:
                needed = max(0, 20 - stats['total'])
                st.caption(f"Label {needed} more clips to unlock training.")

        st.markdown("---")
        if st.button("Logout Dashboard", use_container_width=True):
            st.session_state.clear()
            st.rerun()

    if app_mode == "Film Room & Grading":
        render_film_room()
        
    if app_mode == "Past Film":
        st.title("Historical Film Vault")
        st.caption("Review your archived clips matched alongside numerical performance metric configurations.")
        
        if not st.session_state["past_films"]:
            st.info("No clips have been archived yet. Go to 'Film Room & Grading' to run your first evaluation.")
        else:
            for idx, saved in enumerate(reversed(st.session_state["past_films"])):
                with st.expander(f"{saved['filename']} -- Graded: {saved['timestamp']}", expanded=(idx==0)):
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
        st.title("Scouting Reports & Coaching Logs")
        st.caption("Comprehensive textual technique breakdowns and logs itemized by compilation timestamps.")
        
        if not st.session_state["past_films"]:
            st.info("No logs have been processed yet. Finalize grades in the Film Room to generate an export block.")
        else:
            for idx, saved in enumerate(reversed(st.session_state["past_films"])):
                st.markdown(f"### Report Log: {saved['filename']}")
                st.caption(f"Graded On: {saved['timestamp']} | Unit Grade Score: {saved['total_score']}/15 ({saved['efficiency']}%)")
                
                for pos, info in saved["grades"].items():
                    st.markdown(f"**{pos} Evaluation (Grade: {info['score']}/3)**")
                    st.info(info["note"])
                st.markdown("---")

# --- MAIN ROUTING ENGINE ---
if "user" in st.session_state:
    show_dashboard()
else:
    show_auth_page()