"""Exercise 5 — Streamlit approval UI for the HITL PR review agent.

Run with:
    uv run streamlit run app.py

Goal: wrap the LangGraph built in exercises 1–4 in a web UI that adapts to
the confidence bucket of each PR.

Routing thresholds (common/schemas.py):
    > 72%        auto_approve     UI shows a success card; reviewer does nothing
    58 – 72%     human_approval   UI shows Approve / Reject / Edit buttons
    <  58%       escalate         UI shows a question form for the reviewer
"""

from __future__ import annotations

import asyncio
import uuid

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from common.db import db_conn, db_path
from exercises.exercise_4_audit import build_graph, handle_interrupt


load_dotenv()


# ─── Session state ─────────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "pr_url" not in st.session_state:
    st.session_state.pr_url = ""
if "interrupt_payload" not in st.session_state:
    st.session_state.interrupt_payload = None
if "final" not in st.session_state:
    st.session_state.final = None


# ─── Page setup ────────────────────────────────────────────────────────────
st.set_page_config(page_title="HITL PR Review", layout="wide")
st.title("HITL PR Review Agent")


# ─── Sidebar — recent sessions ─────────────────────────────────────────────
async def _load_recent_threads() -> list[dict]:
    """Query audit_events for the most recent review sessions."""
    async with db_conn() as conn:
        async with conn.execute(
            """
            SELECT thread_id,
                   pr_url,
                   MAX(timestamp)  AS last_event,
                   COUNT(*)         AS events,
                   -- 'high' < 'med' < 'low' alphabetically, so MAX() would
                   -- always return 'low'. Rank explicitly to get the worst risk.
                   MAX(CASE risk_level WHEN 'high' THEN 3 WHEN 'med' THEN 2 ELSE 1 END)
                                      AS risk_rank
              FROM audit_events
             GROUP BY thread_id, pr_url
             ORDER BY MAX(timestamp) DESC
             LIMIT 15
            """
        ) as cur:
            rows = await cur.fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["worst_risk"] = {3: "high", 2: "med"}.get(d.pop("risk_rank"), "low")
        out.append(d)
    return out


with st.sidebar:
    st.header("Recent sessions")
    try:
        _threads = asyncio.run(_load_recent_threads())
    except Exception:
        _threads = []

    if not _threads:
        st.caption("No sessions yet — run a review.")
    else:
        for t in _threads:
            risk = t["worst_risk"]
            icon = {"high": "🔴", "med": "🟡", "low": "🟢"}[risk]
            short = t["pr_url"].rstrip("/").split("/")[-2:]
            label = f"{icon} {'/'.join(short)} · {t['events']}ev"
            if st.button(
                label, key=f"thread-{t['thread_id']}", use_container_width=True,
                disabled=(t["thread_id"] == st.session_state.thread_id),
            ):
                st.session_state.thread_id = t["thread_id"]
                st.session_state.pr_url = t["pr_url"]
                st.session_state.interrupt_payload = None
                st.session_state.final = None
                st.rerun()


# ─── Top form — start a new review ─────────────────────────────────────────
with st.form("start"):
    pr_url = st.text_input(
        "PR URL", value=st.session_state.pr_url,
        placeholder="https://github.com/VinUni-AI20k/PR-Demo/pull/1",
    )
    submitted = st.form_submit_button("Run review")


# ─── Renderers per interrupt kind ──────────────────────────────────────────
def render_approval_card(payload: dict) -> dict | None:
    """58–72% bucket: show the LLM review + 3 buttons. Return resume dict or None."""
    conf = payload["confidence"]
    st.subheader(f"Approval requested — confidence {conf:.0%}")
    st.caption(payload["confidence_reasoning"])
    st.markdown(payload["summary"])

    for c in payload.get("comments", []):
        st.markdown(f"- **[{c['severity']}]** `{c['file']}:{c.get('line') or '?'}` — {c['body']}")

    with st.expander("Diff"):
        st.code(payload.get("diff_preview", ""), language="diff")

    feedback = st.text_input("Feedback (optional)", key="approval_feedback")
    col1, col2, col3 = st.columns(3)
    # TODO: hook up the three buttons. Each click should return one of:
    #   {"choice": "approve", "feedback": feedback}
    #   {"choice": "reject",  "feedback": feedback}
    #   {"choice": "edit",    "feedback": feedback}
    if col1.button("Approve", type="primary", use_container_width=True, key="btn_approve"):
        return {"choice": "approve", "feedback": feedback}
    if col2.button("Reject", use_container_width=True, key="btn_reject"):
        return {"choice": "reject", "feedback": feedback}
    if col3.button("Edit", use_container_width=True, key="btn_edit"):
        return {"choice": "edit", "feedback": feedback or "Please refine the review."}
    return None


def render_escalation_card(payload: dict) -> dict | None:
    """< 58% bucket: show risk factors + question form. Return {question: answer} or None."""
    conf = payload["confidence"]
    st.subheader(f"Strong escalation — confidence {conf:.0%}")
    st.caption(payload["confidence_reasoning"])
    if payload.get("risk_factors"):
        st.error("Risks: " + ", ".join(payload["risk_factors"]))
    st.markdown(payload["summary"])

    with st.form("escalation"):
        # TODO: render one text_input per question in payload["questions"]
        #       collect answers into a dict {question: answer_str}
        #       on submit, return the dict.
        answers: dict[str, str] = {}
    for i, q in enumerate(payload["questions"]):
        answers[q] = st.text_input(f"Q{i + 1}: {q}", key=f"esc_q_{i}")
    if st.form_submit_button("Submit answers"):
        filled = {q: a for q, a in answers.items() if a.strip()}
        if filled:
            return filled
        # Reviewer submitted nothing — make the ambiguity explicit rather than
        # silently synthesising against empty context.
        return {payload["questions"][0]: "No context given — treat every risk as a blocker."}
    return None


# ─── Drive the graph ───────────────────────────────────────────────────────
async def run_graph(pr_url: str, thread_id: str, resume_value=None):
    """Invoke the graph once. Returns the final result or {'__interrupt__': ...}."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        # TODO: build the graph with `cp` as the checkpointer (use the function
        # you imported/copied at the top of this file).
        # app = build_graph(cp)
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}

        # TODO:
        # - If resume_value is None: result = await app.ainvoke(
        #       {"pr_url": pr_url, "thread_id": thread_id}, cfg)
        # - Else:                    result = await app.ainvoke(
        #       Command(resume=resume_value), cfg)
        # - Return result.
        if resume_value is None:
            return await app.ainvoke({"pr_url": pr_url, "thread_id": thread_id}, cfg)
        return await app.ainvoke(Command(resume=resume_value), cfg)


# ─── Main flow ─────────────────────────────────────────────────────────────
if submitted and pr_url:
    st.session_state.pr_url = pr_url
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.interrupt_payload = None
    st.session_state.final = None

    with st.spinner("Fetching PR + asking the LLM..."):
        result = asyncio.run(run_graph(pr_url, st.session_state.thread_id))

    if "__interrupt__" in result:
        st.session_state.interrupt_payload = result["__interrupt__"][0].value
    else:
        st.session_state.final = result

# Render the current interrupt card, if any
payload = st.session_state.interrupt_payload
if payload is not None:
    kind = payload["kind"]
    answer = render_approval_card(payload) if kind == "approval_request" else render_escalation_card(payload)
    if answer is not None:
        with st.spinner("Resuming..."):
            result = asyncio.run(run_graph(
                st.session_state.pr_url, st.session_state.thread_id, resume_value=answer,
            ))
        if "__interrupt__" in result:
            st.session_state.interrupt_payload = result["__interrupt__"][0].value
        else:
            st.session_state.interrupt_payload = None
            st.session_state.final = result
        st.rerun()

# Render final state, if reached
if st.session_state.final is not None:
    final = st.session_state.final
    action = final.get("final_action", "?")
    if action.startswith("auto") or action.startswith("committed"):
        st.success(f"✓ {action} — comment posted to {st.session_state.pr_url}")
    elif action == "rejected":
        st.warning("Rejected — no comment posted")
    else:
        st.info(f"final_action = {action}")
    st.caption(f"thread_id = {st.session_state.thread_id}  ·  replay: "
               f"`uv run python -m audit.replay --thread {st.session_state.thread_id}`")
