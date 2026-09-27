import time
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field


app = FastAPI(title="Vera Merchant AI Assistant", version="1.0.0")

START_TIME = time.time()

# (scope, id) -> {"version": int, "payload": dict}
contexts: dict[tuple[str, str], dict[str, Any]] = {}

# conversation_id -> state
conversations: dict[str, dict[str, Any]] = {}

# suppression keys already acted on
suppressed: set[str] = set()


# ============================================================
# MODELS
# ============================================================

class ContextBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str


class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = Field(default_factory=list)


class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: str | None = None
    customer_id: str | None = None
    from_role: str
    message: str
    received_at: str
    turn_number: int


# ============================================================
# BASIC HELPERS
# ============================================================

def get_context(scope: str, context_id: str | None):
    if not context_id:
        return None
    item = contexts.get((scope, context_id))
    return item["payload"] if item else None


def nested(d: dict, *keys, default=None):
    cur = d
    for key in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
        if cur is None:
            return default
    return cur


def merchant_name(m: dict) -> str:
    identity = m.get("identity", {})
    return (
        identity.get("owner_first_name")
        or identity.get("name")
        or m.get("merchant_id")
        or "there"
    )


def full_merchant_name(m: dict) -> str:
    return m.get("identity", {}).get(
        "name",
        m.get("merchant_id", "your business")
    )


def merchant_id_from_trigger(t: dict) -> str | None:
    return (
        t.get("merchant_id")
        or nested(t, "payload", "merchant_id")
    )


def customer_id_from_trigger(t: dict) -> str | None:
    return (
        t.get("customer_id")
        or nested(t, "payload", "customer_id")
    )


def get_category(m: dict):
    return get_context("category", m.get("category_slug"))


def get_customer(t: dict):
    cid = customer_id_from_trigger(t)
    return get_context("customer", cid)


def active_offer(m: dict):
    for offer in m.get("offers", []):
        if offer.get("status") == "active":
            return offer
    return None


def first_active_offer(m: dict):
    return active_offer(m)


def language_style(m: dict, c: dict | None):
    lang = (
        nested(c or {}, "identity", "language_pref")
        or nested(m, "identity", "languages", default=[])
    )

    if isinstance(lang, str) and ("hi" in lang.lower() or "mix" in lang.lower()):
        return "hinglish"

    if isinstance(lang, list) and any("hi" in str(x).lower() for x in lang):
        return "english_hinglish"

    return "english"


def safe_pct(value):
    if value is None:
        return None
    try:
        return round(float(value) * 100)
    except Exception:
        return None


def offer_price(title: str) -> str | None:
    if "₹" in title:
        idx = title.find("₹")
        return title[idx:].split()[0].rstrip(".,")
    return None


# ============================================================
# DECISION ENGINE
# ============================================================

def decide(category, merchant, trigger, customer=None):
    kind = trigger.get("kind", "")
    payload = trigger.get("payload", {}) or {}
    merchant_name_short = merchant_name(merchant)
    category_name = merchant.get("category_slug", "")

    # --------------------------------------------------------
    # RESEARCH
    # --------------------------------------------------------
    if kind == "research_digest":
        top = payload.get("top_item", {})

        if not top:
            top_id = payload.get("top_item_id")
            for item in category.get("digest", []):
                if item.get("id") == top_id:
                    top = item
                    break

        title = top.get("title", "a new category research update")
        source = top.get("source", "")
        trial_n = top.get("trial_n")
        segment = top.get("patient_segment")

        details = []
        if trial_n:
            details.append(f"{trial_n:,}-patient")
        if segment:
            details.append(f"relevant to {segment.replace('_', ' ')}")

        detail_text = " ".join(details)

        body = f"{merchant_name_short}, a new {category_name} research update is worth a look. "
        if detail_text:
            body += f"{detail_text.capitalize()} evidence: {title}. "
        else:
            body += f"{title}. "

        if source:
            body += f"Source: {source}. "

        body += "Want me to pull the key points and turn them into something useful for your customers?"

        return result(
            body,
            "open_ended",
            trigger,
            "Research trigger matched to the merchant's category; message uses the supplied research facts and offers a concrete next step."
        )

    # --------------------------------------------------------
    # REGULATION
    # --------------------------------------------------------
    if kind == "regulation_change":
        deadline = payload.get("deadline_iso")
        item = None

        top_id = payload.get("top_item_id")
        for x in category.get("digest", []):
            if x.get("id") == top_id:
                item = x
                break

        title = (item or {}).get("title", "a regulatory update")

        body = (
            f"{merchant_name_short}, there's a regulatory update relevant to "
            f"your {category_name} business: {title}."
        )

        if deadline:
            body += f" The supplied deadline is {deadline}."

        body += " Want me to summarise the practical changes you need to check?"

        return result(
            body,
            "open_ended",
            trigger,
            "Regulation-change trigger is relevant to the merchant's category and provides a concrete review action."
        )

    # --------------------------------------------------------
    # RECALL
    # --------------------------------------------------------
    if kind == "recall_due" and customer:
        cname = nested(customer, "identity", "name", default="there")
        p = payload

        slots = p.get("available_slots", [])
        slot_text = ""

        if len(slots) >= 2:
            slot_text = (
                f"{slots[0].get('label')} or {slots[1].get('label')}"
            )
        elif len(slots) == 1:
            slot_text = slots[0].get("label", "")

        offer = first_active_offer(merchant)
        offer_title = offer.get("title") if offer else None

        body = f"Hi {cname}, {full_merchant_name(merchant)} here. "
        body += "Your 6-month cleaning recall is due."

        if p.get("due_date"):
            body += f" Your due date is {p['due_date']}."

        if slot_text:
            body += f" We have {slot_text} available."

        if offer_title:
            body += f" {offer_title}."

        body += " Reply with the slot you prefer, or tell us a better time."

        return result(
            body,
            "open_ended",
            trigger,
            "Customer recall trigger uses the customer's supplied due date, available slots and active merchant offer."
        )

    # --------------------------------------------------------
    # PERFORMANCE DIP
    # --------------------------------------------------------
    if kind == "perf_dip":
        metric = payload.get("metric", "performance")
        delta = safe_pct(payload.get("delta_pct"))
        baseline = payload.get("vs_baseline")
        window = payload.get("window", "recent period")

        body = f"{merchant_name_short}, your {metric} is down"
        if delta is not None:
            body += f" {abs(delta)}%"
        body += f" over the last {window}"

        if baseline is not None:
            body += f" versus a baseline of {baseline}"

        body += ". Rather than pushing another generic promotion, I'd check the conversion/listing side first. Want me to walk through the likely cause?"

        return result(
            body,
            "open_ended",
            trigger,
            "Performance-dip trigger is translated into a diagnostic conversation rather than an unrelated promotion."
        )

    # --------------------------------------------------------
    # PERFORMANCE SPIKE
    # --------------------------------------------------------
    if kind == "perf_spike":
        metric = payload.get("metric", "performance")
        delta = safe_pct(payload.get("delta_pct"))
        window = payload.get("window", "recent period")

        body = f"{merchant_name_short}, your {metric} has spiked"
        if delta is not None:
            body += f" {delta}%"
        body += f" over {window}. "

        offer = active_offer(merchant)
        if offer:
            body += f"Your active offer, {offer.get('title')}, is a natural candidate to amplify. "
        body += "Want me to suggest the lowest-friction way to capture the extra demand?"

        return result(
            body,
            "open_ended",
            trigger,
            "Performance spike is connected to amplification of an existing merchant asset."
        )

    # --------------------------------------------------------
    # RENEWAL
    # --------------------------------------------------------
    if kind == "renewal_due":
        days = payload.get("days_remaining")
        plan = payload.get("plan")
        amount = payload.get("renewal_amount")

        body = f"{merchant_name_short}, your {plan or 'current'} plan renewal is coming up"
        if days is not None:
            body += f" in {days} days"
        if amount is not None:
            body += f" (₹{amount})"
        body += (
    ". The useful next step is to review the renewal before the "
    "deadline rather than letting the plan lapse. Want me to prepare "
    "the renewal action?"
)

        return result(
            body,
            "open_ended",
            trigger,
            "Renewal trigger uses the supplied plan, timing and amount."
        )

    # --------------------------------------------------------
    # FESTIVAL
    # --------------------------------------------------------
    if kind == "festival_upcoming":
        festival = payload.get("festival", "festival")
        date = payload.get("date")
        days = payload.get("days_until")

        body = f"{merchant_name_short}, {festival} is coming up"
        if days is not None:
            body += f" in {days} days"
        if date:
            body += f" ({date})"

        body += (
    ". For this category, the useful move is to adapt an existing "
    "offer to the occasion rather than launch something unrelated. "
    "Want me to turn your current offer into one festival-specific action?"
)

        return result(
            body,
            "open_ended",
            trigger,
            "Festival trigger is connected to the supplied category relevance and asks for one concrete merchant action."
        )

    # --------------------------------------------------------
    # WEDDING FOLLOW-UP
    # --------------------------------------------------------
    if kind == "wedding_package_followup" and customer:
        cname = nested(customer, "identity", "name", default="there")
        wedding = payload.get("wedding_date")
        days = payload.get("days_to_wedding")
        next_step = payload.get("next_step_window_open")

        body = f"Hi {cname}, following up from your earlier consultation."
        if wedding:
            body += f" Your wedding date is {wedding}"
        if days is not None:
            body += f" ({days} days away)"
        body += "."

        if next_step:
            body += f" The next-step window for {next_step.replace('_', ' ')} is open."

        body += " Would you like us to help you plan that next step?"

        return result(
            body,
            "open_ended",
            trigger,
            "Wedding follow-up uses the customer's known event timing and the supplied next-step window."
        )

    # --------------------------------------------------------
    # CURIOUS ASK
    # --------------------------------------------------------
    if kind == "curious_ask_due":
        body = (
            f"{merchant_name_short}, quick one for this week: "
            "what would you like to know about demand or customer behaviour "
            "for your business?"
        )

        return result(
            body,
            "open_ended",
            trigger,
            "Scheduled curiosity trigger opens a useful merchant conversation without inventing a metric."
        )

    # --------------------------------------------------------
    # WINBACK
    # --------------------------------------------------------
    if kind == "winback_eligible":
        expiry = payload.get("days_since_expiry")
        lapsed = payload.get("lapsed_customers_added_since_expiry")
        dip = safe_pct(payload.get("perf_dip_pct"))

        body = f"{merchant_name_short}, there is a win-back opportunity worth revisiting."
        if expiry is not None:
            body += f" Your previous plan expired {expiry} days ago."
        if lapsed is not None:
            body += f" {lapsed} lapsed customers have been added since expiry."
        if dip is not None:
            body += f" Performance is also down {abs(dip)}%."
        body += " Want me to outline the simplest recovery action?"

        return result(
            body,
            "open_ended",
            trigger,
            "Win-back trigger combines the supplied expiry, lapsed-customer and performance signals."
        )

    # --------------------------------------------------------
    # IPL / EVENT
    # --------------------------------------------------------
    if kind == "ipl_match_today":
        match = payload.get("match", "today's match")
        venue = payload.get("venue")
        city = payload.get("city")
        match_time = payload.get("match_time_iso")

        body = f"{merchant_name_short}, {match} is scheduled today"
        if venue:
            body += f" at {venue}"
        if city:
            body += f" in {city}"
        if match_time:
            body += f" ({match_time})"

        body += ". If this audience fits your business, want me to suggest one timely offer angle?"

        return result(
            body,
            "open_ended",
            trigger,
            "Event trigger uses the supplied match, venue and timing without inventing demand."
        )

    # --------------------------------------------------------
    # REVIEW THEME
    # --------------------------------------------------------
    if kind == "review_theme_emerged":
        theme = payload.get("theme", "an issue")
        occurrences = payload.get("occurrences_30d")
        quote = payload.get("common_quote")

        body = f"{merchant_name_short}, a review pattern is emerging around '{theme.replace('_', ' ')}'."

        if occurrences is not None:
            body += f" It appeared in {occurrences} reviews over the last 30 days."

        if quote:
            body += f" One common wording was: “{quote}”."

        body += " This looks like a better fix-now opportunity than another promotion. Want me to map the next action?"

        return result(
            body,
            "open_ended",
            trigger,
            "Review-theme trigger identifies a concrete recurring customer issue before suggesting growth activity."
        )

    # --------------------------------------------------------
    # MILESTONE
    # --------------------------------------------------------
    if kind == "milestone_reached":
        metric = payload.get("metric", "metric")
        value = payload.get("value_now")
        milestone = payload.get("milestone_value")

        body = f"{merchant_name_short}, you're at {value} on {metric.replace('_', ' ')}"
        if milestone:
            body += f" and close to the {milestone} milestone"
        body += ". Worth turning that moment into a simple customer-facing proof point?"

        return result(
            body,
            "open_ended",
            trigger,
            "Milestone trigger uses the supplied current and target values."
        )

    # --------------------------------------------------------
    # ACTIVE PLANNING
    # --------------------------------------------------------
    if kind == "active_planning_intent":
        intent = payload.get("intent_topic", "the idea")
        last_message = payload.get("merchant_last_message")

        body = f"{merchant_name_short}, you were already exploring {intent.replace('_', ' ')}."
        if last_message:
            body += f' You said: "{last_message}"'
        body += " I can turn that into a concrete next step now. Shall I?"

        return result(
            body,
            "open_ended",
            trigger,
            "Existing merchant intent is carried forward instead of restarting the sales pitch."
        )

    # --------------------------------------------------------
    # SEASONAL DIP
    # --------------------------------------------------------
    if kind == "seasonal_perf_dip":
        metric = payload.get("metric", "performance")
        delta = safe_pct(payload.get("delta_pct"))
        note = payload.get("season_note")

        body = f"{merchant_name_short}, {metric} is down"
        if delta is not None:
            body += f" {abs(delta)}%"
        body += ", and the supplied signal marks this as seasonal rather than unexpected."

        if note:
            body += f" Context: {note.replace('_', ' ')}."

        body += " I'd avoid treating this as a generic failure; want a seasonal response plan instead?"

        return result(
            body,
            "open_ended",
            trigger,
            "Seasonal performance trigger is explicitly treated differently from an unexplained performance dip."
        )

    # --------------------------------------------------------
    # CUSTOMER LAPSED
    # --------------------------------------------------------
    if kind == "customer_lapsed_hard" and customer:
        cname = nested(customer, "identity", "name", default="there")
        days = payload.get("days_since_last_visit")
        focus = payload.get("previous_focus")
        months = payload.get("previous_membership_months")

        body = f"Hi {cname}, {full_merchant_name(merchant)} here."
        if days is not None:
            body += f" It's been {days} days since your last visit."
        if focus:
            body += f" We remember your previous focus was {focus.replace('_', ' ')}."
        if months:
            body += f" You were with us for {months} months before."

        body += " If you'd like to restart, reply and we'll help with the next step."

        return result(
            body,
            "open_ended",
            trigger,
            "Customer win-back uses only the supplied relationship history and prior focus."
        )

    # --------------------------------------------------------
    # TRIAL FOLLOW-UP
    # --------------------------------------------------------
    if kind == "trial_followup" and customer:
        cname = nested(customer, "identity", "name", default="there")
        body = f"Hi {cname}, following up after your trial with {full_merchant_name(merchant)}."

        if payload:
            for key in ("trial_date", "days_since_trial", "program"):
                if payload.get(key):
                    body += f" {key.replace('_', ' ').title()}: {payload[key]}."

        body += " Want to continue with the next step?"

        return result(
            body,
            "open_ended",
            trigger,
            "Trial follow-up continues an existing customer journey rather than making a generic acquisition pitch."
        )

    # --------------------------------------------------------
    # SUPPLY ALERT
    # --------------------------------------------------------
    if kind == "supply_alert":
        medication = payload.get("medicine") or payload.get("product") or payload.get("item")
        status = payload.get("status") or payload.get("alert")

        body = f"{merchant_name_short}, there's a supply alert"
        if medication:
            body += f" for {medication}"
        if status:
            body += f": {status}"
        body += ". Want me to help you work through the merchant-side response?"

        return result(
            body,
            "open_ended",
            trigger,
            "Supply alert is surfaced directly without inventing inventory or demand."
        )

    # --------------------------------------------------------
    # CHRONIC REFILL
    # --------------------------------------------------------
    if kind == "chronic_refill_due" and customer:
        cname = nested(customer, "identity", "name", default="there")

        body = f"Hi {cname}, {full_merchant_name(merchant)} here. "
        body += "Your refill reminder is due."

        due = payload.get("due_date")
        if due:
            body += f" The supplied due date is {due}."

        body += " Reply if you'd like us to help with the next step."

        return result(
            body,
            "open_ended",
            trigger,
            "Customer refill message uses the supplied reminder context without making medical claims."
        )

    # --------------------------------------------------------
    # CATEGORY SEASONAL
    # --------------------------------------------------------
    if kind == "category_seasonal":
        topic = payload.get("topic") or payload.get("season") or "a seasonal demand shift"
        body = (
            f"{merchant_name_short}, there's a category-level seasonal shift around "
            f"{str(topic).replace('_', ' ')}. "
            "Want me to translate the supplied signal into one concrete action for your business?"
        )

        return result(
            body,
            "open_ended",
            trigger,
            "Category-seasonal trigger is framed as an opportunity to act on category context."
        )

    # --------------------------------------------------------
    # GBP
    # --------------------------------------------------------
    if kind == "gbp_unverified":
        body = (
            f"{merchant_name_short}, your Google Business Profile verification status "
            "needs attention. This is a concrete listing-health issue rather than a promotion opportunity. "
            "Want me to walk through the verification step?"
        )

        return result(
            body,
            "open_ended",
            trigger,
            "Listing verification issue gets routed to the relevant operational action."
        )

    # --------------------------------------------------------
    # CDE OPPORTUNITY
    # --------------------------------------------------------
    if kind == "cde_opportunity":
        title = payload.get("title") or payload.get("event") or "a category opportunity"

        body = (
            f"{merchant_name_short}, there's a relevant opportunity: {title}. "
            "Want me to turn it into a concrete action for your business?"
        )

        return result(
            body,
            "open_ended",
            trigger,
            "External opportunity is connected to the merchant's category without inventing details."
        )

    # --------------------------------------------------------
    # COMPETITOR
    # --------------------------------------------------------
    if kind == "competitor_opened":
        distance = payload.get("distance_km")
        competitor = payload.get("competitor_name")

        body = f"{merchant_name_short}, a nearby competitor signal just appeared"
        if competitor:
            body += f" for {competitor}"
        if distance:
            body += f" about {distance} km away"

        body += ". I'd focus on your existing customer value rather than reacting blindly. Want me to identify one concrete defensive action?"

        return result(
            body,
            "open_ended",
            trigger,
            "Competitor trigger is treated as a contextual signal, not a reason to invent competitor claims."
        )

    # --------------------------------------------------------
    # DORMANCY
    # --------------------------------------------------------
    if kind == "dormant_with_vera":
        body = (
            f"{merchant_name_short}, we haven't had a merchant conversation recently. "
            "Rather than sending another generic tip, is there one business problem you want Vera to help with today?"
        )

        return result(
            body,
            "open_ended",
            trigger,
            "Dormancy trigger reopens the relationship with a low-friction question."
        )

       # --------------------------------------------------------
    # SMART FALLBACK
    # --------------------------------------------------------

    offer = active_offer(merchant)

    # Collect useful facts from the trigger payload.
    facts = []

    for key, label in [
        ("metric", "metric"),
        ("delta_pct", "change"),
        ("window", "window"),
        ("value_now", "current value"),
        ("target_value", "target"),
        ("title", "topic"),
        ("topic", "topic"),
        ("event", "event"),
        ("product", "product"),
        ("medicine", "medicine"),
        ("status", "status"),
    ]:
        value = payload.get(key)

        if value is not None and value != "":
            if key == "delta_pct":
                try:
                    value = f"{abs(float(value)) * 100:.0f}%"
                except Exception:
                    pass

            facts.append(f"{label}: {value}")

    body = f"{merchant_name_short}, I picked up a {kind.replace('_', ' ')} signal for your business."

    if facts:
        body += " The useful details are " + "; ".join(
            str(x) for x in facts[:3]
        ) + "."

    if offer:
        title = offer.get("title")
        if title:
            body += f" Your active offer is {title}."

    # Convert the signal into a concrete action rather than a generic question.
    if "performance" in kind or "dip" in kind:
        body += (
            " I'd focus on diagnosing the drop before adding another promotion. "
            "Want me to identify the most relevant next step?"
        )

    elif "spike" in kind or "milestone" in kind:
        body += (
            " This is a good moment to capture the momentum with the "
            "existing offer rather than changing everything. "
            "Want me to suggest one amplification step?"
        )

    elif "season" in kind or "festival" in kind:
        body += (
            " The practical next step is to adapt the existing offer "
            "to this seasonal demand. Want me to draft that action?"
        )

    elif "customer" in kind or "followup" in kind or "follow_up" in kind:
        body += (
            " The next step should use the customer's existing context "
            "rather than restarting the conversation. Want me to draft it?"
        )

    else:
        body += (
            " I can turn this specific signal into one concrete action "
            "instead of sending a generic recommendation."
        )

    return result(
        body,
        "open_ended",
        trigger,
        (
            f"Fallback decision uses the actual {kind.replace('_', ' ')} "
            "signal, available trigger facts, merchant offer context and "
            "a concrete next action."
        )
    )


def result(body, cta, trigger, rationale):
    return {
        "body": body,
        "cta": cta,
        "send_as": (
            "merchant_on_behalf"
            if trigger.get("scope") == "customer"
            else "vera"
        ),
        "suppression_key": trigger.get(
            "suppression_key",
            trigger.get("id", trigger.get("kind", "unknown"))
        ),
        "rationale": rationale,
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/v1/healthz")
async def healthz():
    counts = {
        "category": 0,
        "merchant": 0,
        "customer": 0,
        "trigger": 0,
    }

    for scope, _ in contexts:
        if scope in counts:
            counts[scope] += 1

    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": counts,
    }


# ============================================================
# METADATA
# ============================================================

@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": "Vera Decision Engine",
        "team_members": ["Tushar Tyagi"],
        "model": "deterministic-contextual-engine",
        "approach": (
            "Context-aware routing by trigger kind with merchant/category/"
            "customer grounding, suppression and multi-turn intent handling."
        ),
        "contact_email": "team@example.com",
        "version": "1.0.0",
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }


# ============================================================
# CONTEXT
# ============================================================

@app.post("/v1/context")
async def push_context(body: ContextBody):
    valid_scopes = {"category", "merchant", "customer", "trigger"}

    if body.scope not in valid_scopes:
        return {
            "accepted": False,
            "reason": "invalid_scope",
            "details": f"Unsupported scope: {body.scope}",
        }

    key = (body.scope, body.context_id)
    current = contexts.get(key)

    if current is not None and current["version"] >= body.version:
        return {
            "accepted": False,
            "reason": "stale_version",
            "current_version": current["version"],
        }

    contexts[key] = {
        "version": body.version,
        "payload": body.payload,
    }

    return {
        "accepted": True,
        "ack_id": f"ack_{body.context_id}_v{body.version}",
        "stored_at": datetime.now(timezone.utc).isoformat(),
    }


# ============================================================
# TICK
# ============================================================

@app.post("/v1/tick")
async def tick(body: TickBody):
    actions = []

    for trigger_id in body.available_triggers:
        trigger = get_context("trigger", trigger_id)

        if not trigger:
            continue

        mid = merchant_id_from_trigger(trigger)

        if not mid:
            continue

        merchant = get_context("merchant", mid)
        if not merchant:
            continue

        category = get_category(merchant)
        if not category:
            continue

        customer = get_customer(trigger)

        suppression_key = trigger.get(
            "suppression_key",
            trigger_id
        )

        if suppression_key in suppressed:
            continue

        decision = decide(
            category,
            merchant,
            trigger,
            customer,
        )

        conversation_id = (
            f"conv_{mid}_{trigger_id}"
        )

        cid = customer_id_from_trigger(trigger)

        action = {
            "conversation_id": conversation_id,
            "merchant_id": mid,
            "customer_id": cid,
            "send_as": decision["send_as"],
            "trigger_id": trigger_id,
            "template_name": f"vera_{trigger.get('kind', 'context')}_v1",
            "template_params": [
                merchant_name(merchant),
                decision["body"],
            ],
            "body": decision["body"],
            "cta": decision["cta"],
            "suppression_key": decision["suppression_key"],
            "rationale": decision["rationale"],
        }

        actions.append(action)

        suppressed.add(decision["suppression_key"])

        conversations.setdefault(
            conversation_id,
            {
                "merchant_id": mid,
                "customer_id": cid,
                "trigger_id": trigger_id,
                "turns": [],
                "status": "active",
            },
        )

        conversations[conversation_id]["turns"].append(
            {
                "role": "vera",
                "body": decision["body"],
                "timestamp": body.now,
            }
        )

        if len(actions) >= 20:
            break

    return {"actions": actions}


# ============================================================
# REPLY / MULTI-TURN HANDLING
# ============================================================

def is_stop(message: str):
    m = message.lower()

    phrases = [
        "stop",
        "not interested",
        "don't message",
        "dont message",
        "do not message",
        "unsubscribe",
        "remove me",
        "no more",
        "stop messaging",
        "stop sending",
        "this is spam",
        "useless spam",
    ]

    return any(x in m for x in phrases)


def is_auto_reply(message: str, history: list[dict]):
    normalized = message.strip().lower()

    auto_phrases = [
        "thank you for contacting",
        "thanks for contacting",
        "our team will respond",
        "our team will get back",
        "we will respond shortly",
        "automated response",
        "this is an automated",
        "office hours",
        "will get back to you",
    ]

    phrase_match = any(
        x in normalized for x in auto_phrases
    )

    # Count consecutive incoming auto-replies.
    consecutive_auto = 0

    for turn in reversed(history):
        if turn.get("role") not in ("merchant", "customer"):
            break

        text = turn.get("body", "").lower()

        if any(x in text for x in auto_phrases):
            consecutive_auto += 1
        else:
            break

    # Include the current message.
    if phrase_match:
        consecutive_auto += 1

    return phrase_match, consecutive_auto

def is_commitment(message: str):
    m = message.lower()

    phrases = [
        "yes",
        "yes please",
        "go ahead",
        "let's do it",
        "lets do it",
        "do it",
        "proceed",
        "confirm",
        "send it",
        "send me",
        "okay do it",
        "ok do it",
        "i want to join",
        "i want this",
        "sign me up",
    ]

    return any(x in m for x in phrases)


@app.post("/v1/reply")
async def reply(body: ReplyBody):

    conversation = conversations.setdefault(
        body.conversation_id,
        {
            "merchant_id": body.merchant_id,
            "customer_id": body.customer_id,
            "turns": [],
            "status": "active",
        },
    )

    history = conversation["turns"]

    message = body.message.strip()

    history.append(
        {
            "role": body.from_role,
            "body": message,
            "timestamp": body.received_at,
        }
    )

    if is_stop(message):
        conversation["status"] = "ended"

        return {
            "action": "end",
            "rationale": (
                "The recipient explicitly requested that messaging stop; "
                "the conversation is closed."
            ),
        }

    auto_reply, consecutive_auto = is_auto_reply(message, history)

    if auto_reply:

        if consecutive_auto >= 4:
         conversation["status"] = "ended"

        return {
            "action": "end",
            "rationale": (
                "Repeated automated replies were received four times "
                "without meaningful engagement, so Vera ends the "
                "conversation to avoid repeated messaging."
            ),
        }

    return {
        "action": "wait",
        "wait_seconds": 14400,
        "rationale": (
            "The response resembles an automated business reply, "
            "so Vera backs off rather than treating it as customer intent."
        ),
    }

    if is_commitment(message):
        conversation["status"] = "action"

        return {
            "action": "send",
            "body": (
                "Got it — let's move ahead. I'll use the details already "
                "available and keep the next step focused."
            ),
            "cta": "open_ended",
            "rationale": (
                "The recipient expressed clear action intent, so the "
                "conversation transitions from persuasion to execution."
            ),
        }

    # Asking for clarification.
    if "?" in message or any(
        x in message.lower()
        for x in ["how", "what", "why", "which", "when", "price", "cost"]
    ):
        return {
            "action": "send",
            "body": (
                "Sure. I'll keep the answer tied to the context and "
                "specific details already available rather than guessing."
            ),
            "cta": "open_ended",
            "rationale": (
                "The recipient asked a follow-up question; maintain the "
                "conversation instead of restarting the pitch."
            ),
        }

    return {
        "action": "send",
        "body": (
            "Got it. I’ll keep this focused on the current opportunity "
            "and the details already available."
        ),
        "cta": "open_ended",
        "rationale": (
            "Acknowledged the response while preserving the current "
            "conversation context."
        ),
    }