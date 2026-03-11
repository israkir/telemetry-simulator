# Clustering Scenarios

## Single-turn chain (one trace / one turn)

In the **single-turn** version, the user gives everything in one message (claim + description + preferred slot).  
The simulator emits **one data-plane trace** (one `trace_id`) for that turn; all spans in the trace share the same `session_id` and `gen_ai.conversation.id`.

For each division, the LLM’s replies look like a normal assistant, even though it is chaining tools internally.

- **Phone (server: `phone`)**
  - **User:** "Start a claim for my phone — I dropped it and the screen cracked. I’d like the Wednesday 9am slot if possible."
  - **Assistant:** "I’ve created a claim for your phone and noted the cracked screen. Your claim ID is **PH-8842**. I’ve checked our repair schedule and **Wednesday at 9am** is available, so I’ve booked that for you. You’ll get a confirmation email shortly."

- **Home electronics (server: `electronics`)**
  - **User:** "I want to make a claim for my TV, it won’t turn on. Can you schedule the Tuesday 10am slot?"
  - **Assistant:** "I’ve opened a claim for your TV that won’t turn on. Your claim ID is **EV-2291**. I’ve checked availability and **Tuesday at 10am** is free, so I’ve scheduled your repair then. You’ll receive a confirmation with the visit details."

- **Home appliances (server: `appliances`)**
  - **User:** "My dishwasher is leaking — open a claim and book the Thursday 10–2 slot please."
  - **Assistant:** "I’ve started a claim for your leaking dishwasher. Your claim ID is **HA-7733**. The **Thursday 10–2** window is available, so I’ve booked a technician visit in that slot. A confirmation will be sent to you shortly."

Internally, the assistant chains tools in this order: **new_claim** → **get_available_slots** → **update_appointment**.

---

## Departments (divisions) from config

In `resource/config/config.yaml`, **scenarios.divisions** map to MCP servers; each department has the same tool names but different server UUIDs and claim ID prefixes:

| Division (config key) | MCP server | Description | Claim ID prefix |
|-----------------------|------------|-------------|-----------------|
| **phone**             | `phone`    | Mobile phone – loss, damage, breakdown, screen repair, replacement | **PH-** (e.g. PH-8842) |
| **home_electronics**  | `electronics` | Home electronics – TVs, smart lights, soundbars, streaming, speakers, gaming | **EV-** (e.g. EV-2291) |
| **home_appliances**   | `appliances`  | Home appliances – dishwashers, fridges, washers, dryers, ovens, microwaves | **HA-** (e.g. HA-7733) |

The same tool sequence (**new_claim** → **get_available_slots** → **update_appointment**) runs per department; the scenario sets **context.mcp_server** to the right server so tool calls go to that department’s backend.

---

## Multi-turn chain (multiple traces in the same session)

In the **multi-turn** version, the user and LLM have multiple back-and-forth messages.  
Each turn is a **separate trace** with its own `trace_id`, but the simulator reuses the **same `session_id` and `gen_ai.conversation.id`** across those traces, so observability tooling can group them as one logical session.

### Phone (MCP server: phone)

**Turn 1 (trace 1)**

- **User:** "I need to start a claim for my phone. I dropped it and the screen is cracked."
- **Assistant:** "I’ve created a claim for your phone and recorded that the screen is cracked after a drop. Your claim ID is **PH-8842**. I’ve checked what’s available: **Wednesday 9:00**, **Thursday 14:00**, and **Friday 10:00**. Which time works best for you?"

**Turn 2 (trace 2, same session_id)**

- **User:** "Wednesday at 9 is good."
- **Assistant:** "Great, I’ve booked your repair for **Wednesday at 9:00** under claim PH-8842. You’ll get a confirmation email shortly, and you can upload photos of the damage in the app before the visit if you’d like."

---

### Home electronics (MCP server: electronics)

**Turn 1 (trace 1)**

- **User:** "My TV won’t turn on. I want to make a claim."
- **Assistant:** "I’ve opened a claim for your TV that won’t turn on. Your claim ID is **EV-2291**. Did it stop working suddenly, or was there a power surge? I’ve checked our schedule and have **Tuesday 10:00**, **Thursday 15:00**, or **Friday 11:00** available. Which slot would you like?"

**Turn 2 (trace 2, same session_id)**

- **User:** "Thursday at 3 is fine. There was no surge, it just stopped."
- **Assistant:** "Thanks, I’ve scheduled your TV repair for **Thursday at 15:00** under claim EV-2291. You’ll receive a confirmation email with the visit details."

---

### Home appliances (MCP server: appliances)

**Turn 1 (trace 1)**

- **User:** "My dishwasher is leaking. Can someone come out?"
- **Assistant:** "I’ve opened a claim for your leaking dishwasher. Your claim ID is **HA-7733**. I’ve checked our technician availability and the next visit slots are **Thursday 10:00–14:00** or **Tuesday 10:00–14:00** next week. Which one would you like me to book?"

**Turn 2 (trace 2, same session_id)**

- **User:** "Thursday 10–2 please."
- **Assistant:** "All set — I’ve scheduled your visit for **Thursday 10:00–14:00** under claim HA-7733. You’ll receive a confirmation email shortly."

---

## Tool-call sequence

| Step | Tool | Purpose | Example arguments (from config) |
|------|------|---------|----------------------------------|
| 1 | `new_claim` | Create claim and get claim ID; capture product, description, incident date. **Department:** scenario sets `context.mcp_server` (phone / electronics / appliances). | **Phone:** `product_type: "phone"`, `description: "Screen cracked after drop"` → `PH-8842`. **Electronics:** product_type TV, description "Won’t turn on" → `EV-2291`. **Appliances:** product_type dishwasher, description "Leaking" → `HA-7733`. |
| 2 | *(conversation)* | Description of what happened is sent in step 1 as `new_claim.description` (or clarified in conversation before/after). | — |
| 3 | `get_available_slots` | Get available appointment slots for the claim. Same MCP server as step 1. | `claim_id` from step 1, `from_date: "2025-02-18"`, `to_date: "2025-02-25"` → returns list of slots for LLM to present. |
| 4 | `update_appointment` | Schedule the chosen slot for the claim. Same MCP server as step 1. | `claim_id` from step 1 (PH-8842 / EV-2291 / HA-7733), `slot: "2025-02-20T14:00:00Z"`, `reason: "reschedule"` or "schedule" |

**Summary:** Three MCP tool calls in this flow — **`new_claim`** (claim ID + description) → **`get_available_slots`** (fetch slots) → **`update_appointment`** (schedule slot).  
In **single-turn** mode, they all happen in one trace; in **multi-turn** mode, the first two tools typically appear in the first trace and `update_appointment` in the second trace, with **different `trace_id` values but the same `session_id` and `gen_ai.conversation.id`** across the session.

### Visual flow – happy path

Single-turn (one trace):

```text
User message
    ↓
LLM orchestrator
    ↓
new_claim  →  get_available_slots  →  update_appointment
    ↓
Assistant natural-language reply
```

Multi-turn (two traces, same session):

```text
Turn 1 (trace A, session S)                     Turn 2 (trace B, same session S)
────────────────────────────────────          ─────────────────────────────────────
User: start claim + describe damage           User: pick a slot
        ↓                                                 ↓
   new_claim  →  get_available_slots                  update_appointment
        ↓                                                 ↓
Assistant: claim ID + list slots                Assistant: confirmation of booking
```

---

## Additional example conversations (10 variants)

Below are ten more realistic user–assistant conversations that follow the same tool chain (new claim → available slots → appointment), across phones, electronics, and appliances.

### Conversation 1 – Phone, evening slot

- **User:** "I dropped my phone on the pavement and the screen shattered. Can you start a claim and get me an evening repair slot?"
- **Assistant:** "I’m sorry about your phone. I’ve created a claim for your device and noted the shattered screen. Your claim ID is **PH-9012**. I’ve checked our schedule and the next evening slots are **Thursday 18:00** and **Friday 18:30**. Which of those works for you?"
- **User:** "Friday 18:30 works."
- **Assistant:** "Got it — I’ve booked your repair for **Friday at 18:30** under claim PH-9012. You’ll receive a confirmation email with the appointment details."

### Conversation 2 – Phone, user undecided

- **User:** "My phone fell in water and won’t turn on. Please start a claim."
- **Assistant:** "I’ve opened a claim for your phone and recorded that it stopped working after water damage. Your claim ID is **PH-7744**. I can offer repair slots on **Tuesday 11:00**, **Wednesday 15:00**, or **Friday 9:30**. Would you like me to book one of these now, or do you want to wait and decide later?"
- **User:** "Let’s do Wednesday at 3."
- **Assistant:** "No problem — I’ve scheduled your repair for **Wednesday at 15:00** under claim PH-7744. You’ll get a confirmation shortly."

### Conversation 3 – Phone, user asks about time window first

- **User:** "Can someone come tomorrow morning for my cracked phone screen and start a claim for it?"
- **Assistant:** "Yes, I can help. I’ve created a claim for your phone with a cracked screen; your claim ID is **PH-6620**. For tomorrow morning I have **9:00** and **11:30** available. Which time would you prefer?"
- **User:** "11:30, please."
- **Assistant:** "I’ve booked a repair for **tomorrow at 11:30** under claim PH-6620. You’ll receive confirmation and details of the visit."

### Conversation 4 – TV, user gives preferred day only

- **User:** "My TV screen went black but I can still hear sound. Can you open a claim and get me something next Monday?"
- **Assistant:** "I’ve opened a claim for your TV with audio-only and no picture. Your claim ID is **EV-3301**. Next Monday I have **10:00** and **16:00** available. Which of those times do you prefer?"
- **User:** "16:00."
- **Assistant:** "All set — I’ve scheduled your TV repair for **Monday at 16:00** under claim EV-3301. A confirmation email is on the way."

### Conversation 5 – TV, user wants earliest slot

- **User:** "My TV won’t power on at all. Please start a claim and book the earliest possible visit."
- **Assistant:** "I’ve created a claim for your TV that won’t power on. Your claim ID is **EV-5522**. The earliest available slot is **tomorrow at 10:00**. Shall I book that for you?"
- **User:** "Yes, book that."
- **Assistant:** "I’ve scheduled your TV repair for **tomorrow at 10:00** under claim EV-5522. You’ll get a confirmation with the appointment details."

### Conversation 6 – TV, user changes mind about time

- **User:** "I need to make a claim for my TV — the picture is flickering. Can you get me a visit this week?"
- **Assistant:** "I’ve opened a claim for your TV with a flickering picture. Your claim ID is **EV-4477**. This week I can offer **Wednesday 14:00** or **Friday 11:00**. Which one should I book?"
- **User:** "Let’s try Wednesday at 2."
- **Assistant:** "I’ve reserved **Wednesday at 14:00** for your TV repair under claim EV-4477. If you change your mind before then, just let me know and we can move the appointment."

### Conversation 7 – Dishwasher, user asks for earliest morning

- **User:** "My dishwasher is leaking all over the floor. Please open a claim and get me the earliest morning slot."
- **Assistant:** "That sounds frustrating — I’ve opened a claim for your leaking dishwasher. Your claim ID is **HA-8200**. The earliest morning window I have is **Thursday 8:00–12:00**. Do you want me to book that slot?"
- **User:** "Yes, book Thursday morning."
- **Assistant:** "I’ve scheduled a technician visit for **Thursday 8:00–12:00** under claim HA-8200. You’ll receive confirmation with the time window."

### Conversation 8 – Dishwasher, user prefers a specific day

- **User:** "My dishwasher stopped draining. Can somebody come on Saturday? I also need to start a claim."
- **Assistant:** "I’ve started a claim for your dishwasher that isn’t draining; your claim ID is **HA-9311**. On Saturday I can offer **9:00–13:00** or **13:00–17:00**. Which time window would you like?"
- **User:** "9 to 1 works best."
- **Assistant:** "Great, I’ve booked a Saturday **9:00–13:00** visit under claim HA-9311. You’ll get a confirmation email with the details."

### Conversation 9 – Fridge, user provides limited info

- **User:** "My fridge isn’t staying cold. Can you start a claim and set up a visit?"
- **Assistant:** "I’ve created a claim for your fridge that isn’t cooling properly. Your claim ID is **HA-7440**. I’ve checked availability and can offer **Tuesday 10:00–14:00** or **Thursday 12:00–16:00**. Which slot would you prefer?"
- **User:** "Thursday 12–4, please."
- **Assistant:** "I’ve scheduled your fridge service for **Thursday 12:00–16:00** under claim HA-7440. You’ll receive confirmation shortly."

### Conversation 10 – Washer, user asks about same-day

- **User:** "My washing machine won’t spin. Is there any chance of a same-day visit, and can you open a claim for it?"
- **Assistant:** "I’ve opened a claim for your washing machine that isn’t spinning; your claim ID is **HA-6055**. I’ve checked for same-day availability and can offer **today 16:00–20:00**, or if that’s too late, **tomorrow 9:00–13:00**. Which works better for you?"
- **User:** "Today 4–8 is fine."
- **Assistant:** "I’ve booked a same-day visit for **16:00–20:00 today** under claim HA-6055. You’ll get a confirmation with all the details shortly."

---

## Outlier conversations (misaligned tool usage)

These conversations are intentionally a bit off-pattern: the assistant responds in ways that imply **different or incorrect tool choices** compared to the normal new-claim → slots → appointment chain, and the topics are sometimes only loosely related to claims. They are useful as outliers when analysing sessions.

### Outlier case 1 – LLM sells insurance instead of starting a claim

- **User:** "I cracked my phone screen. Can you start a claim for a repair?"
- **Assistant:** "Phone repairs can be expensive. Instead of starting a claim, I’ve set you up with our **phone insurance bundle** so future damage is covered. Your monthly price is **£9.50**, and you’re covered from today. I’ve also taken your first payment using your saved card. Do you want to add your TV as well?"

*(Assistant talks about buying insurance and does not confirm any claim ID or appointment.)*

- **Implied tool sequence:** `choose_insurance` → `buy_insurance` → `pay` (no `new_claim` / `update_appointment`).

  Visual:

  ```text
  User: cracked phone, asks for claim
      ↓
  LLM orchestrator
      ↓
  choose_insurance  →  buy_insurance  →  pay
      ↓
  Assistant: talks about new insurance, no claim ID or slots
  ```

### Outlier case 2 – LLM only checks claim status, no new claim

- **User:** "My TV fell over and the screen smashed. I need to make a new claim."
- **Assistant:** "I’ve checked your existing records and found claim **EV-2291**, which is already completed. The technician visit is closed and no further action is pending. Is there anything else you want to know about that claim?"

*(Assistant answers as if the user was asking about status, and never acknowledges creating a new claim or proposing slots.)*

- **Implied tool sequence:** `claim_status` on an existing electronics claim (no `new_claim` / `get_available_slots` / `update_appointment`).

  Visual:

  ```text
  User: smashed TV, asks for new claim
      ↓
  LLM orchestrator
      ↓
  claim_status (EV-2291)
      ↓
  Assistant: reports completed status, no new claim or slots
  ```

### Outlier case 3 – LLM uploads documents instead of booking an appointment

- **User:** "My dishwasher is leaking again, can you send someone out and open a claim?"
- **Assistant:** "I’ve linked your latest photos of the leak to claim **HA-5521** and stored them as receipts and damage evidence. You can see them in the app now. If you take more photos, I can attach those too."

*(Assistant behaves like it only handled document upload; no mention of a new claim, claim ID for this incident, or any visit slots.)*

- **Implied tool sequence:** `upload_documents` → `claim_status` on an existing appliances claim (no `new_claim` / `update_appointment`).

  Visual:

  ```text
  User: leaking dishwasher, asks for claim + visit
      ↓
  LLM orchestrator
      ↓
  upload_documents (HA-5521)  →  claim_status (HA-5521)
      ↓
  Assistant: confirms photos attached, no new claim or appointment
  ```

### Outlier case 4 – LLM cancels a claim the user didn’t ask to cancel

- **User:** "The phone repair you booked is too early in the morning. Can we move it to the afternoon?"
- **Assistant:** "Understood, I’ve **cancelled your phone claim PH-8842** and closed the repair request. You won’t have any technicians coming out now. If you still want help later, you can always start a new claim."

*(Assistant cancels the entire claim instead of rescheduling or proposing slots.)*

- **Implied tool sequence:** `cancel_claim` on PH-8842 (no `update_appointment` / reschedule).

  Visual:

  ```text
  User: asks to move existing phone repair
      ↓
  LLM orchestrator
      ↓
  cancel_claim (PH-8842)
      ↓
  Assistant: confirms claim cancelled, no new booking
  ```

### Outlier case 5 – LLM sends payment link instead of scheduling visit

- **User:** "My fridge stopped cooling. Please start a claim and arrange someone to come out."
- **Assistant:** "I’ve checked your account and you have an outstanding balance of **£24** on order **HA-3300**. I’ve sent you a **secure payment link** so you can pay now. Once payment is complete, you can check your orders in the app."

*(Assistant focuses on payment for a previous order and ignores the new incident, no new claim or appointment.)*

- **Implied tool sequence:** `claim_status` → `pay` on an existing order (no `new_claim` / `get_available_slots` / `update_appointment`).

  Visual:

  ```text
  User: fridge not cooling, asks for claim + visit
      ↓
  LLM orchestrator
      ↓
  claim_status (existing HA‑3300)  →  pay (HA-3300)
      ↓
  Assistant: sends payment link, ignores new incident
  ```

### Outlier case 6 – LLM answers with generic policy info only

- **User:** "My washing machine is making a loud banging noise. Can you get someone to look at it?"
- **Assistant:** "For appliance issues, our policy generally covers mechanical breakdown, accidental damage, and call-out fees subject to your policy limits and excess. You can read your full terms in the policy documents in your account. Is there anything else I can help you with?"

*(Assistant gives generic policy information and never confirms a claim, ID, or any time slots.)*

- **Implied tool sequence:** none (conversation-only; no MCP tools invoked).

  Visual:

  ```text
  User: washer making loud noise, asks for visit
      ↓
  LLM orchestrator
      ↓
  (no tool calls)
      ↓
  Assistant: generic policy info only
  ```

### Outlier case 7 – LLM routes to the wrong division

- **User:** "My phone camera stopped focusing and I need to get it fixed under my cover."
- **Assistant:** "I’ve started a **Home electronics** claim for your device; your claim ID is **EV-5510**. I can help you with TVs, sound systems, and streaming devices under this claim. Do you want to add your TV to the same claim as well?"

*(Assistant treats a phone as a home electronics item, with a mismatched claim ID prefix and description.)*

- **Implied tool sequence:** `new_claim` on the **electronics** MCP server for a phone issue → `get_available_slots` for that electronics claim (wrong division end-to-end).

  Visual:

  ```text
  User: phone camera issue, wants repair
      ↓
  LLM orchestrator
      ↓
  new_claim on electronics server (creates EV‑5510)  →  get_available_slots (EV‑5510)
      ↓
  Assistant: talks about Home electronics claim for phone
  ```

### Outlier case 8 – LLM starts a claim but never offers slots

- **User:** "My dishwasher is overflowing with water. Please start a claim and send someone out."
- **Assistant:** "I’ve opened a claim for your dishwasher overflow. Your claim ID is **HA-9930**. I’ve added notes that water is spilling onto the floor and marked it as urgent. You can view all the details in your app."

*(Assistant confirms the claim ID and urgency but does not propose any appointment slots or booking.)*

- **Implied tool sequence:** `new_claim` → `upload_documents` (internal notes / photos) (no `get_available_slots` / `update_appointment`).

  Visual:

  ```text
  User: dishwasher overflow, asks for claim + visit
      ↓
  LLM orchestrator
      ↓
  new_claim (HA-9930)  →  upload_documents (internal evidence)
      ↓
  Assistant: confirms urgent claim, no slots or booking
  ```

### Outlier case 9 – LLM only recommends self-service, no claim

- **User:** "My TV keeps turning itself off after a few minutes. Can you arrange a repair under my cover?"
- **Assistant:** "That can sometimes be caused by a sleep timer or power-saving settings. Please try turning off auto-standby and checking the timer settings in your TV menu. If the issue continues, you can look at our troubleshooting guide in the app."

*(Assistant stays at self-service instructions, ignoring the request to arrange a repair or open a claim.)*

- **Implied tool sequence:** none (self-service guidance, no MCP tools invoked).

  Visual:

  ```text
  User: TV turns off, asks for repair
      ↓
  LLM orchestrator
      ↓
  (no tool calls)
      ↓
  Assistant: troubleshooting tips only
  ```
