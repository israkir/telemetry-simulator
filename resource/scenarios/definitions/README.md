# Scenario Definitions

This folder contains **scenario definitions** used by the telemetry simulator to generate traces that look like real product traffic. Each scenario describes a **user journey** (or system outcome) and the **telemetry** we emit so that dashboards, alerts, and ML pipelines see a realistic mix of success, retries, errors, and control-plane decisions.

The document below lists every scenario, grouped by **intention**, with enough detail for observability and usage decisions (e.g. which scenarios to run, how often, and what they represent).

---

## How scenarios are used

- **Single scenario:** `otelsim scenario --name <scenario_name>` runs one scenario only.
- **Mixed workload:** `otelsim run` picks scenarios at random; probability is proportional to **workload_weight** (higher weight = more frequent). This produces a traffic mix that resembles production (mostly success, some retries and higher latency, fewer blocks and errors).
- **Filtering by tags:** `otelsim run --tags=data-plane` or `--tags=control-plane` limits to scenarios with those tags for batch running (e.g. all data-plane, all agent-confusion, or all partial-workflow scenarios).

### Workload weight

**Workload weight** is a per-scenario, non-negative number that controls how often that scenario is chosen when you run a **mixed workload** (`otelsim run`). It is **not** used when you run a single scenario (`otelsim scenario --name ...`) or when you use `--each-once` (which runs each matching scenario exactly once and ignores weights).

- **How selection works:** On each “request” in a mixed run, the simulator picks one scenario at random with probability proportional to its weight. So a scenario with weight **10.0** is about **10× more likely** to be chosen than one with weight **1.0**; with weight **0.5** it is half as likely as weight 1.0.
- **Default:** If a scenario does not set `workload_weight`, it defaults to **1.0**.
- **Zero or negative:** A weight ≤ 0 effectively removes the scenario from random selection in mixed runs (you can still run it explicitly with `otelsim scenario --name ...`).
- **Why use it:** Weights let you shape the mix to look realistic—e.g. high weight for the main happy path (10.0), lower for multi-turn or audit-flagged (2.5–3), and much lower for rare events like policy errors or blocks (0.1–0.5)—so generated traffic resembles production.

**Example:** Suppose only three scenarios are loaded, with weights **10**, **2**, and **0.5**. The total weight is 12.5. On each pick, the probability of choosing each scenario is:


| Scenario   | Weight | Probability (weight ÷ total) | In 100 runs (approx.) |
| ---------- | ------ | ---------------------------- | --------------------- |
| Happy path | 10     | 10 ÷ 12.5 = **80%**          | ~80                   |
| Multi-turn | 2      | 2 ÷ 12.5 = **16%**           | ~16                   |
| Rare error | 0.5    | 0.5 ÷ 12.5 = **4%**          | ~4                    |


So the happy path dominates, with a few multi-turn and only a small number of error cases—similar to real traffic.

---

## 1. Data-plane — Happy path (successful user flows)

These scenarios simulate **successful** agent flows: the user asks for something, the agent calls the right tool(s), and the tool succeeds. They represent the majority of “good” traffic.


| Scenario                    | Division   | Intention                                                                                                                                                                                   | Product context                                                                                                              | Tags |
| --------------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- | ----- |
| **new_claim_phone**         | Phone      | **Primary happy path.** User starts a phone claim; agent runs planner → task → tool recommendation → `new_claim` → response. One turn, one session, one random conversation sample per run. | Bulk of “create claim” traffic. Use for baseline latency and success-rate expectations. **Workload weight: 10.0** (highest). | data-plane, happy-path |
| **cancel_claim_appliances** | Appliances | User asks to cancel an appliances claim with a **valid claim ID** (e.g. HA-7733); agent calls `cancel_claim` with valid reason; tool succeeds and assistant confirms.                       | Represents successful cancellation flows (dishwasher, fridge, washer, etc.). **Workload weight: 2.0.**                       | data-plane, happy-path, cancel-claim, appliances |


---

## 2. Data-plane — Multi-turn and retries

These represent **multi-turn conversations** or **transient failures followed by success**. Important for testing correlation (session/conversation IDs) and retry/backoff behavior.


| Scenario                                        | Division | Intention                                                                                                                                                                              | Product context                                                                                                                                                  | Tags |
| ----------------------------------------------- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----- |
| **new_claim_phone_multi_turn**                  | Phone    | **Multi-turn new claim.** Same session has 3 user/assistant turns; each turn produces one trace (one orchestration tree). Same `session_id` and `gen_ai.conversation.id` across turns. | Use to validate multi-turn correlation and conversation-scoped metrics. **Workload weight: 2.5.**                                                                | data-plane, happy-path |
| **new_claim_phone_mcp_tool_retry_then_success** | Phone    | **Retry then success.** Agent calls `new_claim`; first attempt **times out**, second attempt **succeeds**. Traces show MCP retry (attempt children, retry count, step outcome).        | Validates that retries and timeouts are visible in telemetry and that “success after retry” is distinguishable from first-try success. **Workload weight: 1.0.** | data-plane, mcp-retry, retry-then-success |


---

## 3. Data-plane — Higher latency (degraded but successful)

These scenarios simulate **successful** tool calls that take **longer** (e.g. peak hours, specific zip codes, or backend conditions). They help calibrate latency SLOs and “slow but OK” vs “failure” signals.


| Scenario                                           | Division    | Intention                                                                                                                                                                                                | Product context                                                                                    | Tags |
| -------------------------------------------------- | ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- | ----- |
| **claim_status_phone_higher_latency**              | Phone       | `claim_status` is slower when the status is “technician on-route to scheduled appointment.” Each run emits two traces: one normal reply, one technician-on-route style; condition is on span attributes. | Represents status checks that hit a slower backend path (e.g. live ETA). **Workload weight: 0.7.** | data-plane, higher-latency, claim-status |
| **update_appointment_phone_higher_latency**        | Phone       | `update_appointment` has higher latency when scheduling in **zip 90210**. Condition (e.g. zip_code) is in the scenario so traces can be filtered.                                                        | Tests latency by “region” or zip for scheduling. **Workload weight: 0.7.**                         | data-plane, higher-latency, scheduling |
| **update_appointment_electronics_higher_latency**  | Electronics | Same as above for **electronics** (TV, laptop, etc.) in **zip 10001**.                                                                                                                                   | Same intention as phone variant, different division. **Workload weight: 0.7.**                     | data-plane, higher-latency, scheduling, electronics |
| **cancel_claim_appliances_higher_latency**         | Appliances  | `cancel_claim` is slower when the claim has **an appointment already scheduled** (e.g. cancelling a booked visit). Condition (e.g. `appointment_scheduled: true`) is on span attributes.                 | Represents “cancel + release appointment” backend path. **Workload weight: 0.7.**                  | data-plane, higher-latency, cancel-claim, appliances |
| **generic_higher_latency_peak_hours**              | Phone       | Tool calls are slower during **peak hours** (e.g. weekdays 9am–2pm PT) or **post long weekend**. Condition is defined in the scenario.                                                                   | Tests time-of-day or calendar-based latency behavior. **Workload weight: 0.7.**                    | data-plane, higher-latency, peak-hours |
| **new_claim_appliances_higher_latency_peak_hours** | Appliances  | Same peak-hours / post–long-weekend condition for **new_claim** in appliances (dishwasher, fridge, washer).                                                                                              | Same intention as generic peak-hours, applied to new-claim appliances. **Workload weight: 0.7.**   | data-plane, higher-latency, peak-hours, appliances |


---

## 4. Data-plane — Tool 4xx (invalid parameters)

These simulate **correct tool, wrong or missing parameters** → backend returns **4xx** (e.g. 400, 404, 422). They exercise error handling, user messaging, and observability of “user/agent mistake” vs system failure.


| Scenario                                            | Division    | Intention                                                                                                                                                                                                | Product context                                                                                           | Tags |
| --------------------------------------------------- | ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- | ----- |
| **new_claim_appliances_tool_4xx_invalid_params**    | Appliances  | Agent calls `new_claim` with **wrong or missing parameters** (e.g. date of loss in wrong format, missing required field). One MCP attempt gets `invalid_arguments` and 4xx.                              | Tests validation errors on claim creation (date format, required fields). **Workload weight: 0.5.**       | data-plane, 4xx, invalid-params, entity-format, appliances |
| **new_claim_electronics_tool_4xx_invalid_params**   | Electronics | Same as above for **electronics** (TV, laptop, etc.): wrong date/format or missing fields → 4xx.                                                                                                         | Same intention, electronics division. **Workload weight: 0.5.**                                           | data-plane, 4xx, invalid-params, entity-format, electronics |
| **claim_status_phone_tool_4xx_invalid_params**      | Phone       | Agent calls `claim_status` with **malformed claim ID** (e.g. "8842" instead of "PH-8842"). Backend returns 4xx; exception reflects invalid claim ID format.                                              | Tests “user gave only the number” or wrong prefix for status checks. **Workload weight: 0.5.**            | data-plane, 4xx, invalid-params, claim-id-format, phone |
| **claim_status_appliances_tool_4xx_invalid_params** | Appliances  | Same as above for **appliances**: claim ID without division prefix (e.g. "7733" vs "HA-7733") → 4xx.                                                                                                     | Same intention, appliances. **Workload weight: 0.5.**                                                     | data-plane, 4xx, invalid-params, claim-id-format, appliances |
| **cancel_claim_appliances_tool_4xx_invalid_params** | Appliances  | Agent calls `cancel_claim` with **invalid or missing reason**. Backend expects a reason from an allowed set (e.g. customer_requested, duplicate, no_longer_needed); user gives free text or empty → 4xx. | Tests validation of cancel reason (different 4xx context than claim ID format). **Workload weight: 0.5.** | data-plane, 4xx, invalid-params, cancel-reason, cancel-claim, appliances |


---

## 5. Data-plane — Agent confusion (wrong division / wrong tool)

These scenarios simulate **agent disambiguation failures**: the user’s intent maps to one division or tool, but the agent calls a **different division** (same tool name, wrong `mcp.server.uuid` / `mcp.tool.uuid`) or a **different tool** (e.g. `cancel_product` instead of `cancel_claim`) because user requests are ambiguous and tool names/descriptions are similar across divisions. Useful for testing routing metrics, division/tool confusion detection, and observability of “wrong MCP” or “wrong tool” traffic.

**Errors and exceptions in the flow:**

- **Wrong-division scenarios** (pay, cancel_claim): The flow **includes failure**. The modifier sets the MCP tool call to the wrong division’s server/tool UUID, then marks the call as **failed**: parent MCP span has `step.outcome=fail`, the attempt span has `mcp.attempt.outcome=fail`, `error.type=tool_error`, and status=ERROR. Optional `scenario_overrides.exception_type` / `exception_message` add an exception event on the attempt (e.g. `WrongDivisionError`, “Claim not found in this division”) so traces show a concrete backend error.
- **Tool-confusion scenario** (cancel_product_instead_of_claim): The agent called the wrong tool and the backend **succeeds**. **response_compose** is marked fail (goal: ungrounded_response) with WrongToolError so wrong-tool traffic is visible in error metrics. Use to test “silent” wrong-tool traffic. To simulate “wrong tool → backend rejected,” add a separate scenario with a 4xx or error goal.

| Scenario                                                          | Division / tool                         | Intention                                                                                                                                                                                                                              | Product context                                                                                                        | Tags |
| ----------------------------------------------------------------- | --------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | ----- |
| **agent_confusion_pay_phone_wrong_division_electronics**          | Phone (correct) → Electronics (emitted) | User wants to **pay for a phone claim**; request is ambiguous. Agent routes to **electronics** MCP; trace shows electronics `pay` via `goal: wrong_division`. MCP call **fails** with `step.outcome=fail`, `error.type=tool_error`, and optional exception event. | Same input cluster (pay) → same tool name across divisions; agent picks wrong division. **Workload weight: 0.8.**      | data-plane, division-confusion, wrong-division, agent-confusion, pay |
| **agent_confusion_cancel_claim_phone_wrong_division_electronics** | Phone (correct) → Electronics (emitted) | User wants to **cancel a phone claim**; request is ambiguous. Agent routes to **electronics** MCP; trace shows electronics `cancel_claim` via `goal: wrong_division`. MCP call **fails** with `step.outcome=fail`, `error.type=tool_error`, and optional exception event. | Similar tool names and wording across divisions; agent picks wrong division. **Workload weight: 0.6.**                 | data-plane, division-confusion, wrong-division, agent-confusion, cancel-claim |
| **agent_confusion_cancel_product_instead_of_claim**               | Electronics                             | User says **“cancel my claim”** (intent: `cancel_claim`) but agent calls **`cancel_product`**. Trace shows `cancel_product` succeeding; **response_compose** is marked fail (WrongToolError) so wrong-tool traffic is visible in error metrics. | Same input cluster → similar tools (cancel_claim vs cancel_product); agent picks wrong tool. **Workload weight: 0.5.** | data-plane, tool-confusion, agent-confusion, cancel-product, cancel-claim |
| **agent_confusion_new_claim_partial_tools**                       | Phone                                   | User wants to **start a phone claim**; expected flow is planner → task → tools_recommend → **new_claim** → response_compose. Agent runs planner, task, tools_recommend, then response_compose **without calling new_claim**. Trace has no MCP tool execution; response_compose fail. Cause: no context about sequence. | Partial set of tools in new-claim workflow. **Workload weight: 0.5.**                                                 | data-plane, agent-confusion, partial-workflow, new-claim |
| **agent_confusion_new_claim_wrong_tool_order**                    | Phone                                   | User wants to **start a phone claim**; expected flow has **new_claim** as the (only) tool. Agent calls **update_appointment** first, then new_claim. Trace shows two MCP tool executions in that order; response_compose fail. Cause: no context about sequence. | First tool called is not the first in expected sequence. **Workload weight: 0.5.**                                     | data-plane, agent-confusion, partial-workflow, wrong-order, new-claim |


---

## 6. Control-plane — Allowed but flagged for audit

Request is **allowed** to proceed, but policy **flags it for audit**. Full data-plane flow runs (orchestration, tools, response); root has audit flag set. Used for “allowed but reviewed” traffic mix.


| Scenario                          | Division | Intention                                                                                                                                                                                                       | Product context                                                                                     | Tags |
| --------------------------------- | -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- | ----- |
| **request_allowed_audit_flagged** | Phone    | Policy **allows** the request but sets **audit flag** (e.g. for later review). Root: `outcome=allowed`, `audit.flag=true`; policy span: `decision=allow_with_audit`. Data-plane trace is full happy-path style. | Represents “sensitive but permitted” requests that need audit visibility. **Workload weight: 3.0.** | control-plane |


---

## 7. Control-plane — Blocked (request never reaches data-plane)

Request is **blocked** before or during validation; **no** data-plane orchestration. Different scenarios represent different **block reasons** for filtering and analytics.


| Scenario                                              | Intention                                                                                                                                                                                                            | Product context                                                                                             | Tags |
| ----------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- | ----- |
| **request_blocked_by_policy**                         | Policy **denies** the request (e.g. policy rule). Root: `outcome=blocked`, `block.reason=request_policy`. No data-plane.                                                                                             | Most common “block” type in the mix. **Workload weight: 0.5.**                                              | control-plane |
| **request_blocked_invalid_payload**                   | Request **fails payload validation** (e.g. malformed or invalid structure). Root: `outcome=blocked`, `block.reason=invalid_payload`.                                                                                 | Bad or malformed incoming request. **Workload weight: 0.25.**                                               | control-plane |
| **request_blocked_invalid_payload_multi**             | Same as above but with **multiple validation errors**; payload span has multiple validation error events (path, rule, code).                                                                                         | Same as single invalid payload, with richer error details for debugging. **Workload weight: 0.15.**         | control-plane |
| **request_blocked_rate_limited**                      | Request is **rate limited** early. Root: `outcome=blocked`, `block.reason=rate_limited`. Only root and payload spans (no policy or augmentation).                                                                    | Tests rate-limit visibility and impact. **Workload weight: 0.12.**                                          | control-plane |
| **request_blocked_policy_fail_closed**                | **Policy engine fails** (e.g. timeout); system is **fail-closed** so request is **blocked**. Root: `outcome=blocked`, `block.reason=request_policy`, policy span has ERROR and exception (e.g. PolicyEngineTimeout). | Ensures “policy failure → block” is distinguishable from “policy decided block.” **Workload weight: 0.12.** | control-plane |
| **request_blocked_invalid_context_augment_exception** | **Augmentation step fails** (e.g. context binding error); request blocked with `block.reason=invalid_context`. Augmentation span has failure/exception.                                                              | Tests visibility of context/augmentation failures. **Workload weight: 0.1.**                                | control-plane |


---

## 8. Control-plane — Error (request outcome = error)

Validation or policy results in **request.outcome=error** (e.g. policy engine exception), not “blocked.” Used to test error vs block semantics and alerting.


| Scenario                             | Intention                                                                                                                                                 | Product context                                                                                   | Tags |
| ------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | ----- |
| **request_error_policy_runtime**     | **Policy engine throws** at runtime (e.g. timeout). Request `outcome=error`; policy span has status=ERROR and exception event (e.g. PolicyEngineTimeout). | Distinguishes “policy error” from “policy block” in metrics and traces. **Workload weight: 0.2.** | control-plane |
| **request_error_policy_unavailable** | Same pattern as above but exception is **PolicyEngineUnavailable** (policy service down or unreachable).                                                  | Variant of policy failure for dependency/availability monitoring. **Workload weight: 0.15.**      | control-plane |


---

## 9. Reference only (not part of standard mix)


| Scenario               | Intention                                                                                                                                       | Product context                                                                                 | Tags |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- | ----- |
| ***EXAMPLE_SCENARIO*** | **Template/reference** only. Documents every supported YAML option. Excluded from listing and mixed workload; use when authoring new scenarios. | Not for product mix; for scenario authors. **Workload weight: 1.0** (irrelevant when excluded). | example, data-plane, happy-path |


---

## Summary table (by tag)


| Tag                                                               | Scenarios (examples)                                                                                                | Use for                                          |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| **data-plane**                                                    | All new_claim, claim_status, cancel_claim, update_appointment, pay, cancel_product, agent_confusion_*               | Agent/MCP/tool traces, latency, success/4xx      |
| **control-plane**                                                 | request_blocked_*, request_error_*, request_allowed_audit_flagged                                                   | Validation, policy, blocks, errors               |
| **happy-path**                                                    | new_claim_phone, cancel_claim_appliances                                                                            | Success baselines                                |
| **multi-turn**                                                    | new_claim_phone_multi_turn                                                                                          | Conversation correlation                         |
| **higher-latency**                                                | *_higher_latency*, *_peak_hours                                                                                     | Latency SLOs, slow-path behavior                 |
| **4xx** / **invalid-params**                                      | *_tool_4xx_invalid_params                                                                                           | Tool validation errors                           |
| **division-confusion** / **wrong-division** / **agent-confusion** | agent_confusion_pay_phone_wrong_division_electronics, agent_confusion_cancel_claim_phone_wrong_division_electronics | Wrong MCP division (same tool, wrong server)     |
| **tool-confusion**                                                | agent_confusion_cancel_product_instead_of_claim                                                                     | Wrong tool (e.g. cancel_product vs cancel_claim) |
| **partial-workflow** / **wrong-order**                            | agent_confusion_new_claim_partial_tools, agent_confusion_new_claim_wrong_tool_order                                 | Partial tool set or wrong tool order (no sequence context) |


